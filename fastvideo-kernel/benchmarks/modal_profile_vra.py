from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "fastvideo-vra-kernel-profile"

REPO_MOUNT = Path("/FastVideo")
RESULTS_ROOT = Path("/results")

results_volume = modal.Volume.from_name("fastvideo-vra-kernel-results",
                                        create_if_missing=True)


def _local_repo_for_mount() -> Path:
    here = Path(__file__).resolve()
    try:
        return here.parents[2]
    except IndexError:
        return REPO_MOUNT


_GPU = os.environ.get("MODAL_VRA_GPU", "H100!")
_REPO_ROOT = _local_repo_for_mount()

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    ).apt_install(
        "git",
        "build-essential",
    ).env({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "9.0a",
    }).run_commands(
        "apt-get update && (apt-get install -y nsight-systems-cli || apt-get install -y nsight-systems-2025.6.3 || apt-get install -y nsight-systems-2024.6.2 || true)",
        "python -m pip install -U pip setuptools wheel",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio",
        "python -m pip install pytest numpy einops scikit-build-core cmake ninja",
    ).add_local_dir(
        str(_REPO_ROOT),
        remote_path=str(REPO_MOUNT),
        copy=True,
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.pytest_cache/**",
            "**/*.npz",
            "**/outputs_video/**",
        ],
    ).run_commands(
        "cd /FastVideo/fastvideo-kernel && CC=gcc CXX=g++ "
        "CUDACXX=/usr/local/cuda/bin/nvcc "
        "CMAKE_ARGS='-DFASTVIDEO_KERNEL_BUILD_TK=ON -DCMAKE_CUDA_ARCHITECTURES=90a' "
        "python -m pip install . -v --no-build-isolation",
    ))

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=_GPU,
    volumes={str(RESULTS_ROOT): results_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
def run_vra_kernel_profile(
    seq_shape: str,
    dense_radius: str,
    mid_radius: str,
    num_heads: int,
    head_dim: int,
    target: str,
    tool: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["FASTVIDEO_KERNEL_USE_INSTALLED"] = "1"
    env["PYTHONPATH"] = (
        f"{REPO_MOUNT / 'fastvideo-kernel'}:"
        f"{env.get('PYTHONPATH', '')}")
    kernel_root = REPO_MOUNT / "fastvideo-kernel"

    if tool == "nsys":
        env["FASTVIDEO_VRA_DISABLE_CUDA_PROFILER_API"] = "1"
        nsys = subprocess.run(
            ["bash", "-lc", "command -v nsys"],
            cwd=str(kernel_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if nsys.returncode != 0:
            print("nsys not found in the CUDA image")
            return {"status": "missing-nsys"}

        stem = (
            f"nsys_{target}_{seq_shape}_{dense_radius}_{mid_radius}_h{num_heads}_d{head_dim}"
            .replace(",", "-"))
        out_prefix = RESULTS_ROOT / stem
        out_file = RESULTS_ROOT / f"{stem}.txt"
        cmd = [
            nsys.stdout.strip(),
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--force-overwrite=true",
            "-o",
            str(out_prefix),
            "python",
            "benchmarks/profile_vra_kernel_once.py",
            "--seq-shape",
            seq_shape,
            f"--dense-radius={dense_radius}",
            f"--mid-radius={mid_radius}",
            "--num-heads",
            str(num_heads),
            "--head-dim",
            str(head_dim),
            "--target",
            target,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(kernel_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        stats_cmd = [
            nsys.stdout.strip(),
            "stats",
            "--report",
            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum",
            f"{out_prefix}.nsys-rep",
        ]
        stats = subprocess.run(
            stats_cmd,
            cwd=str(kernel_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        out_file.write_text(
            "NSYS PROFILE STDOUT:\n" + proc.stdout +
            "\nNSYS PROFILE STDERR:\n" + proc.stderr +
            "\nNSYS STATS STDOUT:\n" + stats.stdout +
            "\nNSYS STATS STDERR:\n" + stats.stderr)
        print(proc.stdout)
        if proc.stderr:
            print("NSYS STDERR:")
            print(proc.stderr)
        print(stats.stdout)
        if stats.stderr:
            print("NSYS STATS STDERR:")
            print(stats.stderr)
        results_volume.commit()
        status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"
        if proc.returncode == 0 and stats.returncode != 0:
            status = f"stats-failed:{stats.returncode}"
        return {
            "status": status,
            "profile": str(out_file),
            "nsys_rep": f"{out_prefix}.nsys-rep",
        }

    if tool != "ncu":
        return {"status": f"unknown-tool:{tool}"}

    ncu = subprocess.run(
        ["bash", "-lc", "command -v ncu"],
        cwd=str(kernel_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if ncu.returncode != 0:
        print("ncu not found in the CUDA image")
        return {"status": "missing-ncu"}

    out_file = RESULTS_ROOT / (
        f"ncu_{target}_{seq_shape}_{dense_radius}_{mid_radius}_h{num_heads}_d{head_dim}.txt"
        .replace(",", "-"))
    metrics = ",".join([
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_bytes.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_tensor.sum",
        "smsp__sass_inst_executed_op_memory.sum",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_static",
        "launch__shared_mem_per_block_dynamic",
    ])
    cmd = [
        ncu.stdout.strip(),
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--kernel-name",
        "regex:.*(mixed_vra|packed_attn|stride3).*",
        "--launch-count",
        "1",
        "--metrics",
        metrics,
        "python",
        "benchmarks/profile_vra_kernel_once.py",
        "--seq-shape",
        seq_shape,
        f"--dense-radius={dense_radius}",
        f"--mid-radius={mid_radius}",
        "--num-heads",
        str(num_heads),
        "--head-dim",
        str(head_dim),
        "--target",
        target,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(kernel_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    out_file.write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr)
    print(proc.stdout)
    if proc.stderr:
        print("STDERR:")
        print(proc.stderr)
    results_volume.commit()
    return {
        "status": "ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
        "profile": str(out_file),
    }


@app.local_entrypoint()
def main(
    seq_shape: str = "30x48x80",
    dense_radius: str = "1,1,1",
    mid_radius: str = "2,2,3",
    num_heads: int = 24,
    head_dim: int = 128,
    target: str = "mixed",
    tool: str = "ncu",
) -> None:
    result = run_vra_kernel_profile.remote(
        seq_shape=seq_shape,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        num_heads=num_heads,
        head_dim=head_dim,
        target=target,
        tool=tool,
    )
    print(result)

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "fastvideo-vra-kernel-bench"

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


_GPU = os.environ.get("MODAL_VRA_GPU", "A100-80GB")
_REPO_ROOT = _local_repo_for_mount()
_BUILD_TK = os.environ.get("MODAL_VRA_BUILD_TK", "0") == "1"
_TORCH_CUDA_ARCH_LIST = "9.0a" if _BUILD_TK else "8.0;9.0"
_CMAKE_ARGS = (
    "-DFASTVIDEO_KERNEL_BUILD_TK=ON -DCMAKE_CUDA_ARCHITECTURES=90a"
    if _BUILD_TK else
    "-DFASTVIDEO_KERNEL_BUILD_TK=OFF -DCMAKE_CUDA_ARCHITECTURES=80;90"
)

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
        "TORCH_CUDA_ARCH_LIST": _TORCH_CUDA_ARCH_LIST,
    }).run_commands(
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
        f"cd /FastVideo/fastvideo-kernel && CC=gcc CXX=g++ CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS='{_CMAKE_ARGS}' python -m pip install . -v --no-build-isolation",
    ))

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=_GPU,
    volumes={str(RESULTS_ROOT): results_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
def run_vra_kernel_bench(
    seq_shape: str,
    dense_radius: str,
    mid_radius: str,
    text_length: int,
    num_heads: int,
    head_dim: int,
    warmup: int,
    rep: int,
    include_sdpa: bool,
    include_packed_stride3_sdpa: bool,
    include_packed_stride3_h100: bool,
    include_fused_stride3_h100: bool,
    include_mixed_vra_h100: bool,
    include_pack_timing: bool,
    max_sdpa_tokens: int,
    h100_text_mode: str,
) -> dict[str, str]:
    import torch

    env = os.environ.copy()
    env["FASTVIDEO_VRA_H100_TEXT_MODE"] = h100_text_mode
    env["FASTVIDEO_KERNEL_USE_INSTALLED"] = "1"
    env["PYTHONPATH"] = (
        f"{REPO_MOUNT / 'fastvideo-kernel'}:"
        f"{env.get('PYTHONPATH', '')}")

    kernel_root = REPO_MOUNT / "fastvideo-kernel"
    result_csv = RESULTS_ROOT / (
        f"vra_{seq_shape}_{dense_radius}_{mid_radius}_h{num_heads}_d{head_dim}.csv"
        .replace(",", "-").replace("x", "x"))

    subprocess.check_call(
        [
            "python",
            "-m",
            "pytest",
            "tests/test_vra_reference.py",
            "tests/test_vra_triton.py",
            "-q",
        ],
        cwd=str(kernel_root),
        env=env,
    )

    bench_cmd = [
        "python",
        "benchmarks/bench_vra.py",
        "--seq-shape",
        seq_shape,
        f"--dense-radius={dense_radius}",
        f"--mid-radius={mid_radius}",
        "--text-length",
        str(text_length),
        "--num-heads",
        str(num_heads),
        "--head-dim",
        str(head_dim),
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
        "--run-gpu",
        "--csv",
        str(result_csv),
    ]
    if include_sdpa:
        bench_cmd.extend(
            ["--include-sdpa", "--max-sdpa-tokens",
             str(max_sdpa_tokens)])
    if include_packed_stride3_sdpa:
        bench_cmd.append("--include-packed-stride3-sdpa")
    if include_packed_stride3_h100:
        bench_cmd.append("--include-packed-stride3-h100")
    if include_fused_stride3_h100:
        bench_cmd.append("--include-fused-stride3-h100")
    if include_mixed_vra_h100:
        bench_cmd.append("--include-mixed-vra-h100")
    if include_pack_timing:
        bench_cmd.append("--include-pack-timing")

    subprocess.check_call(
        bench_cmd,
        cwd=str(kernel_root),
        env=env,
    )

    results_volume.commit()
    return {"csv": str(result_csv), "device": torch.cuda.get_device_name(0)}


@app.local_entrypoint()
def main(
    seq_shape: str = "6x8x16",
    dense_radius: str = "0,0,0",
    mid_radius: str = "-1,-1,-1",
    text_length: int = 0,
    num_heads: int = 2,
    head_dim: int = 32,
    warmup: int = 3,
    rep: int = 10,
    include_sdpa: bool = False,
    include_packed_stride3_sdpa: bool = False,
    include_packed_stride3_h100: bool = False,
    include_fused_stride3_h100: bool = False,
    include_mixed_vra_h100: bool = False,
    include_pack_timing: bool = False,
    max_sdpa_tokens: int = 4096,
    h100_text_mode: str = "auto",
) -> None:
    result = run_vra_kernel_bench.remote(
        seq_shape=seq_shape,
        dense_radius=dense_radius,
        mid_radius=mid_radius,
        text_length=text_length,
        num_heads=num_heads,
        head_dim=head_dim,
        warmup=warmup,
        rep=rep,
        include_sdpa=include_sdpa,
        include_packed_stride3_sdpa=include_packed_stride3_sdpa,
        include_packed_stride3_h100=include_packed_stride3_h100,
        include_fused_stride3_h100=include_fused_stride3_h100,
        include_mixed_vra_h100=include_mixed_vra_h100,
        include_pack_timing=include_pack_timing,
        max_sdpa_tokens=max_sdpa_tokens,
        h100_text_mode=h100_text_mode,
    )
    print(result)

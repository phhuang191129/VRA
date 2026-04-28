# SPDX-License-Identifier: Apache-2.0
"""Run HunyuanVideo inference (and optional attention dumps) on Modal.

Patterns follow:
  https://modal.com/docs/examples/ltx
  https://modal.com/docs/guide/model-weights

Local (CPU-only laptop):
  pip install modal
  modal setup

From **repository root**:
  modal run scripts/inference/modal_attention_hunyuan.py

Or:
  bash scripts/inference/attention_hunyuan.sh --modal

Download ``.npz`` for local ``utils/plot_attention.py`` (path is relative to
volume root = mount ``/attn_dumps``):
  modal volume get fastvideo-hunyuan-attn dump ./local_attn

GPU (default A100-80GB; override for H100):
  MODAL_HUNYUAN_GPU=H100 modal run scripts/inference/modal_attention_hunyuan.py

Gated Hugging Face models: create a Modal Secret with ``HF_TOKEN``, then:
  MODAL_USE_HF_SECRET=1 modal run ...

See ``attention_hunyuan_modal.md`` for pricing and volumes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "fastvideo-hunyuan-attention"

hf_volume = modal.Volume.from_name("fastvideo-hf-hunyuan", create_if_missing=True)
outputs_volume = modal.Volume.from_name("fastvideo-hunyuan-outputs",
                                        create_if_missing=True)
attn_volume = modal.Volume.from_name("fastvideo-hunyuan-attn",
                                     create_if_missing=True)

HF_HOME = Path("/models")
OUTPUTS_ROOT = Path("/outputs")
ATTN_ROOT = Path("/attn_dumps")
REPO_MOUNT = Path("/FastVideo")

_PASSTHROUGH_ENV_VARS = (
    "CUDA_LAUNCH_BLOCKING",
    "FASTVIDEO_STA_FORCE_TRITON",
    "FASTVIDEO_VRA_KERNEL_BACKEND",
    "FASTVIDEO_TORCH_PROFILER_DIR",
    "FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES",
    "FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY",
    "FASTVIDEO_TORCH_PROFILER_WITH_STACK",
    "FASTVIDEO_TORCH_PROFILER_WITH_FLOPS",
    "FASTVIDEO_TORCH_PROFILER_WAIT_STEPS",
    "FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS",
    "FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS",
    "FASTVIDEO_TORCH_PROFILE_REGIONS",
)


def _local_repo_for_image_mount() -> Path:
    """Directory to pass to ``add_local_dir`` when defining the Image.

    On your laptop, ``__file__`` is ``.../FastVideo/scripts/inference/...`` and
    ``parents[2]`` is the repo root. Modal imports a copy of this module from
    ``/root/modal_attention_hunyuan.py`` inside the worker, where ``parents[2]``
    does not exist — use the mounted repo path instead (Image was already built
    from the client filesystem).
    """
    here = Path(__file__).resolve()
    try:
        return here.parents[2]
    except IndexError:
        return REPO_MOUNT


_REPO_ROOT = _local_repo_for_image_mount()

_GPU = os.environ.get("MODAL_HUNYUAN_GPU", "A100-80GB")

_HF_SECRETS: list[modal.Secret] = []
if os.environ.get("MODAL_USE_HF_SECRET", "0") == "1":
    _sn = os.environ.get("MODAL_HF_SECRET_NAME", "huggingface")
    _HF_SECRETS.append(modal.Secret.from_name(_sn))

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    ).apt_install(
        "git",
        "build-essential",
        "ffmpeg",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
    ).env({
        "CUDA_HOME": "/usr/local/cuda",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_CUDA_ARCH_LIST": "9.0a",
    }).add_local_dir(
        str(_REPO_ROOT),
        remote_path=str(REPO_MOUNT),
        # Required before ``run_commands``: bake repo into image so build steps
        # can run after (Modal forbids post-``add_local_*`` steps without this).
        copy=True,
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/node_modules/**",
            "**/.pytest_cache/**",
            "**/*.npz",
            "**/outputs_video/**",
            "**/fastvideo-kernel/benchmarks/vra_batch_logs/**",
            "**/fastvideo-kernel/benchmarks/vra_batch_outputs/**",
        ],
    ).run_commands(
        "python -m pip install -U pip setuptools wheel uv",
        # Match ``pyproject.toml`` / ``[tool.uv.sources]`` (linux → cu128).
        "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 "
        f"uv pip install -e {REPO_MOUNT} --system --index-strategy unsafe-best-match",
        "python -m pip install pytest numpy einops scikit-build-core cmake ninja",
        "python -m pip uninstall -y fastvideo-kernel",
        "cd /FastVideo/fastvideo-kernel && "
        "CC=gcc CXX=g++ CUDACXX=/usr/local/cuda/bin/nvcc "
        "CMAKE_ARGS='-DFASTVIDEO_KERNEL_BUILD_TK=ON -DCMAKE_CUDA_ARCHITECTURES=90a' "
        "python -m pip install . -v --no-build-isolation",
    ))

app = modal.App(APP_NAME)

_CLS_KW = {
    "image": image,
    "gpu": _GPU,
    "volumes": {
        str(HF_HOME): hf_volume,
        str(OUTPUTS_ROOT): outputs_volume,
        str(ATTN_ROOT): attn_volume,
    },
    "timeout": 60 * 60,
    "scaledown_window": 15 * 60,
}
if _HF_SECRETS:
    _CLS_KW["secrets"] = _HF_SECRETS


@app.cls(**_CLS_KW)
class HunyuanAttentionRunner:
    """``fastvideo`` installed in image; cold start skips runtime ``pip``."""

    @modal.method()
    def run_generate(
        self,
        model_base: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        prompt: str,
        seed: int,
        attention_backend: str,
        dump_attention: bool,
        dump_slice_len: int,
        target_step: int,
        target_head: int,
        target_block: str,
        target_block_index: int,
        target_cfg: str,
        embedded_cfg_scale: float,
        flow_shift: float,
        vra_sparsity: str,
        vra_kernel_backend: str,
        env_overrides: dict[str, str],
        attention_config: str = "",
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(env_overrides)
        env["HF_HOME"] = str(HF_HOME)
        env["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
        env["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")
        env["FASTVIDEO_ATTENTION_BACKEND"] = attention_backend
        if vra_sparsity:
            env["FASTVIDEO_VRA_SPARSITY"] = vra_sparsity
        if vra_kernel_backend:
            env["FASTVIDEO_VRA_KERNEL_BACKEND"] = vra_kernel_backend
        if attention_config:
            env["FASTVIDEO_ATTENTION_CONFIG"] = attention_config

        out_dir = OUTPUTS_ROOT / "hunyuan_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "out")

        cmd: list[str] = [
            sys.executable,
            "-m",
            "fastvideo.entrypoints.cli.main",
            "generate",
            "--model-path",
            model_base,
            "--sp-size",
            "1",
            "--tp-size",
            "1",
            "--num-gpus",
            "1",
            "--height",
            str(height),
            "--width",
            str(width),
            "--num-frames",
            str(num_frames),
            "--num-inference-steps",
            str(num_inference_steps),
            "--guidance-scale",
            "1",
            "--embedded-cfg-scale",
            str(embedded_cfg_scale),
            "--flow-shift",
            str(flow_shift),
            "--prompt",
            prompt,
            "--seed",
            str(seed),
            "--output-path",
            output_path,
        ]

        attn_rel = ""
        if dump_attention:
            dump_dir = ATTN_ROOT / "dump"
            dump_dir.mkdir(parents=True, exist_ok=True)
            env["FASTVIDEO_ATTENTION_DUMP_DIR"] = str(dump_dir)
            env["FASTVIDEO_ATTENTION_DUMP_SLICE_LEN"] = str(dump_slice_len)
            env["FASTVIDEO_ATTENTION_DUMP_TARGET_STEP"] = str(target_step)
            env["FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD"] = str(target_head)
            env["FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK"] = target_block
            env["FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX"] = str(
                target_block_index)
            env["FASTVIDEO_ATTENTION_DUMP_TARGET_CFG"] = target_cfg
            attn_rel = "dump"

        subprocess.check_call(cmd, env=env, cwd=str(REPO_MOUNT))
        outputs_volume.commit()
        attn_volume.commit()
        return {
            "output_path": output_path,
            "attn_volume_subdir": attn_rel,
            "outputs_volume_subdir": "hunyuan_run",
        }


@app.local_entrypoint()
def main(
    model_base: str = "hunyuanvideo-community/HunyuanVideo",
    height: int = 544,
    width: int = 960,
    num_frames: int = 75,
    num_inference_steps: int = 50,
    prompt: str = "A beautiful woman in a red dress walking down a street",
    seed: int = 1024,
    attention_backend: str = "TORCH_SDPA",
    dump_attention: bool = False,
    dump_slice_len: int = 0,
    target_step: int = 0,
    target_head: int = 0,
    target_block: str = "double_blocks",
    target_block_index: int = 0,
    target_cfg: str = "pos",
    embedded_cfg_scale: float = 6.0,
    flow_shift: float = 7.0,
    vra_sparsity: str = "58",
    vra_kernel_backend: str = "h100",
    attention_config: str = "",
) -> None:
    print(f"GPU (override with MODAL_HUNYUAN_GPU): {_GPU}")
    env_overrides = {
        key: value
        for key in _PASSTHROUGH_ENV_VARS
        if (value := os.environ.get(key)) is not None
    }
    if attention_backend == "SLIDING_TILE_ATTN":
        env_overrides.setdefault("FASTVIDEO_STA_FORCE_TRITON", "1")
    if env_overrides.get("FASTVIDEO_TORCH_PROFILER_WITH_FLOPS", "0") != "0":
        env_overrides.setdefault("FASTVIDEO_TORCH_PROFILER_DIR",
                                 str(OUTPUTS_ROOT / "profiler_traces"))
        env_overrides.setdefault("FASTVIDEO_TORCH_PROFILE_REGIONS",
                                 "profiler_region_inference_pipeline")
        env_overrides.setdefault("FASTVIDEO_TORCH_PROFILER_WAIT_STEPS", "0")
        env_overrides.setdefault("FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS", "0")
        env_overrides.setdefault("FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS", "1")
    result = HunyuanAttentionRunner().run_generate.remote(
        model_base=model_base,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
        seed=seed,
        attention_backend=attention_backend,
        dump_attention=dump_attention,
        dump_slice_len=dump_slice_len,
        target_step=target_step,
        target_head=target_head,
        target_block=target_block,
        target_block_index=target_block_index,
        target_cfg=target_cfg,
        embedded_cfg_scale=embedded_cfg_scale,
        flow_shift=flow_shift,
        vra_sparsity=vra_sparsity,
        vra_kernel_backend=vra_kernel_backend,
        env_overrides=env_overrides,
        attention_config=attention_config,
    )
    print("Done:", result)
    if dump_attention:
        print(
            "Fetch attention dumps for local plotting:\n"
            f"  modal volume get fastvideo-hunyuan-attn {result.get('attn_volume_subdir', 'dump')} ./local_attn"
        )

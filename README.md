# Accelerating Video Generation with VRA

This repository extends [FastVideo](https://github.com/hao-ai-lab/FastVideo) and the [STA](https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete) line with **Variable Rate Attention (VRA)**—a sparse attention backend for DiT-based video models (notably HunyuanVideo)—plus scripts to run **dense FlashAttention 3**, **STA**, and **VRA** locally or on [Modal](https://modal.com), and to **evaluate** generations with [VBench](https://github.com/Vchitect/VBench).

## Repository overview

| Area | Path |
|------|------|
| Core Python package | `fastvideo/` |
| Custom / Triton kernels | `fastvideo-kernel/` |
| Hunyuan inference (local shell) | `scripts/inference/v1_inference_hunyuan*.sh` |
| Hunyuan on Modal | `scripts/inference/modal_attention_hunyuan.py`, `*_modal.sh` |
| Modal details (volumes, pricing, dumps) | `scripts/inference/attention_hunyuan_modal.md` |
| VBench wrapper | `scripts/eval/run_vbench_custom.sh`, `eval/vbench_env/` |
| STA mask JSON (Hunyuan) | `assets/mask_strategy_hunyuan.json`, `assets/mask_strategy_hunyuan_30_40_40.json`, … |

---

## Attention backends

### Baseline: FlashAttention 3 (dense)

Full attention via **`FLASH_ATTN`**. Use this as the quality/speed reference when comparing sparse methods.

### STA: Sliding Tile Attention

**Sliding Tile Attention** uses a fixed sliding-window style sparsity pattern driven by a mask-strategy JSON (`FASTVIDEO_ATTENTION_CONFIG`). See the [STA blog](https://hao-ai-lab.github.io/blogs/sta/) and [paper](https://arxiv.org/abs/2502.04507); in-repo STA docs live under `docs/attention/sta/index.md`.

### VRA: Variable Rate Attention

**Variable Rate Attention** replaces STA’s uniform sliding window with a **per-query concentric multi-ring mask**: denser KV coverage near the query tile and progressively sparser sampling farther away in space and time.

For each query tile at `(t, h, w)` in the tile grid, KV tiles fall into regions by **relative distance** from the query:

| Region | Role |
|--------|------|
| **Core** | Dense cube around the query |
| **Mid ring** | Stride-sampled (e.g. multiples of 2 on `\|Δt\|, \|Δh\|, \|Δw\|`) |
| **Outer ring** | Stride-sampled (e.g. multiples of 3) |
| **Beyond outer** | Skipped |

**Text tokens** use full attention (text sink): text queries see all KV; video queries see all text KV plus VRA-filtered video KV.

**Sparsity presets** (`FASTVIDEO_VRA_SPARSITY`, see `fastvideo/attention/backends/sliding_variable_rate_attn.py`):

| Preset | Notes |
|--------|--------|
| `58` | Default-style sparse preset for the supported tile grids |
| `sta58` | Tuned to approximate STA’s sparse-step tile density (~125 / 300 active tiles on the `30×48×80` grid) |
| `91` | Very sparse preset |

The tuple format for custom tuning is documented in code comments next to `_VRA_CONFIGS`.

---

## Installation

### Main package (inference)

From the repository root:

```bash
uv pip install -e .
```

Build the **`fastvideo-kernel`** package when you use backends that rely on it (STA and other compiled paths):

```bash
cd fastvideo-kernel
./build.sh
cd ..
```

VRA’s variable-rate path is implemented with **Triton** in-tree; Modal bakes the repo and overlays `fastvideo_kernel` without a separate CUDA build for that path (see `modal_attention_hunyuan.py`).

### Modal (optional)

```bash
pip install modal
modal setup
```

Run all `modal run` commands from the **repository root**. For gated Hugging Face checkpoints, create a Modal Secret with `HF_TOKEN` and use `MODAL_USE_HF_SECRET=1` (see `scripts/inference/attention_hunyuan_modal.md`).

### VBench (evaluation)

VBench pulls older `transformers` / `tokenizers`; use **Python 3.10 or 3.11** (not 3.12) for `eval/vbench_env`:

```bash
cd eval/vbench_env
rm -rf .venv && rm -f uv.lock   # optional: force a fresh lock
uv venv --python 3.11 && uv lock && uv sync
```

The eval shell script activates this venv and sets `PYTHONPATH` so `eval/vbench_env/sitecustomize.py` can patch `torch.load` compatibility for VBench.

---

## Running inference

### Environment variables (Hunyuan)

| Variable | Typical values | Purpose |
|----------|----------------|---------|
| `FASTVIDEO_ATTENTION_BACKEND` | `FLASH_ATTN`, `SLIDING_TILE_ATTN`, `SLIDING_VARIABLE_RATE_ATTN` | Attention implementation |
| `FASTVIDEO_ATTENTION_CONFIG` | path to JSON | Required for STA (mask strategy) |
| `FASTVIDEO_VRA_SPARSITY` | `58`, `sta58`, `91` | VRA preset |
| `MODAL_HUNYUAN_GPU` | `H100`, `A100-80GB` | Modal GPU type (read before `modal run`) |

### Local (GPU)

**VRA** (edit `FASTVIDEO_VRA_SPARSITY` and `num_gpus` in the script if needed):

```bash
bash scripts/inference/v1_inference_hunyuan_VRA.sh
```

**STA**:

```bash
bash scripts/inference/v1_inference_hunyuan_STA.sh
```

Uses `assets/mask_strategy_hunyuan.json` by default (resolution in the script: 768×1280, 117 frames).

**Dense baseline (FlashAttention 3)** — example uses multi-GPU `sp-size`; adjust `num_gpus` for your machine:

```bash
bash scripts/inference/v1_inference_hunyuan.sh
```

Other models / STA examples from upstream: `scripts/inference/v1_inference_wan_STA.sh`, Wan mask search under `examples/inference/sta_mask_search/`.

### Modal (cloud GPU)

Wrapper scripts call `modal run scripts/inference/modal_attention_hunyuan.py` with the right backend flags. Examples:

| Script | Backend |
|--------|---------|
| `scripts/inference/v1_inference_hunyuan_FA3_modal.sh` | `FLASH_ATTN` |
| `scripts/inference/v1_inference_hunyuan_STA_modal.sh` | `SLIDING_TILE_ATTN` + `assets/mask_strategy_hunyuan_30_40_40.json` |
| `scripts/inference/v1_inference_hunyuan_VRA_modal.sh` | `SLIDING_VARIABLE_RATE_ATTN` + `--vra-sparsity` (default `sta58` in the script) |

```bash
export MODAL_HUNYUAN_GPU=H100   # optional
uv run bash scripts/inference/v1_inference_hunyuan_VRA_modal.sh
```

Artifacts are written under Modal **Volumes** (e.g. `fastvideo-hunyuan-outputs` → `/outputs/hunyuan_run/` in the container). Download with `modal volume get` (see `attention_hunyuan_modal.md`). For evaluation below, copy or save resulting **`.mp4`** files into the layout VBench expects.

---

## Evaluation (VBench)

`scripts/eval/run_vbench_custom.sh` runs VBench **`custom_input`** mode on six dimensions for videos under:

- `result/sta/` — STA (or any baseline you want to label “sta” for this run)
- `result/vra/` — VRA

Each directory may contain one or more `.mp4` files. Outputs go to `result/vbench_results/<folder>/<dimension>/`.

**Prerequisite:** `eval/vbench_env/.venv` created as in [VBench (evaluation)](#vbench-evaluation).

**Run:**

```bash
./scripts/eval/run_vbench_custom.sh
```

Optional parallelism:

```bash
VBENCH_NGPUS=4 ./scripts/eval/run_vbench_custom.sh
```

Dimensions evaluated: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`. To score a **FlashAttention** baseline alongside STA/VRA, add a folder (e.g. `result/fa3/`) and extend the `for folder in …` loop in the script, or run `vbench evaluate` manually with the same flags.

---

## Original STA documentation (upstream)

STA mask strategy files for Hunyuan / Wan include `assets/mask_strategy_hunyuan.json` and `assets/mask_strategy_wan.json`. Wan mask search:

```bash
bash examples/inference/sta_mask_search/inference_wan_sta.sh
```

(Default script often targets 8 GPUs; see that script’s comments.)

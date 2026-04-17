# `attention_hunyuan.sh`

## Purpose

Runs **HunyuanVideo** text-to-video inference with **FlashAttention** and optional saving of **one** full (or sliced) attention softmax matrix for debugging, matching the hooks in `fastvideo/attention/attention_weight_dump.py`.

Use **one GPU** (`NUM_GPUS=1`) so attention dumps are not split across sequence-parallel head shards.

## Usage

From the **repository root**:

```bash
# Inference only (no attention dump). Clears a stray FASTVIDEO_ATTENTION_DUMP_DIR if unset.
bash scripts/inference/attention_hunyuan.sh
```

```bash
# Inference + optional dump into a directory (enables dump env vars with defaults)
bash scripts/inference/attention_hunyuan.sh /path/to/dump_dir
```

### Modal (remote GPU, no local CUDA)

```bash
pip install modal && modal setup
bash scripts/inference/attention_hunyuan.sh --modal
# optional flags for modal run:
bash scripts/inference/attention_hunyuan.sh --modal -- --prompt "..." --dump-slice-len 2048
```

Volumes, pricing, downloading `.npz` for local `plot_attention.py`: **`attention_hunyuan_modal.md`**.

## Resolution defaults

The script defaults align with a lighter setting (example):

- `HEIGHT=544`, `WIDTH=960`, `NUM_FRAMES=75`

Override with environment variables before invoking the script.

## Environment variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BASE` | `hunyuanvideo-community/HunyuanVideo` | Hugging Face model id or path |
| `NUM_GPUS` | `1` | Passed to `--num-gpus` and `--sp-size` |
| `FASTVIDEO_ATTENTION_BACKEND` | `FLASH_ATTN` | Attention backend |
| `PROMPT`, `SEED`, `OUTPUT_PATH`, `NUM_INFERENCE_STEPS`, `EMBEDDED_CFG_SCALE`, `FLOW_SHIFT` | see script | Passed through to `fastvideo generate` |

### When a dump directory is passed as the first argument

The script sets (defaults can be overridden by exporting variables **before** running the script):

| Variable | Default (in script) | Description |
|----------|----------------------|-------------|
| `FASTVIDEO_ATTENTION_DUMP_DIR` | first argument | Output directory for `.npz` |
| `FASTVIDEO_ATTENTION_DUMP_SLICE_LEN` | `0` | `0` = full sequence length (large memory); set e.g. `2048` to cap |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_STEP` | `0` | Denoise step index to match |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD` | `0` | Global head index |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK` | `double_blocks` | `double_blocks` or `single_blocks` |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX` | `0` | Block index within that group |
| `FASTVIDEO_ATTENTION_DUMP_TARGET_CFG` | `pos` | `pos`, `neg`, or `any` |

See **`fastvideo/attention/attention_weight_dump.md`** for exact matching rules and output naming.

### Memory estimate

When a dump directory is provided, the script prints a **rough** estimate of stored size and peak memory for one \(L \times L\) matrix (float16 + ~2.5× peak heuristic), using the same VAE/patch assumptions as the inline Python snippet (8× spatial, 4× temporal, 1×2×2 patch). Add headroom for the full model and CUDA.

## Plotting outputs

Use `utils/plot_attention.py` (see `utils/plot_attention.md`):

```bash
python utils/plot_attention.py --mode heatmap \
  --npz /path/to/dump_dir/step0000_double_blocks_000_head000_pos.npz \
  --out heatmap.png
```

## Related files

- `fastvideo/attention/attention_weight_dump.py` — dump implementation
- `scripts/inference/v1_inference_hunyuan.sh` — Hunyuan inference without attention dumping

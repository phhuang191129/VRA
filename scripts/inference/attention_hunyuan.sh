#!/usr/bin/env bash
# HunyuanVideo inference with optional **single** (head, layer, step) attention dump.
#
# Usage (local GPU):
#   bash scripts/inference/attention_hunyuan.sh
#       → inference only (no dump).
#   bash scripts/inference/attention_hunyuan.sh /path/to/dump_dir
#       → dumps **one** L×L softmax matrix when all TARGET_* match (see below).
#
# Usage (Modal — no local GPU; see scripts/inference/attention_hunyuan_modal.md):
#   pip install modal && modal setup
#   bash scripts/inference/attention_hunyuan.sh --modal
#   bash scripts/inference/attention_hunyuan.sh --modal -- --prompt "..." --dump-slice-len 2048
# Pull .npz to laptop for utils/plot_attention.py:
#   modal volume get fastvideo-hunyuan-attn dump ./local_attn
#
# With a dump dir, set (or rely on defaults for a quick smoke test):
#   FASTVIDEO_ATTENTION_DUMP_TARGET_STEP
#   FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD
#   FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK   (double_blocks | single_blocks)
#   FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX
#   FASTVIDEO_ATTENTION_DUMP_TARGET_CFG     (pos | neg | any)  default: pos
#   FASTVIDEO_ATTENTION_DUMP_SLICE_LEN      default: 0 (= full sequence; large!)
#
# Memory (544×960, 75 frames, typical Hunyuan VAE 8× spatial / 4× temporal, patch 1×2×2):
#   ~3.9e4 image tokens; joint attn adds text → L often ~4e4…4.5e4.
#   One float16 L×L matrix ≈ 2·L² bytes; peak GPU during matmul+softmax ≈ 2.5× that.
#   Example L=40_000 → ~3.0 GiB stored, ~7.5 GiB peak (order-of-magnitude; request extra
#   for the rest of the model, activations, and CUDA overhead).
#
# Plot:
#   python utils/plot_attention.py --mode heatmap --npz /path/to/step0000_...npz --out p.png
#
# Multi-run sweep: scripts/inference/attention_hunyuan_sweep.sh
#   (default: grid steps×heads×blocks; see script header)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${1:-}" == "--modal" ]]; then
  shift
  cd "${ROOT}"
  exec modal run scripts/inference/modal_attention_hunyuan.py "$@"
fi

cd "${ROOT}"

ATTENTION_DUMP_DIR="${1:-}"
if [[ -z "${ATTENTION_DUMP_DIR}" ]]; then
  unset FASTVIDEO_ATTENTION_DUMP_DIR 2>/dev/null || true
fi

HEIGHT="${HEIGHT:-544}"
WIDTH="${WIDTH:-960}"
NUM_FRAMES="${NUM_FRAMES:-75}"
NUM_GPUS="${NUM_GPUS:-1}"

export MODEL_BASE="${MODEL_BASE:-hunyuanvideo-community/HunyuanVideo}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"

if [[ -n "${ATTENTION_DUMP_DIR}" ]]; then
  mkdir -p "${ATTENTION_DUMP_DIR}"
  export FASTVIDEO_ATTENTION_DUMP_DIR="${ATTENTION_DUMP_DIR}"
  export FASTVIDEO_ATTENTION_DUMP_SLICE_LEN="${FASTVIDEO_ATTENTION_DUMP_SLICE_LEN:-0}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_STEP="${FASTVIDEO_ATTENTION_DUMP_TARGET_STEP:-0}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD="${FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD:-0}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK="${FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK:-double_blocks}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX="${FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX:-0}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_CFG="${FASTVIDEO_ATTENTION_DUMP_TARGET_CFG:-pos}"

  python3 -c "
lt = (${NUM_FRAMES} - 1) // 4 + 1
lh = ${HEIGHT} // 8
lw = ${WIDTH} // 8
n_img = lt * (lh // 2) * (lw // 2)
l_full = n_img + 320
st = 2 * l_full * l_full
pk = int(st * 2.5)
print(
    f'[attention dump mem est.] image_tokens≈{n_img}; L≈{l_full} '
    f'(incl. ~320 text guess): stored_fp16≈{st / 1024**3:.2f} GiB, '
    f'rough_peak≈{pk / 1024**3:.2f} GiB (cap L with '
    f'FASTVIDEO_ATTENTION_DUMP_SLICE_LEN).')
"
fi

fastvideo generate \
  --model-path "${MODEL_BASE}" \
  --sp-size "${NUM_GPUS}" \
  --tp-size 1 \
  --num-gpus "${NUM_GPUS}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num-frames "${NUM_FRAMES}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS:-50}" \
  --guidance-scale 1 \
  --embedded-cfg-scale "${EMBEDDED_CFG_SCALE:-6}" \
  --flow-shift "${FLOW_SHIFT:-7}" \
  --prompt "${PROMPT:-A beautiful woman in a red dress walking down a street}" \
  --seed "${SEED:-1024}" \
  --output-path "${OUTPUT_PATH:-outputs_video/attention_hunyuan_run}"

if [[ -n "${ATTENTION_DUMP_DIR:-}" ]]; then
  echo "If dump ran, matrices under: ${ATTENTION_DUMP_DIR}"
fi

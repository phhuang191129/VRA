#!/usr/bin/env bash
# Sweep Hunyuan attention dumps. Each run is **full** inference.
#
# HunyuanVideo defaults: 20 double_blocks (idx 0–19), 40 single_blocks (idx 20–59
# in param names), 24 heads (0–23). Override with HUNYUAN_* env vars if needed.
#
# --- Grid mode (Cartesian product) ---
#   ATTENTION_SWEEP_MODE=grid
#   ATTENTION_SWEEP_GRID_STEPS="20 40"        # target denoise steps
#   ATTENTION_SWEEP_GRID_HEADS="10 20"        # target head indices
#   ATTENTION_SWEEP_GRID_BLOCKS              # multiline: BLOCK_TYPE:INDEX
#     default:
#       double_blocks:10
#       single_blocks:40
#
# Example (your sweep — 2×2×2 = 8 runs):
#   ATTENTION_SWEEP_MODE=grid bash scripts/inference/attention_hunyuan_sweep.sh
#
# --- Preset mode ---
#   ATTENTION_SWEEP_MODE=preset
#   ATTENTION_SWEEP_STEPS + ATTENTION_SWEEP_PRESETS (NAME:HEAD:BLOCK:INDEX)
#
# --- Uniform random mode ---
#   ATTENTION_SWEEP_UNIFORM_RUNS=N  (N>0; ignores grid/preset)
#
# Modal:  bash .../attention_hunyuan_sweep.sh --modal [-- extra modal args]
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="${ROOT}/scripts/inference/attention_hunyuan.sh"
MODAL_APP="${ROOT}/scripts/inference/modal_attention_hunyuan.py"

NUM_DOUBLE="${HUNYUAN_NUM_DOUBLE_BLOCKS:-20}"
NUM_SINGLE="${HUNYUAN_NUM_SINGLE_BLOCKS:-40}"
NUM_HEADS="${HUNYUAN_NUM_ATTENTION_HEADS:-24}"
SINGLE_IDX_BASE="${HUNYUAN_SINGLE_BLOCK_INDEX_BASE:-${NUM_DOUBLE}}"

UNIFORM_RUNS="${ATTENTION_SWEEP_UNIFORM_RUNS:-0}"
SWEEP_MODE="${ATTENTION_SWEEP_MODE:-grid}"

STEPS_STR="${ATTENTION_SWEEP_STEPS:-20 40}"
read -r -a STEPS <<< "${STEPS_STR}"

GRID_STEPS_STR="${ATTENTION_SWEEP_GRID_STEPS:-20 40}"
read -r -a GRID_STEPS <<< "${GRID_STEPS_STR}"

GRID_HEADS_STR="${ATTENTION_SWEEP_GRID_HEADS:-10 20}"
read -r -a GRID_HEADS <<< "${GRID_HEADS_STR}"

GRID_BLOCK_LINES="${ATTENTION_SWEEP_GRID_BLOCKS:-}"
if [[ -z "${GRID_BLOCK_LINES}" ]]; then
  GRID_BLOCK_LINES=$'double_blocks:10\nsingle_blocks:40'
fi

PRESET_LINES="${ATTENTION_SWEEP_PRESETS:-}"
if [[ -z "${PRESET_LINES}" ]]; then
  PRESET_LINES=$'preset1:0:double_blocks:0\npreset2:8:double_blocks:10'
fi

USE_MODAL=0
EXTRA_MODAL=()
if [[ "${1:-}" == "--modal" ]]; then
  USE_MODAL=1
  shift
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  EXTRA_MODAL=("$@")
  set --
fi

DUMP_ROOT="${ATTENTION_SWEEP_DUMP_ROOT:-${ROOT}/outputs_video/attn_sweep}"
export FASTVIDEO_ATTENTION_DUMP_SLICE_LEN="${FASTVIDEO_ATTENTION_DUMP_SLICE_LEN:-2048}"
export FASTVIDEO_ATTENTION_DUMP_TARGET_CFG="${FASTVIDEO_ATTENTION_DUMP_TARGET_CFG:-pos}"

if [[ "${USE_MODAL}" -eq 0 ]]; then
  export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
  mkdir -p "${DUMP_ROOT}"
fi

if [[ -n "${ATTENTION_SWEEP_SEED:-}" ]]; then
  RANDOM="${ATTENTION_SWEEP_SEED}"
fi

_run_local() {
  local tag="$1"
  local step="$2"
  local head="$3"
  local block="$4"
  local bidx="$5"
  local run_dir="${DUMP_ROOT}/${tag}"
  mkdir -p "${run_dir}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_STEP="${step}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_HEAD="${head}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK="${block}"
  export FASTVIDEO_ATTENTION_DUMP_TARGET_BLOCK_INDEX="${bidx}"
  bash "${WRAPPER}" "${run_dir}"
}

_run_modal() {
  local step="$1"
  local head="$2"
  local block="$3"
  local bidx="$4"
  cd "${ROOT}"
  modal run "${MODAL_APP}" \
    --target-step "${step}" \
    --target-head "${head}" \
    --target-block "${block}" \
    --target-block-index "${bidx}" \
    "${EXTRA_MODAL[@]}"
}

_one_run() {
  local tag="$1"
  local step="$2"
  local head="$3"
  local block="$4"
  local bidx="$5"
  echo "========================================"
  echo "Run: step=${step} head=${head} block=${block} idx=${bidx} tag=${tag}"
  echo "========================================"
  if [[ "${USE_MODAL}" -eq 1 ]]; then
    _run_modal "${step}" "${head}" "${block}" "${bidx}"
  else
    _run_local "${tag}" "${step}" "${head}" "${block}" "${bidx}"
  fi
}

_pick_step_from_list() {
  local n=${#STEPS[@]}
  if [[ "${n}" -lt 1 ]]; then
    echo "ATTENTION_SWEEP_STEPS is empty" >&2
    exit 1
  fi
  echo "${STEPS[$((RANDOM % n))]}"
}

_preset_sweep() {
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    IFS=':' read -r name head block bidx <<< "${line}"
    for step in "${STEPS[@]}"; do
      tag="s${step}_${name}_h${head}_${block}_${bidx}"
      _one_run "${tag}" "${step}" "${head}" "${block}" "${bidx}"
    done
  done <<< "${PRESET_LINES}"
}

_uniform_sweep() {
  local i step_p head_p block_p bidx_p si
  for ((i = 1; i <= UNIFORM_RUNS; i++)); do
    step_p="$(_pick_step_from_list)"
    head_p=$((RANDOM % NUM_HEADS))
    if ((RANDOM % 2 == 0)); then
      block_p="double_blocks"
      bidx_p=$((RANDOM % NUM_DOUBLE))
    else
      block_p="single_blocks"
      bidx_p=$((SINGLE_IDX_BASE + RANDOM % NUM_SINGLE))
    fi
    si=$(printf '%03d' "${i}")
    tag="u${si}_s${step_p}_h${head_p}_${block_p}_${bidx_p}"
    _one_run "${tag}" "${step_p}" "${head_p}" "${block_p}" "${bidx_p}"
  done
}

_grid_sweep() {
  local step head block bidx line tag
  for step in "${GRID_STEPS[@]}"; do
    for head in "${GRID_HEADS[@]}"; do
      while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
        IFS=':' read -r block bidx <<< "${line}"
        tag="g_s${step}_h${head}_${block}_${bidx}"
        _one_run "${tag}" "${step}" "${head}" "${block}" "${bidx}"
      done <<< "${GRID_BLOCK_LINES}"
    done
  done
}

echo "Hunyuan layout: ${NUM_DOUBLE} double_blocks (0..$((NUM_DOUBLE - 1))), \
${NUM_SINGLE} single_blocks (${SINGLE_IDX_BASE}..$((SINGLE_IDX_BASE + NUM_SINGLE - 1))), \
${NUM_HEADS} heads (0..$((NUM_HEADS - 1)))."

if [[ "${UNIFORM_RUNS}" -gt 0 ]]; then
  echo "Uniform mode: ${UNIFORM_RUNS} runs."
  _uniform_sweep
elif [[ "${SWEEP_MODE}" == "preset" ]]; then
  echo "Preset mode: steps [${STEPS_STR}] × preset lines."
  _preset_sweep
elif [[ "${SWEEP_MODE}" == "grid" ]]; then
  echo "Grid mode: steps [${GRID_STEPS_STR}] × heads [${GRID_HEADS_STR}] × block specs."
  _grid_sweep
else
  echo "Unknown ATTENTION_SWEEP_MODE=${SWEEP_MODE} (use grid, preset)" >&2
  exit 1
fi

echo "Sweep finished."

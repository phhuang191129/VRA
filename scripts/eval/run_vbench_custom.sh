#!/usr/bin/env bash
# Run VBench custom_input on the six dimensions supported for arbitrary videos,
# for result/sta and result/vra (each folder may contain one or more mp4 files).
#
# Prereq (once): use Python 3.10 or 3.11 for eval/vbench_env (not 3.12; tokenizers
# 0.13.3 has no cp312 wheels and fails to build from source on current Rust):
#   cd eval/vbench_env && rm -rf .venv && rm -f uv.lock
#   uv venv --python 3.11 && uv lock && uv sync
#
# Usage:
#   ./scripts/eval/run_vbench_custom.sh
# Optional: VBENCH_NGPUS=4 ./scripts/eval/run_vbench_custom.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VBENCH_PROJECT="${REPO_ROOT}/eval/vbench_env"
RESULT="${REPO_ROOT}/result"
OUT_ROOT="${REPO_ROOT}/result/vbench_results"
NGPUS="${VBENCH_NGPUS:-1}"

if [[ ! -d "${VBENCH_PROJECT}/.venv" ]]; then
  echo "Missing ${VBENCH_PROJECT}/.venv — create it with Python 3.11:" >&2
  echo "  cd ${VBENCH_PROJECT} && uv venv --python 3.11 && uv lock && uv sync" >&2
  exit 1
fi

# So sitecustomize.py (torch.load compat for VBench) loads in this shell and in
# vbench's torch.distributed worker subprocesses.
export PYTHONPATH="${VBENCH_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck source=/dev/null
source "${VBENCH_PROJECT}/.venv/bin/activate"
cd "${VBENCH_PROJECT}"

dims=(
  subject_consistency
  background_consistency
  motion_smoothness
  dynamic_degree
  aesthetic_quality
  imaging_quality
)

for folder in sta vra; do
  vid_dir="${RESULT}/${folder}"
  if [[ ! -d "${vid_dir}" ]]; then
    echo "Skip: no directory ${vid_dir}" >&2
    continue
  fi
  for dim in "${dims[@]}"; do
    out="${OUT_ROOT}/${folder}/${dim}"
    mkdir -p "${out}"
    echo "=== ${folder} / ${dim} -> ${out} ==="
    vbench evaluate \
      --ngpus="${NGPUS}" \
      --dimension "${dim}" \
      --videos_path "${vid_dir}" \
      --mode=custom_input \
      --output_path "${out}/"
  done
done

echo "Done. Summaries under ${OUT_ROOT}"

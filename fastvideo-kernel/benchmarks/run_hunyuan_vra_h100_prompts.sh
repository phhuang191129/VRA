#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
# Keep live logs outside the repository. Modal add_local_dir(copy=True) rejects
# files that change while the image is being built.
LOG_DIR="/tmp/vra_batch_logs_${RUN_ID}"
FINAL_LOG_DIR="${ROOT}/fastvideo-kernel/benchmarks/vra_batch_logs/${RUN_ID}"
OUT_DIR="${ROOT}/fastvideo-kernel/benchmarks/vra_batch_outputs"
SUMMARY="${LOG_DIR}/summary.tsv"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

export MODAL_HUNYUAN_GPU="${MODAL_HUNYUAN_GPU:-H100}"
export FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=0
unset FASTVIDEO_TORCH_PROFILER_DIR

HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FRAMES="${FRAMES:-117}"
STEPS="${STEPS:-50}"
SPARSITY="${SPARSITY:-sta58}"
PARALLEL="${PARALLEL:-4}"

cat > "${SUMMARY}" <<'TSV'
idx	status	wall_seconds	generation_seconds	output_file	run_url	prompt
TSV

run_one() {
  local idx="$1"
  local prompt="$2"
  local log="${LOG_DIR}/prompt_${idx}.log"
  local start end wall status
  start="$(date +%s)"
  status="ok"
  {
    echo "Prompt ${idx}"
    echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Prompt: ${prompt}"
    modal run scripts/inference/modal_attention_hunyuan.py \
      --attention-backend "SLIDING_VARIABLE_RATE_ATTN" \
      --vra-sparsity "${SPARSITY}" \
      --vra-kernel-backend "h100" \
      --prompt "${prompt}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num-frames "${FRAMES}" \
      --num-inference-steps "${STEPS}" \
      --embedded-cfg-scale 6 \
      --flow-shift 7 \
      --target-cfg "pos"
    echo "End: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "${log}" 2>&1 || status="failed"
  if grep -Eq "ExecutionError|Stopping app - uncaught exception|Traceback|╭─ Error" "${log}"; then
    status="failed"
  fi
  end="$(date +%s)"
  wall="$((end - start))"

  local gen output run_url
  gen="$(sed -n 's/.*Generated successfully in \([0-9.]*\) seconds.*/\1/p' "${log}" | tail -n 1)"
  output="$(sed -n 's/.*Saved video to \(.*\.mp4\).*/\1/p' "${log}" | tail -n 1)"
  run_url="$(grep -Eo 'https://modal\.com/apps/[^[:space:]]+' "${log}" | tail -n 1)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${idx}" "${status}" "${wall}" "${gen}" "${output}" "${run_url}" "${prompt}" \
    >> "${SUMMARY}"
}

prompts=(
  "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."
  "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
  "A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere."
  "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest. The playful yet serene atmosphere is complemented by soft natural light filtering through the petals. Mid-shot, warm and cheerful tones."
  "A superintelligent humanoid robot waking up. The robot has a sleek metallic body with futuristic design features. Its glowing red eyes are the focal point, emanating a sharp, intense light as it powers on. The scene is set in a dimly lit, high-tech laboratory filled with glowing control panels, robotic arms, and holographic screens. The setting emphasizes advanced technology and an atmosphere of mystery. The ambiance is eerie and dramatic, highlighting the moment of awakening and the robots immense intelligence. Photorealistic style with a cinematic, dark sci-fi aesthetic. Aspect ratio: 16:9 --v 6.1"
  "fox in the forest close-up quickly turned its head to the left"
  "Man walking his dog in the woods on a hot sunny day"
  "A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic."
)

cd "${ROOT}"
printf "Running %s prompts with PARALLEL=%s, kernel=h100, sparsity=%s\n" \
  "${#prompts[@]}" "${PARALLEL}" "${SPARSITY}"

for i in "${!prompts[@]}"; do
  idx="$((i + 1))"
  run_one "${idx}" "${prompts[$i]}" &
  while (( "$(jobs -rp | wc -l | tr -d ' ')" >= PARALLEL )); do
    sleep 10
  done
done
wait

mkdir -p "$(dirname "${FINAL_LOG_DIR}")"
rm -rf "${FINAL_LOG_DIR}"
cp -R "${LOG_DIR}" "${FINAL_LOG_DIR}"

LOCAL_VIDEOS="${OUT_DIR}/hunyuan_run_${RUN_ID}"
rm -rf "${LOCAL_VIDEOS}"
mkdir -p "${LOCAL_VIDEOS}"
modal volume get --force fastvideo-hunyuan-outputs hunyuan_run/ "${LOCAL_VIDEOS}"
echo "Summary: ${SUMMARY}"
echo "Final logs: ${FINAL_LOG_DIR}"
echo "Videos: ${LOCAL_VIDEOS}"

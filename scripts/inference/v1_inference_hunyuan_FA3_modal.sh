#!/bin/bash
# Wrapper script to run baseline HunyuanVideo inference with FlashAttention 3 on Modal.

# PROMPT="Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."
PROMPT="A beautiful woman in a red dress walking down a street"
# PROMPT="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere."

export MODAL_HUNYUAN_GPU=H100
# Keep profiler disabled for latency measurements.
export FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=0
unset FASTVIDEO_TORCH_PROFILER_DIR

echo "Running baseline HunyuanVideo on Modal using FlashAttention..."

modal run scripts/inference/modal_attention_hunyuan.py \
  --attention-backend "FLASH_ATTN" \
  --prompt "${PROMPT}" \
  --height 768 \
  --width 1280 \
  --num-frames 117 \
  --num-inference-steps 50 \
  --embedded-cfg-scale 6 \
  --flow-shift 7 \
  --target-cfg "pos"

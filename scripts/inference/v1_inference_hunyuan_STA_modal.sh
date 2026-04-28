#!/bin/bash
# Wrapper script to run HunyuanVideo inference with STA on Modal.

ATTENTION_CONFIG="assets/mask_strategy_hunyuan_30_40_40.json"
PROMPT="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere."
export MODAL_HUNYUAN_GPU=H100
echo "Running HunyuanVideo STA on Modal using config ${ATTENTION_CONFIG}..."
# export FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=1
export FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=0
unset FASTVIDEO_TORCH_PROFILER_DIR
export FASTVIDEO_STA_FORCE_TRITON=1
# Run Modal. This command executes the modal script remotely, passing the STA backend parameters.
modal run scripts/inference/modal_attention_hunyuan.py \
  --attention-backend "SLIDING_TILE_ATTN" \
  --attention-config "${ATTENTION_CONFIG}" \
  --prompt "${PROMPT}" \
  --height 768 \
  --width 1280 \
  --num-frames 117 \
  --num-inference-steps 50 \
  --embedded-cfg-scale 6 \
  --flow-shift 7 \
  --target-cfg "pos" 

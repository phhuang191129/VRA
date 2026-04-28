#!/bin/bash
# Wrapper script to run HunyuanVideo inference with VRA on Modal.

# You can change the target sparsity here (e.g., "58", "sta58", or "91").
# "sta58" matches STA's sparse-step density (~58.3% sparsity).
SPARSITY="sta58"
# PROMPT="Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."
# PROMPT="A beautiful woman in a red dress walking down a street"
PROMPT="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere."

export MODAL_HUNYUAN_GPU=H100
# export FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=1
export FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=0
unset FASTVIDEO_TORCH_PROFILER_DIR
echo "Running HunyuanVideo VRA on Modal targeting ~${SPARSITY}% sparsity..."

# Run Modal. This command executes the modal script remotely, passing the VRA backend parameters.
# The VRA attention backend can dispatch either to the algorithm-side Triton
# prototype or to our native H100 kernel. This end-to-end script selects the
# optimized native H100 path explicitly.
modal run scripts/inference/modal_attention_hunyuan.py \
  --attention-backend "SLIDING_VARIABLE_RATE_ATTN" \
  --vra-sparsity "${SPARSITY}" \
  --vra-kernel-backend "h100" \
  --prompt "${PROMPT}" \
  --height 768 \
  --width 1280 \
  --num-frames 117 \
  --num-inference-steps 50 \
  --embedded-cfg-scale 6 \
  --flow-shift 7 \
  --target-cfg "pos" 



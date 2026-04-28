#!/bin/bash

num_gpus=1
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_hunyuan.json
export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
export FASTVIDEO_VRA_SPARSITY="58" 
export FASTVIDEO_ATTENTION_BACKEND="SLIDING_VARIABLE_RATE_ATTN"
FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=1
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size ${num_gpus} \
    --tp-size 1 \
    --num-gpus ${num_gpus} \
    --height 544 \
    --width 960 \
    --num-frames 75 \
    --num-inference-steps 50 \
    --guidance-scale 1 \
    --embedded-cfg-scale 6 \
    --flow-shift 7 \
    --prompt "A beautiful woman in a red dress walking down a street" \
    --seed 1024 \
    --output-path outputs_video/

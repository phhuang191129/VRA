#!/bin/bash

num_gpus=4
export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
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
# 544px960p
# 720x1280
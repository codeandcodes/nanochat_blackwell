#!/bin/bash
set -e

IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
MOUNT_DIR=$(pwd)

echo "Running FP8 training test inside Docker..."

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $MOUNT_DIR:/workspace/nanochat \
    -v $HOME/.cache/nanochat:/root/.cache/nanochat \
    -w /workspace/nanochat \
    --rm \
    $IMAGE \
    bash -c "pip install wandb tiktoken tokenizers datasets psutil && python -m scripts.base_train \
        --use_fp8=True \
        --depth=2 \
        --max_seq_len=256 \
        --device_batch_size=4 \
        --num_iterations=10 \
        --run=dummy \
        --save_every=0 \
        --eval_every=50"

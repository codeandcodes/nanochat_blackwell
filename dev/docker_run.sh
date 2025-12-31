#!/bin/bash
set -e

# Image that supports Blackwell and FP8 (CUDA 13.x + PyTorch 2.6+ + TE)
IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"

# Mount the current directory to /workspace/nanochat
MOUNT_DIR=$(pwd)

echo "Pulling latest image: $IMAGE..."
docker pull $IMAGE

echo "Launching container..."
# --gpus all: pass all GPUs
# --ipc=host: critical for avoiding shared memory errors in DDP
# -v: mount source code
# -w: set working directory
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $MOUNT_DIR:/workspace/nanochat \
    -v $HOME/.cache/nanochat:/root/.cache/nanochat \
    -w /workspace/nanochat \
    -it \
    --rm \
    $IMAGE \
    /bin/bash

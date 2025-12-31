#!/bin/bash
set -e

# FP8 Speedrun Script for NVIDIA Blackwell GPUs (Docker)
# Mirrors speedrun_local.sh functionality but runs inside Docker with FP8 enabled.

# Image that supports Blackwell and FP8 (CUDA 13.x + PyTorch 2.6+ + TE)
IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
MOUNT_DIR=$(pwd)
LOG_FILE="speedrun_local_fp8.log"

# Default settings
MODEL_TAG="d20_fp8"
WANDB_PROJECT="nanochat-fp8"
# Default settings
SKIP_TOKENIZER=false
BATCH_SIZE=20


# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip_tokenizer) SKIP_TOKENIZER=true ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="speedrun_fp8"
fi

# Pass WANDB_API_KEY if set
ENV_VARS=""
if [ ! -z "$WANDB_API_KEY" ]; then
    ENV_VARS="-e WANDB_API_KEY=$WANDB_API_KEY"
fi

# Detect wandb credentials file (.netrc)
WANDB_CREDS=""
if [ -f "$HOME/.netrc" ]; then
    echo "Found ~/.netrc, mounting it for W&B authentication..."
    WANDB_CREDS="-v $HOME/.netrc:/root/.netrc:ro"
fi

# Memory fragmentation mitigation
ALLOC_CONF="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "Detailed Speedrun Settings:"
echo "  Container: $IMAGE"
echo "  Model Tag: $MODEL_TAG"
echo "  W&B Project: $WANDB_PROJECT"
echo "  W&B Run Name: $WANDB_RUN"
echo "  Log File: $LOG_FILE"
echo "  Report File: report_local_fp8.md"


# Pass SKIP_TOKENIZER
ENV_VARS="$ENV_VARS -e SKIP_TOKENIZER=$SKIP_TOKENIZER -e PYTHONUNBUFFERED=1"

echo "Launching Docker container... (Logging to $LOG_FILE)"


# Container cleanup trap
CONTAINER_NAME="nanochat-fp8-run"
trap 'docker rm -f $CONTAINER_NAME' EXIT

# We wrap the docker run in a block and redirect output to tee
(
docker run --name $CONTAINER_NAME --gpus all --init --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $MOUNT_DIR:/workspace/nanochat \
    -v $HOME/.cache/nanochat:/root/.cache/nanochat \
    $ENV_VARS \
    $WANDB_CREDS \
    $ALLOC_CONF \
    -w /workspace/nanochat \
    --rm \
    $IMAGE \
    bash -c "
    set -e
    
    echo '-----------------------------------------------------------------------------'
    echo 'Step 0: Setup & Dependencies (Inside Docker)'
    echo 'Installing python dependencies...'
    pip install wandb tiktoken tokenizers datasets psutil files-to-prompt maturin fastapi uvicorn regex
    
    echo 'Installing Rust toolchain (for tokenizer)...'
    command -v cargo &> /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "\$HOME/.cargo/env"
    
    echo 'Building rustbpe tokenizer extension...'
    maturin build --release --manifest-path rustbpe/Cargo.toml --out dist
    pip install dist/*.whl --force-reinstall --no-deps
    
    echo 'Clear report...'
    python -m nanochat.report reset
    
    echo '-----------------------------------------------------------------------------'
    echo 'Step 0.5: Data & Tokenizer'
    
    echo 'Downloading data shards (if needed)...'
    python -m nanochat.dataset -n 8
    
    if [ "$SKIP_TOKENIZER" != "true" ]; then
        # We attempt to train/eval tokenizer. 
        # Note: If rustbpe binary is missing/incompatible, this might fail or fallback.
        echo 'Training Tokenizer...'
        python -m scripts.tok_train --max_chars=2000000000 || echo 'Warning: Tokenizer training failed (maybe missing rust compiler?), hoping for cached tokenizer.'
        
        echo 'Evaluating Tokenizer...'
        python -m scripts.tok_eval || echo 'Warning: Tokenizer eval failed.'
    else
        echo 'Skipping Tokenizer Training/Eval (SKIP_TOKENIZER=true)'
    fi

    echo '-----------------------------------------------------------------------------'
    echo 'Step 1: Base Model Pretraining (FP8)'
    
    # NOTE: user reported hang at step 0. Use torch.compile might take ~60s+ for first step.
    echo 'Starting Base Training... (First step may take >60s due to JIT compilation)'
    
    python -m scripts.base_train \
        --use_fp8=True \
        --model_tag=$MODEL_TAG \
        --wandb_project=$WANDB_PROJECT \
        --run=$WANDB_RUN \
        --log_file=$LOG_FILE \
        --depth=20 \
        --device_batch_size=$BATCH_SIZE
        
    echo 'Evaluating Base Model...'
    python -m scripts.base_loss \
        --model_tag=$MODEL_TAG \
        --run=$WANDB_RUN \
        --wandb_project=$WANDB_PROJECT
        
    python -m scripts.base_eval \
         --model_tag=$MODEL_TAG
        
    echo '-----------------------------------------------------------------------------'
    echo 'Step 2: Midtraining (FP8)'
    
    # Check for midtraining data
    if [ ! -f '/root/.cache/nanochat/identity_conversations.jsonl' ]; then
        echo 'Downloading identity_conversations.jsonl...'
        curl -L -o /root/.cache/nanochat/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi
    
    python -m scripts.mid_train \
        --use_fp8=True \
        --model_tag=$MODEL_TAG \
        --wandb_project=${WANDB_PROJECT}-mid \
        --run=$WANDB_RUN \
        --log_file=$LOG_FILE \
        --device_batch_size=$BATCH_SIZE
        
    python -m scripts.chat_eval --i mid --model_tag=$MODEL_TAG
        
    echo '-----------------------------------------------------------------------------'
    echo 'Step 3: Supervised Finetuning (FP8)'
    python -m scripts.chat_sft \
        --use_fp8=True \
        --model_tag=$MODEL_TAG \
        --wandb_project=${WANDB_PROJECT}-sft \
        --run=$WANDB_RUN \
        --log_file=$LOG_FILE \
        --device_batch_size=$BATCH_SIZE
        
    python -m scripts.chat_eval --i sft --model_tag=$MODEL_TAG
        
    echo '-----------------------------------------------------------------------------'
    echo 'Step 4: Generating Report'
    # Use the new --output argument we just added
    python -m nanochat.report generate --output report_local_fp8.md
    
    echo 'FP8 Speedrun Completed Successfully!'
    " 
) 2>&1 | stdbuf -oL -eL tee $LOG_FILE

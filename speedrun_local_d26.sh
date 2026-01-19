#!/bin/bash

# Modified speedrun.sh for local single-GPU execution
# Original: speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat_d26
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_d26"
mkdir -p $NANOCHAT_BASE_DIR

# Redirect output to log file
exec > >(tee speedrun_d26.log) 2>&1

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=speedrun_d26_local
fi

# -----------------------------------------------------------------------------
# Clear report
python -m nanochat.report reset --filename=report_d26.md

# -----------------------------------------------------------------------------
# Tokenizer
# Install Rust / Cargo
command -v cargo &> /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download data shards
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 450 &
DATASET_DOWNLOAD_PID=$!

# Train tokenizer
python -m scripts.tok_train --max_chars=2000000000
# Evaluate tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
# CUSTOMIZATION: Set to 1 for single-GPU execution
NPROC_PER_NODE=1

# pretrain the d26 model
# Note: With 1 GPU instead of 8, this will take 8x longer to reach the same number of tokens.
BASE_CKPT="$NANOCHAT_BASE_DIR/base_checkpoints/d26/model_040640.pt"
if [ -f "$BASE_CKPT" ]; then
    echo "Base checkpoint found at $BASE_CKPT, skipping base training..."
else
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=26 --device_batch_size=20 --total_batch_size=532480 --run=$WANDB_RUN --wandb_project=d26 --report_filename=report_d26.md --save_every=1000

    # evaluate the model
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
    # evaluate the model on CORE tasks
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
fi

# -----------------------------------------------------------------------------
# Midtraining

MID_CKPT="$NANOCHAT_BASE_DIR/mid_checkpoints/d26/model_000801.pt"
if [ -f "$MID_CKPT" ]; then
    echo "Mid checkpoint found at $MID_CKPT, skipping mid training..."
else
    if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
        curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=20 --total_batch_size=532480 --run=$WANDB_RUN --wandb_project=d26 --report_filename=report_d26.md
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
fi

# -----------------------------------------------------------------------------
# Supervised Finetuning

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device_batch_size=16 --run=$WANDB_RUN --wandb_project=d26 --report_filename=report_d26.md --target_examples_per_step=512 --gradient_checkpointing=True --embedding_lr=0.04 --matrix_lr=0.004 --unembedding_lr=0.0008 --num_epochs=10
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate --filename=report_d26.md

echo "Speedrun completed!"

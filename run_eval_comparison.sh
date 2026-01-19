#!/bin/bash
set -e

# Ensure logs directory exists
mkdir -p logs

echo "Starting evaluations in parallel..."

# Function to run evaluation pipeline
run_eval_pipeline() {
    MODEL_TAG=$1
    REPORT_DIR=$2
    REPORT_FILE="report_${MODEL_TAG}.md"
    
    echo "[$MODEL_TAG] Starting pipeline. Report dir: $REPORT_DIR"
    
    # Export the report directory for this subshell
    export NANOCHAT_REPORT_DIR="$REPORT_DIR"
    
    echo "[$MODEL_TAG] Resetting report..."
    python -m nanochat.report reset --filename "$REPORT_FILE"
    
    echo "[$MODEL_TAG] Evaluating Base model..."
    python -m scripts.base_eval --model-tag "$MODEL_TAG" > "logs/eval_${MODEL_TAG}_base.log" 2>&1
    
    echo "[$MODEL_TAG] Evaluating Mid checkpoint..."
    python -m scripts.chat_eval -i mid -g "$MODEL_TAG" > "logs/eval_${MODEL_TAG}_mid.log" 2>&1
    
    echo "[$MODEL_TAG] Evaluating SFT checkpoint..."
    python -m scripts.chat_eval -i sft -g "$MODEL_TAG" > "logs/eval_${MODEL_TAG}_sft.log" 2>&1
    
    echo "[$MODEL_TAG] Generating report..."
    python -m nanochat.report generate -f "$REPORT_FILE"
    
    echo "[$MODEL_TAG] Done! Created $REPORT_FILE"
}

# Run d26 pipeline
run_eval_pipeline "d26" "report_d26_data"

# Run d20 pipeline
run_eval_pipeline "d20" "report_d20_data"

echo "All evaluations completed."

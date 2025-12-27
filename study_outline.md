# nanochat Study Outline

This guide breaks down the `nanochat` repository into the key stages of an LLM pipeline. Follow this order to understand how a ChatGPT-like model is built from scratch.

## 1. Tokenization
Before any neural network, text must be converted into numbers (tokens). `nanochat` uses a Byte Pair Encoding (BPE) tokenizer, similar to GPT-4.

*   **Key Concept**: BPE (Byte Pair Encoding), Vocab size (65,536).
*   **Files to Study**:
    *   [`nanochat/tokenizer.py`](file:///home/rocketegg/workspace/nanochat/nanochat/tokenizer.py): The Python wrapper that handles encoding/decoding.
    *   [`rustbpe/src/lib.rs`](file:///home/rocketegg/workspace/nanochat/rustbpe/src/lib.rs): The core implementation in Rust for speed.
    *   [`scripts/tok_train.py`](file:///home/rocketegg/workspace/nanochat/scripts/tok_train.py): Logic for training the tokenizer on your dataset.

## 2. Data Pipeline
Efficiently feeding data to the GPU is critical.

*   **Key Concept**: Sharding, Memory Mapping, Distributed Sampling.
*   **Files to Study**:
    *   [`nanochat/dataset.py`](file:///home/rocketegg/workspace/nanochat/nanochat/dataset.py): Utilities for downloading and managing data shards.
    *   [`nanochat/dataloader.py`](file:///home/rocketegg/workspace/nanochat/nanochat/dataloader.py): The `tokenizing_distributed_data_loader` yields batches of tokens. It handles distributed sampling (splitting work across GPUs) and on-the-fly tokenization.

## 3. Model Architecture
The heart of the system is the Transformer.

*   **Key Concept**: GPT (Generative Pre-trained Transformer), Causal Self-Attention, RoPE (Rotary Positional Embeddings).
*   **Files to Study**:
    *   [`nanochat/gpt.py`](file:///home/rocketegg/workspace/nanochat/nanochat/gpt.py): The `GPT` class. Look for `Block` (Transformer layer), `CausalSelfAttention` (the core mechanism), and `MLP` (Feed-Forward Network). It's clean and readable.

## 4. Pretraining (The "Base" Model)
Training the model to predict the next token on a massive text corpus.

*   **Key Concept**: Cross Entropy Loss, Optimizers (AdamW, Muon), Gradient Accumulation, DDP (Distributed Data Parallel).
*   **Files to Study**:
    *   [`scripts/base_train.py`](file:///home/rocketegg/workspace/nanochat/scripts/base_train.py): The main training loop. Concepts like `device_batch_size` vs `total_batch_size`, learning rate scheduling, and the `Muon` optimizer usage.
    *   [`nanochat/muon.py`](file:///home/rocketegg/workspace/nanochat/nanochat/muon.py): A custom optimizer used for 2D parameters (matrices), often more efficient than Adam.
    *   [`nanochat/engine.py`](file:///home/rocketegg/workspace/nanochat/nanochat/engine.py): Used for generating samples during training to monitor progress.

## 5. Post-Training (Making it a "Chat" Assistant)
Raw base models just continue text. To make them helpful assistants, we need further training.

### A. Mid-Training & SFT (Supervised Fine-Tuning)
Teaching the model the "User"/"Assistant" format and specific behaviors.

*   **Files to Study**:
    *   [`scripts/mid_train.py`](file:///home/rocketegg/workspace/nanochat/scripts/mid_train.py): An intermediate stage to stabilize the model for chat.
    *   [`scripts/chat_sft.py`](file:///home/rocketegg/workspace/nanochat/scripts/chat_sft.py): Fine-tuning on high-quality conversation logs.

### B. Evaluation
Measuring performance on standardized tasks.

*   **Files to Study**:
    *   [`scripts/base_eval.py`](file:///home/rocketegg/workspace/nanochat/scripts/base_eval.py): Checking "loss" (perplexity) and simple metrics.
    *   [`scripts/chat_eval.py`](file:///home/rocketegg/workspace/nanochat/scripts/chat_eval.py): Evaluating conversational ability.
    *   [`nanochat/tasks/`](file:///home/rocketegg/workspace/nanochat/nanochat/tasks/): Definitions of tasks like GSM8K (math), MMLU (knowledge).

## 6. Inference & Serving
Using the model to chat.

*   **Key Concept**: KV Cache (Key-Value Cache) for faster generation.
*   **Files to Study**:
    *   [`nanochat/engine.py`](file:///home/rocketegg/workspace/nanochat/nanochat/engine.py): Implements the generation loop with KV caching.
    *   [`scripts/chat_web.py`](file:///home/rocketegg/workspace/nanochat/scripts/chat_web.py): A simple web server/backend.
    *   [`nanochat/ui.html`](file:///home/rocketegg/workspace/nanochat/nanochat/ui.html): The frontend interface.

## 7. The "Speedrun" Script
The glue that holds it all together.

*   [`speedrun.sh`](file:///home/rocketegg/workspace/nanochat/speedrun.sh): Orchestrates the entire pipeline from empty folder to chatted model. It's the best high-level map of the workflow.

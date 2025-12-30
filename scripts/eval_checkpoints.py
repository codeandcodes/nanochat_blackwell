
"""
Evaluate model checkpoints from the d20 run.
"""
import os
import glob
import re
import torch
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.engine import Engine

def main():
    # Configuration
    base_dir = os.path.expanduser("~/.cache/nanochat")
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints", "d20")
    output_file = "checkpoint_evals.txt"
    device_type = "cpu"
    
    prompts = [
        "The capital of France is",
        "The future of AI is",
        "Once upon a time",
        "def fibonacci(n):",
    ]

    print(f"Scanning checkpoints in {checkpoints_dir}...")
    
    # regex to extract step number from filename
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "model_*.pt"))
    steps = []
    for f in checkpoint_files:
        match = re.search(r"model_(\d+)\.pt", f)
        if match:
            step = int(match.group(1))
            if step > 0 and step % 2000 == 0:
                steps.append(step)
    
    steps.sort()
    
    # Check what's already done
    processed_steps = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            content = f.read()
            # Simple heuristic to find processed steps in the text file
            matches = re.findall(r"=== Step (\d+) ===", content)
            processed_steps = set(int(m) for m in matches)

    print(f"Found steps: {steps}")
    print(f"Already processed: {processed_steps}")
    
    # Process remaining steps
    _, _, _, _, device = compute_init(device_type)
    
    for step in steps:
        if step in processed_steps:
            print(f"Skipping step {step} (already done)")
            continue
            
        print(f"Evaluating step {step}...")
        
        try:
            # Load model on CPU
            model, tokenizer, meta = load_model_from_dir(
                os.path.join(base_dir, "base_checkpoints"), 
                device, 
                phase="eval", 
                model_tag="d20", 
                step=step
            )
            
            engine = Engine(model, tokenizer)
            
            results = []
            results.append(f"\n{'='*20} Step {step} {'='*20}\n")
            
            for prompt in prompts:
                tokens = tokenizer.encode(prompt)
                # Generate
                # Engine.generate_batch expects a single list of ints (tokens)
                out_tokens_batch, _ = engine.generate_batch(
                    tokens, 
                    num_samples=1, 
                    max_tokens=64, 
                    temperature=0.8, 
                    top_k=20
                )
                
                out_tokens = out_tokens_batch[0] # Single sample
                decoded = tokenizer.decode(out_tokens)
                
                results.append(f"PROMPT: {prompt}")
                results.append(f"OUTPUT: {decoded}")
                results.append("-" * 40)
                
            # Save results immediately
            with open(output_file, "a") as f:
                f.write("\n".join(results))
                f.write("\n")
                
            print(f"Step {step} completed.")
            
        except Exception as e:
            print(f"Error evaluating step {step}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

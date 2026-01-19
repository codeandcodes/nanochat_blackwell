import time
import torch
import statistics
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model, find_largest_model
from nanochat.engine import Engine

def benchmark_model(model_tag):
    print(f"\nBenchmarking {model_tag}...")
    
    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    
    # Load model
    try:
        model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=model_tag)
    except Exception as e:
        print(f"Could not load sft model for {model_tag}, trying base...")
        try:
             model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag)
        except:
             print(f"Failed to load {model_tag}")
             return

    engine = Engine(model, tokenizer)
    model.eval()

    # 1. Prefill Speed (Prompt Processing)
    # Long prompt (~1000 tokens)
    long_prompt = "The " * 1000
    tokens = tokenizer.encode(long_prompt)
    # Configuration
    prefill_lengths = [128, 1024, 2048]
    gen_tokens = 128
    iterations = 5

    # 1. Prefill Speed (Prompt Processing)
    print(f"\n--- Prefill Speed (Context Processing) ---")
    print(f"{'Context Length':<15} | {'Mean Speed (tok/s)':<20} | {'StdDev':<10}")
    print("-" * 55)

    for length in prefill_lengths:
        speeds = []
        tokens = [1] * length # dummy tokens
        input_ids = torch.tensor([tokens], device=device).long()
        
        # Warmup
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
             model(input_ids)

        for _ in range(iterations):
            t0 = time.time()
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                model(input_ids)
            torch.cuda.synchronize()
            t1 = time.time()
            speeds.append(length / (t1 - t0))
        
        mean_speed = statistics.mean(speeds)
        std_speed = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
        print(f"{length:<15} | {mean_speed:<20.2f} | {std_speed:<10.2f}")

    # 2. Decoding Speed (Generation)
    print(f"\n--- Decoding Speed (Generation of {gen_tokens} tokens) ---")
    print(f"{'Mean Speed (tok/s)':<20} | {'StdDev':<10}")
    print("-" * 35)
    
    decode_speeds = []
    prompt = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(prompt)
    
    # Warmup
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        engine.generate_batch(tokens, num_samples=1, max_tokens=10, temperature=0)

    for _ in range(iterations):
        t0 = time.time()
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            out_tokens, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=gen_tokens, temperature=0)
        torch.cuda.synchronize()
        t1 = time.time()
        
        gen_time = t1 - t0
        generated_count = len(out_tokens[0]) - len(tokens)
        decode_speeds.append(generated_count / gen_time)
        
    mean_decode = statistics.mean(decode_speeds)
    std_decode = statistics.stdev(decode_speeds) if len(decode_speeds) > 1 else 0.0
    print(f"{mean_decode:<20.2f} | {std_decode:<10.2f}")
    
    return mean_decode

if __name__ == "__main__":
    benchmark_model("d26")
    benchmark_model("d20")

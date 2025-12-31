import torch
import torch.nn as nn
try:
    from torchao.quantization import quantize_, float8_weight_only, float8_dynamic_activation_float8_weight
    print("torchao imported successfully.")
except ImportError as e:
    print(f"Failed to import torchao: {e}")
    exit(1)

def test_torchao_fp8():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test.")
        return

    device = "cuda"
    # Create a simple Linear layer
    bs = 16
    in_features = 64
    out_features = 127 # Odd number to check for padding/alignment issues
    
    model = nn.Sequential(
        nn.Linear(in_features, out_features, bias=False)
    ).to(device)
    
    x = torch.randn(bs, in_features, device=device, dtype=torch.bfloat16)
    
    print("Original model:")
    print(model)

    print("\nApplying Float8 Quantization (weights only)...")
    try:
        # Try weight-only first as a baseline
        quantize_(model, float8_weight_only())
        print("Quantization successful.")
        print(model)
        
        # Verify weight type
        print(f"Weight dtype: {model[0].weight.dtype}")
        
    except Exception as e:
        print(f"Quantization failed: {e}")
        return

    print("\nRunning Forward Pass...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
        print(f"Output shape: {y.shape}")
        print("Forward pass successful.")
        
        # Simple backward test (autograd should handle it if supported)
        loss = y.sum()
        try:
            loss.backward()
            print("Backward pass successful.")
        except Exception as e:
            print(f"Backward pass failed (expected for some quantized ops): {e}")

if __name__ == "__main__":
    test_torchao_fp8()

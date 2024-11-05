import torch
from model import GPT, GPTConfig
import math

def try_config(n_layer, n_head, n_embd, vocab_size=32):
    """Try a configuration and get actual PyTorch parameter count"""
    config = GPTConfig()
    config.n_layer = n_layer
    config.n_head = n_head
    config.n_embd = n_embd
    config.vocab_size = vocab_size
    config.block_size = 1024
    config.bias = False
    config.dropout = 0.2

    try:
        model = GPT(config)
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
    except Exception as e:
        return None

def find_optimal_dims(target_params=50_000_000, n_layer=8):
    """Find dimensions that give closest to target parameters"""
    best_diff = float('inf')
    best_config = None

    # Try different embedding dimensions
    for n_embd in range(384, 1025, 64):  # Must be divisible by head count
        # Try different head counts that divide n_embd
        for n_head in [h for h in range(4, 17) if n_embd % h == 0]:
            params = try_config(n_layer, n_head, n_embd)
            if params is None:
                continue

            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = (n_head, n_embd, params)

            # Early stop if we're very close
            if diff < 100_000:
                return best_config

    return best_config

print("\nSearching for optimal dimensions for 50M parameter model with 8 layers...")
n_head, n_embd, total_params = find_optimal_dims()

print(f"\nOptimal configuration:")
print(f"n_layer = 8")
print(f"n_head = {n_head}")
print(f"n_embd = {n_embd}")
print(f"\nTotal parameters: {total_params:,}")
print(f"Difference from target: {abs(total_params - 50_000_000):,}")
print(f"Percent difference: {abs(total_params - 50_000_000) / 50_000_000 * 100:.2f}%")
print(f"\nVerification:")
print(f"- n_embd ({n_embd}) is divisible by n_head ({n_head}): {n_embd % n_head == 0}")
print(f"- head dimension (n_embd/n_head): {n_embd//n_head}")

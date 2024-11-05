import math

def calculate_params(n_layer, n_head, n_embd, vocab_size=32):
    """Calculate total parameters for GPT model."""
    # Parameters in attention layers
    attn_params = n_layer * (3 * n_embd * n_embd + n_embd * n_head + n_embd)
    # Parameters in MLP layers
    mlp_params = n_layer * (4 * n_embd * n_embd + 4 * n_embd)
    # Parameters in embedding and final layer
    other_params = n_embd * vocab_size + n_embd * vocab_size
    return attn_params + mlp_params + other_params

def find_dimensions(target_params=50_000_000, n_layer=8):
    """Find n_head and n_embd that give closest to target parameters."""
    best_diff = float('inf')
    best_config = None

    # Try different n_embd values (must be divisible by n_head)
    for n_embd in range(512, 2049, 64):  # Step by 64 to maintain divisibility
        # Try different numbers of heads (must divide n_embd)
        for n_head in range(8, 33):  # Common head counts
            if n_embd % n_head != 0:  # Skip if not divisible
                continue

            params = calculate_params(n_layer, n_head, n_embd)
            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                best_config = (n_head, n_embd, params)

    return best_config

# Find dimensions for 50M parameter model with 8 layers
n_head, n_embd, total_params = find_dimensions()
percent_off = abs(total_params - 50_000_000) / 50_000_000 * 100

print(f"\nOptimal configuration for ~50M parameters with 8 layers:")
print(f"n_layer = 8")
print(f"n_head = {n_head}")
print(f"n_embd = {n_embd}")
print(f"\nTotal parameters: {total_params:,}")
print(f"Difference from target: {abs(total_params - 50_000_000):,} ({percent_off:.2f}%)")
print(f"\nVerification:")
print(f"- n_embd ({n_embd}) is divisible by n_head ({n_head}): {n_embd % n_head == 0}")
print(f"- head dimension (n_embd/n_head): {n_embd//n_head}")

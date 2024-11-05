import math

def calculate_params(n_layer, n_head, n_embd):
    """Calculate total parameters for GPT model."""
    # Parameters in attention layers
    attn_params = n_layer * (3 * n_embd * n_embd + n_embd * n_head + n_embd)

    # Parameters in MLP layers
    mlp_params = n_layer * (4 * n_embd * n_embd + 4 * n_embd)

    # Parameters in embedding and final layer
    other_params = n_embd * 32  # vocab_size=32

    total = attn_params + mlp_params + other_params
    return total

def find_closest_dimensions(target_params):
    """Find model dimensions that give closest to target parameters."""
    best_diff = float('inf')
    best_config = None

    # Search through reasonable ranges
    for n_layer in range(12, 40, 2):  # Try different layer counts
        for n_embd in range(512, 2049, 64):  # Try different embedding sizes
            # Set n_head to be a divisor of n_embd
            possible_heads = [h for h in range(8, 33, 8) if n_embd % h == 0]
            for n_head in possible_heads:
                params = calculate_params(n_layer, n_head, n_embd)
                diff = abs(params - target_params)

                if diff < best_diff:
                    best_diff = diff
                    best_config = (n_layer, n_head, n_embd, params)

                # If we're within 1% of target, that's good enough
                if diff / target_params < 0.01:
                    return best_config

    return best_config

# Target: 50M parameters
target = 50_000_000

# Find closest configuration
n_layer, n_head, n_embd, actual_params = find_closest_dimensions(target)

print(f"Found configuration for approximately {actual_params:,} parameters:")
print(f"n_layer: {n_layer}")
print(f"n_head: {n_head}")
print(f"n_embd: {n_embd}")
print(f"Difference from target: {abs(actual_params - target):,} ({abs(actual_params - target)/target*100:.2f}%)")

# Update configuration file with calculated dimensions
config_content = f"""# Model/training configuration matching blog specifications (50M parameters)
out_dir = 'out-chess-blog'
eval_interval = 100
eval_iters = 100
log_interval = 10

# Model parameters (~50M parameters)
n_layer = {n_layer}  # Calculated for 50M parameter target
n_head = {n_head}   # Calculated for attention efficiency
n_embd = {n_embd}  # Calculated for 50M parameters
dropout = 0.2
bias = False

# Data
dataset = 'lichess_hf_dataset'
batch_size = 12      # Adjusted for memory constraints
block_size = 1024    # Matches blog's context window
vocab_size = 32      # Matches blog's vocabulary size
device = 'cpu'       # Will be adjusted for GPU when available
dtype = 'float32'

# Training
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 1000
compile = False      # Will enable for GPU training

# Game delimiter configuration
game_delimiter = ';1.'  # Matches blog's specification
"""

with open('config/train_chess_blog.py', 'w') as f:
    f.write(config_content)

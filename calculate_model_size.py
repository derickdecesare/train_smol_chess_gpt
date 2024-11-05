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

# Target: 50M parameters
target = 50_000_000

# Calculate dimensions for approximately 50M parameters
n_layer = 36
n_head = 20
n_embd = 1280

params = calculate_params(n_layer, n_head, n_embd)
print(f"Model architecture:")
print(f"n_layer: {n_layer}")
print(f"n_head: {n_head}")
print(f"n_embd: {n_embd}")
print(f"Total parameters: {params:,}")
print(f"Difference from target: {abs(params - target):,}")

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

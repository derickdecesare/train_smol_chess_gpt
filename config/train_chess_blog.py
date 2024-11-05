# Model/training configuration matching blog specifications (50M parameters)
out_dir = 'out-chess-blog'
eval_interval = 100
eval_iters = 100
log_interval = 10

# Model parameters (48.33M parameters, 3.33% under target)
n_layer = 8   # Blog mentions 8 layers achieving good performance
n_head = 4    # Calculated for optimal parameter count
n_embd = 704  # Calculated for ~50M target, divisible by n_head
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

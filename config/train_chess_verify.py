# Model configuration for setup verification (1.8M parameters)
out_dir = 'out-chess-verify'
eval_interval = 25    # More frequent evaluation for quick verification
eval_iters = 25      # Reduced iterations for faster evaluation
log_interval = 1     # Log every iteration for visibility

# Model parameters (~1.8M parameters for quick CPU training)
n_layer = 4    # Reduced layers for verification
n_head = 4     # Matching blog's attention pattern
n_embd = 256   # Small embedding for CPU training
dropout = 0.2
bias = False

# Data
dataset = 'lichess_hf_dataset'
batch_size = 8       # Small batch for CPU
block_size = 1024    # Matches blog's context window
vocab_size = 32      # Matches blog's vocabulary size
device = 'cpu'
dtype = 'float32'

# Training
max_iters = 500      # Short training for verification
lr_decay_iters = 500
min_lr = 1e-4
learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 50
compile = False

# Game delimiter configuration
game_delimiter = ';1.'  # Matches blog's specification

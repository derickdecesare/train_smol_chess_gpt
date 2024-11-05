# Model/training configuration for initial tiny model test
out_dir = 'out-chess-tiny'
eval_interval = 25 # keep frequent evaluation for testing
eval_iters = 25
log_interval = 1 # don't wait too long to see results

# Tiny model parameters (~1.5M parameters)
n_layer = 20  # Balanced number of layers for pattern recognition
n_head = 20  # Balanced number of attention heads
n_embd = 640  # Balanced embedding dimension
dropout = 0.2  # Added more dropout for regularization
bias = False

# Data
dataset = 'lichess_hf_dataset'  # Set correct dataset directory
batch_size = 2  # Balanced batch size for training stability
block_size = 384  # Balanced context window for move sequences
vocab_size = 32
device = 'cpu'  # Use CPU for training
dtype = 'float32'  # Use float32 for CPU training

# Training
max_iters = 4000  # Balanced number of iterations
lr_decay_iters = 4000
min_lr = 1e-4
learning_rate = 1.5e-4  # Balanced learning rate for stability
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5  # Reduced gradient clipping for stability
decay_lr = True
warmup_iters = 400  # Balanced warmup period
compile = False  # Disable compilation for CPU training

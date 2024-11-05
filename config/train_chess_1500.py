# Small model configuration targeting 1500+ Elo chess play
# Based on blog requirements and nanoGPT architecture

out_dir = 'out-chess-1500'
eval_interval = 250 # keep frequent evaluation for initial training
eval_iters = 100
log_interval = 10 # don't print too often

# we'll start with a small model for quick verification
# but ensure architecture can support chess understanding
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# Adam optimizer with chess-specific learning rate
learning_rate = 1e-4
max_iters = 50000
lr_decay_iters = 45000
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.95

# Data loading
batch_size = 32
block_size = 1024  # context window size from blog
gradient_accumulation_steps = 4

# Compile for slight speedup
compile = True

# Dataset
dataset = 'lichess_hf_dataset'
gradient_checkpointing = True

# Evaluation settings for chess
eval_only = False
always_save_checkpoint = True

# wandb logging
wandb_log = False  # disabled for initial testing
wandb_project = 'chess-1500'
wandb_run_name = 'mini-gpt-chess'

# Sampling settings for move generation
temperature = 0.8  # slightly lower than 1.0 for more focused moves
top_k = 10        # limit to top moves for better quality

# init_from = 'resume'  # instead of default 'scratch' to pickup where we left off
out_dir = "out-chess-mac"  # changed to reflect this is our Mac run
eval_interval = 1000       # reduced since we'll have fewer iterations per epoch with smaller batch
eval_iters = 100
log_interval = 50         # don't log too often

always_save_checkpoint = True

wandb_log = False
wandb_project = "chess-gpt-batch"
wandb_run_name = "mac_test_chess"

dataset = "lichess_hf_dataset"
gradient_accumulation_steps = 4  # added to compensate for smaller batch size
batch_size = 2                  # reduced for Mac memory constraints
block_size = 1023

# Model architecture
n_layer = 8
n_head = 4                # reduced to match smaller n_embd
n_embd = 256             # reduced from 512
dropout = 0.0
# vocab_size = 32  # our chess character vocabulary

# Training parameters
learning_rate = 3e-4
max_iters = 140000       # reduced since this is a test run
lr_decay_iters = max_iters
min_lr = 3e-5
beta2 = 0.95

# dataset_size = 100K games # this is our goal so that we will see each game 2.8 times with max_iters of 140,000


warmup_iters = 1000      # reduced proportionally with max_iters

# Mac specific settings
device = 'mps'
compile = False          # no torch compile for Mac compatibility
# Small chess model configuration for CPU training
out_dir = "out-chess-small"
eval_interval = 200  # evaluate less frequently to save time
eval_iters = 50
log_interval = 20

always_save_checkpoint = True

wandb_log = False
wandb_project = "chess-gpt-batch"
wandb_run_name = "small_chess_test"

dataset = "lichess_hf_dataset"
gradient_accumulation_steps = 8  # increased to simulate larger batch
batch_size = 16  # increased batch size, still manageable for CPU
block_size = 1024  # keep full context window

# small model configuration (~2M parameters)
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0

learning_rate = 1e-3  # slightly higher for faster initial learning
max_iters = 50000  # increased iterations for better coverage
lr_decay_iters = max_iters
min_lr = 1e-4

warmup_iters = 1000  # increased warmup for stability
compile = False  # disable compilation for CPU training

# force CPU training
device = 'cpu'
dtype = 'float32'  # use float32 on CPU

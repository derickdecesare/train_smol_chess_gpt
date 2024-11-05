# Model/training configuration for initial tiny model test
out_dir = 'out-chess-tiny'
eval_interval = 250 # keep frequent evaluation for testing
eval_iters = 100
log_interval = 10 # don't wait too long to see results

# Tiny model parameters (~1.5M parameters)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Data
batch_size = 12
block_size = 1024
vocab_size = 32

# Training
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100

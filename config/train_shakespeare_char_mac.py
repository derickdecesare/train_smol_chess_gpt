# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

# Proposed mini-chess-gpt config
# n_layer = 8
# n_head = 4
# n_embd = 256
# batch_size = 2

out_dir = 'out-shakespeare-char'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'chess-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 2
block_size = 1024 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
# compile = True

# on macbook also add
device = 'mps'  # run on cpu only
compile = False # do not torch compile the model

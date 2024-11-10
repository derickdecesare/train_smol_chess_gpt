"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# python sample.py --init_from=resume --out_dir=out-chess-mac
# python sample.py --init_from=resume --out_dir=out-chess-mac --start=";1."
# python sample.py --init_from=resume --out_dir=out-chess-mac --start=";1. e4"
# python sample.py --init_from=resume --out_dir=out-chess-mac --start=";1.e4"
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device = 'cpu' # Uncomment if on Mac
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) # nullcontext() is a do nothing context manager

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt') # automatically created path compatible with current operating system # out-chess-mac/ckpt.pt
    checkpoint = torch.load(ckpt_path, map_location=device) # load the checkpoint, ensuring to load to the correct device -- eg cpu or gpu (map_location)
    gptconf = GPTConfig(**checkpoint['model_args']) # create GPTConfig object from saved model args imported from model.py ** unpacks the dictionary into keyword arguments
    # Example of ** unpacking
    # config_dict = {
    #     'n_layer': 8,
    #     'n_head': 4,
    #     'n_embd': 256
    # }

    # we initialize with the config from the checkpoint to ensure consistency
    # # These two lines are equivalent:
    # gptconf = GPTConfig(n_layer=8, n_head=4, n_embd=256)
    # gptconf = GPTConfig(**config_dict)

    model = GPT(gptconf) # Creates a new model with random weights based on our config that we just initialized from checkpoint 
    state_dict = checkpoint['model'] # Gets the learned weights from checkpoint
    unwanted_prefix = '_orig_mod.' # Handle pytorch quirk where sometimes model params get saved with an extra prefix "_orig_mod." -- we strip this off
    for k,v in list(state_dict.items()): # Do that for all keys in state_dict--> 'orig_mod.layer1.weight' becomes 'layer1.weight'
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict) # Connects the learned weights to our model after we have cleaned everything up # takes all learned params and assigns them to corresponding layers in our model
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))


# Tell PyTorch we're in inference mode, not training mode
model.eval() # Disables dropout, batch normalization changes, etc.
# Move the model to the specified device (CPU/GPU)
model.to(device)
# Optionally compile the model for speed optimization
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional) --> not for Mac bc designed for Nvidia GPUs to make go faster

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these..
    # Construct path to meta.pkl using dataset name from config
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path) # Check if file exists
if load_meta:
    print(f"Loading meta from {meta_path}...")
    # Load the tokenizer mapping from pickle file
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    # Get string-to-int (stoi) and int-to-string (itos) mappings
    stoi, itos = meta['stoi'], meta['itos']
    # Define encode/decode functions using list comprehension
    encode = lambda s: [stoi[c] for c in s]  # Convert string to token integers
    decode = lambda l: ''.join([itos[i] for i in l]) # Convert token integers back to string
else:
    # Fallback to GPT-2's tokenizer if no custom tokenizer found
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f: # start[5:] removes "FILE:"
        start = f.read() # Read the contents of the file into a string (only for prompt file)
start_ids = encode(start) # if there was no file then we just encode the string directly
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]) # Create tensor from encoded start string, add batch dimension
# The [None, ...] adds a batch dimension: (could also use x = x.unsqueeze(0))
# x.shape = [1, sequence_length]
# pytorch expects x to be a tensor of shape [batch_size, sequence_length]
# So in steps... 
# First start_ids becomes tensor ---> x = [1, 2, 3] --> x.shape = [3]
# Then we add a batch dimension with x = x[None, ...] --> x.shape = [1, 3]
# Appends it to the front of the tensor
# batch_size is number of sequences we are processing in parallel.. in this case 1
# We trained with [batch_size, sequence_length] so we need to add this batch dimension to match the training data format

# run generation
with torch.no_grad(): # with keyword is for context management --ensures proper setup/cleanup
    with ctx: # ctx is a context manager that handles device-specific settings
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # y is a tensor of shape [batch_size, sequence_length]
            print(decode(y[0].tolist())) # Convert tensor to list and then decode to string
            print('---------------')

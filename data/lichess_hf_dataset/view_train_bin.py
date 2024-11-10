import os
import numpy as np
import pickle

# Load the meta.pkl to get itos (integer to string mapping)
meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

itos = meta['itos']

# Function to decode a list of integers to a string
def decode(int_list):
    return ''.join([itos[i] for i in int_list])

# Read the train.bin file
train_bin_path = os.path.join(os.path.dirname(__file__), "train.bin")
train_data = np.memmap(train_bin_path, dtype=np.uint8, mode='r')

# Let's read the first N tokens and decode them
N = 1000  # Adjust this number to read more or fewer tokens
sample_ints = train_data[:N]
sample_text = decode(sample_ints)

print("First {} tokens decoded from train.bin:".format(N))
print(sample_text)
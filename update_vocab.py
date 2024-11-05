import os
import pickle

# Define the complete vocabulary including the semicolon
vocab = {
    'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4,
    'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9,
    'f': 10, 'g': 11, 'h': 12,
    '1': 13, '2': 14, '3': 15, '4': 16,
    '5': 17, '6': 18, '7': 19, '8': 20,
    '+': 21, '#': 22, 'x': 23, '=': 24,
    'O': 25, '-': 26, '.': 27, ' ': 28,
    '\n': 29, ';': 30, '0': 31  # Removed (, ), added ; and 0 for move numbers
}

# Create reverse mapping
itos = {v: k for k, v in vocab.items()}

# Create meta dictionary
meta = {
    'stoi': vocab,
    'itos': itos,
    'vocab_size': len(vocab)
}

# Save to meta.pkl
meta_path = os.path.join('data', 'lichess_hf_dataset', 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Updated meta.pkl with {len(vocab)} tokens including semicolon for game delimiter")
print("Vocabulary includes all necessary chess tokens and game delimiter ';1.'")

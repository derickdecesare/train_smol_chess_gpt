import pickle

# Define chess tokens as per blog requirements
tokens = {
    # Pieces
    'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4,
    # Files
    'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'h': 12,
    # Ranks
    '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20,
    # Special characters
    '+': 21,  # check
    '#': 22,  # checkmate
    'x': 23,  # capture
    '=': 24,  # pawn promotion
    'O': 25,  # castling
    '-': 26,  # castling
    '.': 27,  # move number separator
    ' ': 28,  # space
    '\n': 29, # newline
    '(': 30,  # annotation start
    ')': 31,  # annotation end
}

# Create reverse mapping
itos = {v: k for k, v in tokens.items()}

# Create meta dictionary
meta = {
    'stoi': tokens,  # string to index
    'itos': itos    # index to string
}

# Save to meta.pkl
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print('=== Created meta.pkl with chess tokens ===')
print(f'Total tokens: {len(tokens)}')
print('\nToken mapping:')
for char, idx in sorted(tokens.items(), key=lambda x: x[1]):
    print(f'{idx:2d}: {repr(char)}')

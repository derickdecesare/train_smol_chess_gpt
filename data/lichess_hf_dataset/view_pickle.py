import pickle
import json
import os  # Add this import

# Load the pickle file using the correct path
meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

# Print in a nice format
print("\nString to Integer (stoi) mapping:")
print(json.dumps(meta['stoi'], indent=2))

print("\nInteger to String (itos) mapping:")
print(json.dumps(meta['itos'], indent=2))

# Also show total vocab size
print(f"\nTotal vocabulary size: {len(meta['stoi'])}")

# Maybe print some example mappings
print("\nExample mappings:")
print(f"Space character -> {meta['stoi'][' ']}")
print(f"Number 1 -> {meta['stoi']['1']}")
print(f"Piece 'e' -> {meta['stoi']['e']}")
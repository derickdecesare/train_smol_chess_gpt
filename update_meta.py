import os
import pickle

meta_path = "data/lichess_hf_dataset/meta.pkl"
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

# Add vocab_size while preserving existing mappings
meta["vocab_size"] = len(meta["stoi"])

# Save updated meta dictionary
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

print(f"Updated meta.pkl with vocab_size = {meta['vocab_size']}")

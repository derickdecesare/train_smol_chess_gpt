import os
import torch
import numpy as np
from pathlib import Path
import importlib.util
import pickle

def load_config():
    spec = importlib.util.spec_from_file_location('config', 'config/train_chess_blog.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def calculate_params(config):
    n_layer, n_head, n_embd = config.n_layer, config.n_head, config.n_embd
    # Parameters in attention layers
    attn_params = n_layer * (3 * n_embd * n_embd + n_embd * n_head + n_embd)
    # Parameters in MLP layers
    mlp_params = n_layer * (4 * n_embd * n_embd + 4 * n_embd)
    # Parameters in embedding and final layer
    other_params = n_embd * config.vocab_size
    return attn_params + mlp_params + other_params

def check_data_files():
    data_dir = Path('data/lichess_hf_dataset')
    required_files = ['meta.pkl', 'train.bin', 'val.bin']
    status = {}
    for file in required_files:
        path = data_dir / file
        status[file] = {
            'exists': path.exists(),
            'size': path.stat().st_size if path.exists() else 0
        }
    return status

def check_vocab():
    try:
        with open('data/lichess_hf_dataset/meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        return {
            'vocab_size': len(meta['stoi']),
            'has_delimiter_tokens': all(c in meta['stoi'] for c in ';1.')
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    print("\n=== Training Setup Verification ===\n")

    # Check configuration
    config = load_config()
    params = calculate_params(config)
    print("Model Configuration:")
    print(f"Parameters: {params:,} (~50M target)")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding Dim: {config.n_embd}")
    print(f"Context Window: {config.block_size}")
    print(f"Vocabulary Size: {config.vocab_size}")

    # Check system
    print("\nSystem Resources:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CPU Cores: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        print("Running on CPU only")

    # Check data files
    print("\nData Files:")
    data_status = check_data_files()
    for file, status in data_status.items():
        exists = "✓" if status['exists'] else "✗"
        size = f"{status['size']/1e6:.1f}MB" if status['exists'] else "N/A"
        print(f"{exists} {file}: {size}")

    # Check vocabulary
    print("\nVocabulary Check:")
    vocab_status = check_vocab()
    if 'error' not in vocab_status:
        print(f"Vocabulary Size: {vocab_status['vocab_size']} (target: 32)")
        print(f"Delimiter Tokens: {'✓' if vocab_status['has_delimiter_tokens'] else '✗'}")
    else:
        print(f"Error checking vocabulary: {vocab_status['error']}")

if __name__ == "__main__":
    main()

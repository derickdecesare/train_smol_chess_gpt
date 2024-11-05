import os
import sys
import time
import pickle
import numpy as np
from datetime import datetime

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def load_meta():
    try:
        with open('data/lichess_hf_dataset/meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        return meta
    except Exception as e:
        print(f"Error loading meta.pkl: {e}")
        return None

def verify_binary_files():
    files = {
        'train.bin': {'path': 'data/lichess_hf_dataset/train.bin', 'exists': False, 'size': 0},
        'val.bin': {'path': 'data/lichess_hf_dataset/val.bin', 'exists': False, 'size': 0}
    }

    for name, info in files.items():
        if os.path.exists(info['path']):
            info['exists'] = True
            info['size'] = os.path.getsize(info['path'])

    return files

def verify_vocabulary(meta):
    if not meta:
        return False

    required_tokens = {
        'pieces': {'K', 'Q', 'R', 'B', 'N'},
        'coordinates': set('abcdefgh123456789'),
        'special': {'+', '-', 'x', '#', '=', ';', '.'}
    }

    vocab = set()
    if hasattr(meta, 'stoi'):
        vocab = set(meta.stoi.keys())
    elif isinstance(meta, dict) and 'stoi' in meta:
        vocab = set(meta['stoi'].keys())

    missing = {}
    for category, tokens in required_tokens.items():
        missing_tokens = tokens - vocab
        if missing_tokens:
            missing[category] = missing_tokens

    return not missing, missing

def main():
    print(f"=== Chess Data Verification Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")

    # Check meta.pkl
    print("Checking meta.pkl...")
    meta = load_meta()
    if meta:
        print("✓ meta.pkl found and loaded")
        vocab_ok, missing_tokens = verify_vocabulary(meta)
        if vocab_ok:
            print("✓ All required tokens present in vocabulary")
        else:
            print("✗ Missing tokens:")
            for category, tokens in missing_tokens.items():
                print(f"  - {category}: {', '.join(tokens)}")
    else:
        print("✗ Could not load meta.pkl")

    # Check binary files
    print("\nChecking binary files...")
    files = verify_binary_files()
    for name, info in files.items():
        if info['exists']:
            print(f"✓ {name} found ({format_size(info['size'])})")
        else:
            print(f"✗ {name} not found")

    # Overall status
    all_files_present = all(info['exists'] for info in files.values())
    vocab_ok = meta is not None and vocab_ok

    print("\nOverall Status:")
    if all_files_present and vocab_ok:
        print("✓ All requirements met - ready for training")
    else:
        print("✗ Some requirements not met - not ready for training")
        if not all_files_present:
            print("  - Waiting for binary files to be generated")
        if not vocab_ok:
            print("  - Vocabulary requirements not met")

if __name__ == '__main__':
    main()

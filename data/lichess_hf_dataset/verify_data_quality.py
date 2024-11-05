"""
Helper script to verify data quality and tokenization for chess training.
"""
import os
import pickle
import numpy as np
from datasets import load_dataset

def verify_tokenization():
    """Verify the tokenization mapping is correct."""
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    required_tokens = {
        'pieces': ['K', 'Q', 'R', 'B', 'N'],
        'files': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        'ranks': ['1', '2', '3', '4', '5', '6', '7', '8'],
        'special': ['+', '#', 'x', '=', 'O', '-', '.', ' ', '\n', '(', ')']
    }

    stoi = meta['stoi']
    missing = []

    for category, tokens in required_tokens.items():
        for token in tokens:
            if token not in stoi:
                missing.append(f"{token} ({category})")

    if missing:
        print("❌ Missing required tokens:", ', '.join(missing))
        return False

    print("✅ All required tokens present")
    print("\nToken mapping:")
    for k, v in sorted(stoi.items(), key=lambda x: x[1]):
        print(f"{v:2d}: {repr(k)}")
    return True

def verify_binary_files():
    """Verify the binary training files exist and have correct sizes."""
    expected_files = ['train.bin', 'val.bin']
    all_good = True

    for fname in expected_files:
        if not os.path.exists(fname):
            print(f"❌ Missing {fname}")
            all_good = False
            continue

        size_mb = os.path.getsize(fname) / (1024 * 1024)
        print(f"✅ {fname}: {size_mb:.1f} MB")

        # Quick content check
        data = np.memmap(fname, dtype=np.uint8, mode='r')
        if len(data) == 0:
            print(f"❌ {fname} is empty!")
            all_good = False
        else:
            print(f"   - Contains {len(data):,} tokens")
            print(f"   - Token range: {data.min()}-{data.max()} (should be 0-31)")
            if data.max() > 31:
                print(f"❌ Invalid tokens found in {fname}")
                all_good = False

    return all_good

def verify_dataset_access():
    """Verify we can access the HuggingFace dataset."""
    try:
        dataset = load_dataset("adamkarvonen/chess_games")
        print(f"✅ Dataset loaded successfully")
        print(f"   - Total games: {len(dataset['train']):,}")

        # Sample a few games
        print("\nSample game:")
        game = dataset['train'][0]['game']
        print(game[:200] + "..." if len(game) > 200 else game)
        return True
    except Exception as e:
        print(f"❌ Failed to load dataset: {str(e)}")
        return False

def main():
    print("=== Chess Training Data Verification ===\n")

    # Verify each component
    tok_ok = verify_tokenization()
    print("\n" + "="*50 + "\n")

    data_ok = verify_dataset_access()
    print("\n" + "="*50 + "\n")

    bin_ok = verify_binary_files()

    # Overall status
    print("\n=== Overall Status ===")
    if tok_ok and data_ok and bin_ok:
        print("✅ All checks passed! Ready for training.")
    else:
        print("❌ Some checks failed. Please review the issues above.")

if __name__ == '__main__':
    main()

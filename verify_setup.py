"""
Verify that all required dependencies are correctly installed
and configured for the chess model training environment.
"""

def verify_environment():
    """Check if all dependencies are correctly installed."""
    try:
        # Check PyTorch
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  Metal available: {torch.backends.mps.is_available()}")
        print(f"  Metal built: {torch.backends.mps.is_built()}")

        # Check other packages
        import numpy as np
        print(f"✓ NumPy {np.__version__}")

        from datasets import load_dataset
        print("✓ Hugging Face Datasets")

        import tiktoken
        print("✓ Tiktoken")

        import wandb
        print("✓ Weights & Biases")

        # Additional checks
        if torch.backends.mps.is_available():
            # Test Metal device
            try:
                x = torch.randn(2, 3).to("mps")
                print("✓ Metal device working")
            except Exception as e:
                print(f"✗ Metal device error: {str(e)}")

        # Test dataset access
        try:
            dataset = load_dataset("adamkarvonen/chess_games", split="train")
            print("✓ Chess dataset accessible")
        except Exception as e:
            print(f"✗ Dataset access error: {str(e)}")

        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Chess Training Environment Verification ===\n")
    if verify_environment():
        print("\n✅ Environment setup complete! Ready for training.")
    else:
        print("\n❌ Please install missing dependencies.")

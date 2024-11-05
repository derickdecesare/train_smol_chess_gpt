# RunPod GPU Setup Guide for Chess Model Training

## Overview
This guide details how to set up and train the chess model (47.61M parameters) on RunPod GPUs, following Adam Karvonen's blog specifications.

## Model Specifications
- Parameters: 47.61M (matches blog's ~50M target)
- Architecture:
  - 8 layers (verified in blog)
  - 4 attention heads
  - 704 embedding dimensions
  - 32-token vocabulary
  - 1024 token context window
  - Game delimiter: ";1."

## RunPod Setup Instructions

### 1. Create RunPod Account
1. Sign up at [RunPod.io](https://runpod.io)
2. Add payment method for GPU rental

### 2. Select GPU Configuration
- Recommended: 4x RTX 3090 (matches blog setup)
- Minimum: 1x RTX 3090 (longer training time)
- Container: PyTorch 2.1.0
- Disk Space: 100GB minimum

### 3. Environment Setup
```bash
# Clone repository
git clone https://github.com/derickdecesare/train_smol_chess_gpt.git
cd train_smol_chess_gpt

# Install dependencies
pip install -r requirements.txt
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Prepare dataset
cd data/lichess_hf_dataset
python prepare.py
cd ../..
```

### 4. Training Configuration
Use `config/train_chess_blog.py` for the full 47.61M parameter model:
```python
# Key configurations
n_layer = 8
n_head = 4
n_embd = 704
vocab_size = 32
block_size = 1024
batch_size = 12  # Adjust based on GPU memory
```

### 5. Multi-GPU Training
```bash
# Start training on all available GPUs
torchrun --nproc_per_node=4 train.py config/train_chess_blog.py
```

### 6. Single-GPU Training
```bash
# For single GPU setup
CUDA_VISIBLE_DEVICES=0 python train.py config/train_chess_blog.py
```

### 7. Monitoring Training
- Loss values should start around 3.5
- Target: 1500+ Elo rating
- Training time: ~24-48 hours (based on blog)
- Use TensorBoard or WandB for monitoring

## Checkpointing and Model Export
- Checkpoints saved in `out-chess-blog/`
- Best model saved as `ckpt.pt`
- Use `sample.py` to test move generation

## Troubleshooting
1. Out of Memory (OOM):
   - Reduce batch_size
   - Enable gradient checkpointing
   - Use mixed precision training

2. Training Instability:
   - Adjust learning rate
   - Increase warmup_iters
   - Check gradient clipping

3. Poor Move Generation:
   - Verify game delimiter (";1.")
   - Check vocabulary size (32 tokens)
   - Ensure sufficient training time

## Resource Management
- Monitor GPU utilization with `nvidia-smi`
- Estimated cost: $2-3/hour for 4x RTX 3090
- Save checkpoints regularly
- Use spot instances for cost savings

## References
- [Original Blog Post](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html)
- [RunPod Documentation](https://docs.runpod.io/)

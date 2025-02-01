# Chess GPT Training Guide

## Overview

This guide explains how to train a GPT model on chess games using a custom 32-token vocabulary. The process involves two main steps:

1. Data preparation
2. Model training

## Data Preparation

### 1. Prepare the Chess Dataset

The data preparation script (`data/lichess_hf_dataset/prepare.py`) does the following:

- Downloads chess games from HuggingFace dataset
- Uses a 32-token vocabulary for chess moves
- Creates binary files for training
- Saves vocabulary information

bash
python data/lichess_hf_dataset/prepare.py
bash

This creates:

- `train.bin` and `val.bin`: Binary files containing tokenized chess games
- `meta.pkl`: Contains vocabulary information (size=32) and token mappings

### Vocabulary Details

The chess vocabulary consists of 32 tokens:
{
" ": 0, "#": 1, "+": 2, "-": 3,
".": 4, "0": 5, "1": 6, "2": 7,
"3": 8, "4": 9, "5": 10, "6": 11,
"7": 12, "8": 13, "9": 14, ";": 15,
"=": 16, "B": 17, "K": 18, "N": 19,
"O": 20, "Q": 21, "R": 22, "a": 23,
"b": 24, "c": 25, "d": 26, "e": 27,
"f": 28, "g": 29, "h": 30, "x": 31
}

## Model Training

### 1. Configuration

The training configuration (`config/mac_chess_gpt.py`) specifies:
Model architecture
n_layer = 8
n_head = 4
n_embd = 256
dropout = 0.0
Training parameters
dataset = "lichess_hf_dataset"
batch_size = 2
block_size = 1023
learning_rate = 3e-4
max_iters = 140000
Mac-specific settings
device = 'mps'
compile = False

### 2. Start Training

bash
python train.py config/mac_chess_gpt.py
bash

The training script:

- Automatically detects vocab_size=32 from meta.pkl
- Initializes model with correct architecture
- Loads data from binary files
- Trains the model
- Saves checkpoints to `out-chess-mac` directory

### 3. Monitor Training

- Loss values are printed during training
- Checkpoints saved to `out-chess-mac` directory
- Can enable wandb logging by setting `wandb_log = True` in config

### 4. Generate Samples

After training, generate chess moves using:

bash
python sample.py --out_dir=out-chess-mac
bash

bash
python sample.py --out_dir=out-chess-mac --start=";1.e4"
bash

## Key Files

- `data/lichess_hf_dataset/prepare.py`: Data preparation
- `config/mac_chess_gpt.py`: Training configuration
- `train.py`: Main training script
- `sample.py`: Generation script

## Notes

- Each chess game is exactly 1024 tokens long
- Games start with ";" delimiter
- Model automatically uses correct vocabulary size
- Mac-specific optimizations included in config
- Checkpoints saved regularly during training

## Troubleshooting

- If out of memory, reduce batch_size or model size
- For Mac, ensure device='mps' and compile=False
- Verify meta.pkl exists before training
- Check train.bin and val.bin were created properly

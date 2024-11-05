# Chess GPT Training Guide for Mac

## Getting Started

### Cloning and Exploring the Repository

1. **Clone the Repository**
```bash
# Create a directory for the project
mkdir chess_gpt && cd chess_gpt

# Clone the repository
git clone https://github.com/derickdecesare/train_smol_chess_gpt.git
cd train_smol_chess_gpt
```

2. **Repository Structure**
```
train_smol_chess_gpt/
├── model.py           # Neural network architecture
├── train.py          # Training loop and logic
├── sample.py         # Generate chess moves
├── config/           # Training configurations
│   └── train_chess_small.py  # Small model config for testing
├── data/             # Dataset preparation
│   └── lichess_hf_dataset/   # Chess game data
└── verify_setup.py   # Environment verification
```

3. **Key Components**

- **Model Architecture** (`model.py`):
  - GPT-style transformer for chess moves
  - Configurable size (2M-50M parameters)
  - Optimized for Metal acceleration

- **Training System** (`train.py`):
  - Efficient data loading
  - Metal device support for M3
  - Progress tracking with wandb

- **Data Processing** (`data/lichess_hf_dataset/`):
  - Chess game tokenization
  - Memory-efficient binary format
  - M3-optimized parallel processing

4. **Exploring the Code**
```bash
# View model architecture
less model.py

# Check training configurations
ls config/

# Examine data preparation
cd data/lichess_hf_dataset
less prepare.py
```

5. **Verify Environment**
```bash
# Run the verification script
python3 verify_setup.py
```

This will check:
- Python environment
- Required packages
- Metal support for M3
- Dataset accessibility

### Project Goals

1. **Training Objectives**:
   - Train a language model on chess games
   - Predict next moves in PGN format
   - Achieve 1500+ Elo rating

2. **Development Path**:
   - Start with small model (2M parameters)
   - Verify training pipeline
   - Scale up to 50M parameters
   - Use Metal acceleration on M3

3. **Key Features**:
   - Efficient data processing
   - M3 chip optimization
   - Memory-mapped training data
   - Scalable architecture

Now let's proceed with setting up the environment and preparing the data.

## Data Preparation

### Verifying Data Availability and Format

Before starting training, it's crucial to verify that the data is available and correctly formatted. We provide a verification script that checks:

1. **Dataset Accessibility**
   - Connects to Hugging Face dataset
   - Verifies total game count
   - Checks sample game format

2. **Tokenization Format**
   - Validates 32-token vocabulary
   - Confirms all chess symbols present
   - Checks token mapping integrity

3. **Binary File Status**
   - Verifies file creation
   - Checks file sizes
   - Validates token ranges

Run the verification script:
```bash
# From project root
python3 verify_chess_data.py
```

Expected output:
```
Chess Data Verification Tool

=== Checking Dataset Access ===
✓ Dataset accessible
  - Total games: 8,197,476

Sample game:
1. e4 e5 2. Nf3 Nc6 3. Bb5...

=== Checking Tokenization ===
✓ All required tokens present
Token mapping:
 0: 'K'
 1: 'Q'
...

=== Checking Binary Files ===
✓ train.bin: 2,147.3 MB
✓ val.bin: 21.5 MB

=== Summary ===
✅ All checks passed! Data is ready for training.
```

### Understanding Verification Results

1. **Dataset Check**
   - Should show ~8.2M games
   - Sample game should be valid PGN
   - Common issue: Network connectivity

2. **Token Verification**
   - Must have all 32 tokens:
     ```
     Pieces: K Q R B N         (0-4)
     Files:  a b c d e f g h   (5-12)
     Ranks:  1 2 3 4 5 6 7 8   (13-20)
     Special: + # x = O - . space \n ( )  (21-31)
     ```
   - Common issue: Corrupted meta.pkl

3. **Binary Files**
   - train.bin: ~2GB (99% of data)
   - val.bin: ~20MB (1% of data)
   - Common issue: Incomplete preparation

### Troubleshooting on Mac M3

1. **Dataset Access Issues**
```python
# Manual dataset check
from datasets import load_dataset
dataset = load_dataset("adamkarvonen/chess_games", split="train")
print(f"Games: {len(dataset):,}")
```

2. **Token Mapping Issues**
```python
# Check meta.pkl manually
import pickle
with open('data/lichess_hf_dataset/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
print(f"Tokens: {len(meta['stoi'])}")
```

3. **Binary File Issues**
```bash
# Check file sizes
ls -lh data/lichess_hf_dataset/*.bin

# If files are missing/corrupted:
cd data/lichess_hf_dataset
rm -f train.bin val.bin  # Remove corrupted files
python3 prepare.py       # Regenerate files
```

### Data Pipeline Overview

### Preparation Process

1. **Install Required Packages**
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch datasets tqdm numpy tiktoken
```

2. **Verify Data Access**
```python
# Run this in Python to verify dataset access
from datasets import load_dataset
dataset = load_dataset("adamkarvonen/chess_games")
print(f"Total games: {len(dataset['train'])}")
```

3. **Understanding prepare.py**
The script does several important things:
```python
# Key parts of prepare.py explained:

# 1. Load dataset from Hugging Face
from datasets import load_dataset
dataset = load_dataset("adamkarvonen/chess_games")

# 2. Define our chess move tokenization
tokens = {
    'pieces': ['K', 'Q', 'R', 'B', 'N'],        # Kings, Queens, etc.
    'files': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],  # Chess board columns
    'ranks': ['1', '2', '3', '4', '5', '6', '7', '8'],  # Chess board rows
    'special': ['+', '#', 'x', '=', 'O', '-', '.', ' ', '\n', '(', ')']  # Special moves
}

# 3. Process games in parallel for speed
def process_game(example):
    # Convert PGN text into token indices
    game_text = example['game']
    tokens = []
    for char in game_text:
        if char in stoi:  # stoi = string to index mapping
            tokens.append(stoi[char])
    return {'tokens': tokens}

# 4. Save in efficient binary format
import numpy as np
data = np.memmap('train.bin', dtype=np.uint8, mode='w+', shape=(total_size,))
```

The script processes chess games in these steps:
1. Downloads games from Hugging Face dataset
2. Converts PGN moves into our 32-token vocabulary
3. Uses parallel processing for speed (8 workers by default)
4. Saves as binary files for efficient loading during training

4. **Running Data Preparation**
```bash
# From the project root
cd data/lichess_hf_dataset
python prepare.py
```

The process will:
- Download ~8.19M chess games
- Process them using 8 worker processes
- Create train.bin and val.bin files

5. **Verifying Data Format**

We provide a verification script (`verify_data_quality.py`) to check everything is correct:

```bash
# Run the verification script
python3 verify_data_quality.py
```

This will check:
1. Tokenization mapping (all 32 tokens present)
2. Binary file sizes and content
3. Dataset accessibility
4. Sample game format

The script will output something like:
```
=== Chess Training Data Verification ===

✅ All required tokens present
Token mapping:
 0: 'K'
 1: 'Q'
...

✅ Dataset loaded successfully
   - Total games: 8,197,476

✅ train.bin: 2.1 GB
   - Contains 2,147,483,648 tokens
   - Token range: 0-31 (correct)
```

If you prefer to check manually:
```python
# Verify tokenization
import pickle
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
print("Token mapping:")
for k, v in sorted(meta['stoi'].items(), key=lambda x: x[1]):
    print(f"{v:2d}: {repr(k)}")

# Check binary files
import os
print("\nBinary files:")
for f in ['train.bin', 'val.bin']:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024 * 1024)  # MB
        print(f"{f}: {size:.1f} MB")
```

### Mac-Specific Considerations

1. **Understanding Binary Format**
The binary format (`train.bin` and `val.bin`) is crucial for efficient training:
```python
# How the binary format works:
# 1. Each token is stored as a single byte (uint8)
# 2. Files are memory-mapped for efficient access
# 3. During training, data is loaded in chunks:
#    chunk_size = 1024  # context window size
#    pos = random.randint(0, file_size - chunk_size)
#    chunk = np.memmap('train.bin', mode='r',
#                     offset=pos, shape=(chunk_size,))
```
This format:
- Minimizes memory usage during training
- Enables fast random access to any part of the dataset
- Works well with Apple Silicon's unified memory architecture

2. **M3 Chip Optimization**
The M3 chip's Neural Engine and unified memory architecture provide several advantages:
```python
# In prepare.py:
# For 8GB RAM Macs:
num_proc = 4
total_batches = 4096

# For 16GB RAM Macs:
num_proc = 8
total_batches = 2048

# For 32GB RAM Macs:
num_proc = 12
total_batches = 1024
```

3. **Memory Management**
Different Mac configurations require different settings:
- 8GB RAM: Use smaller batches and fewer workers
- 16GB RAM: Default settings work well
- 32GB RAM: Can increase workers and batch size

4. **Troubleshooting**
Common issues and solutions:
- If preparation is too slow:
  ```python
  # Increase workers if you have available RAM
  num_proc = min(12, os.cpu_count())
  ```
- If you get memory errors:
  ```python
  # Reduce memory usage
  total_batches = 4096  # Smaller chunks
  num_proc = 4  # Fewer workers
  ```
- If binary files are corrupted:
  ```bash
  # Clean up and restart
  rm -f train.bin val.bin
  python3 prepare.py
  ```

### Understanding Data Quality

The binary format ensures efficient training by:
1. **Compact Storage**: Each token uses 1 byte (vs 4+ bytes for text)
2. **Sequential Access**: Data is stored in continuous blocks
3. **Memory Mapping**: Only needed portions are loaded into RAM
4. **Random Access**: Can quickly jump to any game in the dataset

For example, a typical chess game in PGN:
```
1. e4 e5 2. Nf3 Nc6 3. Bb5
```
Becomes a sequence of bytes:
```python
# Each number is a token index (0-31)
[7, 17, 27, 7, 17, 27, 4, 8, 3, 27, 4, 9, 6, 27, 2, 8, 5]
```

### Verifying Success

Your data preparation is successful when:
1. meta.pkl contains the correct 32-token mapping
2. train.bin and val.bin files are created
3. File sizes are reasonable:
   - train.bin: ~2-3GB (99% of data)
   - val.bin: ~20-30MB (1% of data)
4. No errors during processing
5. verify_data_quality.py reports all checks passed

## Dependencies and Environment Setup

### Required Packages

The project requires several Python packages, optimized for Mac M3:

1. **PyTorch**: Deep learning framework with Metal support for M3
2. **Hugging Face Datasets**: For loading the chess dataset
3. **tiktoken**: For tokenization
4. **numpy**: For efficient array operations
5. **tqdm**: For progress bars
6. **wandb**: For experiment tracking (optional)

### Installation Steps

1. **Create Python Environment**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

2. **Install PyTorch with Metal Support**
```bash
# Install PyTorch with M3 optimization
pip install torch torchvision

# Verify Metal support
python3 -c "import torch; print(f'Metal device available: {torch.backends.mps.is_available()}')"
```

3. **Install Other Dependencies**
```bash
# Install required packages
pip install datasets tiktoken numpy tqdm wandb
```

### Verifying Setup

We provide a verification script to check your environment:

```bash
# Run the verification script
python3 verify_setup.py
```

You should see output like:
```
=== Chess Training Environment Verification ===

✓ PyTorch 2.1.0
  Metal available: True
  Metal built: True
✓ NumPy 1.24.3
✓ Hugging Face Datasets
✓ Tiktoken
✓ Weights & Biases
✓ Metal device working
✓ Chess dataset accessible

✅ Environment setup complete! Ready for training.
```

### Understanding the Dependencies

Each package serves a specific purpose:

1. **PyTorch with Metal**
   - Core deep learning framework
   - Uses Mac's Metal API for GPU acceleration
   - Optimized for M3 chip's Neural Engine

2. **Hugging Face Datasets**
   - Efficiently loads the chess dataset
   - Handles downloading and caching
   - Provides parallel processing capabilities

3. **tiktoken**
   - Handles chess move tokenization
   - Converts PGN text to token indices
   - Ensures consistent vocabulary

4. **numpy**
   - Manages binary data format
   - Provides memory-mapped file support
   - Handles efficient array operations

5. **tqdm**
   - Shows progress bars
   - Estimates completion time
   - Monitors data processing

6. **wandb (optional)**
   - Tracks training progress
   - Logs metrics and visualizations
   - Helps compare different runs

### Mac M3 Optimization Tips

1. **Metal Performance**
```python
# In your training code, use Metal device:
device = torch.device("mps")
model = model.to(device)
```

2. **Memory Management**
```python
# Optimize batch size based on your Mac's RAM:
batch_size = {
    "8GB": 32,
    "16GB": 64,
    "32GB": 128
}[your_ram_size]
```

3. **Data Loading**
```python
# Use appropriate number of workers:
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True  # Faster data transfer
)
```

## Training the Small Model

### Model Configuration

The small model is configured for initial testing on Mac M3:

1. **Configuration File** (`config/train_chess_small.py`):
```python
# Model architecture
n_layer = 6           # Number of transformer blocks
n_head = 6           # Number of attention heads
n_embd = 384         # Embedding dimension
block_size = 1024    # Context window size

# Training parameters
batch_size = 64      # Adjusted for M3 memory
learning_rate = 1e-4 # Conservative learning rate
max_iters = 5000     # Initial training iterations
eval_interval = 100  # Evaluate every 100 iterations
eval_iters = 20      # Number of evaluation batches
```

This configuration results in ~2M parameters, suitable for:
- Testing the training pipeline
- Quick iterations on M3
- Memory efficiency (uses ~1GB RAM)

### Starting Training

1. **Verify Environment**
```bash
# Check Metal support
python3 -c "import torch; print(f'Metal available: {torch.backends.mps.is_available()}')"

# Optional: Configure Weights & Biases
wandb login  # Or: export WANDB_MODE=disabled
```

2. **Run Training**
```bash
# From project root
python train.py config/train_chess_small.py
```

Expected output:
```
step 0: train loss 4.8493, val loss 4.8521
Metal device acceleration enabled
Loading dataset from data/lichess_hf_dataset/train.bin...
Number of parameters: 2,034,432

step 100: train loss 2.7841, val loss 2.7892
iter 0100: loss 2.7841, time 23.45ms/iter
...
```

3. **Monitor Progress**
- Training loss should decrease steadily
- Validation loss should track training loss
- Memory usage stable (check Activity Monitor)
- Metal GPU utilization active

### Optimizing Performance

1. **Memory Management**
```python
# In your training script:
import torch

# Use Metal device
device = torch.device("mps")
model = model.to(device)

# Clear cache if needed
torch.mps.empty_cache()

# Adjust batch size based on RAM:
batch_sizes = {
    "8GB": 32,
    "16GB": 64,
    "32GB": 128
}
```

2. **M3 Specific Tips**
- Enable Metal optimizations:
```python
# In train.py
torch.backends.mps.enable_memory_efficient_linear(True)
torch.backends.mps.set_buffer_size(2048)
```

- Use gradient accumulation for larger effective batches:
```python
# Example in train.py
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()
if (iter_num + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

3. **Training Tips**
- Start with small learning rate (1e-4)
- Monitor validation loss for overfitting
- Save checkpoints regularly
- Use Metal Performance HUD to monitor GPU usage

### Checkpoints and Evaluation

1. **Save Progress**
```bash
# Checkpoints saved automatically to:
out/chess_small/
├── config.py          # Configuration backup
├── model.pt          # Latest weights
└── model_best.pt     # Best validation loss
```

2. **Generate Moves**
```bash
# Test model with sample.py
python sample.py --out_dir=out/chess_small --num_samples=5
```

Example output:
```
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6
```

### Troubleshooting

Common issues and solutions:

1. **Memory Errors**
```python
# Reduce memory usage:
batch_size //= 2
gradient_accumulation_steps *= 2
```

2. **Slow Training**
```python
# Check Metal is being used:
print(f"Device: {next(model.parameters()).device}")
# Should show 'mps'
```

3. **Poor Convergence**
```python
# Adjust learning rate:
learning_rate = 1e-4  # Start conservative
warmup_iters = 100    # Gradual warmup
```

4. **Metal-Specific Issues**
```python
# If you get Metal errors:
import torch
torch.mps.empty_cache()  # Clear GPU memory
torch.backends.mps.enable_fallback_implementations(True)  # Enable CPU fallback
```

### Next Steps

After successful training:
1. Verify model quality with sample games
2. Monitor training metrics (loss curves)
3. Consider scaling up to larger model
4. Fine-tune hyperparameters if needed

## Understanding the Code

Let's break down how the core components work together to train our chess model:

### Core Components Overview

1. **Model Architecture** (`model.py`)
```python
# Key components of the model:
class GPT:
    def __init__(self, config):
        # Embedding layer: converts tokens to vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks: process chess moves
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        # Output layer: predicts next move
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
```

The model works by:
- Converting chess moves to numbers (tokens)
- Processing moves through attention layers
- Predicting the most likely next move

2. **Training Process** (`train.py`)
```python
# Main training loop simplified:
for iter in range(max_iters):
    # Get a batch of chess games
    xb, yb = get_batch('train')

    # Forward pass: predict next moves
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)

    # Backward pass: improve predictions
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Training involves:
- Loading chunks of chess games
- Making move predictions
- Learning from mistakes
- Saving progress regularly

3. **Move Generation** (`sample.py`)
```python
# Generating chess moves:
def generate(model, start_tokens, max_moves=100):
    # Start with opening move
    tokens = start_tokens

    # Generate one move at a time
    for _ in range(max_moves):
        # Predict next likely move
        next_token = model.generate_next(tokens)
        tokens.append(next_token)

        # Stop if game ends
        if next_token == END_TOKEN:
            break

    return decode_moves(tokens)
```

Generation works by:
- Starting with an opening
- Predicting each move
- Building complete games

4. **Configuration System** (`config/train_chess_small.py`)
```python
# Training settings made simple:
class TrainConfig:
    # Model size
    n_layer = 6          # Number of layers
    n_head = 6          # Attention heads
    n_embd = 384        # Vector size

    # Training
    batch_size = 64     # Games per batch
    learning_rate = 1e-4 # Learning speed
    max_iters = 5000    # Training steps
```

Configurations control:
- Model size and structure
- Training speed and duration
- Memory usage and efficiency

### How It All Works Together

1. **Data Flow**:
```
PGN Games → Tokens → Model → Predictions → Training → Checkpoints
```

2. **Training Cycle**:
- Load chess games in PGN format
- Convert moves to tokens (0-)
- Process through model
- Compare predictions to actual moves
- Adjust model to improve accuracy

3. **Using the Model**:
```python
# Example of using the trained model:
model = GPT.load('out/chess_small/model.pt')
game = generate_game(model, start="1. e4")
print(game)  # Outputs: "1. e4 e5 2. Nf3 ..."
```

### Key Concepts Simplified

1. **Attention Mechanism**:
- Like a chess player focusing on important pieces
- Helps model understand move relationships
- Learns patterns from millions of games

2. **Token Prediction**:
- Each move broken into tokens
- Model predicts most likely next token
- Builds complete moves from predictions

3. **Learning Process**:
- Starts with random moves
- Improves by studying games
- Learns common patterns and strategies

### Practical Tips

1. **Understanding Output**:
```
step 100: loss 2.7841
```
- Lower loss = better predictions
- Should decrease over time
- Sudden spikes = potential issues

2. **Monitoring Progress**:
- Watch training loss trend
- Check generated moves quality
- Monitor system resources

3. **Common Patterns**:
- Early training: basic moves
- Mid training: common openings
- Late training: strategic play

The next section covers evaluating model performance and scaling to larger architectures.

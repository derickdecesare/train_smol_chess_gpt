# Chess-GPT Model Specifications from Blog

## Model Architecture
- Total Parameters: 50 million
- Context Window: 1024 characters (approximately 180 moves)
- Vocabulary Size: 32 tokens
- Training Framework: nanoGPT

## Training Data
- Dataset Size: 5 million chess games
- Sources:
  - Lichess public database
  - Stockfish-generated games (Elo 1300-3200)
- Data Format: PGN strings (e.g., "1.e4 e5 2.Nf3 ...")

## Training Details
- Hardware: 4 RTX 3090 GPUs
- Training Time: One day
- Training Progress:
  - 99.8% legal moves within one day
  - Reached ~1300 Elo
  - Extended training reached 1500 Elo

## Critical Implementation Details
- Delimiter: ";1." at the start of each game
- Character-level model (not byte-pair encoding)
- Every batch begins with ";1." (delimiter + new game)
- Context length: 1024 tokens
- Move Format: Standard PGN notation

## Performance Metrics
- Initial Performance: 1300 Elo after one day
- Extended Training: 1500 Elo after several days
- Legal Move Rate: 99.8%
- Unique Games: Confirmed unique by 10th turn

## Key Configuration Notes
1. Uses character-level tokenization
2. Reduced vocabulary (32 tokens) compared to standard tokenizers
3. Batch formatting ensures proper game delimiting
4. No explicit board state or chess rules provided during training

## Source Code References
- Training Code: https://github.com/adamkarvonen/nanoGPT
- Evaluation Code: https://github.com/adamkarvonen/chess_gpt_eval/tree/master/nanogpt
- Pretrained Models: https://huggingface.co/adamkarvonen/chess_llms
- Datasets: https://huggingface.co/datasets/adamkarvonen/chess_games

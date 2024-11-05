import os
import pickle
import torch
import chess
import chess.pgn
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
def is_valid_pgn_token(token, stoi):
    """Verify if token is in the vocabulary"""
    return token in stoi

def parse_san_move(move_str):
    """Parse SAN move more flexibly"""
    # Remove check/mate symbols and extra spaces
    move = move_str.replace('+', '').replace('#', '').strip()
    # Remove move numbers and extra spaces
    if '.' in move:
        parts = move.split('.')
        move = parts[-1].strip()
    return move

def is_legal_move(board, move_str):
    """Verify if move is legal in current position"""
    try:
        # Try parsing as SAN first
        move = parse_san_move(move_str)
        if not move:  # Skip empty moves
            return False
        board.push_san(move)
        board.pop()
        return True
    except:
        try:
            # Try parsing as UCI if SAN fails
            move = chess.Move.from_uci(move_str.replace('+','').replace('#',''))
            return move in board.legal_moves
        except:
            return False

def format_chess_moves(tokens, itos, board=None, validate=True):
    """Format tokens into readable chess moves with validation"""
    if board is None:
        board = chess.Board()

    moves = ''.join([itos[int(i)] for i in tokens])
    formatted = []
    current_move = ""
    move_number = len(board.move_stack) // 2 + 1
    is_white_move = len(board.move_stack) % 2 == 0

    for char in moves:
        if char in ['.', '\n', ' ']:
            if current_move.strip():
                try:
                    san_move = parse_san_move(current_move)
                    if validate and is_legal_move(board, san_move):
                        if is_white_move:
                            formatted.append(f"{move_number}. {san_move}")
                        else:
                            formatted.append(san_move)
                            move_number += 1
                        board.push_san(san_move)
                        is_white_move = not is_white_move
                except Exception as e:
                    pass
            current_move = ""
        else:
            current_move += char

    return ' '.join(formatted)

def get_temperature(move_number, board):
    """Dynamic temperature scheduling based on game phase"""
    piece_count = len(board.piece_map())
    if move_number < 5:  # Opening
        return 0.3
    elif piece_count > 20:  # Middlegame
        return 0.5
    else:  # Endgame
        return 0.7

def main():
    out_dir = 'out-chess-tiny'  # or specify your own path
    num_samples = 5  # number of samples to draw
    max_new_tokens = 50  # reduced for more focused generation
    top_k = 10  # reduced for more focused sampling
    seed = 1337
    device = 'cpu'  # force CPU
    dtype = 'float32'  # use float32 for CPU

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    device_type = 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    try:
        # load the meta data for the model
        meta_path = os.path.join('data/lichess_hf_dataset', 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        vocab_size = len(itos)
        print(f'Vocabulary size: {vocab_size}')

        # model
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']

        # convert state dict to CPU tensors if needed
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            if isinstance(v, torch.Tensor) and v.device.type != 'cpu':
                state_dict[k] = v.cpu()

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        # Starting positions for generation
        start_positions = [
            "1. e4",  # Standard king's pawn opening
            "1. d4",  # Standard queen's pawn opening
            "1. Nf3",  # Reti opening
        ]

        print("\nGenerating chess continuations...")
        for start_pos in start_positions:
            print(f"\nStarting position: {start_pos}")
            board = chess.Board()

            # Make the initial move on the board
            try:
                san_move = parse_san_move(start_pos)
                board.push_san(san_move)
            except Exception as e:
                print(f"Invalid starting position: {start_pos} - {str(e)}")
                continue

            # encode the beginning of the prompt
            start_tokens = [stoi[x] for x in start_pos if is_valid_pgn_token(x, stoi)]
            x = (torch.tensor(start_tokens, dtype=torch.long, device=device)[None, ...])

            # run generation with temperature scheduling
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
                    for k in range(num_samples):
                        try:
                            move_count = 1
                            temperature = get_temperature(move_count)
                            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=5)
                            moves = format_chess_moves(y[0].tolist(), itos, validate=True)
                            if moves.strip():  # Only print if we got valid moves
                                print(f'Continuation {k+1}:')
                                print(moves)
                                print('---------------')
                            else:
                                print(f'No valid continuation generated for attempt {k+1}')
                        except Exception as e:
                            print(f"Error generating continuation {k+1}: {str(e)}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()

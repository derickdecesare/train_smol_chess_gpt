import torch
from transformers import GPT2Config, GPT2LMHeadModel
import os
import json

def convert_to_hf_model(checkpoint_path, output_dir):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    model_args = checkpoint.get('model_args', {})
    
    # Infer configuration from the checkpoint. Our training uses a block size of 1023 tokens,
    # but our positional embedding matrix in HF is sized to n_positions = block_size + 1.
    vocab_size = model_args.get('vocab_size', 32)
    block_size = model_args.get('block_size', 1023) + 1  # +1 for positional embeddings
    n_embd = model_args.get('n_embd', 256)
    n_layer = model_args.get('n_layer', 8)
    n_head = model_args.get('n_head', 4)
    n_inner = n_embd * 4  # defaults to 4 * n_embd
    
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        activation_function="gelu",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        bos_token_id=15,  # semicolon token
        eos_token_id=1,   # hash token
        pad_token_id=0,   # space token
    )
    
    # Create a new HuggingFace model
    hf_model = GPT2LMHeadModel(config)
    hf_state_dict = {}
    
    # Token embeddings: copy directly
    hf_state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight']
    
    # Positional embeddings: pad if necessary (training checkpoint usually has shape [block_size-1, n_embd])
    pos_emb = state_dict['transformer.wpe.weight']
    if pos_emb.shape[0] < block_size:
        padded_pos_emb = torch.zeros(block_size, n_embd)
        padded_pos_emb[:pos_emb.shape[0], :] = pos_emb
        hf_state_dict['transformer.wpe.weight'] = padded_pos_emb
    else:
        hf_state_dict['transformer.wpe.weight'] = pos_emb
    
    # Loop over each transformer layer and map the parameters
    for i in range(n_layer):
        # LayerNorm 1
        hf_state_dict[f'transformer.h.{i}.ln_1.weight'] = state_dict[f'transformer.h.{i}.ln_1.weight']
        hf_state_dict[f'transformer.h.{i}.ln_1.bias'] = torch.zeros_like(state_dict[f'transformer.h.{i}.ln_1.weight'])
        
        # Attention: the checkpoint stores the c_attn weights in [n_embd, 3*n_embd] shape.
        # HuggingFace uses [3*n_embd, n_embd], so we need to take the transpose.
        attn_weight = state_dict[f'transformer.h.{i}.attn.c_attn.weight']
        hf_state_dict[f'transformer.h.{i}.attn.c_attn.weight'] = attn_weight.t()
        hf_state_dict[f'transformer.h.{i}.attn.c_attn.bias'] = torch.zeros(3 * n_embd)
        
        # Attention projection (c_proj) – no transpose required if dimensions already match
        proj_weight = state_dict[f'transformer.h.{i}.attn.c_proj.weight']
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.weight'] = proj_weight
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.bias'] = torch.zeros(n_embd)
        
        # LayerNorm 2
        hf_state_dict[f'transformer.h.{i}.ln_2.weight'] = state_dict[f'transformer.h.{i}.ln_2.weight']
        hf_state_dict[f'transformer.h.{i}.ln_2.bias'] = torch.zeros_like(state_dict[f'transformer.h.{i}.ln_2.weight'])
        
        # MLP (Feed-Forward)
        # First linear layer of the MLP (c_fc): transpose required since our checkpoint shape is [n_embd, n_inner]
        mlp_fc_weight = state_dict[f'transformer.h.{i}.mlp.c_fc.weight']
        hf_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mlp_fc_weight.t()
        hf_state_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = torch.zeros(n_inner)
        
        # Second linear layer of the MLP (c_proj): transpose required since our checkpoint shape is [n_inner, n_embd]
        mlp_proj_weight = state_dict[f'transformer.h.{i}.mlp.c_proj.weight']
        hf_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mlp_proj_weight.t()
        hf_state_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = torch.zeros(n_embd)
    
    # Final layer norm before the language modeling head
    hf_state_dict['transformer.ln_f.weight'] = state_dict['transformer.ln_f.weight']
    hf_state_dict['transformer.ln_f.bias'] = torch.zeros_like(state_dict['transformer.ln_f.weight'])
    
    # Language model head: copy the weights for the final output projection
    hf_state_dict['lm_head.weight'] = state_dict['lm_head.weight']
    
    # Load the updated state dict into the HuggingFace model
    hf_model.load_state_dict(hf_state_dict)
    
    # Save the HuggingFace model and configuration to the output directory
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)
    
    # Save the custom vocabulary to vocab.json
    vocab = {
        " ": 0, "#": 1, "+": 2, "-": 3, ".": 4, "0": 5, "1": 6, "2": 7,
        "3": 8, "4": 9, "5": 10, "6": 11, "7": 12, "8": 13, "9": 14, ";": 15,
        "=": 16, "B": 17, "K": 18, "N": 19, "O": 20, "Q": 21, "R": 22, "a": 23,
        "b": 24, "c": 25, "d": 26, "e": 27, "f": 28, "g": 29, "h": 30, "x": 31
    }
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"HuggingFace model saved to {output_dir}")

if __name__ == "__main__":
    # Input checkpoint (from our training run) and output directory for HF model
    checkpoint_path = 'out-chess-mac/ckpt.pt'
    output_dir = 'chess-gpt-4.5M'
    convert_to_hf_model(checkpoint_path, output_dir)
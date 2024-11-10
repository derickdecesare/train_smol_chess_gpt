# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
import pickle

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
dtype = np.uint8  # Currently there are only 32 tokens in the chess LLMs vocab

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc


# THis is what's going on inside pickle
# String to Integer (stoi) mapping:
# {
#   " ": 0,
#   "#": 1,
#   "+": 2,
#   "-": 3,
#   ".": 4,
#   "0": 5,
#   "1": 6,
#   "2": 7,
#   "3": 8,
#   "4": 9,
#   "5": 10,
#   "6": 11,
#   "7": 12,
#   "8": 13,
#   "9": 14,
#   ";": 15,
#   "=": 16,
#   "B": 17,
#   "K": 18,
#   "N": 19,
#   "O": 20,
#   "Q": 21,
#   "R": 22,
#   "a": 23,
#   "b": 24,
#   "c": 25,
#   "d": 26,
#   "e": 27,
#   "f": 28,
#   "g": 29,
#   "h": 30,
#   "x": 31
# }

# Integer to String (itos) mapping:
# {
#   "0": " ",
#   "1": "#",
#   "2": "+",
#   "3": "-",
#   "4": ".",
#   "5": "0",
#   "6": "1",
#   "7": "2",
#   "8": "3",
#   "9": "4",
#   "10": "5",
#   "11": "6",
#   "12": "7",
#   "13": "8",
#   "14": "9",
#   "15": ";",
#   "16": "=",
#   "17": "B",
#   "18": "K",
#   "19": "N",
#   "20": "O",
#   "21": "Q",
#   "22": "R",
#   "23": "a",
#   "24": "b",
#   "25": "c",
#   "26": "d",
#   "27": "e",
#   "28": "f",
#   "29": "g",
#   "30": "h",
#   "31": "x"
# }

# Total vocabulary size: 32

# Example mappings:
# Space character -> 0
# Number 1 -> 6
# Piece 'e' -> 27

if __name__ == "__main__":
    # dataset = load_dataset("csv", data_files={"train": "pgn.csv"}) # For local testing

    dataset_path = "adamkarvonen/chess_games"
    file_path = "lichess_6gb_blocks.zip"
    # file_path = "smaller_pgn_file_blocks.zip"

    # Load the dataset
    dataset = load_dataset(dataset_path, data_files=file_path)


    # new code to fix?? This is proposed...
    # by default only contains the 'train' split
    dataset = dataset["train"]  # Get the training data

    # Limit to 100K games (for Mac memory constraints)
    max_games = 100000  # Limit to 100K games
    dataset = dataset.select(range(max_games))
    print(f"Dataset limited to {len(dataset)} games")  # Confirms we have 100k games

    # Create train/val split
    split_dataset = dataset.train_test_split(
        test_size=0.01, seed=2357, shuffle=True
    )

    # Original code before we limit to 100k games
    # # by default only contains the 'train' split, so create a test split
    # split_dataset = dataset["train"].train_test_split(
    #     test_size=0.01, seed=2357, shuffle=True
    # )
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    # Print dataset structure (not content)
    print("\nDataset splits:")
    print(f"Training examples: {len(split_dataset['train'])}")
    print(f"Validation examples: {len(split_dataset['val'])}")

    print(split_dataset) # this results in:
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # Optional: Print a snippet of the first training example
    print("\nFirst training example transcript snippet:")
    print(split_dataset["train"][0]["transcript"][:100] + "...")  # Print first 100 characters

    # we now want to tokenize the dataset. Using meta.pkl in the same directory as this file
    meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint8, mode='r')
    # print(split_dataset["val"][0])
    # print(len(split_dataset["val"]["transcript"][0]))

    # For verifying that all games are 1024 tokens long
    # for game in split_dataset["train"]["transcript"]:
    #     if len(game) != 1024:
    #         print(len(game))
    #         print(game)
    #         break
    # print(stoi)

    column_name = "transcript"

    def process(example):
        ids = np.array([stoi[c] for c in example[column_name]], dtype=dtype)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # print(tokenized["val"]["ids"])

    # Optional: Print a snippet of the first tokenized training example
    print("\nFirst tokenized training example IDs snippet:")
    print(tokenized["train"][0]["ids"][:20])  # Print first 20 token IDs

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items(): # For each split (train and val)
        # Calculate total length of all tokens
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")

        # Create a memory-mapped file to store all tokens
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        print(arr.shape)

        # This is just for SAVING the data efficiently, not for training
        total_batches = min(1024, len(dset))  # Ensure we don't have more batches than examples # important when training small models on reduced dataset

        # Write data in chunks for memory efficiency
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write # Split dataset into 1024 pieces for writing
            batch = dset.shard( 
                num_shards=total_batches, # Split into 1024 pieces
                index=batch_idx, # Current piece
                contiguous=True # Keep games together
            ).with_format("numpy")

            # print(batch[0])
            # Concatenate all tokenized games in this chunk
            arr_batch = np.concatenate(batch["ids"])
            # print(arr_batch)
            # print(arr_batch.shape)
            # Write into mmap # Write this chunk to the binary file
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        # Verify that all tokens have been written
        assert idx == arr_len, f"Mismatch in total tokens written for {split}"
        # Flush changes to disk
        arr.flush()

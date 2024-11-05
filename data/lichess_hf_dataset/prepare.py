import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import pickle

num_proc = 8
dtype = np.uint8

if __name__ == "__main__":
    dataset_path = "adamkarvonen/chess_games"
    file_path = "lichess_6gb_blocks.zip"

    print("Loading dataset...")
    dataset = load_dataset(dataset_path, data_files=file_path)

    print("Creating train/val split...")
    split_dataset = dataset["train"].train_test_split(
        test_size=0.01, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]
    column_name = "transcript"

    def process(example):
        ids = []
        # Add game delimiter at start
        for c in ";1.":
            ids.append(stoi[c])
        # Process game moves
        for c in example[column_name]:
            ids.append(stoi.get(c, stoi[' ']))
        ids = np.array(ids, dtype=dtype)
        out = {"ids": ids, "len": len(ids)}
        return out

    print("Tokenizing dataset...")
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Process each split separately
    for split, dset in tokenized.items():
        print(f"\nProcessing {split} split...")
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")

        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        total_batches = 2048
        batch_size = len(dset) // total_batches + 1

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")

            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"Finished writing {filename}")

    print("\nData preparation complete!")

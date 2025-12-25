"""
Download and inspect StoryReasoning dataset (Hugging Face) as in the notebook.
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect StoryReasoning dataset.")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--dataset_id", type=str, default="daniel3303/StoryReasoning")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    ds = load_dataset(args.dataset_id, cache_dir=args.cache_dir)
    print(ds)

    for split in ds.keys():
        print(f"\n--- Split: {split} ---")
        print("Num rows:", len(ds[split]))
        print("Columns:", ds[split].column_names)
        print("Features:", ds[split].features)

    sample = ds[list(ds.keys())[0]][0]
    print("\nSample keys:", list(sample.keys()))


if __name__ == "__main__":
    main()

"""
Build index tables for StoryReasoning windowed samples (K-step context -> next step).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .dataloader import build_index_table


def save_index(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare StoryReasoning index tables.")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--dataset_id", type=str, default="daniel3303/StoryReasoning")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_steps", type=int, default=4)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--out_train", type=Path, default=Path("data/index_train.json"))
    parser.add_argument("--out_val", type=Path, default=Path("data/index_val.json"))
    parser.add_argument("--out_test", type=Path, default=Path("data/index_test.json"))
    args = parser.parse_args()

    ds = load_dataset(args.dataset_id, cache_dir=args.cache_dir)
    train_raw = ds["train"]
    test_raw = ds["test"]

    unique_ids = np.unique(np.array(train_raw["story_id"]))
    train_ids, val_ids = train_test_split(unique_ids, test_size=args.val_frac, random_state=args.seed, shuffle=True)
    storyid_to_idx = {sid: i for i, sid in enumerate(train_raw["story_id"])}
    train_story_indices = [storyid_to_idx[sid] for sid in train_ids]
    val_story_indices = [storyid_to_idx[sid] for sid in val_ids]

    train_index = build_index_table(train_raw, train_story_indices, K=args.k_steps, limit_samples=args.limit_train)
    val_index = build_index_table(train_raw, val_story_indices, K=args.k_steps, limit_samples=args.limit_val)
    test_story_indices = list(range(len(test_raw)))
    test_index = build_index_table(test_raw, test_story_indices, K=args.k_steps, limit_samples=args.limit_test)

    save_index(args.out_train, train_index)
    save_index(args.out_val, val_index)
    save_index(args.out_test, test_index)

    print(f"Train index rows: {len(train_index)}")
    print(f"Val index rows: {len(val_index)}")
    print(f"Test index rows: {len(test_index)}")


if __name__ == "__main__":
    main()

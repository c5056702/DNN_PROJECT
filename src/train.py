"""
Training scripts adapted from the graph_reasoning notebook.
Supports baseline (ResNet + TF-IDF) and graph-fused variants.
"""
from __future__ import annotations

import argparse
import os
from functools import partial

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataloader import (
    StoryReasoningWindowDataset,
    build_index_table,
    collate_graph,
    collate_resnet_tfidf,
)
from .encoders_image import ResNet18Embedder
from .encoders_text import build_tfidf_vectorizer, clean_text
from .model_baseline import ResnetTfidfSeqPredictor, cosine_loss, make_txt_target
from .model_graph import GraphFusedSeqPredictor, GraphFusedTextOnlySeqPredictor


def build_splits(train_raw, test_raw, val_frac: float, seed: int, k_steps: int):
    unique_ids = np.unique(np.array(train_raw["story_id"]))
    train_ids, val_ids = train_test_split(unique_ids, test_size=val_frac, random_state=seed, shuffle=True)
    storyid_to_idx = {sid: i for i, sid in enumerate(train_raw["story_id"])}
    train_story_indices = [storyid_to_idx[sid] for sid in train_ids]
    val_story_indices = [storyid_to_idx[sid] for sid in val_ids]

    train_index = build_index_table(train_raw, train_story_indices, K=k_steps, limit_samples=None)
    val_index = build_index_table(train_raw, val_story_indices, K=k_steps, limit_samples=None)
    test_story_indices = list(range(len(test_raw)))
    test_index = build_index_table(test_raw, test_story_indices, K=k_steps, limit_samples=None)
    return train_index, val_index, test_index


def build_vectorizer(train_ds, sample_n: int, max_features: int, seed: int):
    rng = np.random.default_rng(seed)
    sample_n = min(sample_n, len(train_ds))
    sample_idxs = rng.choice(np.arange(len(train_ds)), size=sample_n, replace=False)
    corpus = []
    for i in tqdm(sample_idxs, desc="Building TF-IDF corpus"):
        ex = train_ds[int(i)]
        corpus.extend([clean_text(t) for t in ex["ctx_texts"]])
        corpus.append(clean_text(ex["target_text"]))
    vectorizer = build_tfidf_vectorizer(corpus, max_features=max_features, min_df=2, ngram_range=(1, 2))
    return vectorizer


def eval_epoch_baseline(model, loader):
    model.eval()
    tot = 0
    sum_li = sum_lt = 0.0
    sum_ci = sum_ct = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            pred_i, pred_t = model(batch["ctx_img_emb"], batch["ctx_txt_raw"])
            tgt_t = make_txt_target(model, batch["tgt_txt_raw"])

            li = cosine_loss(pred_i, batch["tgt_img_emb"])
            lt = cosine_loss(pred_t, tgt_t)
            ci = (pred_i * batch["tgt_img_emb"]).sum(dim=-1).mean()
            ct = (pred_t * tgt_t).sum(dim=-1).mean()

            bs = batch["ctx_img_emb"].size(0)
            tot += bs
            sum_li += li.item() * bs
            sum_lt += lt.item() * bs
            sum_ci += ci.item() * bs
            sum_ct += ct.item() * bs
    return {
        "val_loss_img": sum_li / max(tot, 1),
        "val_loss_txt": sum_lt / max(tot, 1),
        "val_cos_img": sum_ci / max(tot, 1),
        "val_cos_txt": sum_ct / max(tot, 1),
    }


def eval_epoch_graph(model, loader):
    model.eval()
    tot = 0
    sum_li = sum_lt = 0.0
    sum_ci = sum_ct = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            pred_i, pred_t = model(
                batch["ctx_img_emb"],
                batch["ctx_txt_raw"],
                batch["g_basic"],
                batch["g_name"],
                batch["g_adj"],
                batch["g_mask"],
            )
            tgt_t = F.normalize(model.txt_proj(batch["tgt_txt_raw"]), dim=-1)
            li = cosine_loss(pred_i, batch["tgt_img_emb"])
            lt = cosine_loss(pred_t, tgt_t)
            ci = (pred_i * batch["tgt_img_emb"]).sum(dim=-1).mean()
            ct = (pred_t * tgt_t).sum(dim=-1).mean()

            bs = batch["ctx_img_emb"].size(0)
            tot += bs
            sum_li += li.item() * bs
            sum_lt += lt.item() * bs
            sum_ci += ci.item() * bs
            sum_ct += ct.item() * bs
    return {
        "val_loss_img": sum_li / max(tot, 1),
        "val_loss_txt": sum_lt / max(tot, 1),
        "val_cos_img": sum_ci / max(tot, 1),
        "val_cos_txt": sum_ct / max(tot, 1),
    }


def train_loop(model, train_loader, val_loader, epochs: int, lr: float, weight_decay: float, device: torch.device, eval_fn):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_score = -1e9
    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0
        sum_loss = sum_ci = sum_ct = 0.0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=True)
        for batch in pbar:
            if "g_basic" in batch:
                pred_i, pred_t = model(
                    batch["ctx_img_emb"],
                    batch["ctx_txt_raw"],
                    batch["g_basic"],
                    batch["g_name"],
                    batch["g_adj"],
                    batch["g_mask"],
                )
                tgt_t = F.normalize(model.txt_proj(batch["tgt_txt_raw"]), dim=-1)
            else:
                pred_i, pred_t = model(batch["ctx_img_emb"], batch["ctx_txt_raw"])
                tgt_t = make_txt_target(model, batch["tgt_txt_raw"])

            li = cosine_loss(pred_i, batch["tgt_img_emb"])
            lt = cosine_loss(pred_t, tgt_t)
            loss = li + lt

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                ci = (pred_i * batch["tgt_img_emb"]).sum(dim=-1).mean()
                ct = (pred_t * tgt_t).sum(dim=-1).mean()

            bs = batch["ctx_img_emb"].size(0)
            tot += bs
            sum_loss += loss.item() * bs
            sum_ci += ci.item() * bs
            sum_ct += ct.item() * bs
            pbar.set_postfix({"loss": sum_loss / tot, "cos_i": sum_ci / tot, "cos_t": sum_ct / tot})

        metrics = eval_fn(model, val_loader)
        score = metrics["val_cos_img"] + metrics["val_cos_txt"]
        print(f"\nEpoch {epoch} summary:")
        print(
            f"  train_loss={sum_loss/tot:.4f}  train_cos_img={sum_ci/tot:.4f}  train_cos_txt={sum_ct/tot:.4f}"
        )
        print(
            f"  val_loss_img={metrics['val_loss_img']:.4f}  val_loss_txt={metrics['val_loss_txt']:.4f}"
        )
        print(
            f"  val_cos_img={metrics['val_cos_img']:.4f}   val_cos_txt={metrics['val_cos_txt']:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics}, "best.pt")
            print("  Saved best checkpoint: best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train StoryReasoning graph models.")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--tfidf_max_features", type=int, default=8000)
    parser.add_argument("--tfidf_corpus_samples", type=int, default=6000)
    parser.add_argument("--max_nodes", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory when using CUDA.")
    parser.add_argument("--mode", choices=["baseline", "graph", "graph_textonly"], default="baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = args.pin_memory and device.type == "cuda"
    ds = load_dataset("daniel3303/StoryReasoning", cache_dir=args.cache_dir)
    train_raw = ds["train"]
    test_raw = ds["test"]

    train_index, val_index, _ = build_splits(train_raw, test_raw, args.val_frac, args.seed, args.k_steps)
    train_ds = StoryReasoningWindowDataset(train_raw, train_index, K=args.k_steps)
    val_ds = StoryReasoningWindowDataset(train_raw, val_index, K=args.k_steps)

    vectorizer = build_vectorizer(train_ds, args.tfidf_corpus_samples, args.tfidf_max_features, args.seed)
    tfidf_dim = len(vectorizer.get_feature_names_out())

    img_encoder = ResNet18Embedder(device)
    encode_images = img_encoder.encode

    if args.mode == "baseline":
        collate = partial(
            collate_resnet_tfidf,
            K=args.k_steps,
            encode_images=encode_images,
            vectorizer=vectorizer,
            device=device,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        model = ResnetTfidfSeqPredictor(img_dim=512, tfidf_dim=tfidf_dim, txt_dim=512, hidden_dim=512, dropout=0.1).to(device)
        train_loop(model, train_loader, val_loader, args.epochs, args.lr, args.weight_decay, device, eval_epoch_baseline)
        return

    collate = partial(
        collate_graph,
        K=args.k_steps,
        max_nodes=args.max_nodes,
        encode_images=encode_images,
        vectorizer=vectorizer,
        device=device,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )

    if args.mode == "graph_textonly":
        model = GraphFusedTextOnlySeqPredictor(
            img_dim=512,
            tfidf_dim=tfidf_dim,
            txt_dim=512,
            hidden_dim=512,
            graph_dim=256,
            dropout=0.1,
        ).to(device)
    else:
        model = GraphFusedSeqPredictor(
            img_dim=512,
            tfidf_dim=tfidf_dim,
            txt_dim=512,
            hidden_dim=512,
            graph_dim=256,
            dropout=0.1,
        ).to(device)

    train_loop(model, train_loader, val_loader, args.epochs, args.lr, args.weight_decay, device, eval_epoch_graph)


if __name__ == "__main__":
    main()

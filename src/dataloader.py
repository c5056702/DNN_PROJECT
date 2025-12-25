"""
Dataset and collation utilities derived from the graph_reasoning notebook.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .encoders_text import clean_text, tfidf_transform

_SENTENCE_RE = re.compile(r"(?<=[\\.!?])\\s+")
ENT_RE = re.compile(r"<gdo\\s+([^>]+)>(.*?)</gdo>", flags=re.IGNORECASE | re.DOTALL)


def split_story_into_frames(story: str, frame_count: int) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split((story or "").strip()) if s.strip()]
    if frame_count <= 0:
        return []
    if not sentences:
        return [""] * frame_count
    chunks: List[List[str]] = [[] for _ in range(frame_count)]
    for i, sent in enumerate(sentences):
        chunks[i * frame_count // len(sentences)].append(sent)
    return [" ".join(c).strip() for c in chunks]


def extract_entities_from_text(tagged_text: str) -> List[Tuple[str, str]]:
    tagged_text = tagged_text or ""
    entities: List[Tuple[str, str]] = []
    for attrs, name in ENT_RE.findall(tagged_text):
        name = re.sub(r"\\s+", " ", name).strip()
        if not name:
            continue
        attrs = attrs.lower()
        if "char" in attrs:
            etype = "char"
        elif "obj" in attrs:
            etype = "obj"
        else:
            etype = "other"
        entities.append((etype, name))
    seen = set()
    out: List[Tuple[str, str]] = []
    for ent in entities:
        if ent not in seen:
            seen.add(ent)
            out.append(ent)
    return out


def build_index_table(split, story_row_indices: Sequence[int], K: int = 4, limit_samples: Optional[int] = None) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    added = 0
    for row_idx in story_row_indices:
        ex = split[int(row_idx)]
        frame_count = int(ex["frame_count"])
        if frame_count < K + 1:
            continue
        for t in range(K, frame_count):
            rows.append((int(row_idx), int(t), int(frame_count)))
            added += 1
            if limit_samples is not None and added >= limit_samples:
                return rows
    return rows


@dataclass
class WindowSample:
    story_id: str
    row_idx: int
    t_index: int
    frame_count: int
    ctx_images: List[Any]
    target_image: Any
    ctx_texts: List[str]
    target_text: str


class StoryReasoningWindowDataset(Dataset):
    def __init__(self, base_split, index_rows: Sequence[Tuple[int, int, int]], K: int = 4):
        self.base = base_split
        self.index = list(index_rows)
        self.K = K

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row_idx, t, frame_count = self.index[i]
        ex = self.base[row_idx]

        imgs = ex["images"]
        story_id = ex["story_id"]
        story = ex["story"]
        frame_texts = split_story_into_frames(story, int(ex["frame_count"]))

        ctx_imgs = imgs[t - self.K : t]
        tgt_img = imgs[t]
        ctx_txts = frame_texts[t - self.K : t]
        tgt_txt = frame_texts[t]

        return {
            "story_id": story_id,
            "row_idx": row_idx,
            "t_index": t,
            "frame_count": frame_count,
            "ctx_images": ctx_imgs,
            "target_image": tgt_img,
            "ctx_texts": ctx_txts,
            "target_text": tgt_txt,
        }


def collate_keep_raw(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        out[key] = [b[key] for b in batch]
    return out


def collate_resnet_tfidf(
    batch: List[Dict[str, Any]],
    K: int,
    encode_images,
    vectorizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    ctx_images_flat: List[Any] = []
    ctx_texts_flat: List[str] = []
    tgt_images: List[Any] = []
    tgt_texts: List[str] = []

    for b in batch:
        if len(b["ctx_images"]) != K or len(b["ctx_texts"]) != K:
            raise ValueError(f"Expected K={K}, got {len(b['ctx_images'])} images and {len(b['ctx_texts'])} texts")
        ctx_images_flat.extend(b["ctx_images"])
        ctx_texts_flat.extend([clean_text(t) for t in b["ctx_texts"]])
        tgt_images.append(b["target_image"])
        tgt_texts.append(clean_text(b["target_text"]))

    ctx_img_emb_flat = encode_images(ctx_images_flat)
    tgt_img_emb = encode_images(tgt_images)

    ctx_txt_raw = torch.from_numpy(tfidf_transform(vectorizer, ctx_texts_flat))
    tgt_txt_raw = torch.from_numpy(tfidf_transform(vectorizer, tgt_texts))

    ctx_img_emb = ctx_img_emb_flat.view(bsz, K, -1).contiguous()
    ctx_txt_raw = ctx_txt_raw.view(bsz, K, -1).contiguous().to(device)
    tgt_txt_raw = tgt_txt_raw.to(device)

    return {
        "ctx_img_emb": ctx_img_emb,
        "ctx_txt_raw": ctx_txt_raw,
        "tgt_img_emb": tgt_img_emb,
        "tgt_txt_raw": tgt_txt_raw,
    }


def build_graph_from_ctx_texts(
    ctx_texts: List[str],
    K: int,
    vectorizer,
    max_nodes: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_ents = [extract_entities_from_text(t) for t in ctx_texts]

    freq: Dict[Tuple[str, str], int] = {}
    meta: Dict[Tuple[str, str], Dict[str, int]] = {}
    for t, ents in enumerate(frame_ents):
        for etype, name in ents:
            key = (etype, name)
            freq[key] = freq.get(key, 0) + 1
            if key not in meta:
                meta[key] = {"first": t, "last": t}
            else:
                meta[key]["last"] = t

    if not freq:
        node_keys = [("other", "<NO_ENTITY>")]
        meta[("other", "<NO_ENTITY>")] = {"first": 0, "last": K - 1}
        freq[("other", "<NO_ENTITY>")] = 1
    else:
        node_keys = sorted(freq.keys(), key=lambda k: (-freq[k], meta[k]["first"]))[:max_nodes]

    n_nodes = len(node_keys)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    key_to_i = {k: i for i, k in enumerate(node_keys)}

    for ents in frame_ents:
        kept = [key_to_i[(etype, name)] for (etype, name) in ents if (etype, name) in key_to_i]
        for i in kept:
            for j in kept:
                if i != j:
                    adj[i, j] = 1.0

    for i, key_i in enumerate(node_keys):
        for j, key_j in enumerate(node_keys):
            if i == j:
                continue
            a0, a1 = meta[key_i]["first"], meta[key_i]["last"]
            b0, b1 = meta[key_j]["first"], meta[key_j]["last"]
            if not (a1 < b0 - 1 or b1 < a0 - 1):
                adj[i, j] = max(adj[i, j], 1.0)

    basic = np.zeros((n_nodes, 6), dtype=np.float32)
    names_clean = [clean_text(k[1]) for k in node_keys]
    name_tfidf = tfidf_transform(vectorizer, names_clean)

    for i, (etype, name) in enumerate(node_keys):
        basic[i, 0] = 1.0 if etype == "char" else 0.0
        basic[i, 1] = 1.0 if etype == "obj" else 0.0
        basic[i, 2] = 1.0 if etype == "other" else 0.0
        basic[i, 3] = freq[(etype, name)] / max(1.0, float(K))
        basic[i, 4] = meta[(etype, name)]["first"] / max(1.0, float(K - 1))
        basic[i, 5] = meta[(etype, name)]["last"] / max(1.0, float(K - 1))

    return basic, name_tfidf, adj


def collate_graph(
    batch: List[Dict[str, Any]],
    K: int,
    max_nodes: int,
    encode_images,
    vectorizer,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    ctx_images_flat: List[Any] = []
    ctx_texts_clean_flat: List[str] = []
    tgt_images: List[Any] = []
    tgt_texts_clean: List[str] = []

    basic_list: List[np.ndarray] = []
    name_tfidf_list: List[np.ndarray] = []
    adj_list: List[np.ndarray] = []
    n_list: List[int] = []

    for b in batch:
        ctx_images_flat.extend(b["ctx_images"])
        ctx_texts_clean_flat.extend([clean_text(t) for t in b["ctx_texts"]])
        tgt_images.append(b["target_image"])
        tgt_texts_clean.append(clean_text(b["target_text"]))

        basic, name_tfidf, adj = build_graph_from_ctx_texts(b["ctx_texts"], K=K, max_nodes=max_nodes, vectorizer=vectorizer)
        basic_list.append(basic)
        name_tfidf_list.append(name_tfidf)
        adj_list.append(adj)
        n_list.append(basic.shape[0])

    ctx_img_emb_flat = encode_images(ctx_images_flat)
    tgt_img_emb = encode_images(tgt_images)

    ctx_txt_raw = torch.from_numpy(tfidf_transform(vectorizer, ctx_texts_clean_flat)).to(device)
    tgt_txt_raw = torch.from_numpy(tfidf_transform(vectorizer, tgt_texts_clean)).to(device)

    ctx_img_emb = ctx_img_emb_flat.view(bsz, K, -1).contiguous()
    ctx_txt_raw = ctx_txt_raw.view(bsz, K, -1).contiguous()

    max_n = min(max_nodes, max(n_list))
    basic_pad = torch.zeros((bsz, max_n, 6), dtype=torch.float32, device=device)
    tfidf_dim = len(vectorizer.get_feature_names_out())
    name_tfidf_pad = torch.zeros((bsz, max_n, tfidf_dim), dtype=torch.float32, device=device)
    adj_pad = torch.zeros((bsz, max_n, max_n), dtype=torch.float32, device=device)
    node_mask = torch.zeros((bsz, max_n), dtype=torch.bool, device=device)

    for i in range(bsz):
        n = min(n_list[i], max_n)
        basic_pad[i, :n] = torch.from_numpy(basic_list[i][:n]).to(device)
        name_tfidf_pad[i, :n] = torch.from_numpy(name_tfidf_list[i][:n]).to(device)
        adj_pad[i, :n, :n] = torch.from_numpy(adj_list[i][:n, :n]).to(device)
        node_mask[i, :n] = True

    return {
        "ctx_img_emb": ctx_img_emb,
        "ctx_txt_raw": ctx_txt_raw,
        "tgt_img_emb": tgt_img_emb,
        "tgt_txt_raw": tgt_txt_raw,
        "g_basic": basic_pad,
        "g_name": name_tfidf_pad,
        "g_adj": adj_pad,
        "g_mask": node_mask,
    }

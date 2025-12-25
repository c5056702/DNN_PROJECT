"""
Evaluation utilities matching the notebook metrics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .model_baseline import cosine_loss, make_txt_target


@torch.no_grad()
def eval_baseline(model, loader):
    model.eval()
    tot = 0
    sum_li = sum_lt = 0.0
    sum_ci = sum_ct = 0.0
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


@torch.no_grad()
def eval_graph(model, loader):
    model.eval()
    tot = 0
    sum_li = sum_lt = 0.0
    sum_ci = sum_ct = 0.0
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

"""
ResNet18 + TF-IDF baseline sequence model from the notebook.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetTfidfSeqPredictor(nn.Module):
    def __init__(self, img_dim: int = 512, tfidf_dim: int = 8000, txt_dim: int = 512, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.txt_proj = nn.Sequential(
            nn.Linear(tfidf_dim, txt_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.in_proj = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_img = nn.Linear(hidden_dim, img_dim)
        self.out_txt = nn.Linear(hidden_dim, txt_dim)

    def forward(self, ctx_img_emb: torch.Tensor, ctx_txt_raw: torch.Tensor):
        ctx_txt = self.txt_proj(ctx_txt_raw)
        x = torch.cat([ctx_img_emb, ctx_txt], dim=-1)
        x = self.in_proj(x)
        _, h_n = self.gru(x)
        h = h_n.squeeze(0)
        pred_img = F.normalize(self.out_img(h), dim=-1)
        pred_txt = F.normalize(self.out_txt(h), dim=-1)
        return pred_img, pred_txt


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - (pred * target).sum(dim=-1)).mean()


@torch.no_grad()
def make_txt_target(model: ResnetTfidfSeqPredictor, tgt_txt_raw: torch.Tensor) -> torch.Tensor:
    tgt = model.txt_proj(tgt_txt_raw)
    return F.normalize(tgt, dim=-1)

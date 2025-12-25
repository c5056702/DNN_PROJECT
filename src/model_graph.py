"""
Graph-fused models from the notebook.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_module import SimpleGraphReasoner


class GraphFusedSeqPredictor(nn.Module):
    def __init__(self, img_dim: int, tfidf_dim: int, txt_dim: int = 512, hidden_dim: int = 512, graph_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.txt_proj = nn.Sequential(
            nn.Linear(tfidf_dim, txt_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph = SimpleGraphReasoner(tfidf_dim=tfidf_dim, node_dim=256, g_dim=graph_dim, steps=2, dropout=dropout)
        self.in_proj = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_img = nn.Linear(hidden_dim, img_dim)
        self.out_txt = nn.Linear(hidden_dim, txt_dim)

    def forward(self, ctx_img_emb, ctx_txt_raw, g_basic, g_name, g_adj, g_mask):
        ctx_txt = self.txt_proj(ctx_txt_raw)
        x = torch.cat([ctx_img_emb, ctx_txt], dim=-1)
        x = self.in_proj(x)
        _, h_n = self.gru(x)
        h = h_n.squeeze(0)
        g = self.graph(g_basic, g_name, g_adj, g_mask)
        h = self.fuse(torch.cat([h, g], dim=-1))
        pred_img = F.normalize(self.out_img(h), dim=-1)
        pred_txt = F.normalize(self.out_txt(h), dim=-1)
        return pred_img, pred_txt


class GraphFusedTextOnlySeqPredictor(nn.Module):
    def __init__(self, img_dim: int, tfidf_dim: int, txt_dim: int = 512, hidden_dim: int = 512, graph_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.txt_proj = nn.Sequential(
            nn.Linear(tfidf_dim, txt_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph = SimpleGraphReasoner(tfidf_dim=tfidf_dim, node_dim=256, g_dim=graph_dim, steps=2, dropout=dropout)
        self.in_proj = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_img = nn.Linear(hidden_dim, img_dim)
        self.fuse_txt = nn.Sequential(
            nn.Linear(hidden_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_txt = nn.Linear(hidden_dim, txt_dim)

    def forward(self, ctx_img_emb, ctx_txt_raw, g_basic, g_name, g_adj, g_mask):
        ctx_txt = self.txt_proj(ctx_txt_raw)
        x = torch.cat([ctx_img_emb, ctx_txt], dim=-1)
        x = self.in_proj(x)
        _, h_n = self.gru(x)
        h = h_n.squeeze(0)
        pred_img = F.normalize(self.out_img(h), dim=-1)
        g = self.graph(g_basic, g_name, g_adj, g_mask)
        h_txt = self.fuse_txt(torch.cat([h, g], dim=-1))
        pred_txt = F.normalize(self.out_txt(h_txt), dim=-1)
        return pred_img, pred_txt

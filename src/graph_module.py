"""
Simple gated message-passing graph reasoner from the notebook.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGraphReasoner(nn.Module):
    """
    Input:
      basic: [B,N,6]
      name_tfidf: [B,N,V]
      adj: [B,N,N]
      mask: [B,N]
    Output:
      graph_emb: [B, G] pooled
    """

    def __init__(self, tfidf_dim: int, node_dim: int = 256, g_dim: int = 256, steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.steps = steps
        self.name_proj = nn.Linear(tfidf_dim, node_dim)
        self.basic_proj = nn.Linear(6, node_dim)
        self.node_in = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.msg = nn.Linear(node_dim, node_dim)
        self.upd = nn.GRUCell(node_dim, node_dim)
        self.out = nn.Sequential(
            nn.Linear(node_dim, g_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, basic: torch.Tensor, name_tfidf: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = basic.shape

        name_h = self.name_proj(name_tfidf)
        basic_h = self.basic_proj(basic)
        h = self.node_in(torch.cat([name_h, basic_h], dim=-1))

        adj = adj * mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj_norm = adj / deg

        for _ in range(self.steps):
            m = torch.bmm(adj_norm, self.msg(h))
            h = self.upd(m.reshape(bsz * n_nodes, -1), h.reshape(bsz * n_nodes, -1)).reshape(bsz, n_nodes, -1)
            h = h * mask.unsqueeze(-1).float()

        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
        pooled = h.sum(dim=1) / denom
        return self.out(pooled)

"""
ResNet18 image embedding utilities used in the notebook.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torchvision import models, transforms


class ResNet18Embedder:
    def __init__(self, device: torch.device):
        self.device = device
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
            ]
        )

    @torch.no_grad()
    def encode(self, pil_images: List) -> torch.Tensor:
        batch = torch.stack([self.transform(im.convert("RGB")) for im in pil_images], dim=0).to(self.device)
        feats = self.model(batch)
        return F.normalize(feats, dim=-1)

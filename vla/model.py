from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, pad_idx: int = 0) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        outputs, _ = self.gru(x)
        mask = (tokens != self.pad_idx).float().unsqueeze(-1)
        summed = (outputs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count


class VisionLanguageActionModel(nn.Module):
    """Predicts a continuous action vector from image + command tokens."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        action_dim: int = 3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder(out_dim=hidden_dim)
        self.language = LanguageEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim, pad_idx=pad_idx)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        vision_features = self.vision(images)
        language_features = self.language(tokens)
        fused = torch.cat([vision_features, language_features], dim=-1)
        hidden = self.fusion(fused)
        return self.action_head(hidden)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

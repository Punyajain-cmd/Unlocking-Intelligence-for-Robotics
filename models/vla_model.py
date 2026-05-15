"""
models/vla_model.py
────────────────────
Vision-Language-Action (VLA) model, inspired by OpenVLA.

Architecture
────────────
                    ┌─────────────────┐
Image (H×W×3) ──→  │  Visual Backbone │  (DINOv2-small or CNN)
                    └──────┬──────────┘
                           │ visual tokens (N_v × D)
                    ┌──────▼──────────┐
NL Command ────→   │  Language Enc.  │  (BERT CLS embedding)
                    └──────┬──────────┘
                           │ fused (N_v+1, D)
                    ┌──────▼──────────┐
                    │  Transformer    │  (L layers, H heads)
                    └──────┬──────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼───┐  ┌─────▼──┐  ┌─────▼──┐
         │ Action │  │ Obj-id │  │Relation│
         │  Head  │  │  Head  │  │  Head  │
         └────────┘  └────────┘  └────────┘

Action head outputs a discrete token sequence:
  [dx, dy, dz, droll, dpitch, dyaw, gripper]
  each discretised into B=256 bins  (similar to RT-2/OpenVLA).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    warnings.warn("torch not installed; VLAModel unavailable.")

from config import DEFAULT_CONFIG, TrainConfig


# ──────────────────────────────────────────────────────────
# Visual Backbone
# ──────────────────────────────────────────────────────────

if _TORCH:

    class CNNVisualBackbone(nn.Module):
        """
        Lightweight CNN backbone (no pretrained weights needed).
        Produces a sequence of patch embeddings from an image.
        """
        def __init__(self, out_dim: int = 512):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3,  32, 7, stride=2, padding=3), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64,128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128,256, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(256,out_dim, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.out_dim = out_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (B, 3, H, W) → (B, N_patches, out_dim)"""
            feat = self.encoder(x)           # (B, D, h, w)
            B, D, h, w = feat.shape
            return feat.flatten(2).transpose(1, 2)  # (B, h*w, D)


    # ── Language Encoder ────────────────────────────────────────

    class LanguageEncoder(nn.Module):
        """
        BERT-based language encoder that produces a single sentence vector.
        """
        def __init__(
            self,
            bert_model: str  = "bert-base-uncased",
            out_dim:    int  = 512,
            freeze:     bool = True,
        ):
            super().__init__()
            try:
                from transformers import BertModel
                self.bert = BertModel.from_pretrained(bert_model)
                if freeze:
                    for p in self.bert.parameters():
                        p.requires_grad = False
                bert_dim = self.bert.config.hidden_size
            except Exception:
                self.bert = None
                bert_dim  = 768

            self.proj = nn.Linear(bert_dim, out_dim)

        def forward(
            self,
            input_ids:      "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """Returns (B, out_dim) sentence embedding."""
            if self.bert is not None:
                out = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
            else:
                cls = torch.zeros(
                    input_ids.size(0), 768, device=input_ids.device
                )
            return self.proj(cls)


    # ── Action Token Discretisation ─────────────────────────────

    class ActionTokeniser:
        """
        Converts continuous action vectors to/from discrete bin tokens.

        Action vector: [dx, dy, dz, droll, dpitch, dyaw, gripper]  (7-D)
        Each dimension → integer in [0, num_bins)
        """
        DOF = 7
        RANGES = [
            (-0.15, 0.15),   # dx   (m)
            (-0.15, 0.15),   # dy
            (-0.15, 0.15),   # dz
            (-0.5,  0.5 ),   # droll  (rad)
            (-0.5,  0.5 ),   # dpitch
            (-0.5,  0.5 ),   # dyaw
            ( 0.0,  1.0 ),   # gripper  (0=closed, 1=open)
        ]

        def __init__(self, num_bins: int = 256):
            self.B = num_bins

        def encode(self, action: np.ndarray) -> List[int]:
            """action: (7,) float → list of 7 bin indices."""
            tokens = []
            for i, (lo, hi) in enumerate(self.RANGES):
                val   = float(np.clip(action[i], lo, hi))
                token = int((val - lo) / (hi - lo) * (self.B - 1))
                tokens.append(token)
            return tokens

        def decode(self, tokens: List[int]) -> np.ndarray:
            """tokens: list of 7 bin indices → (7,) float action."""
            action = np.zeros(self.DOF)
            for i, (lo, hi) in enumerate(self.RANGES):
                t          = np.clip(tokens[i], 0, self.B - 1)
                action[i]  = lo + (t / (self.B - 1)) * (hi - lo)
            return action


    # ── Full VLA Model ───────────────────────────────────────────

    class VLAModel(nn.Module):
        """
        Vision-Language-Action model.

        Inputs
        ──────
        image          : (B, 3, H, W)
        input_ids      : (B, L)   tokenised command
        attention_mask : (B, L)
        action_tokens  : (B, 7)   int  [training only]

        Outputs (training)
        ──────────────────
        action_logits  : (B, 7, num_bins)

        Outputs (inference)
        ───────────────────
        predicted_action : (B, 7) float
        """

        def __init__(self, cfg: TrainConfig = DEFAULT_CONFIG.train):
            super().__init__()
            D = cfg.hidden_dim

            self.visual_enc   = CNNVisualBackbone(out_dim=D)
            self.lang_enc     = LanguageEncoder(
                bert_model=cfg.language_backbone,
                out_dim=D,
                freeze=True,
            )

            encoder_layer  = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.num_heads,
                dim_feedforward=D * 4, dropout=cfg.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.num_transformer_layers
            )

            # Action head: 7 independent classifiers
            self.action_heads = nn.ModuleList([
                nn.Linear(D, cfg.num_action_bins) for _ in range(7)
            ])
            self.tokeniser = ActionTokeniser(cfg.num_action_bins)
            self.D = D

        def forward(
            self,
            image:          "torch.Tensor",
            input_ids:      "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """Returns action_logits: (B, 7, num_bins)"""
            vis_tokens = self.visual_enc(image)                # (B, N_v, D)
            lang_vec   = self.lang_enc(input_ids,
                                        attention_mask)         # (B, D)
            lang_token = lang_vec.unsqueeze(1)                  # (B, 1, D)
            tokens     = torch.cat([lang_token, vis_tokens], 1) # (B, N_v+1, D)

            fused = self.transformer(tokens)                    # (B, N_v+1, D)
            cls   = fused[:, 0, :]                              # CLS position

            logits = torch.stack(
                [head(cls) for head in self.action_heads], dim=1
            )  # (B, 7, num_bins)
            return logits

        @torch.no_grad()
        def predict(
            self,
            image:          "torch.Tensor",
            input_ids:      "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> np.ndarray:
            """Returns predicted action vector (7,) float."""
            self.eval()
            logits  = self(image, input_ids, attention_mask)  # (1, 7, B)
            tokens  = logits.argmax(dim=-1)[0].cpu().tolist()  # (7,)
            return self.tokeniser.decode(tokens)

        def compute_loss(
            self,
            logits:  "torch.Tensor",           # (B, 7, num_bins)
            targets: "torch.Tensor",           # (B, 7) int
        ) -> "torch.Tensor":
            B, dof, bins = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * dof, bins),
                targets.reshape(B * dof),
            )
            return loss


    # ── Checkpoint utilities ─────────────────────────────────────

    def save_checkpoint(model: VLAModel, path: str, epoch: int, metrics: Dict):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }, path)

    def load_checkpoint(model: VLAModel, path: str) -> Tuple[int, Dict]:
        ckpt  = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    if not _TORCH:
        print("torch not installed, skipping VLA self-test.")
    else:
        cfg   = DEFAULT_CONFIG.train
        model = VLAModel(cfg)
        B     = 2

        img   = torch.randn(B, 3, 224, 224)
        ids   = torch.randint(0, 30522, (B, 32))
        mask  = torch.ones(B, 32, dtype=torch.long)
        acts  = torch.randint(0, cfg.num_action_bins, (B, 7))

        logits = model(img, ids, mask)
        loss   = model.compute_loss(logits, acts)
        pred   = model.predict(img[:1], ids[:1], mask[:1])

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"VLAModel  params={params:.1f}M  "
              f"logits={logits.shape}  loss={loss.item():.4f}")
        print(f"Predicted action: {pred}")

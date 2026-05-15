"""
models/temporal_backbone.py
────────────────────────────
Temporal visual encoder: converts a video clip (T frames) into a compact
feature sequence that captures spatial content AND motion dynamics.

Architecture:
  Frame  encoder  : CNN per-frame feature extractor  (shared weights)
  Temporal fusion : Transformer over frame sequence  + optional flow stream
  Output          : (B, D) clip embedding  or  (B, T, D) per-frame features

This feeds into UniversalVLAModel as the "visual spine" for video input.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    warnings.warn("torch not available – TemporalBackbone unavailable.")


if _TORCH:

    # ─────────────────────────────────────────────────────────
    # Positional encoding for Transformer
    # ─────────────────────────────────────────────────────────

    class SinusoidalPositionalEncoding(nn.Module):
        """Sinusoidal 1-D PE (length × dim)."""

        def __init__(self, dim: int, max_len: int = 512):
            super().__init__()
            pe  = torch.zeros(max_len, dim)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, D)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1)]


    # ─────────────────────────────────────────────────────────
    # Per-frame CNN encoder (lightweight, shared across T)
    # ─────────────────────────────────────────────────────────

    class FrameEncoder(nn.Module):
        """
        Lightweight CNN that maps one RGB frame (3, H, W) to a feature vector (D,).
        Shared weights applied independently to each frame in the clip.
        """

        def __init__(self, out_dim: int = 256):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3,  32, 7, stride=2, padding=3), nn.BatchNorm2d(32),  nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128,256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # → (B, 256, 4, 4)
            )
            self.proj = nn.Sequential(
                nn.Flatten(),                              # (B, 256*4*4=4096)
                nn.Linear(4096, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            )
            self.out_dim = out_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, 3, H, W) → (B, out_dim)"""
            return self.proj(self.cnn(x))

        def encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
            """
            clip: (B, T, 3, H, W) → (B, T, out_dim)
            Applies frame encoder to every frame independently.
            """
            B, T, C, H, W = clip.shape
            frames = clip.view(B * T, C, H, W)          # (B*T, 3, H, W)
            feats  = self.forward(frames)                # (B*T, D)
            return feats.view(B, T, -1)                  # (B, T, D)


    # ─────────────────────────────────────────────────────────
    # Flow encoder (optical-flow stream)
    # ─────────────────────────────────────────────────────────

    class FlowEncoder(nn.Module):
        """
        Encodes an optical-flow clip (B, T, 2, H, W) into features.
        Captures explicit motion information.
        """

        def __init__(self, out_dim: int = 128):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(2,  16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 16, out_dim),
                nn.ReLU(),
            )
            self.out_dim = out_dim

        def forward(self, flow: torch.Tensor) -> torch.Tensor:
            """flow: (B, T, 2, H, W) → (B, T, out_dim)"""
            B, T, C, H, W = flow.shape
            x = flow.view(B * T, C, H, W)
            return self.proj(self.cnn(x)).view(B, T, -1)


    # ─────────────────────────────────────────────────────────
    # Temporal Transformer
    # ─────────────────────────────────────────────────────────

    class TemporalTransformer(nn.Module):
        """
        Transformer encoder over the temporal dimension.
        Fuses frame-level and (optionally) flow-level features.

        Input:  (B, T, D_in)    frame features
        Output: (B, T, D)       contextualised frame features
                (B, D)          pooled clip embedding [CLS token]
        """

        def __init__(
            self,
            in_dim:      int = 256,
            flow_dim:    int = 128,
            hidden_dim:  int = 512,
            num_heads:   int = 8,
            num_layers:  int = 4,
            dropout:     float = 0.1,
            use_flow:    bool  = True,
        ):
            super().__init__()
            self.use_flow = use_flow
            self.input_proj  = nn.Linear(in_dim,   hidden_dim)
            self.flow_proj   = nn.Linear(flow_dim, hidden_dim) if use_flow else None
            self.pe          = SinusoidalPositionalEncoding(hidden_dim)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

            # CLS token for clip-level embedding
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)

            self.out_norm = nn.LayerNorm(hidden_dim)
            self.hidden_dim = hidden_dim

        def forward(
            self,
            frame_feats: torch.Tensor,                    # (B, T, D_frame)
            flow_feats:  Optional[torch.Tensor] = None,   # (B, T, D_flow)
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns
            ───────
            per_frame  : (B, T, hidden_dim)
            clip_embed : (B, hidden_dim)    — CLS token output
            """
            x = self.input_proj(frame_feats)             # (B, T, H)
            if self.use_flow and flow_feats is not None and self.flow_proj is not None:
                x = x + self.flow_proj(flow_feats)   # additive flow fusion
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x   = torch.cat([cls, x], dim=1)          # (B, T+1, H)
            x   = self.pe(x)

            x   = self.transformer(x)                 # (B, T+1, H)
            x   = self.out_norm(x)

            clip_embed = x[:, 0, :]                   # CLS → (B, H)
            per_frame  = x[:, 1:, :]                  # frames → (B, T, H)
            return per_frame, clip_embed


    # ─────────────────────────────────────────────────────────
    # Full temporal backbone
    # ─────────────────────────────────────────────────────────

    class TemporalBackbone(nn.Module):
        """
        Full video → feature pipeline.

        Input:
          clip  : (B, T, 3, H, W)  float, normalised
          flow  : (B, T, 2, H, W)  optional optical-flow clip

        Output:
          clip_embed : (B, hidden_dim)    pooled clip representation
          per_frame  : (B, T, hidden_dim) per-frame contextualised features
        """

        def __init__(
            self,
            frame_dim:   int   = 256,
            flow_dim:    int   = 128,
            hidden_dim:  int   = 512,
            num_heads:   int   = 8,
            num_layers:  int   = 4,
            dropout:     float = 0.1,
            use_flow:    bool  = True,
        ):
            super().__init__()
            self.frame_enc = FrameEncoder(out_dim=frame_dim)
            self.flow_enc  = FlowEncoder(out_dim=flow_dim) if use_flow else None
            self.temporal  = TemporalTransformer(
                in_dim      = frame_dim,
                flow_dim    = flow_dim if use_flow else 0,
                hidden_dim  = hidden_dim,
                num_heads   = num_heads,
                num_layers  = num_layers,
                dropout     = dropout,
                use_flow    = use_flow,
            )
            self.hidden_dim = hidden_dim
            self.use_flow   = use_flow

        def forward(
            self,
            clip:  torch.Tensor,
            flow:  Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            frame_feats = self.frame_enc.encode_clip(clip)   # (B, T, D_frame)
            flow_feats  = None
            if self.use_flow and flow is not None and self.flow_enc is not None:
                flow_feats = self.flow_enc(flow)             # (B, T, D_flow)
            return self.temporal(frame_feats, flow_feats)

        @property
        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @torch.no_grad()
        def encode_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
            """
            Convenience: encode a single RGB frame (B, 3, H, W) as a 1-frame clip.
            Returns (B, hidden_dim).
            """
            clip = frame.unsqueeze(1)   # (B, 1, 3, H, W)
            _, embed = self.forward(clip)
            return embed


    # ─────────────────────────────────────────────────────────
    # Lightweight single-frame variant
    # ─────────────────────────────────────────────────────────

    class StaticVisualBackbone(nn.Module):
        """
        Single-frame backbone (no temporal encoding).
        Drop-in replacement for TemporalBackbone when processing
        individual images rather than video clips.
        """

        def __init__(self, out_dim: int = 512):
            super().__init__()
            self.enc = FrameEncoder(out_dim=out_dim)
            self.hidden_dim = out_dim
            self.use_flow = False

        def forward(
            self,
            clip: torch.Tensor,                       # (B, T, 3, H, W) or (B, 3, H, W)
            flow: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if clip.dim() == 5:
                # Use last frame only
                frame = clip[:, -1]                   # (B, 3, H, W)
            else:
                frame = clip                          # (B, 3, H, W)
            embed = self.enc(frame)                   # (B, D)
            return embed.unsqueeze(1), embed          # per_frame, clip_embed

        @torch.no_grad()
        def encode_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
            return self.enc(frame)


    # ─────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────

    def build_temporal_backbone(
        mode:       str = "temporal",   # "temporal" | "static"
        hidden_dim: int = 512,
        use_flow:   bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Build a visual backbone.

        mode:
          "temporal" — full video Transformer (recommended for video input)
          "static"   — single-frame CNN       (for still images)
        """
        if mode == "temporal":
            return TemporalBackbone(
                hidden_dim=hidden_dim, use_flow=use_flow, **kwargs
            )
        return StaticVisualBackbone(out_dim=hidden_dim)


if __name__ == "__main__":
    if not _TORCH:
        print("torch not available")
    else:
        B, T, H, W = 2, 8, 224, 224
        clip = torch.randn(B, T, 3, H, W)
        flow = torch.randn(B, T, 2, H, W)

        # Temporal
        backbone = TemporalBackbone(
            hidden_dim=512, num_heads=8, num_layers=4, use_flow=True
        )
        per_frame, clip_embed = backbone(clip, flow)
        print(f"TemporalBackbone:")
        print(f"  per_frame  shape : {per_frame.shape}")   # (B, T, 512)
        print(f"  clip_embed shape : {clip_embed.shape}")  # (B, 512)
        print(f"  params           : {backbone.num_params/1e6:.1f}M")

        # Static
        static = StaticVisualBackbone(out_dim=512)
        _, se = static(clip)
        print(f"StaticBackbone clip_embed: {se.shape}")

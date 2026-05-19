"""
models/pretrained_backbone.py
───────────────────────────────
Pretrained visual backbones for the Universal VLA model.

Uses timm (PyTorch Image Models) to load state-of-the-art pretrained
feature extractors. This replaces the random-init CNN in temporal_backbone.py
with a properly pretrained model — dramatically improving data efficiency,
transfer learning, and generalisation to new visual environments.

Available backbones (speed-accuracy tradeoff):
  "mobilenetv3_small"  – 2.5M params,   ~4 ms/frame  (fastest, deploy)
  "efficientnet_b0"    – 5.3M params,   ~6 ms/frame  (best balance)
  "efficientnet_b3"    – 12M params,   ~12 ms/frame  (higher accuracy)
  "resnet18"           – 11M params,   ~8 ms/frame   (classic baseline)
  "resnet50"           – 25M params,  ~15 ms/frame   (strong baseline)
  "convnext_tiny"      – 28M params,  ~14 ms/frame   (modern CNN)
  "vit_small_patch16_224" – 22M params ~20 ms/frame  (transformer)

Usage
─────
  backbone = build_pretrained_backbone("efficientnet_b0", out_dim=512)
  clip = torch.randn(B, T, 3, 224, 224)
  per_frame, embed = backbone(clip)
  # embed: (B, 512)
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
    warnings.warn("torch not available – pretrained backbone unavailable.")

try:
    import timm
    _TIMM = True
except ImportError:
    _TIMM = False
    warnings.warn("timm not available – pretrained backbone unavailable. "
                  "Install with: pip install timm")


if _TORCH:

    # ─────────────────────────────────────────────────────────
    # Pretrained frame encoder (drops classification head)
    # ─────────────────────────────────────────────────────────

    class PretrainedFrameEncoder(nn.Module):
        """
        Per-frame feature extractor based on a timm pretrained model.

        Input:  (B, 3, H, W)  — normalised RGB, ImageNet stats
        Output: (B, out_dim)  — feature vector

        Parameters
        ──────────
        model_name : timm model name (see module docstring)
        out_dim    : output embedding dimension
        freeze     : freeze pretrained weights (only train projection)
        """

        # ImageNet normalisation constants
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        def __init__(
            self,
            model_name: str  = "efficientnet_b0",
            out_dim:    int  = 512,
            freeze:     bool = False,
            pretrained: bool = True,
        ):
            super().__init__()
            self.out_dim = out_dim

            if _TIMM:
                try:
                    self.encoder = timm.create_model(
                        model_name,
                        pretrained   = pretrained,
                        num_classes  = 0,      # drop classifier head
                        global_pool  = "avg",  # global average pooling
                    )
                    feat_dim = self.encoder.num_features
                    self._using_timm = True
                except Exception as e:
                    warnings.warn(f"timm model {model_name!r} failed ({e}); "
                                  "falling back to random CNN.")
                    self._using_timm = False
                    self.encoder = self._make_fallback_cnn()
                    feat_dim = 512
            else:
                warnings.warn("timm not installed; using random CNN fallback.")
                self._using_timm = False
                self.encoder = self._make_fallback_cnn()
                feat_dim = 512

            if freeze and self._using_timm:
                for p in self.encoder.parameters():
                    p.requires_grad = False

            self.proj = nn.Sequential(
                nn.Linear(feat_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

            # Register normalisation buffer (applied before encoder)
            mean = torch.tensor(self.MEAN).view(1, 3, 1, 1)
            std  = torch.tensor(self.STD).view(1, 3, 1, 1)
            self.register_buffer("norm_mean", mean)
            self.register_buffer("norm_std",  std)

        @staticmethod
        def _make_fallback_cnn() -> nn.Module:
            """Lightweight random-init CNN fallback (no timm required)."""
            return nn.Sequential(
                nn.Conv2d(3,  32, 7, stride=2, padding=3), nn.BatchNorm2d(32),  nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128,256,3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256,512,3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        def _normalise(self, x: torch.Tensor) -> torch.Tensor:
            """Apply ImageNet normalisation.  x: float in [0, 1]."""
            return (x - self.norm_mean) / (self.norm_std + 1e-6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (B, 3, H, W) — float in [0, 1] OR [0, 255]
            Returns: (B, out_dim)
            """
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            if x.max() > 1.1:           # assume [0, 255]
                x = x / 255.0

            x = self._normalise(x)

            if self._using_timm:
                feats = self.encoder(x)   # (B, feat_dim)
            else:
                feats = self.encoder(x)   # (B, 512)

            return self.proj(feats)

        def encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
            """
            clip: (B, T, 3, H, W) → (B, T, out_dim)
            Applies per-frame encoding to every frame independently.
            """
            B, T, C, H, W = clip.shape
            frames = clip.reshape(B * T, C, H, W)
            feats  = self.forward(frames)        # (B*T, out_dim)
            return feats.reshape(B, T, -1)

        @property
        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @property
        def trainable_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    # ─────────────────────────────────────────────────────────
    # Positional encoding (re-used from temporal_backbone)
    # ─────────────────────────────────────────────────────────

    class SinusoidalPE(nn.Module):
        def __init__(self, dim: int, max_len: int = 512):
            super().__init__()
            pe  = torch.zeros(max_len, dim)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


    # ─────────────────────────────────────────────────────────
    # Full pretrained temporal backbone
    # ─────────────────────────────────────────────────────────

    class PretrainedTemporalBackbone(nn.Module):
        """
        Full video → feature pipeline using a pretrained per-frame encoder
        + Transformer temporal fusion.

        Replaces TemporalBackbone in universal_vla.py when pretrained=True.

        Architecture:
          PretrainedFrameEncoder (EfficientNet / MobileNet / etc.)
          ↓  per-frame features  (B, T, frame_dim)
          Transformer  (temporal attention + CLS pooling)
          ↓  (B, T, hidden_dim),  (B, hidden_dim)
        """

        def __init__(
            self,
            model_name:  str   = "efficientnet_b0",
            frame_dim:   int   = 512,
            hidden_dim:  int   = 512,
            num_heads:   int   = 8,
            num_layers:  int   = 4,
            dropout:     float = 0.1,
            freeze_enc:  bool  = False,
            pretrained:  bool  = True,
        ):
            super().__init__()
            self.frame_enc = PretrainedFrameEncoder(
                model_name = model_name,
                out_dim    = frame_dim,
                freeze     = freeze_enc,
                pretrained = pretrained,
            )
            self.input_proj = nn.Linear(frame_dim, hidden_dim)
            self.pe         = SinusoidalPE(hidden_dim)

            enc_layer = nn.TransformerEncoderLayer(
                d_model       = hidden_dim,
                nhead         = num_heads,
                dim_feedforward = hidden_dim * 4,
                dropout       = dropout,
                batch_first   = True,
                norm_first    = True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.cls_token   = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            self.out_norm    = nn.LayerNorm(hidden_dim)

            self.hidden_dim = hidden_dim
            self.use_flow   = False   # no flow stream for speed

        def forward(
            self,
            clip: torch.Tensor,
            flow: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            clip: (B, T, 3, H, W)
            Returns: per_frame (B, T, H), clip_embed (B, H)
            """
            frame_feats = self.frame_enc.encode_clip(clip)   # (B, T, frame_dim)
            x   = self.input_proj(frame_feats)               # (B, T, H)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x   = torch.cat([cls, x], dim=1)                 # (B, T+1, H)
            x   = self.pe(x)
            x   = self.transformer(x)
            x   = self.out_norm(x)
            return x[:, 1:, :], x[:, 0, :]                   # per_frame, CLS

        @torch.no_grad()
        def encode_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
            """(B, 3, H, W) → (B, hidden_dim)"""
            clip = frame.unsqueeze(1)
            _, embed = self.forward(clip)
            return embed

        @property
        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @property
        def trainable_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    # ─────────────────────────────────────────────────────────
    # Lightweight single-frame pretrained backbone
    # ─────────────────────────────────────────────────────────

    class PretrainedStaticBackbone(nn.Module):
        """
        Single-frame pretrained backbone.
        Fastest option — use for real-time deployment.
        """

        def __init__(
            self,
            model_name: str  = "mobilenetv3_small_100",
            out_dim:    int  = 512,
            freeze:     bool = False,
            pretrained: bool = True,
        ):
            super().__init__()
            self.enc = PretrainedFrameEncoder(
                model_name = model_name,
                out_dim    = out_dim,
                freeze     = freeze,
                pretrained = pretrained,
            )
            self.hidden_dim = out_dim
            self.use_flow   = False

        def forward(
            self,
            clip: torch.Tensor,
            flow: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if clip.dim() == 5:
                frame = clip[:, -1]   # use last frame
            else:
                frame = clip
            embed = self.enc(frame)
            return embed.unsqueeze(1), embed

        @torch.no_grad()
        def encode_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
            return self.enc(frame)


    # ─────────────────────────────────────────────────────────
    # Factory function
    # ─────────────────────────────────────────────────────────

    def build_pretrained_backbone(
        mode:        str  = "temporal",    # "temporal" | "static"
        model_name:  str  = "efficientnet_b0",
        hidden_dim:  int  = 512,
        freeze_enc:  bool = False,
        pretrained:  bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Build a pretrained visual backbone.

        Parameters
        ──────────
        mode       : "temporal" (video, best quality) or "static" (single frame, fastest)
        model_name : timm backbone (see module docstring for options)
        hidden_dim : output embedding dimension
        freeze_enc : freeze pretrained weights (only train projection + transformer)
        pretrained : load ImageNet pretrained weights

        Speed tradeoffs
        ───────────────
        Deploy on robot (latency critical):
            mode="static", model_name="mobilenetv3_small_100", freeze_enc=True

        Best quality (GPU available):
            mode="temporal", model_name="efficientnet_b3", freeze_enc=False

        Good balance (default):
            mode="temporal", model_name="efficientnet_b0", freeze_enc=False
        """
        if mode == "temporal":
            return PretrainedTemporalBackbone(
                model_name  = model_name,
                hidden_dim  = hidden_dim,
                freeze_enc  = freeze_enc,
                pretrained  = pretrained,
                **kwargs,
            )
        return PretrainedStaticBackbone(
            model_name = model_name,
            out_dim    = hidden_dim,
            freeze     = freeze_enc,
            pretrained = pretrained,
        )


    # ─────────────────────────────────────────────────────────
    # Available backbones info
    # ─────────────────────────────────────────────────────────

    AVAILABLE_BACKBONES = {
        "mobilenetv3_small_100": {
            "params_M": 2.5, "latency_ms": 4, "description": "Fastest — real-time deploy"
        },
        "efficientnet_b0": {
            "params_M": 5.3, "latency_ms": 6, "description": "Best speed/accuracy balance"
        },
        "efficientnet_b3": {
            "params_M": 12,  "latency_ms": 12, "description": "High accuracy"
        },
        "resnet18": {
            "params_M": 11,  "latency_ms": 8,  "description": "Classic baseline"
        },
        "resnet50": {
            "params_M": 25,  "latency_ms": 15, "description": "Strong baseline"
        },
        "convnext_tiny": {
            "params_M": 28,  "latency_ms": 14, "description": "Modern CNN"
        },
    }

    def list_backbones() -> None:
        print("\nAvailable pretrained backbones:")
        print(f"  {'Name':<30} {'Params':>8}  {'Latency':>10}  Description")
        print("  " + "-"*70)
        for name, info in AVAILABLE_BACKBONES.items():
            print(f"  {name:<30} {info['params_M']:>6}M  "
                  f"{info['latency_ms']:>7}ms/fr  {info['description']}")


if __name__ == "__main__":
    if not _TORCH:
        print("torch not available")
    else:
        B, T, H, W = 2, 8, 224, 224
        clip = torch.rand(B, T, 3, H, W)

        list_backbones()

        # Test temporal backbone
        backbone = build_pretrained_backbone(
            mode="temporal", model_name="efficientnet_b0",
            hidden_dim=512, pretrained=False,
        )
        per_frame, embed = backbone(clip)
        print(f"\nPretrainedTemporalBackbone (EfficientNet-B0):")
        print(f"  per_frame shape : {per_frame.shape}")     # (2, 8, 512)
        print(f"  embed shape     : {embed.shape}")          # (2, 512)
        print(f"  total params    : {backbone.num_params/1e6:.1f}M")
        print(f"  trainable       : {backbone.trainable_params/1e6:.1f}M")

        # Test static backbone (fastest)
        static = build_pretrained_backbone(
            mode="static", model_name="mobilenetv3_small_100",
            hidden_dim=256, pretrained=False,
        )
        _, se = static(clip)
        print(f"\nPretrainedStaticBackbone (MobileNetV3-Small):")
        print(f"  embed shape : {se.shape}")
        print(f"  total params: {static.enc.num_params/1e6:.1f}M")

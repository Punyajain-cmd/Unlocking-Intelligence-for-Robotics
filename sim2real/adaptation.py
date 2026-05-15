"""
sim2real/adaptation.py
───────────────────────
Test-time domain adaptation for closing the sim-to-real gap.

Techniques implemented:
  1. AdaptiveBNUpdater     — updates BatchNorm running stats on unlabelled
                             real frames (no labels needed).
  2. TTAAdapter            — Test-Time Adaptation via entropy minimisation
                             on the AdaptiveLayerNorm params.
  3. EMAAdapter            — Exponential Moving Average model for stable inference.
  4. VisualFeatureAdapter  — lightweight adapter layers (LoRA-style) that map
                             simulated visual features to real-world space.
  5. DomainClassifier      — adversarial domain discriminator for training
                             domain-invariant representations.

Usage (inference)
─────────────────
  tta = TTAAdapter(model, steps=3, lr=1e-4)
  tta.adapt(real_frames)          # update AdaptiveLayerNorm stats
  action, gripper = model.predict(...)
"""

from __future__ import annotations

import copy
import warnings
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    warnings.warn("torch not available – adaptation modules unavailable.")


# ─────────────────────────────────────────────────────────
# 1. Adaptive BatchNorm updater
# ─────────────────────────────────────────────────────────

if _TORCH:

    class AdaptiveBNUpdater:
        """
        Updates running mean/var of all BatchNorm layers using a batch
        of unlabelled real-world frames.  Zero gradient computation needed.

        This is the cheapest sim2real adaptation: just forward-pass
        a few real frames through the feature extractor before deployment.
        """

        def __init__(self, model: nn.Module):
            self.model  = model
            self._bn_layers = [m for m in model.modules()
                               if isinstance(m, (nn.BatchNorm1d,
                                                 nn.BatchNorm2d,
                                                 nn.BatchNorm3d))]

        def reset(self):
            """Restore BN layers to training mode so stats update."""
            for bn in self._bn_layers:
                bn.training = True
                bn.momentum = None    # use cumulative average

        def update(self, images: "torch.Tensor", n_passes: int = 4):
            """
            Forward-pass images through the model in inference-BN-update
            mode.  Call this on a few hundred real frames before deploying.

            images : (B, 3, H, W)  or  (B, T, 3, H, W)
            """
            was_training = self.model.training
            self.reset()
            with torch.no_grad():
                for _ in range(n_passes):
                    # Only need the visual backbone
                    vis = getattr(self.model, "visual_backbone",
                                  getattr(self.model, "visual_enc", None))
                    if vis is not None:
                        if images.dim() == 4:
                            images_in = images.unsqueeze(1)
                        else:
                            images_in = images
                        vis(images_in)
                    else:
                        # Fall back to full forward (may need dummy inputs)
                        pass
            # Restore
            self.model.train(was_training)

        @property
        def num_bn_layers(self) -> int:
            return len(self._bn_layers)


    # ─────────────────────────────────────────────────────────
    # 2. Test-Time Adaptation (entropy minimisation)
    # ─────────────────────────────────────────────────────────

    class TTAAdapter:
        """
        Test-time adaptation via entropy minimisation on AdaptiveLayerNorm
        parameters.

        Only the γ and β parameters of AdaptiveLayerNorm are updated —
        all other weights stay frozen.  This is safe, fast, and reversible.

        References: TENT (Wang et al., 2021), TTT++ (Liu et al., 2021)
        """

        def __init__(
            self,
            model:      nn.Module,
            lr:         float = 1e-4,
            steps:      int   = 3,
            reset_each: bool  = False,     # reset before each new episode?
        ):
            self.model      = model
            self.lr         = lr
            self.steps      = steps
            self.reset_each = reset_each

            # Collect AdaptiveLayerNorm params
            self._adapt_params = self._get_adapt_params()
            self._optimizer    = torch.optim.Adam(self._adapt_params, lr=lr)

            # Save original values for reset
            self._original = [p.data.clone() for p in self._adapt_params]

        def _get_adapt_params(self) -> List[nn.Parameter]:
            params = []
            for m in self.model.modules():
                cls_name = type(m).__name__
                if cls_name == "AdaptiveLayerNorm":
                    params += list(m.parameters())
            if not params:
                # Fall back to all LayerNorm affine params
                for m in self.model.modules():
                    if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                        params += list(m.parameters())
            return params

        def adapt(self, frames: "torch.Tensor") -> float:
            """
            Run TTA for self.steps gradient steps on unlabelled real frames.

            frames : (B, 3, H, W) or (B, T, 3, H, W)
            Returns mean entropy loss.
            """
            if not self._adapt_params:
                return 0.0

            if self.reset_each:
                self.reset()

            total_loss = 0.0
            vis = getattr(self.model, "visual_backbone",
                          getattr(self.model, "visual_enc", None))
            if vis is None:
                return 0.0

            self.model.eval()
            for _ in range(self.steps):
                self._optimizer.zero_grad()
                with torch.enable_grad():
                    if frames.dim() == 4:
                        frames_in = frames.unsqueeze(1)
                    else:
                        frames_in = frames
                    _, feat = vis(frames_in)               # (B, D)
                    # Entropy of softmax over features as a proxy
                    p    = F.softmax(feat, dim=-1)
                    ent  = -(p * torch.log(p + 1e-9)).sum(dim=-1).mean()
                    ent.backward()
                    self._optimizer.step()
                    total_loss += ent.item()

            return total_loss / max(self.steps, 1)

        def reset(self):
            """Restore AdaptiveLayerNorm params to their original values."""
            for p, orig in zip(self._adapt_params, self._original):
                p.data.copy_(orig)

        @property
        def num_adapt_params(self) -> int:
            return sum(p.numel() for p in self._adapt_params)


    # ─────────────────────────────────────────────────────────
    # 3. EMA model wrapper
    # ─────────────────────────────────────────────────────────

    class EMAModel:
        """
        Exponential Moving Average over model parameters.
        Produces smoother, more stable inference.

        Usage:
            ema = EMAModel(model, decay=0.999)
            # After each gradient step:
            ema.update(model)
            # For inference:
            ema.apply()         # copy EMA params into model
            result = model.predict(...)
            ema.restore()       # restore original params
        """

        def __init__(self, model: nn.Module, decay: float = 0.999):
            self.decay  = decay
            self._ema   = copy.deepcopy(model)
            self._backup: Optional[Dict] = None
            for p in self._ema.parameters():
                p.requires_grad = False

        @torch.no_grad()
        def update(self, model: nn.Module):
            """Call after every gradient step."""
            for ema_p, model_p in zip(self._ema.parameters(),
                                      model.parameters()):
                ema_p.data.mul_(self.decay).add_(
                    model_p.data, alpha=1.0 - self.decay
                )

        def apply(self, model: nn.Module):
            """Copy EMA params into model for inference."""
            self._backup = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            model.load_state_dict(self._ema.state_dict())

        def restore(self, model: nn.Module):
            """Restore original params after inference."""
            if self._backup is not None:
                model.load_state_dict(self._backup)
                self._backup = None


    # ─────────────────────────────────────────────────────────
    # 4. LoRA-style visual feature adapter
    # ─────────────────────────────────────────────────────────

    class VisualFeatureAdapter(nn.Module):
        """
        Lightweight LoRA-style adapter that maps sim visual features
        to real-world feature space.

        Inserted after the visual backbone:
          sim_features → Adapter → real_features

        Only this adapter is fine-tuned on a small real-world dataset.
        The backbone remains frozen.

        r : rank of the low-rank factorisation (typically 4–16).
        """

        def __init__(self, dim: int, r: int = 8, alpha: float = 1.0):
            super().__init__()
            self.down  = nn.Linear(dim, r, bias=False)
            self.up    = nn.Linear(r, dim, bias=False)
            self.scale = alpha / r
            self.norm  = nn.LayerNorm(dim)
            nn.init.normal_(self.down.weight, std=0.01)
            nn.init.zeros_(self.up.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(x + self.scale * self.up(self.down(x)))


    class MultiScaleAdapter(nn.Module):
        """
        Stack of VisualFeatureAdapters that adapts at multiple feature scales.
        Useful when bridging large domain gaps.
        """

        def __init__(self, dim: int, ranks: List[int] = (4, 8, 16)):
            super().__init__()
            self.adapters = nn.ModuleList([
                VisualFeatureAdapter(dim, r=r) for r in ranks
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for adapter in self.adapters:
                x = adapter(x)
            return x


    # ─────────────────────────────────────────────────────────
    # 5. Domain adversarial discriminator
    # ─────────────────────────────────────────────────────────

    class GradientReversal(torch.autograd.Function):
        """Gradient reversal layer for adversarial domain training."""

        @staticmethod
        def forward(ctx, x, alpha):
            ctx.save_for_backward(torch.tensor(alpha))
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            alpha, = ctx.saved_tensors
            return -alpha * grad, None


    class DomainClassifier(nn.Module):
        """
        Binary domain discriminator: sim (0) vs real (1).
        Used with gradient reversal for domain-adversarial training
        (DANN — Domain Adversarial Neural Networks).

        Insert after the shared visual backbone.
        """

        def __init__(self, in_dim: int, hidden_dim: int = 256, alpha: float = 1.0):
            super().__init__()
            self.alpha = alpha
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
            )

        def forward(
            self,
            features: torch.Tensor,
            alpha:    Optional[float] = None,
        ) -> torch.Tensor:
            """
            Returns (B, 2) domain logits.
            During training, gradient reversal makes the backbone
            produce domain-invariant features.
            """
            a   = alpha if alpha is not None else self.alpha
            rev = GradientReversal.apply(features, a)
            return self.net(rev)

        def compute_loss(
            self,
            sim_feats:  torch.Tensor,
            real_feats: torch.Tensor,
            alpha:      float = 1.0,
        ) -> torch.Tensor:
            B_sim  = sim_feats.size(0)
            B_real = real_feats.size(0)
            feats  = torch.cat([sim_feats, real_feats], dim=0)
            labels = torch.cat([
                torch.zeros(B_sim,  dtype=torch.long, device=feats.device),
                torch.ones( B_real, dtype=torch.long, device=feats.device),
            ])
            logits = self(feats, alpha)
            return F.cross_entropy(logits, labels)


    # ─────────────────────────────────────────────────────────
    # Convenience wrapper: all adaptation in one place
    # ─────────────────────────────────────────────────────────

    class Sim2RealAdapter:
        """
        High-level sim2real adaptation manager.

        Combines:
          - EMAModel                  (always on)
          - AdaptiveBNUpdater         (on first real-world encounter)
          - TTAAdapter                (optional, for extra robustness)
          - VisualFeatureAdapter      (optional, for fine-tuning)
        """

        def __init__(
            self,
            model:      nn.Module,
            ema_decay:  float = 0.999,
            use_tta:    bool  = True,
            tta_lr:     float = 1e-4,
            tta_steps:  int   = 3,
        ):
            self.model   = model
            self.ema     = EMAModel(model, decay=ema_decay)
            self.bn_upd  = AdaptiveBNUpdater(model)
            self.tta     = TTAAdapter(model, lr=tta_lr, steps=tta_steps) if use_tta else None
            self._adapted = False

        def adapt_to_environment(
            self,
            real_frames: "torch.Tensor",
            verbose:     bool = False,
        ):
            """
            One-time call when the robot enters a new environment.
            Runs BN update + optional TTA on a batch of real frames.

            real_frames : (B, 3, H, W)  unlabelled frames from the real world.
            """
            self.bn_upd.update(real_frames, n_passes=4)
            if self.tta is not None:
                loss = self.tta.adapt(real_frames)
                if verbose:
                    print(f"TTA adaptation loss: {loss:.4f}")
            self._adapted = True

        def update_ema(self):
            """Call after each training/rollout gradient step."""
            self.ema.update(self.model)

        def inference_context(self):
            """Context manager that applies EMA weights for inference."""
            return _EMAContext(self.model, self.ema)

        @property
        def is_adapted(self) -> bool:
            return self._adapted


    class _EMAContext:
        def __init__(self, model, ema):
            self._m   = model
            self._ema = ema

        def __enter__(self):
            self._ema.apply(self._m)
            return self._m

        def __exit__(self, *_):
            self._ema.restore(self._m)


if __name__ == "__main__":
    if not _TORCH:
        print("torch not available")
    else:
        from models.universal_vla import UniversalVLAModel

        model = UniversalVLAModel(
            hidden_dim=128, num_bins=64, max_dof=8,
            num_heads=2, num_layers=1, use_flow=False,
            use_temporal=False, use_bert=False,
        )

        adapter = Sim2RealAdapter(model, use_tta=True)
        frames  = torch.randn(4, 3, 224, 224)
        adapter.adapt_to_environment(frames, verbose=True)

        print(f"BN layers adapted: {adapter.bn_upd.num_bn_layers}")
        print(f"TTA adapt params:  {adapter.tta.num_adapt_params}")
        print(f"Is adapted:        {adapter.is_adapted}")

        # Test LoRA adapter
        feat  = torch.randn(4, 128)
        lora  = VisualFeatureAdapter(128, r=8)
        out   = lora(feat)
        print(f"LoRA adapter: {feat.shape} → {out.shape}")

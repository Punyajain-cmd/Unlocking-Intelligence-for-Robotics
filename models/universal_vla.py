"""
models/universal_vla.py
────────────────────────
Universal Vision-Language-Action (VLA) Model.

This model takes:
  • A video clip   (T RGB frames)     — temporal visual context
  • Optical flow   (T-1 flow frames)  — motion information
  • A language command  (text)        — instruction
  • Robot DOF info (continuous embed) — which robot is being controlled

And outputs:
  • Joint position deltas OR absolute joint targets  (any DOF N)
  • Gripper command
  • Optional: per-object trajectory predictions (auxiliary head)

Key design decisions for generalisation + sim2real:
  1. Robot-agnostic: a RobotEmbedding layer conditions the model on
     robot morphology so ONE model works for any robot.
  2. Action space scaling: outputs are normalised to [-1, 1] and
     the RobotAdapter denormalises to real joint limits.
  3. Domain adaptation: AdaptiveLayerNorm replaces standard LN
     so the model can adapt statistics at test time.
  4. History conditioning: LSTM or Transformer over past actions
     prevents mode-collapse and improves temporal consistency.
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
    warnings.warn("torch not available – UniversalVLAModel unavailable.")

if _TORCH:
    from models.temporal_backbone import (
        TemporalBackbone, StaticVisualBackbone, build_temporal_backbone
    )


# ─────────────────────────────────────────────────────────
# Robot morphology embedding
# ─────────────────────────────────────────────────────────

if _TORCH:

    class RobotMorphologyEmbedding(nn.Module):
        """
        Conditions the model on robot morphology so one model can control
        any robot.

        Input features per joint (concatenated):
          - joint type (one-hot: revolute, prismatic, fixed)
          - axis direction (3D unit vector)
          - limit lo/hi normalised
          - max velocity normalised
          - max effort normalised
          → 9 features × N_joints → Linear → embedding

        A learned CLS-like 'robot token' is appended to the token sequence.
        """

        JOINT_TYPES = {"revolute": 0, "prismatic": 1, "fixed": 2}
        FEAT_PER_JOINT = 9     # type(3) + axis(3) + limit(2) + max_vel(1)

        def __init__(self, out_dim: int = 256, max_joints: int = 32):
            super().__init__()
            self.out_dim    = out_dim
            self.max_joints = max_joints
            self.joint_enc  = nn.Sequential(
                nn.Linear(self.FEAT_PER_JOINT, 64), nn.ReLU(),
            )
            self.pool     = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=256,
                batch_first=True, dropout=0.0,
            )
            self.proj     = nn.Linear(64, out_dim)
            self.robot_token = nn.Parameter(torch.zeros(1, 1, 64))
            nn.init.normal_(self.robot_token, std=0.02)

        def forward(self, joint_feats: torch.Tensor) -> torch.Tensor:
            """
            joint_feats : (B, N_joints, FEAT_PER_JOINT)
            Returns     : (B, out_dim)
            """
            x   = self.joint_enc(joint_feats)              # (B, N, 64)
            cls = self.robot_token.expand(x.size(0), -1, -1)
            x   = torch.cat([cls, x], dim=1)               # (B, N+1, 64)
            x   = self.pool(x)
            return self.proj(x[:, 0, :])                   # CLS → (B, out_dim)

        @classmethod
        def from_robot_config(
            cls,
            cfg,             # RobotConfig
            device: str = "cpu",
        ) -> torch.Tensor:
            """
            Build a joint_feats tensor from a RobotConfig.
            Returns (1, N_joints, FEAT_PER_JOINT).
            """
            from robot.robot_config import RobotConfig
            feats = []
            for j in cfg.active_joints:
                jtype = [0.0] * 3
                jtype[cls.JOINT_TYPES.get(j.type, 0)] = 1.0
                axis  = list(j.axis) if len(j.axis) == 3 else [0, 0, 1]
                lo_n  = (j.limit[0] + 3.15) / 6.3    # rough normalisation
                hi_n  = (j.limit[1] + 3.15) / 6.3
                vel_n = min(j.max_vel / 10.0, 1.0)
                feat  = jtype + axis + [lo_n, hi_n, vel_n]
                feats.append(feat)
            if not feats:
                feats = [[0.0] * cls.FEAT_PER_JOINT]
            return torch.tensor(feats, dtype=torch.float32,
                                device=device).unsqueeze(0)


    # ─────────────────────────────────────────────────────────
    # Adaptive Layer Norm (for domain adaptation)
    # ─────────────────────────────────────────────────────────

    class AdaptiveLayerNorm(nn.Module):
        """
        Layer normalisation with learnable per-domain scale / shift.
        During test-time adaptation, only γ and β are updated — the
        rest of the model stays frozen.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.ln = nn.LayerNorm(dim, elementwise_affine=False)
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta  = nn.Parameter(torch.zeros(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ln(x) * self.gamma + self.beta


    # ─────────────────────────────────────────────────────────
    # Language encoder
    # ─────────────────────────────────────────────────────────

    class UniversalLanguageEncoder(nn.Module):
        """
        Language encoder that works with or without BERT.
        Falls back to a learned embedding lookup + GRU.
        """

        def __init__(
            self,
            out_dim:    int  = 512,
            vocab_size: int  = 30522,
            use_bert:   bool = True,
        ):
            super().__init__()
            self.out_dim = out_dim
            self._use_bert = False

            if use_bert:
                try:
                    from transformers import BertModel
                    self.bert = BertModel.from_pretrained("bert-base-uncased")
                    for p in self.bert.parameters():
                        p.requires_grad = False
                    self.bert_proj = nn.Linear(768, out_dim)
                    self._use_bert = True
                except Exception as e:
                    warnings.warn(f"BERT unavailable ({e}); using GRU fallback.")

            if not self._use_bert:
                self.embed = nn.Embedding(vocab_size, 256, padding_idx=0)
                self.gru   = nn.GRU(256, out_dim // 2, 2,
                                    batch_first=True, bidirectional=True)
                self.proj  = nn.Linear(out_dim, out_dim)

        def forward(
            self,
            input_ids:      torch.Tensor,       # (B, L)
            attention_mask: torch.Tensor,       # (B, L)
        ) -> torch.Tensor:
            """Returns (B, out_dim) language embedding."""
            if self._use_bert:
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                return self.bert_proj(cls)
            else:
                emb = self.embed(input_ids)          # (B, L, 256)
                lengths = attention_mask.sum(1)
                out, _  = self.gru(emb)              # (B, L, out_dim)
                # Last real token
                idx = (lengths - 1).clamp(0).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(-1))
                last = out.gather(1, idx).squeeze(1)
                return self.proj(last)


    # ─────────────────────────────────────────────────────────
    # Action tokeniser (N-DOF aware)
    # ─────────────────────────────────────────────────────────

    class UniversalActionTokeniser:
        """
        Discretises continuous action vectors for any DOF.
        Action vector: normalised joint deltas ∈ [-1, 1] per DOF.
        Each dim → integer bin in [0, num_bins).
        """

        def __init__(self, num_bins: int = 256, max_dof: int = 32):
            self.B    = num_bins
            self.max_dof = max_dof

        def encode(self, action: np.ndarray) -> List[int]:
            """(N,) float in [-1,1] → N bin indices."""
            clipped = np.clip(action, -1.0, 1.0)
            return [int((v + 1.0) / 2.0 * (self.B - 1)) for v in clipped]

        def decode(self, tokens: List[int]) -> np.ndarray:
            """N bin indices → (N,) float in [-1, 1]."""
            return np.array(
                [t / (self.B - 1) * 2.0 - 1.0 for t in tokens],
                dtype=np.float32,
            )


    # ─────────────────────────────────────────────────────────
    # Universal VLA Model
    # ─────────────────────────────────────────────────────────

    class UniversalVLAModel(nn.Module):
        """
        One model for all robots.

        Architecture summary:
          TemporalBackbone  → clip_embed (B, D_vis)
          LanguageEncoder   → lang_embed (B, D_lang)
          RobotMorphEmbed   → robot_embed (B, D_robot)
          ─────────── concatenate + project → fused (B, D) ───────────
          Transformer cross-attention (fused tokens + action history)
          ─────────── per-DOF action heads ────────────────────────────
          N parallel linear heads → logits (B, N, num_bins)
        """

        def __init__(
            self,
            hidden_dim:    int   = 512,
            num_bins:      int   = 256,
            max_dof:       int   = 32,
            num_heads:     int   = 8,
            num_layers:    int   = 6,
            dropout:       float = 0.1,
            use_flow:      bool  = True,
            use_temporal:  bool  = True,
            use_bert:      bool  = True,
            lang_dim:      int   = 512,
            robot_dim:     int   = 256,
            hist_len:      int   = 8,
        ):
            super().__init__()
            D = hidden_dim

            # ── Visual backbone ──────────────────────────────
            if use_temporal:
                self.visual_backbone = TemporalBackbone(
                    hidden_dim=D, use_flow=use_flow,
                    num_heads=num_heads, num_layers=min(num_layers, 4),
                )
            else:
                self.visual_backbone = StaticVisualBackbone(out_dim=D)

            # ── Language encoder ─────────────────────────────
            self.lang_encoder = UniversalLanguageEncoder(
                out_dim=lang_dim, use_bert=use_bert
            )

            # ── Robot morphology embedding ────────────────────
            self.robot_embed = RobotMorphologyEmbedding(out_dim=robot_dim)

            # ── Fusion projection ─────────────────────────────
            fused_in = D + lang_dim + robot_dim
            self.fusion_proj = nn.Sequential(
                nn.Linear(fused_in, D),
                AdaptiveLayerNorm(D),
                nn.GELU(),
            )

            # ── Action history encoder ────────────────────────
            self.hist_len    = hist_len
            self.action_hist = nn.GRU(max_dof + 1, D // 4,
                                      num_layers=1, batch_first=True)
            self.hist_proj   = nn.Linear(D // 4, D)

            # ── Cross-attention Transformer ───────────────────
            cross_layer = nn.TransformerDecoderLayer(
                d_model=D, nhead=num_heads,
                dim_feedforward=D * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.cross_transformer = nn.TransformerDecoder(
                cross_layer, num_layers=num_layers
            )

            # ── Per-DOF action heads ──────────────────────────
            # Shared MLP backbone + per-DOF heads (up to max_dof)
            self.shared_head = nn.Sequential(
                nn.Linear(D, D), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(D, D // 2), nn.GELU(),
            )
            self.action_heads = nn.ModuleList([
                nn.Linear(D // 2, num_bins) for _ in range(max_dof)
            ])
            # Gripper head (binary-ish, but use num_bins for consistency)
            self.gripper_head = nn.Linear(D // 2, num_bins)

            self.tokeniser = UniversalActionTokeniser(num_bins, max_dof)
            self.D         = D
            self.num_bins  = num_bins
            self.max_dof   = max_dof

        # ── Forward pass ─────────────────────────────────────

        def forward(
            self,
            clip:           torch.Tensor,          # (B, T, 3, H, W)
            input_ids:      torch.Tensor,          # (B, L)
            attention_mask: torch.Tensor,          # (B, L)
            joint_feats:    torch.Tensor,          # (B, N_joints, 9)  robot morphology
            n_dof:          int,                   # actual DOF for this robot
            flow:           Optional[torch.Tensor] = None,  # (B, T, 2, H, W)
            action_history: Optional[torch.Tensor] = None,  # (B, hist, max_dof+1)
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns
            ───────
            action_logits  : (B, n_dof, num_bins)
            gripper_logits : (B, num_bins)
            """
            B = clip.size(0)

            # 1. Visual
            per_frame, clip_embed = self.visual_backbone(clip, flow)

            # 2. Language
            lang_vec = self.lang_encoder(input_ids, attention_mask)    # (B, D)

            # 3. Robot morphology
            robot_vec = self.robot_embed(joint_feats)                  # (B, D_robot)

            # 4. Fuse
            fused = self.fusion_proj(
                torch.cat([clip_embed, lang_vec, robot_vec], dim=-1)
            ).unsqueeze(1)   # (B, 1, D)

            # 5. Action history
            if action_history is not None:
                _, h = self.action_hist(action_history)    # h: (1, B, D//4)
                hist_vec = self.hist_proj(h.squeeze(0)).unsqueeze(1)  # (B, 1, D)
                query    = fused + hist_vec
            else:
                query = fused

            # 6. Cross-attend to visual frame tokens
            ctx = self.cross_transformer(query, per_frame)  # (B, 1, D)
            ctx = ctx.squeeze(1)                             # (B, D)

            # 7. Action heads
            h_shared = self.shared_head(ctx)                # (B, D//2)
            logits_list = [
                self.action_heads[i](h_shared)              # (B, num_bins)
                for i in range(n_dof)
            ]
            action_logits  = torch.stack(logits_list, dim=1)  # (B, n_dof, num_bins)
            gripper_logits = self.gripper_head(h_shared)       # (B, num_bins)

            return action_logits, gripper_logits

        # ── Inference ────────────────────────────────────────

        @torch.no_grad()
        def predict(
            self,
            clip:           torch.Tensor,
            input_ids:      torch.Tensor,
            attention_mask: torch.Tensor,
            joint_feats:    torch.Tensor,
            n_dof:          int,
            flow:           Optional[torch.Tensor] = None,
            action_history: Optional[torch.Tensor] = None,
        ) -> Tuple[np.ndarray, float]:
            """
            Returns
            ───────
            action_normalised : (n_dof,) float in [-1, 1]
            gripper           : float in [0, 1]  (0=closed, 1=open)
            """
            self.eval()
            a_logits, g_logits = self(
                clip, input_ids, attention_mask,
                joint_feats, n_dof, flow, action_history,
            )
            a_tokens   = a_logits.argmax(dim=-1)[0].cpu().tolist()   # (n_dof,)
            g_token    = int(g_logits.argmax(dim=-1)[0])
            action     = self.tokeniser.decode(a_tokens)
            gripper    = g_token / (self.num_bins - 1)
            return action, gripper

        # ── Loss ─────────────────────────────────────────────

        def compute_loss(
            self,
            action_logits:  torch.Tensor,   # (B, n_dof, num_bins)
            gripper_logits: torch.Tensor,   # (B, num_bins)
            action_targets: torch.Tensor,   # (B, n_dof) int
            gripper_target: torch.Tensor,   # (B,) int
            dof_weight:     Optional[torch.Tensor] = None,  # (B, n_dof) float
        ) -> Dict[str, torch.Tensor]:
            B, n_dof, bins = action_logits.shape

            # Action cross-entropy
            act_loss = F.cross_entropy(
                action_logits.reshape(B * n_dof, bins),
                action_targets.reshape(B * n_dof),
                reduction="none",
            ).reshape(B, n_dof)

            if dof_weight is not None:
                act_loss = (act_loss * dof_weight).mean()
            else:
                act_loss = act_loss.mean()

            # Gripper cross-entropy
            g_loss = F.cross_entropy(gripper_logits, gripper_target)

            total = act_loss + 0.5 * g_loss
            return {"total": total, "action": act_loss, "gripper": g_loss}

        @property
        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())


    # ─────────────────────────────────────────────────────────
    # Checkpoint utilities
    # ─────────────────────────────────────────────────────────

    def save_universal_checkpoint(
        model:   UniversalVLAModel,
        path:    str,
        epoch:   int,
        metrics: Dict,
        cfg:     Optional[Dict] = None,
    ):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "epoch":       epoch,
            "model":       model.state_dict(),
            "metrics":     metrics,
            "model_cfg":   cfg or {},
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_universal_checkpoint(
        model:  UniversalVLAModel,
        path:   str,
    ) -> Tuple[int, Dict]:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return ckpt.get("epoch", 0), ckpt.get("metrics", {})


if __name__ == "__main__":
    if not _TORCH:
        print("torch not available")
    else:
        from robot.robot_config import get_robot

        cfg     = get_robot("kuka_iiwa7")
        B, T, H, W = 2, 8, 224, 224

        model = UniversalVLAModel(
            hidden_dim=256, num_bins=128, max_dof=32,
            num_heads=4, num_layers=2, use_flow=True,
            use_temporal=True, use_bert=False,
        )
        print(f"UniversalVLAModel  params={model.num_params/1e6:.1f}M")

        clip       = torch.randn(B, T, 3, H, W)
        flow       = torch.randn(B, T, 2, H, W)
        ids        = torch.randint(0, 1000, (B, 32))
        mask       = torch.ones(B, 32, dtype=torch.long)
        jfeats     = RobotMorphologyEmbedding.from_robot_config(cfg).expand(B, -1, -1)
        n_dof      = cfg.dof

        a_logits, g_logits = model(clip, ids, mask, jfeats, n_dof, flow)
        print(f"action_logits  : {a_logits.shape}")    # (B, 7, 128)
        print(f"gripper_logits : {g_logits.shape}")    # (B, 128)

        # Inference
        a, g = model.predict(clip[:1], ids[:1], mask[:1],
                             jfeats[:1], n_dof, flow[:1])
        print(f"Predicted action (normalised): {a.round(3)}")
        print(f"Predicted gripper: {g:.3f}")

        # Loss
        targets = torch.randint(0, 128, (B, n_dof))
        g_tgt   = torch.randint(0, 128, (B,))
        losses  = model.compute_loss(a_logits, g_logits, targets, g_tgt)
        print(f"Loss: {losses}")

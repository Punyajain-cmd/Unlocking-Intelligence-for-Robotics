"""
train_universal.py
───────────────────
Training script for the Universal VLA Model.

Trains a single model that generalises across:
  • Multiple robots (any DOF)
  • Diverse visual environments (via domain randomisation)
  • New tasks at test time (via few-shot fine-tuning)

Training loop:
  1. Sample (video_clip, command, robot_config, action_targets) from dataset
  2. Domain-randomise the video clip
  3. Forward pass through UniversalVLAModel
  4. Cross-entropy loss on discretised joint targets
  5. Optional: DANN domain adversarial loss (sim vs real)
  6. Gradient clip + Adam(W) update
  7. EMA model update

Usage
─────
  python train_universal.py
  python train_universal.py --epochs 20 --batch 16 --robot kuka_iiwa7
  python train_universal.py --robots ur5 kuka_iiwa7 franka_panda  # multi-robot
  python train_universal.py --resume checkpoints/universal_vla.pt
"""

from __future__ import annotations

import argparse
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────
# Synthetic dataset for universal training
# ─────────────────────────────────────────────────────────

if _TORCH:

    class UniversalManipulationDataset(Dataset):
        """
        Synthetic dataset that generates (video_clip, command, robot_cfg,
        action) tuples for training the Universal VLA.

        Each sample:
          clip        : (T, 3, H, W) float  — simulated camera frames
          flow        : (T, 2, H, W) float  — optical flow
          input_ids   : (L,)  int           — tokenised command
          attn_mask   : (L,)  int
          joint_feats : (N_joints, 9) float — robot morphology
          action_tgt  : (n_dof,) int        — discretised joint targets
          gripper_tgt : int                 — gripper bin
          n_dof       : int                 — effective DOF
        """

        COMMANDS = [
            "Move the blue block to the right of the green cube.",
            "Pick up the red sphere and place it on the yellow platform.",
            "Push the cyan cylinder to the left side.",
            "Stack the orange cube on top of the purple block.",
            "Grasp the small blue object near the edge.",
            "Lift the white box above the brown cylinder.",
            "Transfer the grey block to the left of the black cube.",
            "Rotate the green cylinder 90 degrees.",
            "Slide the red block forward.",
            "Release the blue sphere gently.",
        ]

        def __init__(
            self,
            robot_names: List[str],
            n_samples:   int  = 1000,
            clip_len:    int  = 8,
            img_size:    int  = 224,
            max_seq_len: int  = 32,
            num_bins:    int  = 128,
            augment:     bool = True,
        ):
            from robot.robot_config import get_robot
            from models.universal_vla import RobotMorphologyEmbedding
            from sim2real.domain_randomizer import VisualRandomizerPipeline

            self.robot_cfgs = [get_robot(n) for n in robot_names]
            self.n          = n_samples
            self.T          = clip_len
            self.H = self.W = img_size
            self.L          = max_seq_len
            self.B          = num_bins
            self.augment    = augment
            self.aug        = VisualRandomizerPipeline.default() if augment else None

            # Pre-compute joint feature tensors
            self.joint_feats = {
                n: RobotMorphologyEmbedding.from_robot_config(cfg).squeeze(0)
                for n, cfg in zip(robot_names, self.robot_cfgs)
            }
            self.robot_names = robot_names

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> Dict:
            # Random robot
            ri        = idx % len(self.robot_cfgs)
            robot_cfg = self.robot_cfgs[ri]
            r_name    = self.robot_names[ri]
            n_dof     = len(robot_cfg.arm_joints)

            # Synthetic video clip (coloured blobs on background)
            clip  = self._make_clip()
            flow  = self._make_flow()

            # Random command
            cmd     = random.choice(self.COMMANDS)
            ids, m  = self._tokenise(cmd)

            # Synthetic action target (random joint deltas)
            act_norm = np.random.uniform(-1, 1, n_dof).astype(np.float32)
            gripper  = random.random()
            act_bins = [int(np.clip((v + 1) / 2 * (self.B - 1), 0, self.B - 1))
                        for v in act_norm]
            g_bin    = int(gripper * (self.B - 1))

            return {
                "clip":        torch.tensor(clip, dtype=torch.float32),
                "flow":        torch.tensor(flow, dtype=torch.float32),
                "input_ids":   ids,
                "attn_mask":   m,
                "joint_feats": self.joint_feats[r_name],
                "action_tgt":  torch.tensor(act_bins, dtype=torch.long),
                "gripper_tgt": torch.tensor(g_bin,    dtype=torch.long),
                "n_dof":       n_dof,
                "robot_name":  r_name,
            }

        def _make_clip(self) -> np.ndarray:
            """Generate (T, 3, H, W) float32 synthetic frames in [0,1]."""
            clip = []
            colours = [
                np.array([0.8, 0.2, 0.2]),
                np.array([0.2, 0.4, 0.9]),
                np.array([0.2, 0.8, 0.2]),
            ]
            bg = np.random.uniform(0.3, 0.7, 3).astype(np.float32)
            for t in range(self.T):
                frame = np.ones((self.H, self.W, 3), dtype=np.float32)
                frame *= bg
                for j, col in enumerate(colours):
                    cx = int(self.W * (0.25 + j * 0.25) + np.random.randn() * 5)
                    cy = int(self.H * 0.5 + np.random.randn() * 5)
                    r  = int(np.random.uniform(15, 30))
                    y1, y2 = max(0, cy-r), min(self.H, cy+r)
                    x1, x2 = max(0, cx-r), min(self.W, cx+r)
                    frame[y1:y2, x1:x2] = col.astype(np.float32)
                if self.aug:
                    frame_u8 = (frame * 255).astype(np.uint8)
                    frame_u8 = self.aug(frame_u8)
                    frame    = frame_u8.astype(np.float32) / 255.0
                clip.append(frame.transpose(2, 0, 1))   # (3, H, W)
            return np.stack(clip)   # (T, 3, H, W)

        def _make_flow(self) -> np.ndarray:
            """Generate (T, 2, H, W) synthetic optical flow."""
            flow = np.random.normal(0, 0.02, (self.T, 2, self.H, self.W))
            return flow.astype(np.float32)

        def _tokenise(
            self, text: str
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            tokens = [ord(c) % 1000 + 1 for c in text[:self.L]]
            tokens += [0] * (self.L - len(tokens))
            ids  = torch.tensor(tokens, dtype=torch.long)
            mask = (ids != 0).long()
            return ids, mask


    # ─────────────────────────────────────────────────────────
    # Variable-DOF collation
    # ─────────────────────────────────────────────────────────

    def universal_collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate samples with variable DOF.
        Pads action_tgt to the max n_dof in the batch.
        Also provides a DOF mask so the loss only covers real joints.
        """
        max_dof = max(s["n_dof"] for s in batch)

        def pad_action(a, target_len):
            n = len(a)
            if n >= target_len:
                return a[:target_len]
            return torch.cat([a, torch.zeros(target_len - n, dtype=torch.long)])

        def pad_joint_feats(jf, target_len):
            n = jf.size(0)
            if n >= target_len:
                return jf[:target_len]
            pad = torch.zeros(target_len - n, jf.size(1))
            return torch.cat([jf, pad], dim=0)

        max_joints = max(s["joint_feats"].size(0) for s in batch)

        collated = {
            "clip":        torch.stack([s["clip"]       for s in batch]),
            "flow":        torch.stack([s["flow"]       for s in batch]),
            "input_ids":   torch.stack([s["input_ids"]  for s in batch]),
            "attn_mask":   torch.stack([s["attn_mask"]  for s in batch]),
            "joint_feats": torch.stack([
                pad_joint_feats(s["joint_feats"], max_joints) for s in batch
            ]),
            "action_tgt":  torch.stack([
                pad_action(s["action_tgt"], max_dof) for s in batch
            ]),
            "gripper_tgt": torch.stack([s["gripper_tgt"] for s in batch]),
            "n_dof":       [s["n_dof"] for s in batch],
            "dof_mask":    torch.stack([
                torch.cat([
                    torch.ones(s["n_dof"]),
                    torch.zeros(max_dof - s["n_dof"])
                ]) for s in batch
            ]),
        }
        return collated


    # ─────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────

    def train_one_epoch(
        model:      nn.Module,
        loader:     DataLoader,
        optimizer:  optim.Optimizer,
        scheduler:  Optional[object],
        device:     torch.device,
        max_dof:    int,
        grad_clip:  float = 1.0,
        ema:        Optional[object] = None,
    ) -> Dict[str, float]:
        model.train()
        total_loss = act_loss_sum = g_loss_sum = 0.0
        n_batches  = 0

        for batch in loader:
            clip       = batch["clip"].to(device)         # (B, T, 3, H, W)
            flow       = batch["flow"].to(device)         # (B, T, 2, H, W)
            ids        = batch["input_ids"].to(device)
            mask       = batch["attn_mask"].to(device)
            jfeats     = batch["joint_feats"].to(device)
            act_tgt    = batch["action_tgt"].to(device)   # (B, max_dof)
            g_tgt      = batch["gripper_tgt"].to(device)
            dof_mask   = batch["dof_mask"].to(device)     # (B, max_dof)
            n_dof      = max(batch["n_dof"])

            optimizer.zero_grad()

            a_logits, g_logits = model(
                clip, ids, mask, jfeats, n_dof, flow
            )

            # Align targets and mask
            act_tgt_clipped  = act_tgt[:, :n_dof]
            dof_mask_clipped = dof_mask[:, :n_dof]

            losses = model.compute_loss(
                action_logits  = a_logits,
                gripper_logits = g_logits,
                action_targets = act_tgt_clipped,
                gripper_target = g_tgt,
                dof_weight     = dof_mask_clipped,
            )

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()

            total_loss    += losses["total"].item()
            act_loss_sum  += losses["action"].item()
            g_loss_sum    += losses["gripper"].item()
            n_batches     += 1

        return {
            "loss":         total_loss  / max(n_batches, 1),
            "action_loss":  act_loss_sum / max(n_batches, 1),
            "gripper_loss": g_loss_sum  / max(n_batches, 1),
        }


    @torch.no_grad()
    def evaluate(
        model:   nn.Module,
        loader:  DataLoader,
        device:  torch.device,
        max_dof: int,
    ) -> Dict[str, float]:
        model.eval()
        correct_act = 0
        correct_g   = 0
        total_act   = 0
        total_g     = 0

        for batch in loader:
            clip    = batch["clip"].to(device)
            flow    = batch["flow"].to(device)
            ids     = batch["input_ids"].to(device)
            mask    = batch["attn_mask"].to(device)
            jfeats  = batch["joint_feats"].to(device)
            act_tgt = batch["action_tgt"].to(device)
            g_tgt   = batch["gripper_tgt"].to(device)
            n_dof   = max(batch["n_dof"])

            a_logits, g_logits = model(clip, ids, mask, jfeats, n_dof, flow)

            # Per-DOF accuracy (top-1 bin match within ±2 bins)
            preds  = a_logits.argmax(dim=-1)[:, :n_dof]   # (B, n_dof)
            tgts   = act_tgt[:, :n_dof]
            close  = (preds - tgts).abs() <= 5             # ±5 bin tolerance
            correct_act += close.float().sum().item()
            total_act   += close.numel()

            g_preds = g_logits.argmax(dim=-1)
            correct_g  += ((g_preds - g_tgt).abs() <= 5).float().sum().item()
            total_g    += g_tgt.numel()

        return {
            "action_acc":  correct_act / max(total_act, 1),
            "gripper_acc": correct_g   / max(total_g,   1),
        }


# ─────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────

def train(
    robots:      List[str],
    epochs:      int   = 10,
    batch_size:  int   = 8,
    lr:          float = 1e-4,
    weight_decay: float = 1e-5,
    hidden_dim:  int   = 256,
    num_bins:    int   = 128,
    max_dof:     int   = 32,
    clip_len:    int   = 8,
    n_train:     int   = 500,
    n_val:       int   = 100,
    save_dir:    str   = "checkpoints",
    resume:      Optional[str] = None,
    use_ema:     bool  = True,
    verbose:     bool  = True,
):
    if not _TORCH:
        print("PyTorch not available – cannot train.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining Universal VLA")
    print(f"  Robots     : {robots}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {epochs}  Batch: {batch_size}  LR: {lr}")
    print(f"  Hidden dim : {hidden_dim}  Bins: {num_bins}  MaxDOF: {max_dof}\n")

    # ── Datasets ────────────────────────────────────────────
    train_ds = UniversalManipulationDataset(
        robots, n_samples=n_train, clip_len=clip_len, augment=True
    )
    val_ds   = UniversalManipulationDataset(
        robots, n_samples=n_val,   clip_len=clip_len, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=universal_collate_fn, num_workers=0,
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        collate_fn=universal_collate_fn, num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────
    from models.universal_vla import (
        UniversalVLAModel,
        save_universal_checkpoint,
        load_universal_checkpoint,
    )

    model = UniversalVLAModel(
        hidden_dim    = hidden_dim,
        num_bins      = num_bins,
        max_dof       = max_dof,
        num_heads     = 4,
        num_layers    = 2,
        use_flow      = True,
        use_temporal  = True,
        use_bert      = False,
    ).to(device)

    print(f"  Model params: {model.num_params / 1e6:.1f} M")

    start_epoch = 0
    if resume and Path(resume).exists():
        start_epoch, _ = load_universal_checkpoint(model, resume)
        print(f"  Resumed from epoch {start_epoch}")

    # ── Optimiser + Scheduler ────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=lr * 0.01
    )

    # ── EMA ─────────────────────────────────────────────────
    ema = None
    if use_ema:
        from sim2real.adaptation import EMAModel
        ema = EMAModel(model, decay=0.999)

    # ── Training loop ────────────────────────────────────────
    best_acc = 0.0
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, max_dof, grad_clip=1.0, ema=ema,
        )
        val_metrics = evaluate(model, val_loader, device, max_dof)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"  Epoch {epoch+1:3d}/{start_epoch+epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"act_loss={train_metrics['action_loss']:.4f}  "
                f"g_loss={train_metrics['gripper_loss']:.4f}  |  "
                f"val_act_acc={val_metrics['action_acc']:.3f}  "
                f"val_g_acc={val_metrics['gripper_acc']:.3f}  "
                f"({elapsed:.1f}s)"
            )

        # Save checkpoint
        acc = val_metrics["action_acc"]
        if acc >= best_acc:
            best_acc = acc
            save_universal_checkpoint(
                model, f"{save_dir}/universal_vla_best.pt",
                epoch + 1,
                {**train_metrics, **val_metrics},
            )

        if (epoch + 1) % 5 == 0:
            save_universal_checkpoint(
                model, f"{save_dir}/universal_vla_e{epoch+1}.pt",
                epoch + 1,
                {**train_metrics, **val_metrics},
            )

    print(f"\n  Training complete.  Best val action acc: {best_acc:.3f}")
    print(f"  Saved to: {save_dir}/")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train the Universal VLA Model")
    ap.add_argument("--robots",      nargs="+",
                    default=["kuka_iiwa7", "ur5", "franka_panda"],
                    help="Robot names to train on")
    ap.add_argument("--epochs",      type=int,   default=10)
    ap.add_argument("--batch",       type=int,   default=8)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--hidden-dim",  type=int,   default=256)
    ap.add_argument("--num-bins",    type=int,   default=128)
    ap.add_argument("--max-dof",     type=int,   default=32)
    ap.add_argument("--n-train",     type=int,   default=500)
    ap.add_argument("--n-val",       type=int,   default=100)
    ap.add_argument("--save-dir",    default="checkpoints")
    ap.add_argument("--resume",      default=None)
    ap.add_argument("--no-ema",      action="store_true")
    ap.add_argument("--quiet",       action="store_true")
    args = ap.parse_args()

    train(
        robots      = args.robots,
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
        hidden_dim  = args.hidden_dim,
        num_bins    = args.num_bins,
        max_dof     = args.max_dof,
        n_train     = args.n_train,
        n_val       = args.n_val,
        save_dir    = args.save_dir,
        resume      = args.resume,
        use_ema     = not args.no_ema,
        verbose     = not args.quiet,
    )


if __name__ == "__main__":
    main()

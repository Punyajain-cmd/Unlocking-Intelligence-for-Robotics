"""
train.py
─────────
Training loop for the VLA model.

Usage
─────
  python train.py                                  # train with synthetic data
  python train.py --dataset path/to/data.jsonl     # train on real dataset
  python train.py --config configs/train_config.yaml
  python train.py --epochs 10 --batch_size 16 --lr 5e-5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    _TORCH = True
except ImportError:
    _TORCH = False
    print("[ERROR] torch not installed. Install with: pip install torch torchvision")
    sys.exit(1)

try:
    from transformers import BertTokenizer
    _HF = True
except ImportError:
    _HF  = False
    BertTokenizer = None

from config import DEFAULT_CONFIG, TrainConfig, MODEL_DIR
from models.vla_model import VLAModel, ActionTokeniser, save_checkpoint, load_checkpoint
from data.dataset_loader import get_dataloaders
from data.augmentation import ManipulationAugmentor


# ──────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _continuous_to_tokens(
    actions:    torch.Tensor,    # (B, 7) float
    tokeniser:  ActionTokeniser,
    num_bins:   int = 256,
) -> torch.Tensor:               # (B, 7) int
    B, dof = actions.shape
    tokens = torch.zeros(B, dof, dtype=torch.long, device=actions.device)
    for i, (lo, hi) in enumerate(ActionTokeniser.RANGES):
        clipped      = actions[:, i].clamp(lo, hi)
        tokens[:, i] = ((clipped - lo) / (hi - lo) * (num_bins - 1)).long()
    return tokens


# ──────────────────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────────────────

def train_one_epoch(
    model:      VLAModel,
    loader,
    optimiser:  optim.Optimizer,
    scaler:     Optional["torch.cuda.amp.GradScaler"],
    cfg:        TrainConfig,
    device:     torch.device,
    act_tok:    ActionTokeniser,
    epoch:      int,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    t0         = time.time()

    for step, batch in enumerate(loader):
        images    = batch["image"].to(device, non_blocking=True)
        actions   = batch["action_vector"].to(device, non_blocking=True)

        if "input_ids" in batch:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
        else:
            # Fallback: dummy token tensors (train vision without language)
            B = images.size(0)
            input_ids = torch.zeros(B, 32, dtype=torch.long, device=device)
            attn_mask = torch.ones(B, 32, dtype=torch.long, device=device)

        target_tokens = _continuous_to_tokens(actions, act_tok, cfg.num_action_bins)

        optimiser.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images, input_ids, attn_mask)
                loss   = model.compute_loss(logits, target_tokens)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimiser)
            scaler.update()
        else:
            logits = model(images, input_ids, attn_mask)
            loss   = model.compute_loss(logits, target_tokens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()

        loss_meter.update(loss.item(), images.size(0))

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:03d}  Step {step:04d}/{len(loader):04d}  "
                  f"Loss={loss_meter.avg:.4f}  "
                  f"({elapsed:.0f}s)")

    return loss_meter.avg


@torch.no_grad()
def validate(
    model:   VLAModel,
    loader,
    cfg:     TrainConfig,
    device:  torch.device,
    act_tok: ActionTokeniser,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    mae_meter  = AverageMeter()

    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        actions   = batch["action_vector"].to(device, non_blocking=True)

        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
        else:
            B = images.size(0)
            input_ids = torch.zeros(B, 32, dtype=torch.long, device=device)
            attn_mask = torch.ones(B, 32, dtype=torch.long, device=device)

        target_tokens = _continuous_to_tokens(actions, act_tok, cfg.num_action_bins)
        logits        = model(images, input_ids, attn_mask)
        loss          = model.compute_loss(logits, target_tokens)

        # Decode to continuous for MAE
        pred_tokens  = logits.argmax(dim=-1)  # (B, 7)
        pred_cont    = torch.stack([
            torch.tensor(act_tok.decode(t.cpu().tolist()), dtype=torch.float32)
            for t in pred_tokens
        ]).to(device)
        mae = (pred_cont - actions).abs().mean()

        loss_meter.update(loss.item(), images.size(0))
        mae_meter.update(mae.item(),  images.size(0))

    return {"val_loss": loss_meter.avg, "val_mae": mae_meter.avg}


# ──────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────

def train(cfg: TrainConfig = DEFAULT_CONFIG.train, dataset_path: Optional[str] = None):
    device = _get_device()
    print(f"\n{'='*60}")
    print(f"  RoboLang VLA Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {cfg.num_epochs}   Batch : {cfg.batch_size}   LR : {cfg.learning_rate}")
    print(f"{'='*60}\n")

    # ── Tokeniser (optional) ─────────────────────────────────
    tokeniser = None
    if _HF:
        try:
            tokeniser = BertTokenizer.from_pretrained(cfg.language_backbone)
            print(f"  Tokeniser: {cfg.language_backbone}")
        except Exception as e:
            print(f"  Tokeniser unavailable ({e}); skipping language conditioning.")

    # ── Data ─────────────────────────────────────────────────
    print("  Loading datasets …")
    train_loader, val_loader, _ = get_dataloaders(
        cfg=cfg, dataset_path=dataset_path, tokeniser=tokeniser
    )
    print(f"  Train: {len(train_loader.dataset):,}  "
          f"Val: {len(val_loader.dataset):,}")

    # ── Model ────────────────────────────────────────────────
    model   = VLAModel(cfg).to(device)
    act_tok = ActionTokeniser(cfg.num_action_bins)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {n_params:.1f}M\n")

    # Optionally resume
    start_epoch = 0
    if cfg.resume_from and Path(cfg.resume_from).exists():
        start_epoch, _ = load_checkpoint(model, cfg.resume_from)
        print(f"  Resumed from epoch {start_epoch}")

    # ── Optimiser & scheduler ────────────────────────────────
    optimiser = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    warmup_sched = LinearLR(
        optimiser, start_factor=0.1, end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )
    cosine_sched = CosineAnnealingLR(
        optimiser,
        T_max=cfg.num_epochs * len(train_loader) - cfg.warmup_steps,
        eta_min=cfg.learning_rate * 0.01,
    )
    scheduler = SequentialLR(
        optimiser, [warmup_sched, cosine_sched],
        milestones=[cfg.warmup_steps],
    )

    # Mixed precision
    scaler = (
        torch.cuda.amp.GradScaler()
        if cfg.mixed_precision and device.type == "cuda"
        else None
    )

    # ── Training loop ────────────────────────────────────────
    best_val_loss = float("inf")
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        print(f"\n── Epoch {epoch}/{cfg.num_epochs} ──────────────────────────")
        train_loss = train_one_epoch(
            model, train_loader, optimiser, scaler, cfg, device, act_tok, epoch
        )
        scheduler.step()

        if epoch % cfg.eval_every == 0:
            val_metrics = validate(model, val_loader, cfg, device, act_tok)
            print(f"  → Val loss: {val_metrics['val_loss']:.4f}  "
                  f"MAE: {val_metrics['val_mae']:.4f}")
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_mae"].append(val_metrics["val_mae"])

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                ckpt_path = str(cfg.checkpoint_dir / "best_model.pt")
                save_checkpoint(model, ckpt_path, epoch, val_metrics)
                print(f"  ★ New best model saved → {ckpt_path}")

        history["train_loss"].append(train_loss)

        if epoch % cfg.save_every == 0:
            ckpt_path = str(cfg.checkpoint_dir / f"epoch_{epoch:03d}.pt")
            save_checkpoint(model, ckpt_path, epoch, {"train_loss": train_loss})
            print(f"  Checkpoint saved → {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete.  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    return model, history


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train the RoboLang VLA model.")
    ap.add_argument("--config",     type=str, default=None, help="YAML config path")
    ap.add_argument("--dataset",    type=str, default=None, help="JSONL dataset path")
    ap.add_argument("--epochs",     type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr",         type=float, default=None)
    ap.add_argument("--resume",     type=str, default=None)
    args = ap.parse_args()

    from config import Config
    cfg = Config.from_yaml(args.config).train if args.config else DEFAULT_CONFIG.train

    if args.epochs:     cfg.num_epochs    = args.epochs
    if args.batch_size: cfg.batch_size    = args.batch_size
    if args.lr:         cfg.learning_rate = args.lr
    if args.resume:     cfg.resume_from   = args.resume

    train(cfg=cfg, dataset_path=args.dataset)


if __name__ == "__main__":
    main()

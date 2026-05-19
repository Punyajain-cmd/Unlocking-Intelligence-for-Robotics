"""
train_universal.py
───────────────────
Training script for the Universal VLA Model.

Uses the best-practice training recipe:
  • Pretrained EfficientNet backbone (timm) — better visual features
  • Cosine LR schedule + warmup — faster convergence
  • Mixed precision (AMP) — 2-3x speedup on GPU
  • Gradient clipping — prevents instability
  • Label smoothing — better generalisation
  • EMA model — smoother final weights
  • Curriculum domain randomisation — progressive sim2real gap closing
  • Multi-robot balanced sampling — prevents robot-specific overfitting

Supported datasets:
  "synthetic"  — always works, no download (default)
  "bridgev2"   — BridgeData V2 (60k real demos, WidowX robot)
  "openx"      — Open X-Embodiment (500k demos, 22 robot types)
  "mixed"      — combine any of the above

Open-source papers leveraged:
  RT-2    : https://arxiv.org/abs/2307.15818
  OpenVLA : https://arxiv.org/abs/2406.09246
  Octo    : https://arxiv.org/abs/2405.12213
  MAML    : https://arxiv.org/abs/1703.03400
  TENT    : https://arxiv.org/abs/2006.10726

Usage
─────
  # Fastest (CPU, synthetic data)
  python train_universal.py --epochs 5 --batch 8

  # Best quality (GPU, pretrained backbone)
  python train_universal.py --epochs 50 --batch 32 --pretrained --amp

  # With real data (Open X-Embodiment)
  python train_universal.py --dataset openx --data-dir ./data/openx/ \\
      --pretrained --amp --epochs 20

  # Fine-tune on a specific robot
  python train_universal.py --robot ur5 --resume checkpoints/universal_vla.pt \\
      --epochs 10 --lr 1e-4

  # Multi-robot training (all presets)
  python train_universal.py --robots ur5 kuka_iiwa7 franka_panda simple_2dof
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False
    print("ERROR: torch not installed. Run: pip install torch")
    sys.exit(1)


# ─────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal VLA Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    parser.add_argument("--hidden-dim",   type=int,   default=256)
    parser.add_argument("--num-bins",     type=int,   default=128)
    parser.add_argument("--max-dof",      type=int,   default=32)
    parser.add_argument("--num-heads",    type=int,   default=4)
    parser.add_argument("--num-layers",   type=int,   default=4)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--pretrained",   action="store_true",
                        help="Use timm pretrained backbone (EfficientNet-B0)")
    parser.add_argument("--backbone",     default="efficientnet_b0",
                        help="timm backbone name")
    parser.add_argument("--no-temporal",  action="store_true",
                        help="Single-frame mode (faster inference)")
    parser.add_argument("--use-flow",     action="store_true",
                        help="Enable optical flow stream (slower, marginally better)")

    # Dataset
    parser.add_argument("--dataset",      default="synthetic",
                        choices=["synthetic", "bridgev2", "openx", "mixed"],
                        help="Training dataset")
    parser.add_argument("--data-dir",     default=None,
                        help="Path to dataset (needed for bridgev2/openx)")
    parser.add_argument("--n-episodes",   type=int,   default=2000)
    parser.add_argument("--clip-len",     type=int,   default=8)
    parser.add_argument("--img-size",     type=int,   default=224)
    parser.add_argument("--robot",        default=None,
                        help="Train on single robot (default: all presets)")
    parser.add_argument("--robots",       nargs="+",
                        default=["simple_2dof", "kuka_iiwa7", "ur5", "franka_panda"])
    parser.add_argument("--no-domain-rand", action="store_true",
                        help="Disable domain randomisation")

    # Optimiser
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.05)
    parser.add_argument("--ema-decay",    type=float, default=0.999)

    # Schedule
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch",        type=int,   default=16)
    parser.add_argument("--warmup-steps", type=int,   default=500)
    parser.add_argument("--no-cosine",    action="store_true",
                        help="Use constant LR instead of cosine annealing")

    # Mixed precision
    parser.add_argument("--amp",          action="store_true",
                        help="Use automatic mixed precision (GPU only)")

    # Curriculum
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum domain randomisation")

    # Checkpointing
    parser.add_argument("--ckpt-dir",     default="checkpoints")
    parser.add_argument("--save-every",   type=int,   default=5)
    parser.add_argument("--log-every",    type=int,   default=50)
    parser.add_argument("--resume",       default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--finetune",     default=None,
                        help="Fine-tune from checkpoint (reset optimiser)")

    # Hardware
    parser.add_argument("--workers",      type=int,   default=2)
    parser.add_argument("--device",       default="auto")

    # Testing
    parser.add_argument("--smoke-test",   action="store_true",
                        help="Quick 2-epoch test (tiny model + data)")
    parser.add_argument("--benchmark-latency", action="store_true",
                        help="Run latency benchmark after training")
    parser.add_argument("--show-backbones", action="store_true",
                        help="List available pretrained backbones and exit")

    args = parser.parse_args()

    # ── Show backbones ──────────────────────────────────────
    if args.show_backbones:
        try:
            from models.pretrained_backbone import list_backbones
            list_backbones()
        except Exception as e:
            print(f"Could not list backbones: {e}")
        return

    # ── Smoke test override ─────────────────────────────────
    if args.smoke_test:
        print("\n=== SMOKE TEST MODE ===")
        args.epochs      = 2
        args.batch       = 4
        args.n_episodes  = 20
        args.hidden_dim  = 64
        args.num_bins    = 32
        args.num_heads   = 2
        args.num_layers  = 1
        args.clip_len    = 4
        args.img_size    = 64
        args.pretrained  = False
        args.log_every   = 5
        args.save_every  = 2
        args.warmup_steps = 5
        args.workers     = 0

    # ── Build training config ───────────────────────────────
    from training.training_recipe import TrainingConfig, Trainer

    robots = [args.robot] if args.robot else args.robots

    cfg = TrainingConfig(
        # Architecture
        hidden_dim   = args.hidden_dim,
        num_bins     = args.num_bins,
        max_dof      = args.max_dof,
        num_heads    = args.num_heads,
        num_layers   = args.num_layers,
        dropout      = args.dropout,
        use_flow     = args.use_flow,
        use_temporal = not args.no_temporal,
        use_bert     = False,
        pretrained   = args.pretrained,
        backbone     = args.backbone,

        # Dataset
        dataset_type = args.dataset,
        data_dir     = args.data_dir,
        n_episodes   = args.n_episodes,
        clip_len     = args.clip_len,
        img_size     = args.img_size,
        robots       = robots,
        domain_rand  = not args.no_domain_rand,

        # Optimiser
        lr           = args.lr,
        weight_decay = args.weight_decay,
        grad_clip    = args.grad_clip,
        label_smooth = args.label_smooth,
        ema_decay    = args.ema_decay,

        # Schedule
        epochs       = args.epochs,
        batch_size   = args.batch,
        warmup_steps = args.warmup_steps,
        cosine_schedule = not args.no_cosine,

        # Misc
        amp          = args.amp,
        curriculum   = not args.no_curriculum,
        ckpt_dir     = args.ckpt_dir,
        save_every   = args.save_every,
        log_every    = args.log_every,
        num_workers  = args.workers,
        device       = args.device,
    )

    # ── Build trainer ───────────────────────────────────────
    trainer = Trainer(cfg)

    # ── Load checkpoint (resume or fine-tune) ───────────────
    model = None
    resume_path = args.resume or args.finetune
    if resume_path and Path(resume_path).exists():
        from models.universal_vla import UniversalVLAModel, load_universal_checkpoint
        from training.training_recipe import build_model
        model = build_model(cfg)
        ep, met = load_universal_checkpoint(model, resume_path)
        print(f"Loaded checkpoint: {resume_path}  (epoch={ep}  "
              f"loss={met.get('loss/total', '?')})")
        if args.finetune:
            print("Fine-tuning mode: resetting optimiser.")

    # ── Train ───────────────────────────────────────────────
    model = trainer.train(model=model)

    # ── Latency benchmark ───────────────────────────────────
    if args.benchmark_latency:
        print("\nRunning latency benchmark...")
        from models.latency_optimizer import LatencyOptimizer
        device = cfg.device

        opt = LatencyOptimizer(model, device=device)
        print(f"Warmup: {opt.warmup():.1f} ms")

        stats = opt.benchmark(n_dof=7, clip_len=cfg.clip_len, n_runs=10)
        print(f"FP32 baseline: {stats['mean_ms']:.1f} ms  "
              f"({stats['throughput_fps']:.1f} Hz)")

        opt.quantize_dynamic()
        stats2 = opt.benchmark(n_dof=7, clip_len=cfg.clip_len, n_runs=10)
        print(f"INT8 quantised: {stats2['mean_ms']:.1f} ms  "
              f"({stats2['throughput_fps']:.1f} Hz)")
        print(f"Speedup: {stats['mean_ms']/max(stats2['mean_ms'],0.1):.2f}x")

    print("\nDone. Checkpoints saved to:", cfg.ckpt_dir)


if __name__ == "__main__":
    main()

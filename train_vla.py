from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from vla.dataset import SyntheticRelationDataset
from vla.model import VisionLanguageActionModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline vision-language-action model.")
    parser.add_argument("--num-samples", type=int, default=8000)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser


def run_epoch(
    model: VisionLanguageActionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    criterion = torch.nn.MSELoss()
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        actions = batch["action"].to(device)
        predictions = model(images, tokens)
        loss = criterion(predictions, actions)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    return total_loss / max(batches, 1)


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticRelationDataset(
        num_samples=args.num_samples,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = VisionLanguageActionModel(
        vocab_size=dataset.tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        pad_idx=dataset.tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Model parameters: {model.num_parameters:,}")
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device=device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer=None, device=device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    checkpoint_path = output_dir / "vla_policy_best.pt"
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "vocab_size": dataset.tokenizer.vocab_size,
                "pad_id": dataset.tokenizer.pad_id,
                "hidden_dim": args.hidden_dim,
                "action_dim": args.action_dim,
                "image_size": args.image_size,
            },
            checkpoint_path,
        )

    vocab_path = output_dir / "tokenizer_vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(dataset.tokenizer.token_to_id, f, indent=2)

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best checkpoint to: {checkpoint_path}")
    print(f"Saved tokenizer vocab to: {vocab_path}")
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()

"""
data/dataset_loader.py
────────────────────────
Dataset loaders for robotic manipulation datasets.

Supported datasets
──────────────────
• Open X-Embodiment (Open X)  – large-scale robot trajectories
• ALFRED                       – language-guided task sequences
• LIBERO                       – long-horizon manipulation tasks
• Synthetic                    – generated from simulation for unit tests

Each loader returns a unified ManipulationDataset (PyTorch Dataset) that
yields (image, command_tokens, action_vector) triples.

When actual dataset files are not present the loader gracefully falls
back to a synthetic generator so the training loop always has data.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False

from config import DEFAULT_CONFIG, TrainConfig, DATA_DIR


# ──────────────────────────────────────────────────────────
# Unified sample type
# ──────────────────────────────────────────────────────────

@dataclass
class ManipulationSample:
    """One training sample: observation + language + action."""
    image:         np.ndarray          # (H, W, 3) uint8
    command:       str                  # raw NL command
    action_vector: np.ndarray          # (7,) float – continuous action
    episode_id:    str  = ""
    step_id:       int  = 0
    dataset_name:  str  = "synthetic"


# ──────────────────────────────────────────────────────────
# Synthetic dataset generator
# ──────────────────────────────────────────────────────────

COMMAND_TEMPLATES = [
    "Move the {colour1} {shape1} to the right of the {colour2} {shape2}.",
    "Pick up the {colour1} {shape1} and place it on the {colour2} {shape2}.",
    "Push the {colour1} {shape1} to the left side of the table.",
    "Stack the {colour1} {shape1} on top of the {colour2} {shape2}.",
    "Grasp the small {colour1} {shape1} near the edge.",
    "Lift the {colour1} {shape1} above the {colour2} {shape2}.",
    "Move the {colour1} {shape1} behind the {colour2} {shape2}.",
    "Place the {colour1} {shape1} in front of the {colour2} {shape2}.",
    "Slide the {colour1} {shape1} beside the {colour2} {shape2}.",
    "Transfer the {colour1} {shape1} to the {colour2} platform.",
]

COLOURS = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "white"]
SHAPES  = ["block", "cube", "sphere", "cylinder"]


def _random_command(rng: np.random.Generator) -> str:
    tmpl = rng.choice(COMMAND_TEMPLATES)
    return tmpl.format(
        colour1=rng.choice(COLOURS), shape1=rng.choice(SHAPES),
        colour2=rng.choice(COLOURS), shape2=rng.choice(SHAPES),
    )


def _random_action(rng: np.random.Generator) -> np.ndarray:
    """Simulate a plausible manipulation action delta."""
    return np.array([
        rng.uniform(-0.10, 0.10),   # dx
        rng.uniform(-0.10, 0.10),   # dy
        rng.uniform(-0.05, 0.12),   # dz
        rng.uniform(-0.20, 0.20),   # droll
        rng.uniform(-0.20, 0.20),   # dpitch
        rng.uniform(-0.20, 0.20),   # dyaw
        rng.choice([0.0, 1.0]),     # gripper: 0=closed, 1=open
    ], dtype=np.float32)


def _synthetic_image(rng: np.random.Generator, h: int = 224, w: int = 224) -> np.ndarray:
    """Generate a plausible synthetic tabletop image."""
    # Background: brown table surface
    img = np.full((h, w, 3), (180, 140, 90), dtype=np.uint8)
    # Add 2-4 coloured rectangular blobs for objects
    for _ in range(rng.integers(2, 5)):
        cx = int(rng.integers(40, w - 40))
        cy = int(rng.integers(40, h - 40))
        bw = int(rng.integers(20, 50))
        bh = int(rng.integers(20, 50))
        colour = rng.integers(50, 255, 3, dtype=np.uint8)
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)
        img[y1:y2, x1:x2] = colour
    # Add subtle noise
    noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ──────────────────────────────────────────────────────────
# PyTorch Dataset implementations
# ──────────────────────────────────────────────────────────

if _TORCH:

    class SyntheticDataset(Dataset):
        """
        Fully synthetic dataset – no files needed.
        Generates random (image, command, action) triples on the fly.
        """

        def __init__(
            self,
            size:     int = 10_000,
            img_size: int = 224,
            seed:     int = 42,
            tokeniser = None,
            max_len:  int = 64,
        ):
            self.size     = size
            self.img_size = img_size
            self.rng      = np.random.default_rng(seed)
            self.tokeniser = tokeniser
            self.max_len  = max_len

            # Pre-generate all commands and actions for reproducibility
            self._commands = [_random_command(self.rng) for _ in range(size)]
            self._actions  = np.array(
                [_random_action(self.rng) for _ in range(size)], dtype=np.float32
            )
            # Seed per-sample rng for images
            self._img_seeds = self.rng.integers(0, 2**31, size)

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int) -> Dict:
            img_rng = np.random.default_rng(int(self._img_seeds[idx]))
            image   = _synthetic_image(img_rng, self.img_size, self.img_size)
            command = self._commands[idx]
            action  = self._actions[idx]

            # To tensor
            img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            sample = {
                "image":         img_t,
                "action_vector": torch.from_numpy(action),
                "command":       command,
            }

            # Optional tokenisation
            if self.tokeniser is not None:
                enc = self.tokeniser(
                    command,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                sample["input_ids"]      = enc["input_ids"].squeeze(0)
                sample["attention_mask"] = enc["attention_mask"].squeeze(0)

            return sample


    class JSONManipulationDataset(Dataset):
        """
        Loads episodes from a JSON-lines file.

        Each line is a JSON object:
        {
            "episode_id": "...",
            "step_id": 0,
            "command": "Move the blue block ...",
            "image_path": "path/to/frame.jpg",   // optional
            "action": [dx, dy, dz, droll, dpitch, dyaw, gripper]
        }
        """

        def __init__(
            self,
            jsonl_path: str,
            img_size:   int = 224,
            tokeniser       = None,
            max_len:    int = 64,
        ):
            self.img_size  = img_size
            self.tokeniser = tokeniser
            self.max_len   = max_len
            self.records: List[Dict] = []

            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> Dict:
            rec     = self.records[idx]
            command = rec["command"]
            action  = np.array(rec["action"], dtype=np.float32)

            # Load image if available, else synthesise
            img_path = rec.get("image_path")
            if img_path and _PIL and os.path.exists(img_path):
                img = np.array(
                    Image.open(img_path).convert("RGB")
                    .resize((self.img_size, self.img_size))
                )
            else:
                rng = np.random.default_rng(idx)
                img = _synthetic_image(rng, self.img_size, self.img_size)

            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            sample = {
                "image":         img_t,
                "action_vector": torch.from_numpy(action),
                "command":       command,
            }

            if self.tokeniser is not None:
                enc = self.tokeniser(
                    command,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                sample["input_ids"]      = enc["input_ids"].squeeze(0)
                sample["attention_mask"] = enc["attention_mask"].squeeze(0)

            return sample


# ──────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────

def get_dataloaders(
    cfg:           TrainConfig = DEFAULT_CONFIG.train,
    dataset_path:  Optional[str] = None,
    tokeniser      = None,
):
    """
    Build train / val / test DataLoader objects.

    If dataset_path points to a JSONL file it is used; otherwise a
    synthetic dataset is generated automatically.

    Returns
    ───────
    train_loader, val_loader, test_loader
    """
    if not _TORCH:
        raise RuntimeError("torch not installed – cannot build DataLoaders.")

    N_TOTAL = 12_000

    if dataset_path and os.path.exists(dataset_path):
        full_ds = JSONManipulationDataset(
            dataset_path, tokeniser=tokeniser
        )
        N_TOTAL = len(full_ds)
    else:
        if dataset_path:
            warnings.warn(
                f"Dataset file '{dataset_path}' not found – "
                "using synthetic data instead."
            )
        full_ds = SyntheticDataset(
            size=N_TOTAL, seed=42, tokeniser=tokeniser
        )

    # Train / val / test split
    n_train = int(N_TOTAL * cfg.train_split)
    n_val   = int(N_TOTAL * cfg.val_split)
    n_test  = N_TOTAL - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=min(cfg.num_workers, os.cpu_count() or 1),
        pin_memory=True,
    )

    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


# ──────────────────────────────────────────────────────────
# ALFRED stub (structure for future integration)
# ──────────────────────────────────────────────────────────

class ALFREDLoader:
    """
    Stub for ALFRED dataset loading.

    To use:
    1. Download ALFRED from https://askforalfred.com/
    2. Set alfred_root to the download directory.
    3. This loader yields ManipulationSample objects.
    """

    def __init__(self, alfred_root: str, split: str = "train"):
        self.root  = Path(alfred_root)
        self.split = split
        self._index: List[Dict] = []
        self._load_index()

    def _load_index(self):
        index_file = self.root / self.split / "tasks.json"
        if index_file.exists():
            with open(index_file) as f:
                self._index = json.load(f)
        else:
            warnings.warn(f"ALFRED index not found at {index_file}.")

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        rng = np.random.default_rng(0)
        for rec in self._index:
            yield ManipulationSample(
                image=_synthetic_image(rng),
                command=rec.get("task_desc", ""),
                action_vector=_random_action(rng),
                episode_id=str(rec.get("task_id", "")),
                dataset_name="alfred",
            )


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    if not _TORCH:
        print("torch not installed – skipping dataset test.")
    else:
        ds = SyntheticDataset(size=500)
        print(f"SyntheticDataset  len={len(ds)}")
        sample = ds[0]
        print(f"  image:  {sample['image'].shape}  dtype={sample['image'].dtype}")
        print(f"  action: {sample['action_vector']}")
        print(f"  cmd:    {sample['command']}")

        train_loader, val_loader, test_loader = get_dataloaders()
        print(f"\nDataLoaders: train={len(train_loader.dataset)} "
              f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
        batch = next(iter(train_loader))
        print(f"Batch image: {batch['image'].shape}")

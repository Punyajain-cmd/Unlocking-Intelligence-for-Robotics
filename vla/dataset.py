from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


COLORS = ("red", "green", "blue", "yellow")
RELATION_TEXT = {
    "right_of": "to the right of",
    "left_of": "to the left of",
    "in_front_of": "in front of",
    "behind": "behind",
    "next_to": "next to",
    "on_top_of": "on top of",
}
RELATION_OFFSETS = {
    "right_of": np.array([0.22, 0.0], dtype=np.float32),
    "left_of": np.array([-0.22, 0.0], dtype=np.float32),
    "in_front_of": np.array([0.0, 0.22], dtype=np.float32),
    "behind": np.array([0.0, -0.22], dtype=np.float32),
    "next_to": np.array([0.16, 0.16], dtype=np.float32),
    "on_top_of": np.array([0.0, 0.0], dtype=np.float32),
}
COLOR_RGB = {
    "red": (230, 60, 60),
    "green": (60, 220, 80),
    "blue": (70, 100, 230),
    "yellow": (230, 220, 60),
}


@dataclass(frozen=True)
class SyntheticSample:
    command: str
    positions: dict[str, np.ndarray]
    action: np.ndarray


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.strip().lower())


class WordTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, token_to_id: dict[str, int], max_length: int = 20) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.max_length = max_length
        self.pad_id = token_to_id[self.PAD_TOKEN]
        self.unk_id = token_to_id[self.UNK_TOKEN]

    @classmethod
    def from_corpus(cls, corpus: list[str], max_length: int = 20) -> "WordTokenizer":
        counts: Counter[str] = Counter()
        for text in corpus:
            counts.update(_normalize_text(text).split())
        token_to_id = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
        }
        for token, _ in counts.most_common():
            if token in token_to_id:
                continue
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id, max_length=max_length)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> np.ndarray:
        tokens = _normalize_text(text).split()
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))
        else:
            ids = ids[: self.max_length]
        return np.asarray(ids, dtype=np.int64)


class SyntheticRelationDataset(Dataset):
    """Synthetic dataset for training a baseline VLA action regressor."""

    def __init__(
        self,
        num_samples: int = 8000,
        image_size: int = 96,
        max_seq_len: int = 20,
        seed: int = 7,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.samples = self._generate_samples(num_samples, seed=seed)
        self.tokenizer = WordTokenizer.from_corpus(
            [sample.command for sample in self.samples],
            max_length=max_seq_len,
        )

    @staticmethod
    def _sample_scene_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        for color in COLORS:
            for _ in range(100):
                candidate = rng.uniform(-0.8, 0.8, size=(2,)).astype(np.float32)
                if not positions:
                    positions[color] = candidate
                    break
                min_dist = min(np.linalg.norm(candidate - p) for p in positions.values())
                if min_dist >= 0.30:
                    positions[color] = candidate
                    break
            if color not in positions:
                positions[color] = candidate
        return positions

    @staticmethod
    def _world_to_pixel(value_xy: np.ndarray, image_size: int) -> tuple[int, int]:
        x = int((value_xy[0] + 1.0) * 0.5 * (image_size - 1))
        y = int((1.0 - (value_xy[1] + 1.0) * 0.5) * (image_size - 1))
        return x, y

    def _render_scene(self, positions: dict[str, np.ndarray]) -> np.ndarray:
        image = np.full((self.image_size, self.image_size, 3), fill_value=24, dtype=np.uint8)
        block_size = max(int(self.image_size * 0.07), 4)
        for color, pos in positions.items():
            px, py = self._world_to_pixel(pos, self.image_size)
            x1 = max(px - block_size, 0)
            y1 = max(py - block_size, 0)
            x2 = min(px + block_size, self.image_size - 1)
            y2 = min(py + block_size, self.image_size - 1)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_RGB[color], thickness=-1)
        return image

    @staticmethod
    def _compose_command(src_color: str, tgt_color: str, relation: str) -> str:
        relation_text = RELATION_TEXT[relation]
        return f"move the {src_color} block {relation_text} the {tgt_color} cube"

    def _generate_samples(self, num_samples: int, seed: int) -> list[SyntheticSample]:
        samples: list[SyntheticSample] = []
        for idx in range(num_samples):
            rng = np.random.default_rng(seed + idx)
            relation = rng.choice(tuple(RELATION_TEXT.keys()))
            src_color, tgt_color = rng.choice(COLORS, size=2, replace=False)
            positions = self._sample_scene_positions(rng)
            target_xy = positions[tgt_color] + RELATION_OFFSETS[relation]
            target_xy = np.clip(target_xy, -0.9, 0.9)
            action = np.array(
                [
                    target_xy[0],
                    target_xy[1],
                    1.0 if relation == "on_top_of" else 0.0,
                ],
                dtype=np.float32,
            )
            command = self._compose_command(src_color, tgt_color, relation)
            samples.append(
                SyntheticSample(
                    command=command,
                    positions={color: pos.copy() for color, pos in positions.items()},
                    action=action,
                )
            )
        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = self._render_scene(sample.positions)
        tokens = self.tokenizer.encode(sample.command)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        token_tensor = torch.from_numpy(tokens).long()
        action_tensor = torch.from_numpy(sample.action).float()

        return {
            "image": image_tensor,
            "tokens": token_tensor,
            "action": action_tensor,
        }

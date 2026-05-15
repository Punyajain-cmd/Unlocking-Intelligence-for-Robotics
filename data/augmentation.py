"""
data/augmentation.py
─────────────────────
Data augmentation strategies for the visual and language modalities.

Visual augmentations  – applied to training images only:
  • Random colour jitter (brightness, contrast, saturation, hue)
  • Random horizontal flip (with mirrored spatial relation labels)
  • Random crop + resize
  • Gaussian blur
  • Random erasing (simulate occlusion)

Language augmentations – applied to commands:
  • Synonym replacement for colours / shapes / actions
  • Random word dropout (mild)
  • Paraphrase templates (rule-based)
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    _TORCH = True
except ImportError:
    _TORCH = False


# ──────────────────────────────────────────────────────────
# Visual augmentation pipeline
# ──────────────────────────────────────────────────────────

if _TORCH:

    class VisualAugmentor:
        """
        Applies a stochastic augmentation pipeline to training images.

        Parameters
        ──────────
        img_size    : output spatial size (square)
        flip_prob   : probability of horizontal flip
        jitter_prob : probability of colour jitter
        blur_prob   : probability of Gaussian blur
        erase_prob  : probability of random erasing
        """

        def __init__(
            self,
            img_size:    int   = 224,
            flip_prob:   float = 0.5,
            jitter_prob: float = 0.8,
            blur_prob:   float = 0.2,
            erase_prob:  float = 0.3,
        ):
            self.img_size    = img_size
            self.flip_prob   = flip_prob

            self.jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            )
            self.blur   = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            self.eraser = T.RandomErasing(
                p=erase_prob, scale=(0.02, 0.15), value="random"
            )
            self.jitter_prob = jitter_prob
            self.blur_prob   = blur_prob

            self.base_transforms = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                )
            ])

        def __call__(
            self,
            image:    "torch.Tensor",              # (3, H, W) float [0,1]
            flipped:  Optional[bool] = None,       # if None, sample randomly
        ) -> Tuple["torch.Tensor", bool]:
            """
            Returns (augmented_image, was_flipped).
            `was_flipped` is needed to mirror spatial relation labels.
            """
            # Colour jitter
            if random.random() < self.jitter_prob:
                image = self.jitter(image)

            # Horizontal flip
            do_flip = random.random() < self.flip_prob if flipped is None else flipped
            if do_flip:
                image = TF.hflip(image)

            # Gaussian blur
            if random.random() < self.blur_prob:
                image = self.blur(image)

            # Base: crop + resize + normalise
            image = self.base_transforms(image)

            # Random erasing (after normalisation)
            image = self.eraser(image)

            return image, do_flip

    class ValTransform:
        """Deterministic transform for validation / test."""

        def __init__(self, img_size: int = 224):
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                )
            ])

        def __call__(self, image: "torch.Tensor") -> "torch.Tensor":
            return self.transform(image)


# ──────────────────────────────────────────────────────────
# Language augmentation
# ──────────────────────────────────────────────────────────

# Synonym tables
_COLOUR_SYNONYMS = {
    "red":    ["crimson", "scarlet", "red"],
    "blue":   ["azure", "cobalt", "blue"],
    "green":  ["emerald", "lime", "green"],
    "yellow": ["golden", "amber", "yellow"],
    "orange": ["tangerine", "orange"],
    "purple": ["violet", "indigo", "purple"],
    "cyan":   ["teal", "aqua", "cyan"],
    "white":  ["ivory", "white"],
    "black":  ["dark", "black"],
    "grey":   ["gray", "silver", "grey"],
}

_SHAPE_SYNONYMS = {
    "block":    ["block", "brick", "object"],
    "cube":     ["cube", "box", "block"],
    "sphere":   ["sphere", "ball", "orb"],
    "cylinder": ["cylinder", "tube", "rod"],
    "platform": ["platform", "tray", "base"],
}

_ACTION_SYNONYMS = {
    "move":  ["move", "transfer", "carry", "bring"],
    "pick":  ["pick up", "grab", "grasp", "take"],
    "place": ["place", "put down", "set", "deposit"],
    "push":  ["push", "slide", "nudge"],
    "stack": ["stack", "place on top of", "put on"],
    "lift":  ["lift", "raise", "elevate"],
}

_RELATION_SYNONYMS = {
    "to the right of":  ["to the right of", "right of", "on the right side of"],
    "to the left of":   ["to the left of",  "left of",  "on the left side of"],
    "on top of":        ["on top of", "on", "above", "stacked on"],
    "in front of":      ["in front of", "before", "ahead of"],
    "behind":           ["behind", "in back of"],
    "beside":           ["beside", "next to", "alongside"],
    "near":             ["near", "close to", "by"],
}

# Spatial flip: when image is horizontally flipped, left↔right
FLIP_RELATION_MAP = {
    "right_of": "left_of",
    "left_of":  "right_of",
    "right":    "left",
    "left":     "right",
}


class LanguageAugmentor:
    """
    Rule-based language augmentation for manipulation commands.

    Usage
    ─────
    >>> aug = LanguageAugmentor()
    >>> aug.augment("Move the blue block to the right of the green cube.")
    'Transfer the cobalt brick to the right side of the emerald box.'
    """

    def __init__(self, synonym_prob: float = 0.4, dropout_prob: float = 0.1):
        self.synonym_prob = synonym_prob
        self.dropout_prob = dropout_prob

    def augment(self, command: str) -> str:
        tokens = command.split()
        result = []
        i = 0
        while i < len(tokens):
            tok = tokens[i].lower().rstrip(".,!?")
            replaced = False

            # Try multi-word lookups first (e.g. "to the right of")
            for length in (4, 3, 2):
                phrase = " ".join(t.lower().rstrip(".,!?") for t in tokens[i:i+length])
                for canonical, synonyms in {**_RELATION_SYNONYMS, **_ACTION_SYNONYMS}.items():
                    if phrase in synonyms and random.random() < self.synonym_prob:
                        result.append(random.choice(synonyms))
                        i += length
                        replaced = True
                        break
                if replaced:
                    break

            if replaced:
                continue

            # Single-word lookups
            for synonyms in [_COLOUR_SYNONYMS, _SHAPE_SYNONYMS]:
                for canonical, syns in synonyms.items():
                    if tok in syns and random.random() < self.synonym_prob:
                        result.append(random.choice(syns))
                        replaced = True
                        break
                if replaced:
                    break

            if not replaced:
                # Mild word dropout (skip stop words randomly)
                if tok in {"the", "a", "an"} and random.random() < self.dropout_prob:
                    i += 1
                    continue
                result.append(tokens[i])

            i += 1

        return " ".join(result)

    def mirror_relation(self, command: str) -> str:
        """
        Swap left/right in a command when the image has been h-flipped.
        """
        result = command
        for original, flipped in [
            ("to the right of", "to the left of"),
            ("to the left of",  "to the right of"),
            ("right side",      "left side"),
            ("left side",       "right side"),
        ]:
            result = result.replace(original, "__TEMP__")
            if "__TEMP__" in result:
                result = result.replace("__TEMP__", flipped)
                break
        return result


# ──────────────────────────────────────────────────────────
# Combined augmentation wrapper
# ──────────────────────────────────────────────────────────

class ManipulationAugmentor:
    """
    Applies both visual and language augmentations in a coordinated way:
    if the image is flipped, the command's spatial relations are mirrored.
    """

    def __init__(
        self,
        img_size:     int   = 224,
        lang_aug_prob: float = 0.5,
    ):
        self.lang_aug_prob = lang_aug_prob
        self.lang_aug = LanguageAugmentor()
        if _TORCH:
            self.vis_aug = VisualAugmentor(img_size=img_size)
            self.val_tfm = ValTransform(img_size=img_size)

    def augment_train(
        self,
        image:   "torch.Tensor",
        command: str,
    ):
        """Returns (augmented_image, augmented_command)."""
        aug_img, flipped = self.vis_aug(image)
        aug_cmd = command
        if flipped:
            aug_cmd = self.lang_aug.mirror_relation(aug_cmd)
        if random.random() < self.lang_aug_prob:
            aug_cmd = self.lang_aug.augment(aug_cmd)
        return aug_img, aug_cmd

    def transform_val(self, image: "torch.Tensor") -> "torch.Tensor":
        return self.val_tfm(image)


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    aug = LanguageAugmentor(synonym_prob=0.6)

    CMDS = [
        "Move the blue block to the right of the green cube.",
        "Pick up the red sphere and place it on the yellow platform.",
        "Push the cyan cylinder to the left side of the table.",
        "Stack the orange cube on top of the purple block.",
        "Grasp the small blue object near the edge.",
    ]

    print("\nLanguage Augmentation Examples")
    print("=" * 65)
    for cmd in CMDS:
        print(f"  Original : {cmd}")
        print(f"  Augmented: {aug.augment(cmd)}")
        print()

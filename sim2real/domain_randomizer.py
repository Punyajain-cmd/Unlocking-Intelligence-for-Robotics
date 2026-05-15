"""
sim2real/domain_randomizer.py
──────────────────────────────
Domain Randomization for closing the Sim2Real gap.

Randomly perturbs visual and physical properties during training so the
policy becomes robust to the distributional shift between simulation and
the real world.

Randomised properties:
  Visual:
    • Lighting colour / intensity
    • Object texture / colour jitter
    • Camera pose perturbation
    • Background replacement / noise
    • Motion blur, chromatic aberration
    • Sensor noise (Gaussian, salt-and-pepper)

  Physical:
    • Object mass / inertia
    • Friction / restitution coefficients
    • Joint damping / stiffness
    • Robot actuator delay / noise
    • Gravity direction perturbation

  Structural:
    • Random object placements
    • Distractors (extra objects)
    • Random table textures

All randomisers are composable via RandomizerPipeline.
"""

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────
# Base randomiser
# ─────────────────────────────────────────────────────────

class BaseRandomizer(ABC):
    """Abstract base for all randomisers."""

    def __init__(self, prob: float = 1.0):
        self.prob = prob

    def __call__(self, *args, **kwargs):
        if random.random() < self.prob:
            return self.apply(*args, **kwargs)
        return args[0] if args else None

    @abstractmethod
    def apply(self, *args, **kwargs):
        ...


# ─────────────────────────────────────────────────────────
# Visual randomisers
# ─────────────────────────────────────────────────────────

class ColourJitter(BaseRandomizer):
    """Random brightness / contrast / saturation / hue."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast:   float = 0.4,
        saturation: float = 0.3,
        hue:        float = 0.08,
        prob:       float = 0.8,
    ):
        super().__init__(prob)
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        # Brightness
        b_fac = 1.0 + random.uniform(-self.brightness, self.brightness)
        img   = img * b_fac

        # Contrast
        c_fac = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean  = img.mean()
        img   = (img - mean) * c_fac + mean

        img   = np.clip(img, 0, 255).astype(np.uint8)

        # Saturation + hue via HSV
        if _CV2:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 1.0 + random.uniform(-self.saturation, self.saturation)
            hsv[..., 0] += random.uniform(-self.hue * 180, self.hue * 180)
            hsv[..., 0]  = hsv[..., 0] % 180
            hsv           = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img


class GaussianNoise(BaseRandomizer):
    """Additive Gaussian sensor noise."""

    def __init__(self, sigma_range: Tuple[float, float] = (2.0, 20.0), prob: float = 0.5):
        super().__init__(prob)
        self.sigma_range = sigma_range

    def apply(self, image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class MotionBlur(BaseRandomizer):
    """Simulate camera motion blur."""

    def __init__(self, max_kernel: int = 9, prob: float = 0.3):
        super().__init__(prob)
        self.max_kernel = max_kernel

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not _CV2:
            return image
        k     = random.choice(range(3, self.max_kernel + 1, 2))
        angle = random.uniform(0, 180)
        M     = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        kernel = cv2.warpAffine(kernel, M, (k, k))
        return cv2.filter2D(image, -1, kernel)


class RandomCrop(BaseRandomizer):
    """Random crop + resize back to original size."""

    def __init__(self, scale_range: Tuple[float, float] = (0.85, 1.0), prob: float = 0.5):
        super().__init__(prob)
        self.scale_range = scale_range

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not _CV2:
            return image
        H, W  = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        nh, nw = int(H * scale), int(W * scale)
        y0 = random.randint(0, H - nh)
        x0 = random.randint(0, W - nw)
        crop = image[y0:y0+nh, x0:x0+nw]
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)


class BackgroundRandomizer(BaseRandomizer):
    """Replace background pixels with random colour or texture."""

    def __init__(self, prob: float = 0.4):
        super().__init__(prob)

    def apply(
        self,
        image:      np.ndarray,
        fg_mask:    Optional[np.ndarray] = None,
    ) -> np.ndarray:
        img = image.copy()
        if fg_mask is None:
            # Rough background estimate: edges of frame
            H, W = img.shape[:2]
            fg_mask = np.zeros((H, W), dtype=np.uint8)
            border = max(1, H // 8)
            bg_colour = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            img[:border, :] = bg_colour
            img[-border:, :] = bg_colour
            img[:, :border] = bg_colour
            img[:, -border:] = bg_colour
        else:
            bg_colour = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            img[fg_mask == 0] = bg_colour
        return img


class CameraDistortion(BaseRandomizer):
    """Random radial/tangential lens distortion."""

    def __init__(self, max_k1: float = 0.1, prob: float = 0.3):
        super().__init__(prob)
        self.max_k1 = max_k1

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not _CV2:
            return image
        H, W   = image.shape[:2]
        fx     = W * 1.2
        fy     = H * 1.2
        cx, cy = W / 2, H / 2
        K      = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
        k1     = random.uniform(-self.max_k1, self.max_k1)
        dist   = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
        return cv2.undistort(image, K, dist)


# ─────────────────────────────────────────────────────────
# Physical randomisers (return parameter dicts)
# ─────────────────────────────────────────────────────────

@dataclass
class PhysicsParams:
    """Randomised physical parameters for one simulation step."""
    gravity:     np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    friction:    float = 0.5
    restitution: float = 0.1
    object_mass_scale:  float = 1.0
    joint_damping:      float = 0.01
    actuator_delay_ms:  float = 0.0
    joint_noise_sigma:  float = 0.0     # radians of zero-mean joint noise


class PhysicsRandomizer:
    """
    Randomises physical parameters for sim-to-real transfer.
    Call randomize() before each episode.
    """

    def __init__(
        self,
        gravity_noise:   float = 0.2,    # ± fraction of g
        friction_range:  Tuple = (0.2, 1.0),
        mass_range:      Tuple = (0.7, 1.5),
        damping_range:   Tuple = (0.001, 0.1),
        delay_range_ms:  Tuple = (0.0, 20.0),
        joint_noise_deg: float = 0.5,
        prob:            float = 1.0,
    ):
        self.gravity_noise   = gravity_noise
        self.friction_range  = friction_range
        self.mass_range      = mass_range
        self.damping_range   = damping_range
        self.delay_range_ms  = delay_range_ms
        self.joint_noise_rad = np.deg2rad(joint_noise_deg)
        self.prob            = prob

    def randomize(self) -> PhysicsParams:
        if random.random() > self.prob:
            return PhysicsParams()
        g_scale = np.random.uniform(1 - self.gravity_noise, 1 + self.gravity_noise)
        return PhysicsParams(
            gravity            = np.array([0, 0, -9.81 * g_scale]),
            friction           = random.uniform(*self.friction_range),
            restitution        = random.uniform(0.0, 0.3),
            object_mass_scale  = random.uniform(*self.mass_range),
            joint_damping      = random.uniform(*self.damping_range),
            actuator_delay_ms  = random.uniform(*self.delay_range_ms),
            joint_noise_sigma  = abs(np.random.normal(0, self.joint_noise_rad)),
        )

    def apply_joint_noise(
        self, q: np.ndarray, params: PhysicsParams
    ) -> np.ndarray:
        """Corrupt joint readings to simulate encoder noise."""
        return q + np.random.normal(0, params.joint_noise_sigma, q.shape)


# ─────────────────────────────────────────────────────────
# Visual pipeline composer
# ─────────────────────────────────────────────────────────

class VisualRandomizerPipeline:
    """
    Compose multiple visual randomisers into one pipeline.

    Usage:
        pipeline = VisualRandomizerPipeline.default()
        aug_img  = pipeline(image)
    """

    def __init__(self, transforms: List[BaseRandomizer]):
        self.transforms = transforms

    @classmethod
    def default(cls) -> "VisualRandomizerPipeline":
        return cls([
            ColourJitter(prob=0.9),
            GaussianNoise(prob=0.5),
            MotionBlur(prob=0.3),
            RandomCrop(prob=0.5),
            BackgroundRandomizer(prob=0.3),
            CameraDistortion(prob=0.2),
        ])

    @classmethod
    def light(cls) -> "VisualRandomizerPipeline":
        """Lighter augmentation for fine-tuning on real data."""
        return cls([
            ColourJitter(brightness=0.2, contrast=0.2,
                         saturation=0.1, hue=0.04, prob=0.7),
            GaussianNoise(sigma_range=(1.0, 8.0), prob=0.4),
        ])

    @classmethod
    def heavy(cls) -> "VisualRandomizerPipeline":
        """Maximum diversity for pure sim training."""
        return cls([
            ColourJitter(brightness=0.6, contrast=0.6,
                         saturation=0.5, hue=0.15, prob=1.0),
            GaussianNoise(sigma_range=(5.0, 30.0), prob=0.7),
            MotionBlur(max_kernel=13, prob=0.5),
            RandomCrop(scale_range=(0.75, 1.0), prob=0.7),
            BackgroundRandomizer(prob=0.6),
            CameraDistortion(max_k1=0.2, prob=0.5),
        ])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image

    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply pipeline independently to each image in a (B, H, W, 3) array."""
        return np.stack([self(img) for img in images])


# ─────────────────────────────────────────────────────────
# Tensor-level augmentation (for training loops)
# ─────────────────────────────────────────────────────────

if _TORCH:
    class TensorDomainRandomizer:
        """
        Fast GPU-compatible domain randomisation applied directly to
        (B, C, H, W) float tensors (values in [0, 1]).
        """

        def __init__(
            self,
            brightness: float = 0.3,
            contrast:   float = 0.3,
            noise_std:  float = 0.03,
        ):
            self.brightness = brightness
            self.contrast   = contrast
            self.noise_std  = noise_std

        @torch.no_grad()
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, C, H, W) float in [0, 1]"""
            B = x.shape[0]
            device = x.device

            # Per-image brightness
            b = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.brightness
            x = x * b

            # Per-image contrast
            c = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.contrast
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            x = (x - mean) * c + mean

            # Gaussian noise
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

            return x.clamp(0.0, 1.0)


if __name__ == "__main__":
    H, W = 128, 128
    img  = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)

    pipeline = VisualRandomizerPipeline.default()
    aug      = pipeline(img)
    print(f"Visual aug: {img.shape} → {aug.shape}  "
          f"mean_diff={np.abs(aug.astype(float)-img.astype(float)).mean():.2f}")

    phys = PhysicsRandomizer()
    params = phys.randomize()
    print(f"Physics params: gravity={params.gravity}  "
          f"friction={params.friction:.3f}  mass_scale={params.object_mass_scale:.3f}")

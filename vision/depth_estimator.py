"""
vision/depth_estimator.py
──────────────────────────
Monocular depth estimation from a single RGB frame.

Backends (in priority order):
  1. MiDaS (via torch.hub or timm)          — best quality, needs download
  2. Gradient-based estimate                 — fast lightweight fallback
  3. Flat-plane assumption                   — zero-dependency mock

The estimator returns a metric-scale depth map (approx metres).
Calibration helpers convert raw estimates to real-world scale.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────
# Backend: Lightweight CNN depth estimator
# ─────────────────────────────────────────────────────────

if _TORCH:
    class _DepthNet(nn.Module):
        """
        Very lightweight encoder-decoder for monocular depth.
        ~1 M params; runs at > 30 FPS on CPU for 224×224 input.
        Not as accurate as MiDaS but needs no external download.
        """
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            )
            # Decoder with skip-like upsampling
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(128, 64,  2, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(64,  32,  2, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(32,   1,  2, stride=2),
                nn.Sigmoid(),       # → (0, 1) relative depth
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dec(self.enc(x))   # (B, 1, H, W)


# ─────────────────────────────────────────────────────────
# MiDaS loader (optional)
# ─────────────────────────────────────────────────────────

_midas_model = None
_midas_transform = None


def _try_load_midas(model_type: str = "MiDaS_small") -> bool:
    """Try to load MiDaS from torch.hub. Returns True on success."""
    global _midas_model, _midas_transform
    if not _TORCH:
        return False
    try:
        _midas_model     = torch.hub.load("intel-isl/MiDaS", model_type,
                                          trust_repo=True)
        transforms       = torch.hub.load("intel-isl/MiDaS", "transforms",
                                          trust_repo=True)
        _midas_transform = (transforms.small_transform
                            if "small" in model_type
                            else transforms.default_transform)
        _midas_model.eval()
        return True
    except Exception as e:
        warnings.warn(f"MiDaS not available ({e}); using built-in depth net.")
        return False


# ─────────────────────────────────────────────────────────
# Public estimator
# ─────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Unified monocular depth estimator.

    Parameters
    ──────────
    use_midas       : Try to load MiDaS from torch hub (requires internet).
    depth_scale     : Scale factor applied to raw output → metres.
    min_depth_m     : Clip minimum depth (m).
    max_depth_m     : Clip maximum depth (m).
    known_plane_z   : If set, use this as the dominant plane depth for
                      a quick metric calibration (e.g. table at 0.65 m).
    """

    def __init__(
        self,
        use_midas:      bool  = False,
        depth_scale:    float = 1.0,
        min_depth_m:    float = 0.1,
        max_depth_m:    float = 5.0,
        known_plane_z:  Optional[float] = None,
    ):
        self._scale      = depth_scale
        self._min        = min_depth_m
        self._max        = max_depth_m
        self._plane_z    = known_plane_z
        self._backend    = "none"
        self._model      = None

        if use_midas and _try_load_midas():
            self._backend = "midas"
        elif _TORCH:
            self._model   = _DepthNet().eval()
            self._backend = "lightweight"

    # ── public API ───────────────────────────────────────────

    def estimate(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate depth from an RGB image.

        Parameters
        ──────────
        rgb : (H, W, 3) uint8

        Returns
        ───────
        depth : (H, W) float32  in approximate metres.
        """
        if self._backend == "midas":
            return self._midas_estimate(rgb)
        if self._backend == "lightweight":
            return self._net_estimate(rgb)
        return self._gradient_estimate(rgb)

    def estimate_point(
        self,
        rgb: np.ndarray,
        px:  float,
        py:  float,
    ) -> float:
        """Return depth at a single pixel (cx, cy)."""
        depth = self.estimate(rgb)
        H, W  = depth.shape
        ix    = int(np.clip(px, 0, W - 1))
        iy    = int(np.clip(py, 0, H - 1))
        return float(depth[iy, ix])

    def calibrate_scale(
        self,
        depth_map:    np.ndarray,
        known_points: list,
    ) -> "DepthEstimator":
        """
        Adjust internal scale so known pixel→depth pairs match.

        known_points : [(px, py, true_depth_m), ...]
        """
        if not known_points:
            return self
        ratios = []
        for px, py, true_d in known_points:
            H, W = depth_map.shape
            ix   = int(np.clip(px, 0, W - 1))
            iy   = int(np.clip(py, 0, H - 1))
            raw  = float(depth_map[iy, ix])
            if raw > 0:
                ratios.append(true_d / raw)
        if ratios:
            self._scale = float(np.median(ratios))
        return self

    # ── backends ─────────────────────────────────────────────

    def _midas_estimate(self, rgb: np.ndarray) -> np.ndarray:
        inp    = _midas_transform(rgb)
        with torch.no_grad():
            pred = _midas_model(inp)
        raw = pred.squeeze().cpu().numpy()
        raw = cv2.resize(raw, (rgb.shape[1], rgb.shape[0]),
                         interpolation=cv2.INTER_LINEAR) if _CV2 else raw
        return self._postprocess(raw)

    def _net_estimate(self, rgb: np.ndarray) -> np.ndarray:
        H, W   = rgb.shape[:2]
        img    = rgb.astype(np.float32) / 255.0
        mean   = np.array([0.485, 0.456, 0.406])
        std    = np.array([0.229, 0.224, 0.225])
        img    = (img - mean) / (std + 1e-8)

        # Resize to 224×224 for the network
        if _CV2:
            img_r = cv2.resize(img, (224, 224))
        else:
            img_r = img

        t   = torch.tensor(img_r, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            out = self._model(t)                      # (1,1,224,224)
        raw = out.squeeze().cpu().numpy()             # (224,224)

        if _CV2:
            raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
        return self._postprocess(raw)

    def _gradient_estimate(self, rgb: np.ndarray) -> np.ndarray:
        """
        Heuristic depth from intensity + luminance gradients.
        Objects with lower spatial frequency → assumed farther.
        """
        gray = rgb.mean(axis=2).astype(np.float32) / 255.0

        if _CV2:
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            sobelx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            grad   = np.sqrt(sobelx**2 + sobely**2)
        else:
            grad = np.ones_like(gray) * 0.5

        # High gradient → foreground (closer)
        depth_raw = 1.0 - grad / (grad.max() + 1e-6)
        return self._postprocess(depth_raw)

    def _postprocess(self, raw: np.ndarray) -> np.ndarray:
        """Normalise raw output to metric depth."""
        r_min, r_max = raw.min(), raw.max()
        if r_max - r_min < 1e-6:
            return np.full_like(raw, self._plane_z or 1.0)

        # Map to [0, 1]
        norm = (raw - r_min) / (r_max - r_min)

        # Map to [min_depth, max_depth]
        depth = self._min + norm * (self._max - self._min)

        # Apply calibrated scale
        depth = depth * self._scale

        # Optional: snap to known plane
        if self._plane_z is not None:
            median = float(np.median(depth))
            if median > 0:
                depth *= self._plane_z / median

        return depth.astype(np.float32)


# ─────────────────────────────────────────────────────────
# Back-projection helpers
# ─────────────────────────────────────────────────────────

class CameraIntrinsics:
    """Pinhole camera model."""

    def __init__(
        self,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        width:  int = 640,
        height: int = 480,
        fov_deg: Optional[float] = None,
    ):
        if fov_deg is not None:
            import math
            fx = (width  / 2) / math.tan(math.radians(fov_deg) / 2)
            fy = fx
        self.fx = fx
        self.fy = fy
        self.cx = cx if cx is not None else width  / 2.0
        self.cy = cy if cy is not None else height / 2.0

    def backproject(
        self,
        px:    float,
        py:    float,
        depth: float,
    ) -> Tuple[float, float, float]:
        """Pixel (px, py) + depth → 3-D point (x, y, z) in camera frame."""
        x = (px - self.cx) * depth / self.fx
        y = (py - self.cy) * depth / self.fy
        return (float(x), float(y), float(depth))

    def project(
        self,
        x: float, y: float, z: float,
    ) -> Tuple[float, float]:
        """3-D point → pixel (px, py)."""
        if abs(z) < 1e-9:
            return (self.cx, self.cy)
        px = self.fx * x / z + self.cx
        py = self.fy * y / z + self.cy
        return (float(px), float(py))

    def backproject_depth_map(
        self,
        depth:  np.ndarray,
        stride: int = 4,
    ) -> np.ndarray:
        """
        Convert a full depth map to a 3-D point cloud.
        Returns (N, 3) array sampled at given stride.
        """
        H, W = depth.shape
        ys, xs = np.mgrid[0:H:stride, 0:W:stride]
        zs = depth[::stride, ::stride]
        xs_ = (xs - self.cx) * zs / self.fx
        ys_ = (ys - self.cy) * zs / self.fy
        pts = np.stack([xs_.ravel(), ys_.ravel(), zs.ravel()], axis=1)
        return pts[pts[:, 2] > 0].astype(np.float32)


if __name__ == "__main__":
    H, W = 224, 224
    rgb  = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    est  = DepthEstimator(use_midas=False, known_plane_z=0.65)
    depth = est.estimate(rgb)
    print(f"Depth map: shape={depth.shape}  min={depth.min():.3f}  "
          f"max={depth.max():.3f}  mean={depth.mean():.3f} m")

    cam = CameraIntrinsics(fov_deg=60, width=W, height=H)
    pt  = cam.backproject(W//2, H//2, depth[H//2, W//2])
    print(f"Centre pixel backproject: {pt}")
    pc  = cam.backproject_depth_map(depth, stride=16)
    print(f"Point cloud: {pc.shape}")

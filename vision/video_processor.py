"""
vision/video_processor.py
──────────────────────────
Video input handler.  Converts raw video (file, webcam, byte-stream)
into temporally-ordered RGB + optional depth frames.

Key capabilities:
  • Frame extraction at configurable FPS
  • Temporal frame buffer (sliding window)
  • Optical flow computation (dense / sparse)
  • Motion magnitude maps
  • Frame preprocessing (resize, normalise, batch)
  • Generator interface for live streams
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Generator, Iterator, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    warnings.warn("opencv not installed – VideoProcessor in mock mode.")

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────

@dataclass
class VideoFrame:
    """One timestep of video data."""
    frame_idx:   int
    timestamp_s: float
    rgb:         np.ndarray                    # (H, W, 3) uint8
    depth:       Optional[np.ndarray] = None   # (H, W)   float32 metres
    flow:        Optional[np.ndarray] = None   # (H, W, 2) optical flow

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rgb.shape[:2]   # (H, W)


@dataclass
class VideoConfig:
    target_fps:    float = 10.0
    resize_hw:     Tuple[int, int] = (224, 224)
    buffer_len:    int   = 16          # temporal context window
    normalise:     bool  = True        # subtract mean, divide std
    compute_flow:  bool  = True        # compute optical flow
    flow_scale:    float = 20.0        # px→ normalised scale
    mean:          Tuple = (0.485, 0.456, 0.406)
    std:           Tuple = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────
# Core processor
# ─────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Reads a video source and produces VideoFrame objects.

    Usage
    ─────
    proc = VideoProcessor("path/to/video.mp4")
    for frame in proc.stream():
        detect(frame.rgb)

    Or batch:
        frames = proc.read_clip(start=0.0, end=5.0)  # list of VideoFrames
    """

    def __init__(
        self,
        source: Optional[str | int] = None,
        cfg:    VideoConfig = None,
    ):
        self.cfg    = cfg or VideoConfig()
        self.source = source
        self._cap   = None
        self._frame_idx  = 0
        self._prev_gray  = None
        self._buffer: Deque[VideoFrame] = deque(maxlen=self.cfg.buffer_len)

    # ── video capture lifecycle ──────────────────────────────

    def open(self, source: Optional[str | int] = None) -> "VideoProcessor":
        if not _CV2:
            return self
        src = source or self.source or 0
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {src!r}")
        return self

    def close(self):
        if self._cap and _CV2:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    # ── streaming ────────────────────────────────────────────

    def stream(self) -> Generator[VideoFrame, None, None]:
        """
        Yield VideoFrames until the source is exhausted or camera closed.
        """
        if not _CV2:
            for frame in self._mock_stream():
                yield frame
            return

        if self._cap is None:
            self.open()

        native_fps  = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip_factor = max(1, int(native_fps / self.cfg.target_fps))
        raw_idx     = 0

        while True:
            ret, bgr = self._cap.read()
            if not ret:
                break
            raw_idx += 1
            if (raw_idx - 1) % skip_factor != 0:
                continue

            ts = raw_idx / max(native_fps, 1.0)
            frame = self._process_raw(bgr, ts)
            self._buffer.append(frame)
            yield frame

    def read_clip(
        self,
        source: Optional[str] = None,
        start_s: float = 0.0,
        end_s:   float = float("inf"),
    ) -> List[VideoFrame]:
        """Read a time-bounded clip into a list."""
        frames = []
        src = source or self.source
        with VideoProcessor(src, self.cfg) as proc:
            for f in proc.stream():
                if f.timestamp_s < start_s:
                    continue
                if f.timestamp_s > end_s:
                    break
                frames.append(f)
        return frames

    def push_frame(self, rgb: np.ndarray, timestamp_s: float = 0.0,
                   depth: Optional[np.ndarray] = None) -> VideoFrame:
        """
        Manually push an RGB frame (e.g. from a ROS topic or simulator).
        Returns the processed VideoFrame and adds it to the internal buffer.
        """
        if _CV2:
            bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame = self._process_raw(bgr, timestamp_s, depth=depth)
        else:
            frame = VideoFrame(
                frame_idx   = self._frame_idx,
                timestamp_s = timestamp_s,
                rgb         = self._resize(rgb),
                depth       = depth,
            )
            self._frame_idx += 1
        self._buffer.append(frame)
        return frame

    # ── buffer / clip utilities ──────────────────────────────

    @property
    def buffer(self) -> List[VideoFrame]:
        return list(self._buffer)

    def get_clip_tensor(self) -> Optional["torch.Tensor"]:
        """
        Return the current buffer as a (T, 3, H, W) float tensor.
        Pads with the first frame if buffer is shorter than buffer_len.
        """
        if not _TORCH:
            return None
        frames = list(self._buffer)
        if not frames:
            return None
        T = self.cfg.buffer_len
        while len(frames) < T:
            frames.insert(0, frames[0])
        frames = frames[-T:]

        imgs = []
        for f in frames:
            img = f.rgb.astype(np.float32) / 255.0
            if self.cfg.normalise:
                mean = np.array(self.cfg.mean, dtype=np.float32)
                std  = np.array(self.cfg.std,  dtype=np.float32)
                img  = (img - mean) / (std + 1e-8)
            imgs.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))

        return torch.stack(imgs)   # (T, 3, H, W)

    def get_flow_tensor(self) -> Optional["torch.Tensor"]:
        """Return optical-flow stack (T-1, 2, H, W)."""
        if not _TORCH:
            return None
        frames = [f for f in self._buffer if f.flow is not None]
        if not frames:
            return None
        flows = [torch.tensor(f.flow.astype(np.float32)).permute(2, 0, 1)
                 for f in frames]
        return torch.stack(flows)

    # ── frame preprocessing ──────────────────────────────────

    def _process_raw(
        self,
        bgr:         np.ndarray,
        timestamp_s: float,
        depth:       Optional[np.ndarray] = None,
    ) -> VideoFrame:
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if _CV2 else bgr
        rgb   = self._resize(rgb)
        flow  = self._compute_flow(rgb) if self.cfg.compute_flow else None

        frame = VideoFrame(
            frame_idx   = self._frame_idx,
            timestamp_s = timestamp_s,
            rgb         = rgb,
            depth       = depth,
            flow        = flow,
        )
        self._frame_idx += 1
        return frame

    def _resize(self, img: np.ndarray) -> np.ndarray:
        H, W = self.cfg.resize_hw
        if img.shape[:2] == (H, W):
            return img
        if _CV2:
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return img

    def _compute_flow(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if not _CV2:
            return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return np.zeros((*gray.shape, 2), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        self._prev_gray = gray
        # Normalise by scale
        flow = flow / (self.cfg.flow_scale + 1e-6)
        return flow.astype(np.float32)

    # ── mock stream ─────────────────────────────────────────

    def _mock_stream(self, n: int = 10) -> Iterator[VideoFrame]:
        H, W = self.cfg.resize_hw
        for i in range(n):
            rgb = (np.random.randint(50, 200, (H, W, 3), dtype=np.uint8))
            frame = VideoFrame(
                frame_idx   = i,
                timestamp_s = i / self.cfg.target_fps,
                rgb         = rgb,
                depth       = np.ones((H, W), dtype=np.float32) * 0.8,
                flow        = np.zeros((H, W, 2), dtype=np.float32),
            )
            self._buffer.append(frame)
            yield frame


# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────

def frames_to_tensor(
    frames: List[VideoFrame],
    normalise: bool = True,
    mean: Tuple = (0.485, 0.456, 0.406),
    std:  Tuple = (0.229, 0.224, 0.225),
) -> Optional["torch.Tensor"]:
    """Convert a list of VideoFrames → (T, 3, H, W) float tensor."""
    if not _TORCH or not frames:
        return None
    imgs = []
    for f in frames:
        img = f.rgb.astype(np.float32) / 255.0
        if normalise:
            img = (img - np.array(mean, dtype=np.float32)) / (np.array(std, dtype=np.float32) + 1e-8)
        imgs.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
    return torch.stack(imgs)


def motion_magnitude(flow: np.ndarray) -> np.ndarray:
    """Compute per-pixel motion magnitude from optical flow (H,W,2)→(H,W)."""
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)


def detect_moving_regions(
    flow: np.ndarray,
    threshold: float = 0.02,
) -> np.ndarray:
    """
    Return binary mask of pixels with significant motion.
    flow : (H, W, 2), threshold in normalised flow units.
    """
    mag  = motion_magnitude(flow)
    return (mag > threshold).astype(np.uint8)


if __name__ == "__main__":
    cfg  = VideoConfig(target_fps=5, resize_hw=(112, 112), buffer_len=8)
    proc = VideoProcessor(cfg=cfg)

    print("Mock stream test:")
    for frame in proc._mock_stream(n=4):
        print(f"  Frame {frame.frame_idx}  ts={frame.timestamp_s:.2f}s  "
              f"shape={frame.rgb.shape}  flow={frame.flow is not None}")

    tensor = proc.get_clip_tensor()
    if tensor is not None:
        print(f"Clip tensor shape: {tensor.shape}")   # (T, 3, 112, 112)

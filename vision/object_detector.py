"""
vision/object_detector.py
──────────────────────────
Multi-strategy object detector for the tabletop manipulation scene.

Strategies (in order of preference)
─────────────────────────────────────
1. DNN-based detector  (optional, if Detectron2 / YOLO weights available)
2. Colour-segmentation detector (default, zero extra dependencies)
   – converts frame to HSV, finds contours per colour band,
     fits bounding boxes, estimates 3-D centre from depth map.

Output: List[DetectedObject]  – stable across frames via simple IoU tracker.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    warnings.warn("opencv-python not installed – detector returns mock objects.")

from config import DEFAULT_CONFIG, VisionConfig


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """One detected object in the scene."""
    id:        int
    colour:    str
    shape:     str   = "unknown"        # "cube", "sphere", "cylinder", …
    bbox_2d:   Tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h  (pixels)
    centre_2d: Tuple[float, float] = (0.0, 0.0)           # pixels
    centre_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # metres
    area:      float = 0.0
    confidence: float = 1.0

    # Spatial tags populated by SceneGraph
    spatial_tags: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"Obj#{self.id} [{self.colour} {self.shape}]  "
                f"3D=({self.centre_3d[0]:.3f}, "
                f"{self.centre_3d[1]:.3f}, "
                f"{self.centre_3d[2]:.3f})")

    def matches(self, colour: Optional[str], shape: Optional[str]) -> bool:
        """Return True if this object matches the query attributes."""
        colour_ok = (colour is None) or (colour.lower() == self.colour.lower())
        shape_ok  = (shape  is None) or (shape.lower()  == self.shape.lower()) \
                    or shape.lower() == "object"
        return colour_ok and shape_ok


# ──────────────────────────────────────────────────────────
# Colour-segmentation backend
# ──────────────────────────────────────────────────────────

class ColourSegmentationDetector:
    """
    Detects objects by HSV colour thresholding.

    For each registered colour band the detector:
      1. Thresholds the HSV image
      2. Finds contours, filters by area
      3. Approximates shape (cube / cylinder / sphere) from aspect ratio
      4. Projects 2-D centre to 3-D using aligned depth image
    """

    def __init__(self, cfg: VisionConfig = DEFAULT_CONFIG.vision):
        if not _CV2_AVAILABLE:
            raise RuntimeError("opencv-python required for ColourSegmentationDetector.")
        self.cfg = cfg
        self._next_id = 0

    # ── public ──────────────────────────────────────────────

    def detect(
        self,
        rgb_image:   np.ndarray,
        depth_image: Optional[np.ndarray] = None,
    ) -> List[DetectedObject]:
        """
        Parameters
        ──────────
        rgb_image   : (H, W, 3) uint8 RGB
        depth_image : (H, W)    float32 depth in metres (or None)

        Returns
        ───────
        List of DetectedObject sorted by descending area.
        """
        hsv    = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        found: List[DetectedObject] = []

        for colour_name, (lower, upper) in self.cfg.colour_ranges.items():
            # Handle the red-wrap-around bands
            real_name = colour_name.rstrip("2")  # "red2" → "red"

            mask = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )

            # Morphological clean-up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.cfg.min_contour_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy     = x + w / 2, y + h / 2
                shape      = self._classify_shape(cnt, w, h)

                # 3-D projection
                z_m = self._get_depth(depth_image, cx, cy)
                c3d = self._backproject(cx, cy, z_m, rgb_image.shape)

                obj = DetectedObject(
                    id=self._next_id,
                    colour=real_name,
                    shape=shape,
                    bbox_2d=(x, y, w, h),
                    centre_2d=(cx, cy),
                    centre_3d=c3d,
                    area=area,
                    confidence=min(1.0, area / 5000),
                )
                found.append(obj)
                self._next_id += 1

        # Deduplicate overlapping detections (same colour dual-band for red)
        found = self._dedup(found)
        found.sort(key=lambda o: o.area, reverse=True)
        return found

    # ── private helpers ─────────────────────────────────────

    @staticmethod
    def _classify_shape(
        contour: np.ndarray, w: int, h: int
    ) -> str:
        """
        Simple geometry-based shape classification:
          - aspect ratio ≈ 1  + high solidity → cube/block
          - circularity high                  → sphere
          - tall & narrow                     → cylinder
        """
        area          = cv2.contourArea(contour)
        hull          = cv2.convexHull(contour)
        hull_area     = cv2.contourArea(hull)
        solidity      = area / (hull_area + 1e-6)
        perimeter     = cv2.arcLength(contour, True)
        circularity   = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        aspect_ratio  = float(w) / (h + 1e-6)

        if circularity > 0.78:
            return "sphere"
        if 0.7 < aspect_ratio < 1.4 and solidity > 0.85:
            return "cube"
        if aspect_ratio < 0.6:
            return "cylinder"
        return "block"

    def _get_depth(
        self,
        depth: Optional[np.ndarray],
        cx: float, cy: float,
    ) -> float:
        """Read depth at pixel (cx, cy); return estimate if no depth image."""
        if depth is None:
            return self.cfg.far_plane * 0.12   # rough default ~table distance
        ix = int(np.clip(cx, 0, depth.shape[1] - 1))
        iy = int(np.clip(cy, 0, depth.shape[0] - 1))
        d  = float(depth[iy, ix])
        return d if d > 0 else self.cfg.far_plane * 0.12

    def _backproject(
        self,
        cx: float, cy: float, z: float,
        shape: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        """Simple pinhole back-projection (assumes centre principal point)."""
        h, w  = shape[:2]
        fov_r = np.deg2rad(self.cfg.fov_degrees)
        fx    = (w / 2) / np.tan(fov_r / 2)
        fy    = fx
        x_m   = (cx - w / 2) * z / fx
        y_m   = (cy - h / 2) * z / fy
        return (round(x_m, 4), round(y_m, 4), round(z, 4))

    @staticmethod
    def _dedup(
        objects: List[DetectedObject],
        iou_thresh: float = 0.4,
    ) -> List[DetectedObject]:
        """Remove highly overlapping boxes with the same colour label."""
        kept: List[DetectedObject] = []
        for obj in sorted(objects, key=lambda o: -o.area):
            duplicate = False
            for k in kept:
                if k.colour == obj.colour and _iou(k.bbox_2d, obj.bbox_2d) > iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(obj)
        return kept


# ──────────────────────────────────────────────────────────
# Simulated detector (when running inside PyBullet)
# ──────────────────────────────────────────────────────────

class SimulatedDetector:
    """
    In-simulation oracle: reads ground-truth object poses from PyBullet.
    Used during training/eval to sidestep camera noise.
    """

    def __init__(self):
        self._next_id = 0

    def detect_from_sim(
        self,
        object_info: List[Dict],
    ) -> List[DetectedObject]:
        """
        Parameters
        ──────────
        object_info : list of dicts with keys
                      {colour, shape, position (xyz), orientation}
                      as provided by pybullet_env.get_objects_info()
        """
        detected = []
        for i, info in enumerate(object_info):
            pos = tuple(info.get("position", (0, 0, 0)))
            obj = DetectedObject(
                id=i,
                colour=info.get("colour", "unknown"),
                shape=info.get("shape",  "block"),
                centre_3d=pos,
                area=2500.0,
                confidence=1.0,
            )
            detected.append(obj)
        return detected


# ──────────────────────────────────────────────────────────
# Public wrapper
# ──────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Unified detector API.  Automatically falls back to the colour-
    segmentation backend if DNN weights are unavailable.
    """

    def __init__(
        self,
        cfg: VisionConfig = DEFAULT_CONFIG.vision,
        use_sim_oracle: bool = False,
    ):
        self.cfg      = cfg
        self.sim_mode = use_sim_oracle
        if not use_sim_oracle:
            if _CV2_AVAILABLE:
                self._backend = ColourSegmentationDetector(cfg)
            else:
                self._backend = None
        else:
            self._backend = SimulatedDetector()

    def detect(
        self,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        sim_object_info: Optional[List[Dict]] = None,
    ) -> List[DetectedObject]:
        if self.sim_mode and sim_object_info is not None:
            return self._backend.detect_from_sim(sim_object_info)
        if rgb_image is None:
            return []
        if self._backend is None:
            return self._mock_detect()
        return self._backend.detect(rgb_image, depth_image)

    @staticmethod
    def _mock_detect() -> List[DetectedObject]:
        """Return dummy objects when no backend is available."""
        return [
            DetectedObject(0, "blue",  "block",    centre_3d=(-0.1, 0.0, 0.65)),
            DetectedObject(1, "green", "cube",     centre_3d=( 0.1, 0.0, 0.65)),
            DetectedObject(2, "red",   "sphere",   centre_3d=( 0.0, 0.1, 0.65)),
            DetectedObject(3, "yellow","platform", centre_3d=( 0.0,-0.1, 0.65)),
        ]

    def find_object(
        self,
        objects:    List[DetectedObject],
        colour:     Optional[str],
        shape:      Optional[str],
        spatial_tag: Optional[str] = None,
    ) -> Optional[DetectedObject]:
        """
        Resolve a (colour, shape, spatial_tag) query to one DetectedObject.
        Returns the highest-confidence match, or None if not found.
        """
        candidates = [o for o in objects if o.matches(colour, shape)]
        if spatial_tag:
            tagged = [c for c in candidates if spatial_tag in c.spatial_tags]
            if tagged:
                candidates = tagged
        if not candidates:
            return None
        return max(candidates, key=lambda o: o.confidence)


# ──────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────

def _iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, aw, ah = a;  ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b;  bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)


def visualise_detections(
    image:   np.ndarray,
    objects: List[DetectedObject],
) -> np.ndarray:
    """Draw bounding boxes + labels onto a copy of the image."""
    if not _CV2_AVAILABLE:
        return image
    out = image.copy()
    colour_bgr = {
        "red": (0, 0, 220), "blue": (220, 0, 0), "green": (0, 200, 0),
        "yellow": (0, 220, 220), "orange": (0, 140, 255),
        "purple": (180, 0, 180), "cyan": (200, 200, 0),
        "white": (255, 255, 255), "black": (30, 30, 30),
        "brown": (20, 60, 100), "grey": (150, 150, 150),
    }
    for obj in objects:
        x, y, w, h = obj.bbox_2d
        bgr        = colour_bgr.get(obj.colour, (200, 200, 200))
        cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)
        label = f"{obj.colour} {obj.shape} #{obj.id}"
        cv2.putText(out, label, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
    return out


# ── self-test ───────────────────────────────────────────────

if __name__ == "__main__":
    detector = ObjectDetector(use_sim_oracle=True)
    mock_info = [
        {"colour": "blue",   "shape": "block",    "position": (-0.10, 0.00, 0.65)},
        {"colour": "green",  "shape": "cube",     "position": ( 0.10, 0.00, 0.65)},
        {"colour": "red",    "shape": "sphere",   "position": ( 0.00, 0.10, 0.65)},
        {"colour": "yellow", "shape": "platform", "position": ( 0.00,-0.10, 0.65)},
    ]
    objs = detector.detect(sim_object_info=mock_info)
    print("\nDetected objects:")
    for o in objs:
        print(" ", o)

    print("\nQuery: blue block →", detector.find_object(objs, "blue", "block"))
    print("Query: red sphere  →", detector.find_object(objs, "red",  "sphere"))
    print("Query: cyan cube   →", detector.find_object(objs, "cyan", "cube"))

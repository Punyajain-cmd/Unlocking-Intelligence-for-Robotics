from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


COLOR_ANCHORS_RGB = {
    "red": np.array([220, 40, 40], dtype=np.float32),
    "green": np.array([40, 220, 40], dtype=np.float32),
    "blue": np.array([40, 40, 220], dtype=np.float32),
    "yellow": np.array([220, 220, 40], dtype=np.float32),
}


@dataclass(frozen=True)
class DetectedObject:
    color: str
    centroid_px: tuple[int, int]
    bbox_xywh: tuple[int, int, int, int]
    world_xy: tuple[float, float]
    contour_area: float


class ColorObjectPerceiver:
    """Color-segmentation based object perceiver for top-down RGB frames."""

    def __init__(
        self,
        workspace_x: tuple[float, float],
        workspace_y: tuple[float, float],
        color_distance_threshold: float = 80.0,
        min_area_px: float = 12.0,
    ) -> None:
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.color_distance_threshold = color_distance_threshold
        self.min_area_px = min_area_px

    def pixel_to_world(self, px: int, py: int, width: int, height: int) -> tuple[float, float]:
        x_min, x_max = self.workspace_x
        y_min, y_max = self.workspace_y
        world_x = x_min + (px / max(width - 1, 1)) * (x_max - x_min)
        world_y = y_max - (py / max(height - 1, 1)) * (y_max - y_min)
        return float(world_x), float(world_y)

    def detect(self, rgb_image: np.ndarray) -> dict[str, DetectedObject]:
        if rgb_image.dtype != np.uint8:
            rgb = np.clip(rgb_image, 0, 255).astype(np.uint8)
        else:
            rgb = rgb_image

        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Expected an RGB image with shape [H, W, 3].")

        rgb = rgb[:, :, :3]
        height, width = rgb.shape[:2]
        rgb_f = rgb.astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.uint8)

        detections: dict[str, DetectedObject] = {}
        for color_name, anchor in COLOR_ANCHORS_RGB.items():
            distance = np.linalg.norm(rgb_f - anchor, axis=2)
            mask = (distance < self.color_distance_threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + (w // 2)
            cy = y + (h // 2)
            world_xy = self.pixel_to_world(cx, cy, width, height)
            detections[color_name] = DetectedObject(
                color=color_name,
                centroid_px=(cx, cy),
                bbox_xywh=(x, y, w, h),
                world_xy=world_xy,
                contour_area=area,
            )

        return detections

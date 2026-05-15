from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


@dataclass(frozen=True)
class WorkspaceBounds:
    x: tuple[float, float]
    y: tuple[float, float]


class ManipulationSimEnv:
    """Minimal PyBullet object-manipulation environment."""

    _COLOR_RGBA = {
        "red": (0.95, 0.15, 0.15, 1.0),
        "green": (0.15, 0.9, 0.2, 1.0),
        "blue": (0.15, 0.2, 0.95, 1.0),
        "yellow": (0.95, 0.9, 0.15, 1.0),
    }

    def __init__(
        self,
        gui: bool = False,
        seed: int = 0,
        workspace_bounds: WorkspaceBounds | None = None,
    ) -> None:
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)
        self._rng = np.random.default_rng(seed)
        self.workspace_bounds = workspace_bounds or WorkspaceBounds(
            x=(-0.30, 0.30),
            y=(-0.30, 0.30),
        )
        self.object_z = 0.025
        self.object_half_extent = 0.02
        self.object_ids: dict[str, int] = {}
        self.reset(seed=seed)

    def _sample_non_colliding_xy(self, existing: list[np.ndarray], min_dist: float = 0.11) -> np.ndarray:
        x_min, x_max = self.workspace_bounds.x
        y_min, y_max = self.workspace_bounds.y
        for _ in range(100):
            candidate = np.array(
                [
                    self._rng.uniform(x_min + 0.03, x_max - 0.03),
                    self._rng.uniform(y_min + 0.03, y_max - 0.03),
                ],
                dtype=np.float32,
            )
            if not existing:
                return candidate
            if min(np.linalg.norm(candidate - e) for e in existing) >= min_dist:
                return candidate
        return candidate

    def _create_block(self, color: str, xy: np.ndarray) -> int:
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.object_half_extent] * 3,
            physicsClientId=self.client_id,
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.object_half_extent] * 3,
            rgbaColor=self._COLOR_RGBA[color],
            physicsClientId=self.client_id,
        )
        body_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[float(xy[0]), float(xy[1]), self.object_z],
            physicsClientId=self.client_id,
        )
        p.changeDynamics(body_id, -1, lateralFriction=1.0, physicsClientId=self.client_id)
        return body_id

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.object_ids.clear()
        occupied: list[np.ndarray] = []

        for color in self._COLOR_RGBA.keys():
            xy = self._sample_non_colliding_xy(occupied)
            occupied.append(xy)
            self.object_ids[color] = self._create_block(color, xy)

        for _ in range(30):
            p.stepSimulation(physicsClientId=self.client_id)

    def get_workspace_bounds(self) -> WorkspaceBounds:
        return self.workspace_bounds

    def get_object_positions(self) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        for color, object_id in self.object_ids.items():
            position, _ = p.getBasePositionAndOrientation(object_id, physicsClientId=self.client_id)
            positions[color] = np.array(position, dtype=np.float32)
        return positions

    def set_object_xy(self, color: str, xy: np.ndarray, z: float | None = None) -> None:
        if color not in self.object_ids:
            raise KeyError(f"Unknown object color '{color}'.")
        pos = self.get_object_positions()[color]
        z_val = float(z if z is not None else pos[2])
        p.resetBasePositionAndOrientation(
            self.object_ids[color],
            [float(xy[0]), float(xy[1]), z_val],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )

    def execute_pick_place(self, color: str, target_xy: np.ndarray, steps: int = 90) -> None:
        current = self.get_object_positions()[color]
        start_xy = current[:2].copy()
        target_xy = target_xy.astype(np.float32)
        for i in range(steps):
            alpha = (i + 1) / max(steps, 1)
            xy = (1.0 - alpha) * start_xy + alpha * target_xy
            z = self.object_z + 0.06 * np.sin(np.pi * alpha)
            self.set_object_xy(color, xy, z=z)
            p.stepSimulation(physicsClientId=self.client_id)

        self.set_object_xy(color, target_xy, z=self.object_z)
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client_id)

    def step(self, num_steps: int = 1) -> None:
        for _ in range(max(num_steps, 1)):
            p.stepSimulation(physicsClientId=self.client_id)

    def render_topdown(self, width: int = 256, height: int = 256) -> np.ndarray:
        view = p.computeViewMatrix(
            cameraEyePosition=[0.0, 0.0, 0.90],
            cameraTargetPosition=[0.0, 0.0, 0.0],
            cameraUpVector=[0.0, 1.0, 0.0],
        )
        projection = p.computeProjectionMatrixFOV(
            fov=58.0,
            aspect=float(width) / float(height),
            nearVal=0.05,
            farVal=2.5,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        rgba_np = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
        return rgba_np[:, :, :3]

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

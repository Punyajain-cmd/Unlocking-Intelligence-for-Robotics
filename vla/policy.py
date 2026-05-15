from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .language import CommandIntent
from .perception import DetectedObject
from .simulation import ManipulationSimEnv, WorkspaceBounds


RELATION_OFFSETS = {
    "right_of": np.array([0.12, 0.00], dtype=np.float32),
    "left_of": np.array([-0.12, 0.00], dtype=np.float32),
    "in_front_of": np.array([0.00, 0.12], dtype=np.float32),
    "behind": np.array([0.00, -0.12], dtype=np.float32),
    "next_to": np.array([0.09, 0.09], dtype=np.float32),
    "on_top_of": np.array([0.00, 0.00], dtype=np.float32),
}


@dataclass(frozen=True)
class ExecutionInfo:
    target_xy: tuple[float, float]
    source_xy_before: tuple[float, float]
    source_xy_after: tuple[float, float]
    relation_satisfied: bool


def _clip_to_workspace(xy: np.ndarray, bounds: WorkspaceBounds) -> np.ndarray:
    clipped = xy.copy()
    clipped[0] = np.clip(clipped[0], bounds.x[0] + 0.02, bounds.x[1] - 0.02)
    clipped[1] = np.clip(clipped[1], bounds.y[0] + 0.02, bounds.y[1] - 0.02)
    return clipped


def relation_satisfied(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    relation: str,
    margin: float = 0.05,
    tol: float = 0.08,
) -> bool:
    dx = float(source_xy[0] - target_xy[0])
    dy = float(source_xy[1] - target_xy[1])
    if relation == "right_of":
        return dx > margin and abs(dy) < tol
    if relation == "left_of":
        return dx < -margin and abs(dy) < tol
    if relation == "in_front_of":
        return dy > margin and abs(dx) < tol
    if relation == "behind":
        return dy < -margin and abs(dx) < tol
    if relation == "next_to":
        dist = np.sqrt(dx * dx + dy * dy)
        return 0.05 < dist < 0.20
    if relation == "on_top_of":
        return abs(dx) < 0.04 and abs(dy) < 0.04
    return False


class RelationActionPlanner:
    """Plans target XY locations from parsed relation intents."""

    def __init__(self, relation_offsets: dict[str, np.ndarray] | None = None) -> None:
        self.relation_offsets = relation_offsets or RELATION_OFFSETS

    def plan_target_xy(
        self,
        intent: CommandIntent,
        object_positions_xyz: dict[str, np.ndarray],
        workspace_bounds: WorkspaceBounds,
    ) -> np.ndarray:
        if intent.target_color not in object_positions_xyz:
            raise KeyError(f"Target object '{intent.target_color}' not found in scene.")
        offset = self.relation_offsets.get(intent.relation)
        if offset is None:
            raise KeyError(f"Unsupported relation '{intent.relation}'.")
        target_xy = object_positions_xyz[intent.target_color][:2] + offset
        return _clip_to_workspace(target_xy, workspace_bounds)


class TaskExecutor:
    """Executes parsed intents in the simulation environment."""

    def __init__(self, planner: RelationActionPlanner | None = None) -> None:
        self.planner = planner or RelationActionPlanner()

    def execute(
        self,
        intent: CommandIntent,
        detections: dict[str, DetectedObject],
        env: ManipulationSimEnv,
    ) -> tuple[bool, ExecutionInfo]:
        scene_positions = env.get_object_positions()

        for color, obj in detections.items():
            if color in scene_positions:
                scene_positions[color][:2] = np.array(obj.world_xy, dtype=np.float32)

        source_before = scene_positions[intent.source_color][:2].copy()
        target_xy = self.planner.plan_target_xy(intent, scene_positions, env.get_workspace_bounds())
        env.execute_pick_place(intent.source_color, target_xy)

        final_positions = env.get_object_positions()
        source_after = final_positions[intent.source_color][:2]
        target_after = final_positions[intent.target_color][:2]
        ok = bool(relation_satisfied(source_after, target_after, intent.relation))

        info = ExecutionInfo(
            target_xy=(float(target_xy[0]), float(target_xy[1])),
            source_xy_before=(float(source_before[0]), float(source_before[1])),
            source_xy_after=(float(source_after[0]), float(source_after[1])),
            relation_satisfied=bool(ok),
        )
        return bool(ok), info

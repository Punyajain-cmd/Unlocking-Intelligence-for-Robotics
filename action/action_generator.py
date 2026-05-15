"""
action/action_generator.py
───────────────────────────
Converts a ParsedCommand + SceneGraph into a concrete ActionPlan:
an ordered list of primitive robot actions (grasp, move, place, …).

Grounding pipeline
──────────────────
ParsedCommand
  ├── subject  → resolve_object(colour, shape)   → DetectedObject
  ├── target   → resolve_object(colour, shape)   → DetectedObject (or position)
  └── relation → compute_target_position()       → goal xyz
         ↓
  ActionPlan: [PreGrasp, Grasp, Lift, MoveTo, Place, Release]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG, ActionConfig
from language.command_parser import ParsedCommand
from vision.object_detector import DetectedObject
from vision.scene_graph import SceneGraph


# ──────────────────────────────────────────────────────────
# Primitive action types
# ──────────────────────────────────────────────────────────

class PrimitiveType(Enum):
    MOVE_JOINTS    = auto()
    OPEN_GRIPPER   = auto()
    CLOSE_GRIPPER  = auto()
    MOVE_CARTESIAN = auto()
    PUSH           = auto()
    PULL           = auto()
    ROTATE_JOINT   = auto()
    WAIT           = auto()


@dataclass
class Primitive:
    """A single low-level robot command."""
    ptype:   PrimitiveType
    target:  Optional[Tuple[float, float, float]] = None  # xyz in world frame
    joints:  Optional[List[float]] = None                  # joint angles (rad)
    speed:   float = 0.3        # fraction of max velocity
    comment: str   = ""

    def __str__(self) -> str:
        if self.target:
            t = f"({self.target[0]:.3f},{self.target[1]:.3f},{self.target[2]:.3f})"
        else:
            t = str(self.joints)
        return f"{self.ptype.name:<18} → {t}  [{self.comment}]"


@dataclass
class ActionPlan:
    """Ordered sequence of primitives with metadata."""
    command_raw:  str
    action_type:  str
    subject_obj:  Optional[DetectedObject]
    target_pos:   Optional[Tuple[float, float, float]]
    primitives:   List[Primitive] = field(default_factory=list)
    success:      bool = True
    error_msg:    str  = ""

    def __str__(self) -> str:
        hdr = (f"ActionPlan [{self.action_type}]  "
               f"subject={self.subject_obj}  "
               f"target_pos={self.target_pos}")
        steps = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(self.primitives))
        return f"{hdr}\n{steps}"


# ──────────────────────────────────────────────────────────
# Grounding: command → objects + goal position
# ──────────────────────────────────────────────────────────

class Grounder:
    """
    Resolves symbolic references in a ParsedCommand to concrete
    DetectedObject instances and 3-D goal positions.
    """

    def __init__(self, cfg: ActionConfig = DEFAULT_CONFIG.action):
        self.cfg = cfg

    def ground(
        self,
        cmd:    ParsedCommand,
        scene:  SceneGraph,
    ) -> Tuple[Optional[DetectedObject], Optional[DetectedObject],
               Optional[Tuple[float, float, float]], str]:
        """
        Returns
        ───────
        (subject_obj, target_obj, goal_position, error_message)
        """
        # Resolve subject
        subj_obj = scene.get_by_colour_shape(
            cmd.subject.colour, cmd.subject.shape
        )
        if subj_obj is None:
            return None, None, None, (
                f"Subject '{cmd.subject}' not found in scene. "
                f"Available: {[str(o) for o in scene.get_all()]}"
            )

        # Resolve target
        tgt_obj = None
        goal_pos: Optional[Tuple[float, float, float]] = None

        if cmd.target.colour or cmd.target.shape:
            tgt_obj = scene.get_by_colour_shape(
                cmd.target.colour, cmd.target.shape
            )
            if tgt_obj is None:
                return subj_obj, None, None, (
                    f"Target '{cmd.target}' not found in scene."
                )
            # Compute spatial goal from relation
            if cmd.relation:
                goal_pos = scene.compute_target_position(
                    tgt_obj, cmd.relation,
                    offset_m=self.cfg.grasp_height_offset * 10 + 0.07
                )
            else:
                # Default: place on top of target
                tx, ty, tz = tgt_obj.centre_3d
                goal_pos   = (tx, ty, tz + self.cfg.place_height_offset + 0.05)
        else:
            # No explicit target – use relation + scene context
            if cmd.relation:
                goal_pos = self._infer_goal_from_relation(cmd, subj_obj, scene)

        return subj_obj, tgt_obj, goal_pos, ""

    @staticmethod
    def _infer_goal_from_relation(
        cmd:     ParsedCommand,
        subject: DetectedObject,
        scene:   SceneGraph,
    ) -> Tuple[float, float, float]:
        """Fallback: push/pull relative to current subject position."""
        sx, sy, sz = subject.centre_3d
        offsets = {
            "right_of":   ( 0.12,  0.0,  0.0),
            "left_of":    (-0.12,  0.0,  0.0),
            "in_front_of":( 0.0,  -0.12, 0.0),
            "behind":     ( 0.0,   0.12, 0.0),
            "above":      ( 0.0,   0.0,  0.12),
        }
        dx, dy, dz = offsets.get(cmd.relation, (0.10, 0.0, 0.0))
        return (sx + dx, sy + dy, sz + dz)


# ──────────────────────────────────────────────────────────
# Trajectory builders
# ──────────────────────────────────────────────────────────

class TrajectoryBuilder:
    """
    Converts a (subject_obj, goal_position, action_type) triple into
    an ordered list of Primitive commands.
    """

    def __init__(self, cfg: ActionConfig = DEFAULT_CONFIG.action):
        self.cfg = cfg

    # ── Public ──────────────────────────────────────────────

    def build(
        self,
        action_type:  str,
        subject_obj:  DetectedObject,
        goal_pos:     Optional[Tuple[float, float, float]],
        tgt_obj:      Optional[DetectedObject] = None,
    ) -> List[Primitive]:
        builders = {
            "pick_and_place": self._pick_and_place,
            "grasp":          self._grasp_only,
            "place":          self._place_only,
            "push":           self._push,
            "pull":           self._pull,
            "lift":           self._lift,
            "stack":          self._stack,
            "rotate":         self._rotate,
        }
        fn = builders.get(action_type, self._pick_and_place)
        return fn(subject_obj, goal_pos, tgt_obj)

    # ── Private builders ─────────────────────────────────────

    def _pick_and_place(
        self,
        subject:  DetectedObject,
        goal_pos: Optional[Tuple[float, float, float]],
        _target:  Optional[DetectedObject],
    ) -> List[Primitive]:
        sx, sy, sz = subject.centre_3d
        gx, gy, gz = goal_pos if goal_pos else (sx + 0.10, sy, sz)
        pre_z      = sz + self.cfg.pre_grasp_height
        prims: List[Primitive] = [
            Primitive(PrimitiveType.OPEN_GRIPPER,  comment="open before approach"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, pre_z), comment="pre-grasp"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + self.cfg.grasp_height_offset), comment="descend to grasp"),
            Primitive(PrimitiveType.CLOSE_GRIPPER, comment="close on object"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, pre_z), comment="lift"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (gx, gy, pre_z + 0.05), comment="transport"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (gx, gy, gz + self.cfg.place_height_offset), comment="descend to place"),
            Primitive(PrimitiveType.OPEN_GRIPPER,  comment="release"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (gx, gy, gz + self.cfg.pre_grasp_height), comment="retreat"),
        ]
        return prims

    def _grasp_only(
        self,
        subject:  DetectedObject,
        _goal:    Optional[Tuple],
        _target:  Optional[DetectedObject],
    ) -> List[Primitive]:
        sx, sy, sz = subject.centre_3d
        pre_z      = sz + self.cfg.pre_grasp_height
        return [
            Primitive(PrimitiveType.OPEN_GRIPPER,  comment="open"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, pre_z), comment="pre-grasp"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + self.cfg.grasp_height_offset), comment="descend"),
            Primitive(PrimitiveType.CLOSE_GRIPPER, comment="grasp"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, pre_z), comment="lift"),
        ]

    def _place_only(
        self,
        _subject: DetectedObject,
        goal_pos: Optional[Tuple],
        _target:  Optional[DetectedObject],
    ) -> List[Primitive]:
        gx, gy, gz = goal_pos if goal_pos else (0.0, 0.0, 0.65)
        return [
            Primitive(PrimitiveType.MOVE_CARTESIAN, (gx, gy, gz + self.cfg.pre_grasp_height), comment="move above"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (gx, gy, gz + self.cfg.place_height_offset), comment="descend"),
            Primitive(PrimitiveType.OPEN_GRIPPER,  comment="release"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (gx, gy, gz + self.cfg.pre_grasp_height), comment="retreat"),
        ]

    def _push(
        self,
        subject:  DetectedObject,
        goal_pos: Optional[Tuple],
        _target:  Optional[DetectedObject],
    ) -> List[Primitive]:
        sx, sy, sz = subject.centre_3d
        gx, gy, gz = goal_pos if goal_pos else (sx + 0.10, sy, sz)
        return [
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + 0.01), comment="approach level"),
            Primitive(PrimitiveType.CLOSE_GRIPPER,  comment="close (push posture)"),
            Primitive(PrimitiveType.PUSH, (gx, gy, gz), comment="push to goal"),
        ]

    def _pull(
        self,
        subject:  DetectedObject,
        goal_pos: Optional[Tuple],
        _target:  Optional[DetectedObject],
    ) -> List[Primitive]:
        sx, sy, sz = subject.centre_3d
        gx, gy, gz = goal_pos if goal_pos else (sx - 0.10, sy, sz)
        return [
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx + 0.03, sy, sz + 0.01), comment="hook from behind"),
            Primitive(PrimitiveType.CLOSE_GRIPPER, comment="hook close"),
            Primitive(PrimitiveType.PULL, (gx, gy, gz), comment="pull toward robot"),
        ]

    def _lift(
        self,
        subject:  DetectedObject,
        _goal:    Optional[Tuple],
        _target:  Optional[DetectedObject],
    ) -> List[Primitive]:
        sx, sy, sz = subject.centre_3d
        return [
            Primitive(PrimitiveType.OPEN_GRIPPER,  comment="open"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + self.cfg.pre_grasp_height), comment="pre-grasp"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + 0.005), comment="descend"),
            Primitive(PrimitiveType.CLOSE_GRIPPER, comment="grasp"),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + 0.20), comment="lift high"),
        ]

    def _stack(
        self,
        subject:  DetectedObject,
        goal_pos: Optional[Tuple],
        target:   Optional[DetectedObject],
    ) -> List[Primitive]:
        if target is None:
            return self._pick_and_place(subject, goal_pos, target)
        tx, ty, tz = target.centre_3d
        stack_z    = tz + 0.055   # height of one block ≈ 5 cm
        return self._pick_and_place(subject, (tx, ty, stack_z), target)

    def _rotate(
        self,
        subject:  DetectedObject,
        _goal:    Optional[Tuple],
        _target:  Optional[DetectedObject],
    ) -> List[Primitive]:
        sx, sy, sz = subject.centre_3d
        return [
            Primitive(PrimitiveType.OPEN_GRIPPER),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + self.cfg.pre_grasp_height)),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + 0.005)),
            Primitive(PrimitiveType.CLOSE_GRIPPER),
            Primitive(PrimitiveType.ROTATE_JOINT, comment="rotate 90° around z"),
            Primitive(PrimitiveType.OPEN_GRIPPER),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (sx, sy, sz + self.cfg.pre_grasp_height)),
        ]


# ──────────────────────────────────────────────────────────
# Public ActionGenerator
# ──────────────────────────────────────────────────────────

class ActionGenerator:
    """
    Converts (ParsedCommand, SceneGraph) → ActionPlan.

    Usage
    ─────
    >>> gen   = ActionGenerator()
    >>> plan  = gen.generate(parsed_cmd, scene_graph)
    >>> print(plan)
    """

    def __init__(self, cfg: ActionConfig = DEFAULT_CONFIG.action):
        self.cfg      = cfg
        self.grounder = Grounder(cfg)
        self.builder  = TrajectoryBuilder(cfg)

    def generate(
        self,
        cmd:   ParsedCommand,
        scene: SceneGraph,
    ) -> ActionPlan:
        if not cmd.is_valid:
            return ActionPlan(
                command_raw=cmd.raw,
                action_type="unknown",
                subject_obj=None,
                target_pos=None,
                success=False,
                error_msg=f"Invalid command: {cmd.error_msg}",
            )

        # Ground language → objects + goal
        subj_obj, tgt_obj, goal_pos, err = self.grounder.ground(cmd, scene)
        if err:
            return ActionPlan(
                command_raw=cmd.raw,
                action_type=cmd.action,
                subject_obj=subj_obj,
                target_pos=None,
                success=False,
                error_msg=err,
            )

        # Build trajectory
        primitives = self.builder.build(
            cmd.action, subj_obj, goal_pos, tgt_obj
        )

        return ActionPlan(
            command_raw=cmd.raw,
            action_type=cmd.action,
            subject_obj=subj_obj,
            target_pos=goal_pos,
            primitives=primitives,
            success=True,
        )


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    from language.command_parser import CommandParser
    from vision.object_detector import DetectedObject
    from vision.scene_graph import SceneGraph

    objects = [
        DetectedObject(0, "blue",   "block",  centre_3d=(-0.10, 0.00, 0.65)),
        DetectedObject(1, "green",  "cube",   centre_3d=( 0.00, 0.00, 0.65)),
        DetectedObject(2, "red",    "sphere", centre_3d=( 0.10, 0.00, 0.65)),
        DetectedObject(3, "yellow", "block",  centre_3d=( 0.00,-0.10, 0.65)),
    ]
    scene = SceneGraph().build(objects)

    parser = CommandParser()
    gen    = ActionGenerator()

    COMMANDS = [
        "Move the blue block to the right of the green cube.",
        "Pick up the red sphere.",
        "Stack the blue block on top of the green cube.",
        "Push the yellow block to the left.",
    ]

    for cmd_str in COMMANDS:
        cmd  = parser.parse(cmd_str)
        plan = gen.generate(cmd, scene)
        print("\n" + "─" * 60)
        print(f"COMMAND: {cmd_str}")
        print(plan)

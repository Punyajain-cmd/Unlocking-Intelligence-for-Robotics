"""
robot/robot_config.py
──────────────────────
Universal robot configuration schema.

Any robot — from a 2-DOF planar arm to a 23-DOF dexterous hand — is
described by a plain YAML / dict that gets loaded into a RobotConfig.

Users only need to supply:
  1. Joint names, types, and limits
  2. Parent-child linkage (kinematic tree)
  3. Actuator type (position / velocity / torque)
  4. Optional: end-effector name(s) and gripper joint names

The rest of the system (adapter, IK, motor controller) reads from this
config and works automatically regardless of DOF or morphology.

Example YAML (ur5.yaml):
  name: UR5
  dof: 6
  control_mode: position
  joints:
    - name: shoulder_pan_joint
      type: revolute
      parent: base_link
      child: shoulder_link
      axis: [0, 0, 1]
      limit: [-3.14159, 3.14159]
      max_vel: 3.14
      max_effort: 150.0
    ...
  end_effectors:
    - name: tool0
      parent: wrist_3_link
  gripper_joints: []   # UR5 has no integrated gripper by default
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────
# Joint description
# ─────────────────────────────────────────────────────────

@dataclass
class JointSpec:
    name:       str
    type:       str             = "revolute"     # revolute | prismatic | fixed
    parent:     str             = ""
    child:      str             = ""
    axis:       List[float]     = field(default_factory=lambda: [0, 0, 1])
    limit:      Tuple[float, float] = (-math.pi, math.pi)
    max_vel:    float           = 3.14           # rad/s  or  m/s (prismatic)
    max_effort: float           = 100.0          # N·m    or  N
    damping:    float           = 0.0
    friction:   float           = 0.0
    home_angle: float           = 0.0            # resting position

    @classmethod
    def from_dict(cls, d: dict) -> "JointSpec":
        limit = d.get("limit", [-math.pi, math.pi])
        return cls(
            name       = d["name"],
            type       = d.get("type", "revolute"),
            parent     = d.get("parent", ""),
            child      = d.get("child", ""),
            axis       = d.get("axis", [0, 0, 1]),
            limit      = (float(limit[0]), float(limit[1])),
            max_vel    = float(d.get("max_vel",    3.14)),
            max_effort = float(d.get("max_effort", 100.0)),
            damping    = float(d.get("damping",    0.0)),
            friction   = float(d.get("friction",   0.0)),
            home_angle = float(d.get("home_angle", 0.0)),
        )


# ─────────────────────────────────────────────────────────
# End-effector description
# ─────────────────────────────────────────────────────────

@dataclass
class EndEffectorSpec:
    name:   str
    parent: str
    offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, d: dict) -> "EndEffectorSpec":
        return cls(
            name        = d["name"],
            parent      = d.get("parent", ""),
            offset_xyz  = tuple(d.get("offset_xyz", [0, 0, 0])),
            offset_rpy  = tuple(d.get("offset_rpy", [0, 0, 0])),
        )


# ─────────────────────────────────────────────────────────
# Master robot config
# ─────────────────────────────────────────────────────────

@dataclass
class RobotConfig:
    """
    Complete description of one robot.

    Attributes
    ──────────
    name          : Human-readable robot name ("UR5", "Kuka iiwa-7", …)
    dof           : Total active DOF (computed from joints if 0)
    control_mode  : "position" | "velocity" | "torque"
    joints        : Ordered list of JointSpec (active + fixed)
    end_effectors : List of tool-frame specs
    gripper_joints: Names of joints that control the gripper/fingers
    urdf_path     : Optional path to URDF for full FK/IK
    description   : Free-text notes
    """

    name:           str                   = "robot"
    dof:            int                   = 0
    control_mode:   str                   = "position"
    joints:         List[JointSpec]       = field(default_factory=list)
    end_effectors:  List[EndEffectorSpec] = field(default_factory=list)
    gripper_joints: List[str]             = field(default_factory=list)
    urdf_path:      Optional[str]         = None
    description:    str                   = ""

    def __post_init__(self):
        if self.dof == 0:
            self.dof = sum(1 for j in self.joints
                          if j.type in ("revolute", "prismatic"))

    # ── convenience accessors ────────────────────────────────

    @property
    def active_joints(self) -> List[JointSpec]:
        return [j for j in self.joints if j.type != "fixed"]

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.active_joints]

    @property
    def joint_limits(self) -> np.ndarray:
        """(N, 2) array of [lower, upper] limits for each active joint."""
        lims = [(j.limit[0], j.limit[1]) for j in self.active_joints]
        return np.array(lims, dtype=np.float32)

    @property
    def home_config(self) -> np.ndarray:
        """Home/resting joint angles as a (N,) array."""
        return np.array([j.home_angle for j in self.active_joints],
                        dtype=np.float32)

    @property
    def max_velocities(self) -> np.ndarray:
        return np.array([j.max_vel for j in self.active_joints],
                        dtype=np.float32)

    @property
    def max_efforts(self) -> np.ndarray:
        return np.array([j.max_effort for j in self.active_joints],
                        dtype=np.float32)

    @property
    def primary_ee(self) -> Optional[EndEffectorSpec]:
        return self.end_effectors[0] if self.end_effectors else None

    @property
    def has_gripper(self) -> bool:
        return len(self.gripper_joints) > 0

    @property
    def arm_joints(self) -> List[JointSpec]:
        """Active joints NOT used for grasping."""
        g = set(self.gripper_joints)
        return [j for j in self.active_joints if j.name not in g]

    def clip_joints(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to their physical limits."""
        limits = self.joint_limits
        return np.clip(q, limits[:, 0], limits[:, 1])

    def normalise_joints(self, q: np.ndarray) -> np.ndarray:
        """Map joint angles to [-1, 1] using their limits."""
        limits = self.joint_limits
        lo, hi = limits[:, 0], limits[:, 1]
        return 2.0 * (q - lo) / (hi - lo + 1e-8) - 1.0

    def denormalise_joints(self, q_norm: np.ndarray) -> np.ndarray:
        """Inverse of normalise_joints."""
        limits = self.joint_limits
        lo, hi = limits[:, 0], limits[:, 1]
        return lo + (q_norm + 1.0) * 0.5 * (hi - lo)

    # ── serialisation ────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "RobotConfig":
        joints = [JointSpec.from_dict(j) for j in d.get("joints", [])]
        ees    = [EndEffectorSpec.from_dict(e) for e in d.get("end_effectors", [])]
        cfg = cls(
            name           = d.get("name",         "robot"),
            dof            = int(d.get("dof", 0)),
            control_mode   = d.get("control_mode", "position"),
            joints         = joints,
            end_effectors  = ees,
            gripper_joints = d.get("gripper_joints", []),
            urdf_path      = d.get("urdf_path"),
            description    = d.get("description", ""),
        )
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "RobotConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def __str__(self) -> str:
        return (f"RobotConfig({self.name}  dof={self.dof}  "
                f"mode={self.control_mode}  "
                f"gripper={'yes' if self.has_gripper else 'no'})")


# ─────────────────────────────────────────────────────────
# Built-in presets
# ─────────────────────────────────────────────────────────

def _make_ur5() -> RobotConfig:
    joints = [
        JointSpec("shoulder_pan_joint",  "revolute", "base_link",        "shoulder_link",      [0,0,1], (-math.pi,  math.pi),  3.14, 150.0),
        JointSpec("shoulder_lift_joint", "revolute", "shoulder_link",    "upper_arm_link",     [0,1,0], (-math.pi,  math.pi),  3.14, 150.0),
        JointSpec("elbow_joint",         "revolute", "upper_arm_link",   "forearm_link",       [0,1,0], (-math.pi,  math.pi),  3.14, 150.0),
        JointSpec("wrist_1_joint",       "revolute", "forearm_link",     "wrist_1_link",       [0,1,0], (-math.pi,  math.pi),  3.14,  28.0),
        JointSpec("wrist_2_joint",       "revolute", "wrist_1_link",     "wrist_2_link",       [0,0,1], (-math.pi,  math.pi),  3.14,  28.0),
        JointSpec("wrist_3_joint",       "revolute", "wrist_2_link",     "wrist_3_link",       [0,1,0], (-math.pi,  math.pi),  3.14,  28.0),
    ]
    ees = [EndEffectorSpec("tool0", "wrist_3_link", (0, 0, 0.1))]
    return RobotConfig("UR5", 6, "position", joints, ees, [], description="6-DOF industrial arm")


def _make_kuka_iiwa7() -> RobotConfig:
    joints = [
        JointSpec("iiwa_joint_1", "revolute", "iiwa_link_0", "iiwa_link_1", [0,0,1], (-2.97,  2.97), 1.48, 320.0),
        JointSpec("iiwa_joint_2", "revolute", "iiwa_link_1", "iiwa_link_2", [0,1,0], (-2.09,  2.09), 1.48, 320.0),
        JointSpec("iiwa_joint_3", "revolute", "iiwa_link_2", "iiwa_link_3", [0,0,1], (-2.97,  2.97), 1.75, 176.0),
        JointSpec("iiwa_joint_4", "revolute", "iiwa_link_3", "iiwa_link_4", [0,1,0], (-2.09,  2.09), 1.75, 176.0),
        JointSpec("iiwa_joint_5", "revolute", "iiwa_link_4", "iiwa_link_5", [0,0,1], (-2.97,  2.97), 2.27,  110.0),
        JointSpec("iiwa_joint_6", "revolute", "iiwa_link_5", "iiwa_link_6", [0,1,0], (-2.09,  2.09), 2.27,  110.0),
        JointSpec("iiwa_joint_7", "revolute", "iiwa_link_6", "iiwa_link_7", [0,0,1], (-2.97,  2.97), 2.27,  110.0),
    ]
    ees = [EndEffectorSpec("iiwa_link_ee", "iiwa_link_7", (0, 0, 0.126))]
    return RobotConfig("Kuka_iiwa7", 7, "position", joints, ees, [], description="7-DOF lightweight arm")


def _make_franka_panda() -> RobotConfig:
    joints = [
        JointSpec("panda_joint1", "revolute", "panda_link0", "panda_link1", [0,0,1], (-2.8973,  2.8973), 2.175, 87.0),
        JointSpec("panda_joint2", "revolute", "panda_link1", "panda_link2", [0,1,0], (-1.7628,  1.7628), 2.175, 87.0),
        JointSpec("panda_joint3", "revolute", "panda_link2", "panda_link3", [0,0,1], (-2.8973,  2.8973), 2.175, 87.0),
        JointSpec("panda_joint4", "revolute", "panda_link3", "panda_link4", [0,1,0], (-3.0718, -0.0698), 2.175, 87.0),
        JointSpec("panda_joint5", "revolute", "panda_link4", "panda_link5", [0,0,1], (-2.8973,  2.8973), 2.610, 12.0),
        JointSpec("panda_joint6", "revolute", "panda_link5", "panda_link6", [0,1,0], (-0.0175,  3.7525), 2.610, 12.0),
        JointSpec("panda_joint7", "revolute", "panda_link6", "panda_link7", [0,0,1], (-2.8973,  2.8973), 2.610, 12.0),
        JointSpec("panda_finger_joint1", "prismatic", "panda_hand", "panda_leftfinger",  [0,1,0], (0.0, 0.04), 0.2, 70.0),
        JointSpec("panda_finger_joint2", "prismatic", "panda_hand", "panda_rightfinger", [0,1,0], (0.0, 0.04), 0.2, 70.0),
    ]
    ees = [EndEffectorSpec("panda_hand_tcp", "panda_link8", (0, 0, 0.1034))]
    gripper = ["panda_finger_joint1", "panda_finger_joint2"]
    return RobotConfig("Franka_Panda", 9, "position", joints, ees, gripper, description="7-DOF arm + 2-DOF parallel gripper")


def _make_shadow_hand() -> RobotConfig:
    """24-DOF Shadow Dexterous Hand (simplified joint set)."""
    fingers = {
        "FF": ["J4", "J3", "J2", "J1"],
        "MF": ["J4", "J3", "J2", "J1"],
        "RF": ["J4", "J3", "J2", "J1"],
        "LF": ["J5", "J4", "J3", "J2", "J1"],
        "TH": ["J5", "J4", "J3", "J2", "J1"],
        "WR": ["J1", "J2"],
    }
    joints = []
    for finger, jnames in fingers.items():
        for jn in jnames:
            lo, hi = (-0.35, 1.57) if jn != "J1" else (-0.5, 0.5)
            joints.append(JointSpec(f"{finger}{jn}", "revolute",
                                    f"{finger}_proximal", f"{finger}_distal",
                                    [0,1,0], (lo, hi), 2.0, 0.5,
                                    home_angle=0.0))
    # Wrist joints act as the "gripper" (opening/closing the hand posture);
    # all finger + thumb joints are the arm (manipulation DOF).
    gripper_jnames = ["WRJ1", "WRJ2"]
    ees = [EndEffectorSpec("palm_ee", "palm", (0, 0, 0))]
    return RobotConfig("Shadow_Hand", len(joints), "position", joints, ees,
                       gripper_jnames, description="24-DOF dexterous hand")


def _make_simple_arm_2dof() -> RobotConfig:
    joints = [
        JointSpec("joint1", "revolute", "base", "link1", [0,0,1], (-math.pi, math.pi), 5.0, 50.0),
        JointSpec("joint2", "revolute", "link1", "link2", [0,1,0], (-math.pi, math.pi), 5.0, 50.0),
    ]
    ees = [EndEffectorSpec("tip", "link2", (0, 0, 0.3))]
    return RobotConfig("Simple2DOF", 2, "position", joints, ees, description="2-DOF planar arm")


# ─────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────

ROBOT_PRESETS: Dict[str, RobotConfig] = {}


def _lazy_build_presets():
    global ROBOT_PRESETS
    if not ROBOT_PRESETS:
        ROBOT_PRESETS = {
            "ur5":          _make_ur5(),
            "kuka_iiwa7":   _make_kuka_iiwa7(),
            "franka_panda": _make_franka_panda(),
            "shadow_hand":  _make_shadow_hand(),
            "simple_2dof":  _make_simple_arm_2dof(),
        }


def get_robot(name: str) -> RobotConfig:
    """
    Retrieve a preset robot by name OR load from YAML path.

    Parameters
    ──────────
    name : "ur5" | "kuka_iiwa7" | "franka_panda" | "shadow_hand"
           | path/to/my_robot.yaml
    """
    _lazy_build_presets()
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key in ROBOT_PRESETS:
        return ROBOT_PRESETS[key]
    p = Path(name)
    if p.exists():
        return RobotConfig.from_yaml(str(p))
    raise ValueError(
        f"Unknown robot '{name}'. "
        f"Available presets: {list(ROBOT_PRESETS.keys())} "
        f"or provide a path to a YAML file."
    )


def list_presets() -> List[str]:
    _lazy_build_presets()
    return list(ROBOT_PRESETS.keys())


if __name__ == "__main__":
    for name in ["ur5", "kuka_iiwa7", "franka_panda", "shadow_hand", "simple_2dof"]:
        r = get_robot(name)
        print(r)
        print(f"  active joints: {r.joint_names}")
        print(f"  home config:   {r.home_config}")
        print()

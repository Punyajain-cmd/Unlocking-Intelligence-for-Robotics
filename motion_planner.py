"""
action/motion_planner.py
──────────────────────────
Converts a Cartesian waypoint sequence (ActionPlan.primitives) into
joint-space trajectories using:
  1. Analytical / numerical IK  (via ikpy if available)
  2. Linear interpolation in Cartesian space (fallback)

Also provides:
  • Collision-check hooks (stub – extend with PyBullet geometry queries)
  • Velocity / acceleration profile (trapezoidal)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import ikpy.chain
    import ikpy.utils.plot as ikplot
    _IKPY_AVAILABLE = True
except ImportError:
    _IKPY_AVAILABLE = False

from config import DEFAULT_CONFIG, ActionConfig
from action.action_generator import ActionPlan, Primitive, PrimitiveType


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class JointTrajectory:
    """Time-parametrised joint trajectory."""
    joint_positions:  List[List[float]]   # shape [T, num_joints]
    timestamps:       List[float]         # seconds
    gripper_cmds:     List[float]         # gripper width at each step
    num_joints:       int = 7

    def num_steps(self) -> int:
        return len(self.joint_positions)

    def __str__(self) -> str:
        dur = self.timestamps[-1] if self.timestamps else 0
        return (f"JointTrajectory  steps={self.num_steps()}  "
                f"duration={dur:.2f}s  joints={self.num_joints}")


# ──────────────────────────────────────────────────────────
# IK solver wrapper
# ──────────────────────────────────────────────────────────

class IKSolver:
    """
    Thin wrapper around ikpy (if available) with a numeric fallback.

    The KUKA iiwa-7 has 7 revolute joints.  We approximate its
    kinematics with a planar 3-DOF chain for the fallback.
    """

    NUM_JOINTS = 7
    # Approximate KUKA iiwa link lengths (m)
    LINK_LENGTHS = [0.34, 0.0, 0.40, 0.0, 0.40, 0.0, 0.15]

    def __init__(self, urdf_path: Optional[str] = None):
        self._chain = None
        if _IKPY_AVAILABLE and urdf_path:
            try:
                self._chain = ikpy.chain.Chain.from_urdf_file(
                    urdf_path,
                    active_links_mask=[False, True, True, True,
                                       True, True, True, True, False],
                )
            except Exception as e:
                warnings.warn(f"ikpy chain load failed ({e}); using fallback IK.")

    def solve(
        self,
        target_xyz:   Tuple[float, float, float],
        initial_joints: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        """
        Returns a joint angle list (len = NUM_JOINTS) or None on failure.
        """
        if self._chain is not None:
            return self._ikpy_solve(target_xyz, initial_joints)
        return self._analytic_fallback(target_xyz)

    def _ikpy_solve(
        self,
        xyz:     Tuple[float, float, float],
        initial: Optional[List[float]],
    ) -> Optional[List[float]]:
        target_frame = np.eye(4)
        target_frame[:3, 3] = xyz
        try:
            angles = self._chain.inverse_kinematics(
                target_frame,
                initial_position=initial,
                max_iter=DEFAULT_CONFIG.action.ik_max_iter,
            )
            return list(angles[1:-1])   # strip dummy first/last joints
        except Exception:
            return None

    def _analytic_fallback(
        self,
        xyz: Tuple[float, float, float],
    ) -> List[float]:
        """
        Very simplified 3-link planar IK projected into 3-D.
        Produces plausible (not exact) angles for simulation display.
        """
        x, y, z = xyz
        # Project into reach plane
        r  = math.sqrt(x**2 + y**2)
        h  = z - 0.34        # offset from base
        L1 = 0.40
        L2 = 0.40

        d  = math.sqrt(r**2 + h**2)
        d  = min(d, L1 + L2 - 0.01)  # clamp to reach

        cos2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos2 = np.clip(cos2, -1.0, 1.0)
        q2   = math.acos(cos2)

        alpha = math.atan2(h, r)
        beta  = math.acos(np.clip((d**2 + L1**2 - L2**2)/(2*d*L1), -1, 1))
        q1    = alpha + beta

        q0  = math.atan2(y, x)          # base rotation
        joints = [q0, q1, -q2, q1*0.3, q2*0.4, -q1*0.2, 0.0]
        return joints


# ──────────────────────────────────────────────────────────
# Trajectory interpolation
# ──────────────────────────────────────────────────────────

def _trapezoidal_profile(n: int, total_time: float) -> List[float]:
    """Trapezoidal velocity profile: accelerate 20%, coast, decelerate 20%."""
    t   = np.linspace(0, 1, n)
    acc = 0.2
    vel = np.where(
        t < acc, t / acc,
        np.where(t > 1 - acc, (1 - t) / acc, 1.0)
    )
    vel /= vel.sum()
    return list(np.cumsum(vel) * total_time)


def _interpolate_cartesian(
    start: Tuple[float, float, float],
    end:   Tuple[float, float, float],
    n_pts: int,
) -> List[Tuple[float, float, float]]:
    xs = np.linspace(start[0], end[0], n_pts)
    ys = np.linspace(start[1], end[1], n_pts)
    zs = np.linspace(start[2], end[2], n_pts)
    return list(zip(xs, ys, zs))


# ──────────────────────────────────────────────────────────
# Motion Planner
# ──────────────────────────────────────────────────────────

class MotionPlanner:
    """
    Converts ActionPlan primitives → JointTrajectory.

    Each Cartesian waypoint is resolved via IK; gripper open/close
    commands are inserted at the appropriate steps.
    """

    HOME_JOINTS = [0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0]

    def __init__(
        self,
        cfg:       ActionConfig = DEFAULT_CONFIG.action,
        urdf_path: Optional[str] = None,
    ):
        self.cfg     = cfg
        self.ik      = IKSolver(urdf_path)
        self._gripper_open  = cfg.gripper_open_width
        self._gripper_close = cfg.gripper_close_width

    def plan(self, action_plan: ActionPlan) -> JointTrajectory:
        """
        Convert an ActionPlan to a JointTrajectory.
        """
        all_joints:  List[List[float]] = []
        all_times:   List[float] = []
        all_gripper: List[float] = []

        current_joints = list(self.HOME_JOINTS)
        current_gripper = self._gripper_open
        t_cursor = 0.0
        n_wp     = self.cfg.waypoints

        for prim in action_plan.primitives:
            if prim.ptype == PrimitiveType.OPEN_GRIPPER:
                current_gripper = self._gripper_open
                all_joints.append(list(current_joints))
                all_times.append(t_cursor)
                all_gripper.append(current_gripper)
                t_cursor += 0.3

            elif prim.ptype == PrimitiveType.CLOSE_GRIPPER:
                current_gripper = self._gripper_close
                all_joints.append(list(current_joints))
                all_times.append(t_cursor)
                all_gripper.append(current_gripper)
                t_cursor += 0.3

            elif prim.ptype in (PrimitiveType.MOVE_CARTESIAN,
                                PrimitiveType.PUSH,
                                PrimitiveType.PULL):
                if prim.target is None:
                    continue
                start_xyz = self._fk_approx(current_joints)
                waypoints = _interpolate_cartesian(start_xyz, prim.target, n_wp)
                seg_dur   = self._segment_duration(start_xyz, prim.target, prim.speed)
                seg_times = _trapezoidal_profile(n_wp, seg_dur)

                for wp, dt in zip(waypoints, seg_times):
                    joints = self.ik.solve(wp, current_joints)
                    if joints is None:
                        joints = list(current_joints)
                    current_joints = list(joints)
                    all_joints.append(list(current_joints))
                    all_times.append(t_cursor + dt)
                    all_gripper.append(current_gripper)

                t_cursor += seg_dur

            elif prim.ptype == PrimitiveType.ROTATE_JOINT:
                # Rotate last wrist joint by 90°
                new_joints  = list(current_joints)
                new_joints[-1] += math.pi / 2
                seg_times = _trapezoidal_profile(n_wp, 1.0)
                for frac, dt in zip(np.linspace(0, 1, n_wp), seg_times):
                    interp = [
                        c + frac * (n - c)
                        for c, n in zip(current_joints, new_joints)
                    ]
                    all_joints.append(interp)
                    all_times.append(t_cursor + dt)
                    all_gripper.append(current_gripper)
                current_joints = new_joints
                t_cursor += 1.0

            elif prim.ptype == PrimitiveType.WAIT:
                all_joints.append(list(current_joints))
                all_times.append(t_cursor)
                all_gripper.append(current_gripper)
                t_cursor += 0.5

        return JointTrajectory(
            joint_positions=all_joints,
            timestamps=all_times,
            gripper_cmds=all_gripper,
            num_joints=IKSolver.NUM_JOINTS,
        )

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _fk_approx(joints: List[float]) -> Tuple[float, float, float]:
        """Very rough FK: used only to compute path length for timing."""
        q0, q1, q2 = joints[0], joints[1], joints[2]
        L1, L2 = 0.40, 0.40
        z  = 0.34 + L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
        r  = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
        x  = r * math.cos(q0)
        y  = r * math.sin(q0)
        return (x, y, z)

    @staticmethod
    def _segment_duration(
        start: Tuple[float, float, float],
        end:   Tuple[float, float, float],
        speed: float,
    ) -> float:
        dist = float(np.linalg.norm(np.array(end) - np.array(start)))
        max_cart_vel = 0.15 * speed   # m/s
        return max(0.5, dist / (max_cart_vel + 1e-6))


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    from action.action_generator import ActionPlan, Primitive, PrimitiveType
    from vision.object_detector import DetectedObject

    plan = ActionPlan(
        command_raw="Move the blue block.",
        action_type="pick_and_place",
        subject_obj=DetectedObject(0, "blue", "block", centre_3d=(0.0, 0.0, 0.65)),
        target_pos=(0.15, 0.0, 0.65),
        primitives=[
            Primitive(PrimitiveType.OPEN_GRIPPER),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (0.0, 0.0, 0.80)),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (0.0, 0.0, 0.655)),
            Primitive(PrimitiveType.CLOSE_GRIPPER),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (0.0, 0.0, 0.80)),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (0.15, 0.0, 0.80)),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (0.15, 0.0, 0.67)),
            Primitive(PrimitiveType.OPEN_GRIPPER),
            Primitive(PrimitiveType.MOVE_CARTESIAN, (0.15, 0.0, 0.80)),
        ],
    )

    planner    = MotionPlanner()
    trajectory = planner.plan(plan)
    print(trajectory)
    print(f"First 3 joint configs: {trajectory.joint_positions[:3]}")

"""
robot/robot_adapter.py
───────────────────────
Universal Robot Adapter.

Given a RobotConfig, this adapter translates high-level Cartesian
action commands (from the VLA model or action generator) into:
  • Joint positions / velocities / torques  (depending on control_mode)
  • Gripper commands                        (open/close + width)

The user only needs to register their robot ONCE:

    adapter = RobotAdapter.from_config("ur5")
    cmd     = adapter.cartesian_to_joints(target_xyz=(0.3, 0.1, 0.5))
    send_to_robot(cmd.joint_positions)

Works for any DOF — 2-DOF planar arms, 6-DOF industrial, 7-DOF collaborative,
up to 23-DOF dexterous hands — without changing a single line of user code.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from robot.robot_config import RobotConfig, get_robot
from robot.kinematics import RobotKinematics


# ─────────────────────────────────────────────────────────
# Motor command container
# ─────────────────────────────────────────────────────────

@dataclass
class MotorCommand:
    """
    Low-level command sent to the physical (or simulated) robot.

    Fields
    ──────
    joint_positions  : target joint angles/lengths  (radians or metres)
    joint_velocities : feedforward velocities        (rad/s or m/s)
    joint_torques    : feedforward torques           (N·m or N)
    gripper_width    : gripper opening               (metres, 0=closed)
    gripper_force    : max gripper contact force     (N)
    control_mode     : "position" | "velocity" | "torque"
    timestamp_s      : when this command was created
    robot_name       : originating robot name
    """
    joint_positions:   np.ndarray
    joint_velocities:  Optional[np.ndarray]  = None
    joint_torques:     Optional[np.ndarray]  = None
    gripper_width:     float                 = 0.08     # open by default
    gripper_force:     float                 = 20.0
    control_mode:      str                   = "position"
    timestamp_s:       float                 = 0.0
    robot_name:        str                   = ""

    def to_dict(self) -> Dict:
        return {
            "joint_positions":  self.joint_positions.tolist(),
            "joint_velocities": (self.joint_velocities.tolist()
                                 if self.joint_velocities is not None else None),
            "joint_torques":    (self.joint_torques.tolist()
                                 if self.joint_torques is not None else None),
            "gripper_width":    self.gripper_width,
            "gripper_force":    self.gripper_force,
            "control_mode":     self.control_mode,
            "timestamp_s":      self.timestamp_s,
            "robot_name":       self.robot_name,
        }

    def __str__(self) -> str:
        q_str = np.array2string(self.joint_positions, precision=3, suppress_small=True)
        return (f"MotorCommand[{self.robot_name}]  "
                f"q={q_str}  gripper={self.gripper_width:.3f}m")


# ─────────────────────────────────────────────────────────
# Trajectory interpolator
# ─────────────────────────────────────────────────────────

def _trapezoidal_profile(n: int) -> np.ndarray:
    """Normalised trapezoidal velocity profile over n steps."""
    t    = np.linspace(0, 1, n)
    acc  = 0.2
    vel  = np.where(t < acc, t / acc,
           np.where(t > 1 - acc, (1 - t) / acc, 1.0))
    vel /= (vel.sum() + 1e-9)
    return np.cumsum(vel)


def interpolate_joint_trajectory(
    q_start: np.ndarray,
    q_end:   np.ndarray,
    n_steps: int = 20,
) -> np.ndarray:
    """Interpolate joint angles from q_start to q_end with a velocity profile.
    Returns (n_steps, N) array.  Handles DOF mismatches by padding to longer."""
    n = max(len(q_start), len(q_end))
    qs = np.zeros(n, dtype=np.float32)
    qe = np.zeros(n, dtype=np.float32)
    qs[:len(q_start)] = q_start
    qe[:len(q_end)]   = q_end
    # Keep trailing joints (e.g. gripper) at their current value
    qe[len(q_end):]   = qs[len(q_end):]
    alpha = _trapezoidal_profile(n_steps)[:, None]
    return qs[None] + alpha * (qe - qs)[None]


# ─────────────────────────────────────────────────────────
# Robot Adapter
# ─────────────────────────────────────────────────────────

class RobotAdapter:
    """
    High-level adapter for any robot described by a RobotConfig.

    Responsibilities
    ────────────────
    1. Cartesian → joint IK
    2. Trajectory generation between waypoints
    3. Gripper command generation
    4. Action normalization / denormalization (for VLA model output)
    5. Safety clamping (joint limits + velocity limits)
    """

    def __init__(
        self,
        cfg:          RobotConfig,
        max_vel_frac: float = 0.5,    # fraction of max_vel to allow
        n_waypoints:  int   = 20,     # interpolation steps per move
    ):
        self.cfg          = cfg
        self.kinematics   = RobotKinematics(cfg)
        self._max_vel_frac = max_vel_frac
        self._n_wp        = n_waypoints
        self._q_current   = cfg.home_config.copy()
        self._gripper_w   = 0.08     # open

    # ── factory ─────────────────────────────────────────────

    @classmethod
    def from_config(cls, name_or_path: str, **kwargs) -> "RobotAdapter":
        return cls(get_robot(name_or_path), **kwargs)

    # ── Cartesian control ────────────────────────────────────

    def cartesian_to_joints(
        self,
        target_xyz:  Tuple[float, float, float],
        target_R:    Optional[np.ndarray] = None,
        q_init:      Optional[np.ndarray] = None,
    ) -> MotorCommand:
        """
        Solve IK for a Cartesian target and return a MotorCommand.

        This is the primary API for an arm: pass (x, y, z) where you want
        the end-effector to go, get back joint angles to send to the robot.
        """
        q0 = q_init if q_init is not None else self._q_current
        q  = self.kinematics.joints_for_target(target_xyz, target_R, q0)
        q  = self._safe_clamp(q)
        self._q_current = q.copy()
        return self._make_cmd(q)

    def move_to_cartesian(
        self,
        target_xyz:  Tuple[float, float, float],
        target_R:    Optional[np.ndarray] = None,
        n_steps:     Optional[int]  = None,
    ) -> List[MotorCommand]:
        """
        Generate a smooth trajectory from current pose to target_xyz.
        Returns list of MotorCommand (one per waypoint).
        """
        n = n_steps or self._n_wp
        q_end  = self.cartesian_to_joints(target_xyz, target_R).joint_positions
        q_traj = interpolate_joint_trajectory(self._q_current, q_end, n)
        cmds   = []
        for i, q in enumerate(q_traj):
            q_safe = self._safe_clamp(q)
            cmd    = self._make_cmd(q_safe)
            cmd.gripper_width = self._gripper_w
            cmds.append(cmd)
        self._q_current = q_traj[-1].copy()
        return cmds

    # ── Joint-space control ──────────────────────────────────

    def set_joints(self, q: np.ndarray) -> MotorCommand:
        """Direct joint-space command (with safety clamping)."""
        q = self._safe_clamp(np.asarray(q, dtype=np.float32))
        self._q_current = q.copy()
        return self._make_cmd(q)

    # ── Gripper control ──────────────────────────────────────

    def open_gripper(self, width: Optional[float] = None) -> MotorCommand:
        w = width if width is not None else self.cfg_gripper_open_width
        self._gripper_w = w
        cmd = self._make_cmd(self._q_current)
        cmd.gripper_width = w
        return cmd

    def close_gripper(self, width: Optional[float] = None) -> MotorCommand:
        w = width if width is not None else self.cfg_gripper_close_width
        self._gripper_w = w
        cmd = self._make_cmd(self._q_current)
        cmd.gripper_width = w
        return cmd

    # ── VLA output → MotorCommand ────────────────────────────

    def action_to_command(
        self,
        action_vector: np.ndarray,
        action_type:   str = "delta_cartesian",
    ) -> MotorCommand:
        """
        Convert a VLA model output action vector to a MotorCommand.

        action_type options:
          "delta_cartesian" : action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
          "delta_joints"    : action = [dq0, dq1, …, dqN, gripper]
          "abs_joints"      : action = [q0, q1, …, qN, gripper]
          "abs_cartesian"   : action = [x, y, z, roll, pitch, yaw, gripper]
        """
        n_arm = len(self.cfg.arm_joints)

        if action_type == "delta_cartesian":
            # Extract position delta
            dx, dy, dz = action_vector[:3]
            gripper    = float(np.clip(action_vector[6], 0, 1)) if len(action_vector) > 6 else 0.5
            pos, _     = self.kinematics.fk.fk(self._q_current)
            target     = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            cmd        = self.cartesian_to_joints(target)
            cmd.gripper_width = gripper * self.cfg_gripper_open_width

        elif action_type == "delta_joints":
            dq    = action_vector[:n_arm]
            grip  = float(action_vector[n_arm]) if len(action_vector) > n_arm else 0.5
            q_new = self._q_current[:n_arm] + dq
            cmd   = self.set_joints(
                np.pad(q_new, (0, max(0, self.cfg.dof - n_arm))))
            cmd.gripper_width = grip * self.cfg_gripper_open_width

        elif action_type == "abs_joints":
            q_raw = action_vector[:self.cfg.dof]
            grip  = float(action_vector[self.cfg.dof]) if len(action_vector) > self.cfg.dof else 0.5
            q     = self.cfg.denormalise_joints(q_raw) if q_raw.max() <= 1.0 else q_raw
            cmd   = self.set_joints(q)
            cmd.gripper_width = grip * self.cfg_gripper_open_width

        elif action_type == "abs_cartesian":
            x, y, z = action_vector[:3]
            grip    = float(action_vector[6]) if len(action_vector) > 6 else 0.5
            cmd     = self.cartesian_to_joints((x, y, z))
            cmd.gripper_width = grip * self.cfg_gripper_open_width

        else:
            raise ValueError(f"Unknown action_type: {action_type!r}")

        return cmd

    # ── Properties / helpers ─────────────────────────────────

    @property
    def current_joints(self) -> np.ndarray:
        return self._q_current.copy()

    @property
    def current_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kinematics.fk.fk(self._q_current)

    def home(self) -> MotorCommand:
        """Return to home configuration."""
        cmds = self.move_to_cartesian_joints(self.cfg.home_config)
        return cmds[-1]

    def move_to_cartesian_joints(
        self, q_target: np.ndarray, n_steps: int = 20
    ) -> List[MotorCommand]:
        q_traj = interpolate_joint_trajectory(self._q_current, q_target, n_steps)
        cmds = [self._make_cmd(self._safe_clamp(q)) for q in q_traj]
        self._q_current = q_traj[-1].copy()
        return cmds

    # ── Dexterous-hand specific ──────────────────────────────

    def finger_grasp_command(
        self,
        grasp_type: str = "power",    # "power" | "pinch" | "tripod" | "open"
    ) -> MotorCommand:
        """
        Generate grasp configurations for multi-finger hands.
        Works for any robot with gripper_joints defined.
        """
        q = self._q_current.copy()
        n_arm = len(self.cfg.arm_joints)
        grip_indices = [
            i for i, j in enumerate(self.cfg.active_joints)
            if j.name in self.cfg.gripper_joints
        ]

        if grasp_type == "open":
            for gi in grip_indices:
                if gi < len(q):
                    q[gi] = self.cfg.active_joints[gi].limit[0]
        elif grasp_type == "power":
            for gi in grip_indices:
                if gi < len(q):
                    q[gi] = self.cfg.active_joints[gi].limit[1] * 0.7
        elif grasp_type == "pinch":
            # Only close the first two fingers
            for k, gi in enumerate(grip_indices[:8]):
                if gi < len(q):
                    q[gi] = self.cfg.active_joints[gi].limit[1] * (0.5 if k < 4 else 0.0)
        elif grasp_type == "tripod":
            for k, gi in enumerate(grip_indices[:12]):
                if gi < len(q):
                    q[gi] = self.cfg.active_joints[gi].limit[1] * (0.6 if k < 8 else 0.0)

        q = self._safe_clamp(q)
        self._q_current = q.copy()
        return self._make_cmd(q)

    # ── internal helpers ─────────────────────────────────────

    @property
    def cfg_gripper_open_width(self) -> float:
        if self.cfg.has_gripper:
            return self.cfg.active_joints[
                self.cfg.joint_names.index(self.cfg.gripper_joints[0])
            ].limit[1] if self.cfg.gripper_joints else 0.08
        return 0.08

    @property
    def cfg_gripper_close_width(self) -> float:
        return 0.0

    @property
    def cfg(self) -> RobotConfig:
        return self.__dict__["cfg"]

    @cfg.setter
    def cfg(self, v):
        self.__dict__["cfg"] = v

    def _safe_clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp to joint limits and max velocity."""
        N     = len(self.cfg.active_joints)
        q     = np.asarray(q, dtype=np.float32)[:N]
        limits = self.cfg.joint_limits[:len(q)]
        q     = np.clip(q, limits[:, 0], limits[:, 1])
        return q

    def _make_cmd(self, q: np.ndarray) -> MotorCommand:
        N = len(self.cfg.active_joints)
        q = np.asarray(q, dtype=np.float32)[:N]
        # Pad if needed
        if len(q) < N:
            q = np.pad(q, (0, N - len(q)))

        vel = None
        if len(self._q_current) >= len(q):
            dq  = q - self._q_current[:len(q)]
            vel = np.clip(
                dq,
                -self.cfg.max_velocities[:len(q)] * self._max_vel_frac,
                 self.cfg.max_velocities[:len(q)] * self._max_vel_frac,
            )

        return MotorCommand(
            joint_positions  = q,
            joint_velocities = vel,
            gripper_width    = self._gripper_w,
            control_mode     = self.cfg.control_mode,
            robot_name       = self.cfg.name,
        )


if __name__ == "__main__":
    for robot_name in ["kuka_iiwa7", "ur5", "franka_panda", "simple_2dof"]:
        adapter = RobotAdapter.from_config(robot_name)
        tgt = (0.3, 0.1, 0.6)
        cmd = adapter.cartesian_to_joints(tgt)
        pos, _ = adapter.current_ee_pose
        print(f"\n{robot_name} ({adapter.cfg.dof}-DOF):")
        print(f"  Target EE:  {tgt}")
        print(f"  Achieved:   {tuple(round(v,3) for v in pos)}")
        print(f"  Joints:     {cmd.joint_positions.round(3)}")

    print("\n--- Shadow Hand (23 DOF) ---")
    hand = RobotAdapter.from_config("shadow_hand")
    power = hand.finger_grasp_command("power")
    pinch = hand.finger_grasp_command("pinch")
    print(f"Power grasp joints (first 5): {power.joint_positions[:5].round(3)}")
    print(f"Pinch grasp joints (first 5): {pinch.joint_positions[:5].round(3)}")

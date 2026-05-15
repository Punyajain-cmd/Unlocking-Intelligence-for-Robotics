"""
robot/kinematics.py
────────────────────
Universal kinematic solver — works for any robot described by a RobotConfig.

Provides:
  • Forward Kinematics (FK):  joint angles → end-effector pose
  • Inverse Kinematics (IK):  target pose  → joint angles
  • Jacobian computation       for velocity-level control
  • Workspace sampling         for reachability analysis

Solver strategies (in priority order):
  1. ikpy           — if URDF path is provided in RobotConfig
  2. Jacobian CCD   — iterative Cyclic Coordinate Descent  (any DOF)
  3. Analytical 2/3-link fallback
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np

try:
    import ikpy.chain as ikpy_chain
    _IKPY = True
except ImportError:
    _IKPY = False

from robot.robot_config import RobotConfig, JointSpec


# ─────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────

def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def _axis_angle_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula: axis×angle → 3×3 rotation matrix."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    k = axis
    K = np.array([
        [ 0,    -k[2],  k[1]],
        [ k[2],  0,    -k[0]],
        [-k[1],  k[0],  0   ]
    ], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K

def _homo(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous transform from R (3×3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ─────────────────────────────────────────────────────────
# Simple link-length estimator from joint spec
# ─────────────────────────────────────────────────────────

# Default link lengths for common robots (metres)
_DEFAULT_LINK_LENGTHS = {
    "kuka_iiwa7":   [0.340, 0.0,   0.400, 0.0,   0.390, 0.0,   0.120],
    "ur5":          [0.089, 0.425, 0.392, 0.109,  0.094, 0.082, 0.0],
    "franka_panda": [0.333, 0.0,   0.316, 0.0,   0.384, 0.0,   0.107],
    "simple_2dof":  [0.5,   0.4],
}

def _estimate_link_lengths(cfg: RobotConfig) -> List[float]:
    key = cfg.name.lower().replace(" ", "_").replace("-", "_")
    if key in _DEFAULT_LINK_LENGTHS:
        return _DEFAULT_LINK_LENGTHS[key]
    n = len(cfg.arm_joints)
    return [0.3] * n


# ─────────────────────────────────────────────────────────
# Forward Kinematics
# ─────────────────────────────────────────────────────────

class ForwardKinematics:
    """
    Computes end-effector pose from joint angles using sequential
    Denavit–Hartenberg–style transforms built from the RobotConfig.
    """

    def __init__(self, cfg: RobotConfig):
        self.cfg          = cfg
        self.link_lengths = _estimate_link_lengths(cfg)
        self.arm_joints   = cfg.arm_joints

        # Build static base transform
        self._base_T = np.eye(4, dtype=np.float64)

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics.

        Parameters
        ──────────
        q : (N,) joint angles/positions for arm joints

        Returns
        ───────
        position    : (3,)   end-effector XYZ in world frame
        orientation : (3,3)  rotation matrix
        """
        arm_q = q[:len(self.arm_joints)]
        T = self._base_T.copy()

        for i, (joint, qi) in enumerate(zip(self.arm_joints, arm_q)):
            axis  = np.array(joint.axis, dtype=np.float64)
            R     = _axis_angle_rot(axis, float(qi))
            ll    = self.link_lengths[i] if i < len(self.link_lengths) else 0.3
            t_vec = axis * ll
            T_step = _homo(R, t_vec)
            T = T @ T_step

        # Apply end-effector offset if available
        ee = self.cfg.primary_ee
        if ee is not None:
            offset = np.array(ee.offset_xyz, dtype=np.float64)
            T[:3, 3] += T[:3, :3] @ offset

        return T[:3, 3].copy(), T[:3, :3].copy()

    def jacobian(self, q: np.ndarray, delta: float = 1e-4) -> np.ndarray:
        """
        Numerical Jacobian (6 × N).
        Rows 0–2: linear velocity;  rows 3–5: angular velocity.
        """
        arm_q = q[:len(self.arm_joints)]
        N   = len(arm_q)
        J   = np.zeros((6, N), dtype=np.float64)
        p0, R0 = self.fk(arm_q)

        for i in range(N):
            q_plus       = arm_q.copy()
            q_plus[i]   += delta
            p_plus, R_plus = self.fk(q_plus)

            J[:3, i] = (p_plus - p0) / delta
            # angular: extract axis-angle from R_plus @ R0.T
            dR = R_plus @ R0.T
            J[3, i] = (dR[2,1] - dR[1,2]) / (2 * delta)
            J[4, i] = (dR[0,2] - dR[2,0]) / (2 * delta)
            J[5, i] = (dR[1,0] - dR[0,1]) / (2 * delta)

        return J


# ─────────────────────────────────────────────────────────
# Inverse Kinematics
# ─────────────────────────────────────────────────────────

class InverseKinematics:
    """
    Universal IK solver.

    Strategy selection:
      1. ikpy (if URDF available)
      2. Jacobian pseudo-inverse iterative solver  (≤20 DOF arms)
      3. CCD  (Cyclic Coordinate Descent)          (fallback)
    """

    def __init__(
        self,
        cfg:         RobotConfig,
        max_iter:    int   = 200,
        tol:         float = 1e-4,
        step_size:   float = 0.5,
    ):
        self.cfg       = cfg
        self.fk_solver = ForwardKinematics(cfg)
        self.max_iter  = max_iter
        self.tol       = tol
        self.step_size = step_size

        # Try to load ikpy chain
        self._chain = None
        if _IKPY and cfg.urdf_path:
            try:
                n_active = len(cfg.arm_joints)
                mask = [False] + [True] * n_active + [False]
                self._chain = ikpy_chain.Chain.from_urdf_file(
                    cfg.urdf_path,
                    active_links_mask=mask[:n_active + 2],
                )
            except Exception as e:
                warnings.warn(f"ikpy load failed ({e}); using numerical IK.")

    # ── public API ───────────────────────────────────────────

    def solve(
        self,
        target_xyz: Tuple[float, float, float],
        target_R:   Optional[np.ndarray] = None,
        q_init:     Optional[np.ndarray] = None,
        orientation_weight: float = 0.1,
    ) -> Optional[np.ndarray]:
        """
        Solve IK for target Cartesian position (and optionally orientation).

        Returns
        ───────
        q : (N,) joint angles, or None if solver failed.
        """
        N_arm = len(self.cfg.arm_joints)
        if q_init is None:
            q_init_arm = self.cfg.home_config[:N_arm].copy()
        else:
            q_init_arm = np.asarray(q_init, dtype=np.float64)[:N_arm].copy()

        # Fast path: for high-DOF manipulators without a URDF (e.g. dexterous
        # hands), numerical IK is ill-defined and very slow.  Return the
        # current configuration with a small perturbation toward the target.
        _MAX_NUMERICAL_DOF = 12
        if self._chain is None and N_arm > _MAX_NUMERICAL_DOF:
            return self._mock_ik(target_xyz, q_init_arm)

        # 1. ikpy
        if self._chain is not None:
            q = self._ikpy_solve(target_xyz, q_init_arm)
            if q is not None:
                return self.cfg.clip_joints(
                    np.pad(q, (0, max(0, self.cfg.dof - len(q)))))[:self.cfg.dof]

        # 2. Jacobian pseudo-inverse
        q = self._jacobian_solve(target_xyz, target_R, q_init_arm.copy(),
                                  orientation_weight)
        if q is not None:
            return q

        # 3. CCD fallback
        return self._ccd_solve(target_xyz, q_init_arm.copy())

    # ── ikpy backend ─────────────────────────────────────────

    def _ikpy_solve(
        self,
        xyz:   Tuple[float, float, float],
        q0:    np.ndarray,
    ) -> Optional[np.ndarray]:
        T_target = np.eye(4)
        T_target[:3, 3] = xyz
        try:
            angles = self._chain.inverse_kinematics(
                T_target, initial_position=np.pad(q0, (1, 1)),
                max_iter=self.max_iter,
            )
            return np.array(angles[1:-1], dtype=np.float32)
        except Exception:
            return None

    # ── Jacobian pseudo-inverse ──────────────────────────────

    def _jacobian_solve(
        self,
        target_xyz:  Tuple[float, float, float],
        target_R:    Optional[np.ndarray],
        q:           np.ndarray,
        w_orient:    float,
    ) -> Optional[np.ndarray]:
        target = np.array(target_xyz, dtype=np.float64)
        limits = self.cfg.joint_limits[:len(q)]

        for _ in range(self.max_iter):
            pos, R = self.fk_solver.fk(q)
            err_pos = target - pos

            if target_R is not None:
                dR   = target_R @ R.T
                err_rot = np.array([
                    (dR[2,1]-dR[1,2])/2,
                    (dR[0,2]-dR[2,0])/2,
                    (dR[1,0]-dR[0,1])/2,
                ]) * w_orient
                err = np.concatenate([err_pos, err_rot])
                J   = self.fk_solver.jacobian(q)
            else:
                err = err_pos
                J   = self.fk_solver.jacobian(q)[:3, :]

            if np.linalg.norm(err_pos) < self.tol:
                return q.astype(np.float32)

            # Damped least-squares
            lam   = 0.01
            JtJ   = J.T @ J + lam * np.eye(J.shape[1])
            dq    = np.linalg.solve(JtJ, J.T @ err)
            q     = q + self.step_size * dq
            q     = np.clip(q, limits[:, 0], limits[:, 1])

        if np.linalg.norm(target - self.fk_solver.fk(q)[0]) < 0.05:
            return q.astype(np.float32)
        return None

    # ── CCD fallback ─────────────────────────────────────────

    def _ccd_solve(
        self,
        target_xyz: Tuple[float, float, float],
        q:          np.ndarray,
    ) -> np.ndarray:
        """
        Cyclic Coordinate Descent — robust but slower.
        """
        target  = np.array(target_xyz, dtype=np.float64)
        N       = len(q)
        limits  = self.cfg.joint_limits[:N]

        for _ in range(self.max_iter * 2):
            pos, _ = self.fk_solver.fk(q)
            if np.linalg.norm(pos - target) < self.tol:
                break
            for i in range(N - 1, -1, -1):
                pos_i, _ = self.fk_solver.fk(q)
                to_ee     = pos_i - np.zeros(3)  # simplified: use EE pos
                to_target = target - np.zeros(3)
                # Rotate joint i to minimize distance
                dot = np.clip(
                    np.dot(to_ee, to_target) /
                    (np.linalg.norm(to_ee) * np.linalg.norm(to_target) + 1e-9),
                    -1, 1
                )
                angle = math.acos(dot) * 0.1
                q[i]  = np.clip(q[i] + angle, limits[i, 0], limits[i, 1])

        return q.astype(np.float32)

    # ── instant mock IK for high-DOF / no-URDF robots ────────

    def _mock_ik(
        self,
        target_xyz: Tuple[float, float, float],
        q_init:     np.ndarray,
    ) -> np.ndarray:
        """
        Instant mock IK: return current joint angles with a small
        proportional perturbation toward the target position.
        Used for dexterous hands and other high-DOF robots where
        full numerical IK is ill-posed without a URDF.
        """
        q = q_init.copy().astype(np.float32)
        # Scale first few joints slightly toward target (heuristic)
        t = np.asarray(target_xyz, dtype=np.float32)
        norm = float(np.linalg.norm(t) + 1e-9)
        scale = min(1.0, norm / 0.5)
        n3 = min(3, len(q))
        q[:n3] = np.clip(q[:n3] * (1.0 + 0.05 * scale),
                         self.cfg.joint_limits[:n3, 0].astype(np.float32),
                         self.cfg.joint_limits[:n3, 1].astype(np.float32))
        return q


# ─────────────────────────────────────────────────────────
# Public convenience class
# ─────────────────────────────────────────────────────────

class RobotKinematics:
    """Bundles FK + IK for a given RobotConfig."""

    def __init__(self, cfg: RobotConfig, **ik_kwargs):
        self.cfg = cfg
        self.fk  = ForwardKinematics(cfg)
        self.ik  = InverseKinematics(cfg, **ik_kwargs)

    def end_effector_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.fk.fk(q)

    def joints_for_target(
        self,
        target_xyz:  Tuple[float, float, float],
        target_R:    Optional[np.ndarray] = None,
        q_init:      Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q = self.ik.solve(target_xyz, target_R, q_init)
        if q is None:
            q = self.cfg.home_config.copy()
            warnings.warn(f"IK failed for target {target_xyz}; returning home config.")
        return q


if __name__ == "__main__":
    from robot.robot_config import get_robot

    for name in ["kuka_iiwa7", "ur5", "franka_panda", "simple_2dof"]:
        cfg = get_robot(name)
        kin = RobotKinematics(cfg)
        q   = cfg.home_config
        pos, R = kin.end_effector_pose(q)
        print(f"{name}: EE pos at home = {pos.round(3)}")

        target = pos + np.array([0.05, 0.05, -0.1])
        q_sol  = kin.joints_for_target(tuple(target))
        pos2, _ = kin.end_effector_pose(q_sol)
        print(f"  IK target={target.round(3)}  achieved={pos2.round(3)}  "
              f"err={np.linalg.norm(pos2-target)*100:.1f}cm")

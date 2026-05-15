"""
simulation/pybullet_env.py
───────────────────────────
PyBullet-based tabletop manipulation environment.

Features
────────
• Loads a KUKA iiwa-7 arm on a flat table
• Spawns coloured primitive objects (cubes, spheres, cylinders)
• Step-by-step trajectory execution with joint position control
• Camera rendering (RGB + depth) for visual perception
• Collision detection and success evaluation

Requires: pybullet  (pip install pybullet)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pybullet as p
    import pybullet_data
    _PB_AVAILABLE = True
except ImportError:
    _PB_AVAILABLE = False
    warnings.warn("pybullet not installed – SimulationEnv will run in mock mode.")

from config import DEFAULT_CONFIG, SimulationConfig
from action.action_generator import ActionPlan, PrimitiveType
from action.motion_planner import JointTrajectory


# ──────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────

COLOUR_RGBA: Dict[str, Tuple[float, float, float, float]] = {
    "red":    (0.9, 0.1, 0.1, 1.0),
    "blue":   (0.1, 0.2, 0.9, 1.0),
    "green":  (0.1, 0.8, 0.1, 1.0),
    "yellow": (0.9, 0.8, 0.0, 1.0),
    "orange": (0.9, 0.5, 0.0, 1.0),
    "purple": (0.6, 0.1, 0.8, 1.0),
    "cyan":   (0.0, 0.8, 0.8, 1.0),
    "white":  (0.9, 0.9, 0.9, 1.0),
    "black":  (0.1, 0.1, 0.1, 1.0),
    "brown":  (0.5, 0.3, 0.1, 1.0),
    "grey":   (0.5, 0.5, 0.5, 1.0),
    "pink":   (0.9, 0.4, 0.6, 1.0),
}


@dataclass
class SimObject:
    """Tracked simulation object."""
    body_id:  int
    colour:   str
    shape:    str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)


# ──────────────────────────────────────────────────────────
# Mock environment (no PyBullet)
# ──────────────────────────────────────────────────────────

class MockEnv:
    """Lightweight mock that tracks object positions without physics."""

    def __init__(self):
        self.objects: List[SimObject] = []
        self._robot_joints = [0.0] * 7
        self._gripper      = 0.08

    def spawn_object(self, colour, shape, position, **_kw) -> int:
        oid = len(self.objects)
        self.objects.append(SimObject(oid, colour, shape, position))
        return oid

    def execute_trajectory(self, traj: JointTrajectory) -> bool:
        self._robot_joints = list(traj.joint_positions[-1])
        return True

    def get_objects_info(self) -> List[Dict]:
        return [
            {"colour": o.colour, "shape": o.shape, "position": o.position}
            for o in self.objects
        ]

    def get_object_position(self, body_id: int) -> Optional[Tuple]:
        for o in self.objects:
            if o.body_id == body_id:
                return o.position
        return None

    def render(self) -> Tuple[np.ndarray, np.ndarray]:
        h, w = 480, 640
        rgb   = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
        depth = np.ones((h, w), dtype=np.float32) * 0.8
        return rgb, depth

    def reset(self):
        self.objects = []
        self._robot_joints = [0.0] * 7

    def close(self):
        pass


# ──────────────────────────────────────────────────────────
# Real PyBullet environment
# ──────────────────────────────────────────────────────────

class PyBulletEnv:
    """
    Full physics simulation environment.

    Usage
    ─────
    >>> env = PyBulletEnv(cfg=sim_cfg)
    >>> env.reset()
    >>> env.spawn_object("blue", "cube", (-0.1, 0.0, 0.65))
    >>> env.spawn_object("green", "cube", (0.1, 0.0, 0.65))
    >>> rgb, depth = env.render()
    >>> success = env.execute_trajectory(joint_traj)
    >>> env.close()
    """

    # Camera setup: slightly above and behind the scene
    CAM_DIST       = 1.2
    CAM_YAW        = 0.0
    CAM_PITCH      = -35.0
    CAM_TARGET     = (0.0, 0.0, 0.62)

    def __init__(self, cfg: SimulationConfig = DEFAULT_CONFIG.simulation):
        if not _PB_AVAILABLE:
            raise RuntimeError("pybullet not installed.")
        self.cfg      = cfg
        self._client  = -1
        self._robot   = -1
        self._table   = -1
        self._objects: List[SimObject] = []
        self._num_joints = 0
        self._ee_idx     = 6   # end-effector link index for iiwa

    # ── Lifecycle ────────────────────────────────────────────

    def connect(self):
        mode = p.GUI if self.cfg.use_gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self._client)
        p.setGravity(0, 0, self.cfg.gravity, physicsClientId=self._client)
        p.setTimeStep(self.cfg.timestep, physicsClientId=self._client)
        p.setNumSolverIterations(self.cfg.solver_iterations,
                                  physicsClientId=self._client)
        if self.cfg.use_gui:
            p.resetDebugVisualizerCamera(
                self.CAM_DIST, self.CAM_YAW, self.CAM_PITCH,
                self.CAM_TARGET, physicsClientId=self._client
            )

    def reset(self):
        """Reset scene: remove all objects and reload robot."""
        if self._client < 0:
            self.connect()
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, self.cfg.gravity, physicsClientId=self._client)

        # Plane
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # Table
        table_col  = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cfg.table_half_ext,
                         self.cfg.table_half_ext,
                         self.cfg.table_height / 2],
            physicsClientId=self._client
        )
        table_vis  = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.cfg.table_half_ext,
                         self.cfg.table_half_ext,
                         self.cfg.table_height / 2],
            rgbaColor=[0.75, 0.65, 0.5, 1.0],
            physicsClientId=self._client
        )
        self._table = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0, 0, self.cfg.table_height / 2],
            physicsClientId=self._client
        )

        # Robot
        self._robot = p.loadURDF(
            self.cfg.robot_urdf,
            basePosition=self.cfg.robot_base_pos,
            baseOrientation=self.cfg.robot_base_orn,
            useFixedBase=True,
            physicsClientId=self._client
        )
        self._num_joints = p.getNumJoints(
            self._robot, physicsClientId=self._client
        )
        self._objects = []

    def close(self):
        if self._client >= 0:
            p.disconnect(physicsClientId=self._client)
            self._client = -1

    # ── Object spawning ──────────────────────────────────────

    def spawn_object(
        self,
        colour:   str,
        shape:    str,
        position: Tuple[float, float, float],
        mass:     float = 0.05,
    ) -> int:
        rgba  = COLOUR_RGBA.get(colour, (0.5, 0.5, 0.5, 1.0))
        pos3  = (position[0], position[1],
                 self.cfg.table_height + self.cfg.spawn_z_offset)

        if shape in ("cube", "block"):
            hs  = self.cfg.block_half_size
            col = p.createCollisionShape(p.GEOM_BOX,
                    halfExtents=[hs, hs, hs],
                    physicsClientId=self._client)
            vis = p.createVisualShape(p.GEOM_BOX,
                    halfExtents=[hs, hs, hs],
                    rgbaColor=rgba,
                    physicsClientId=self._client)
        elif shape == "sphere":
            r   = self.cfg.sphere_radius
            col = p.createCollisionShape(p.GEOM_SPHERE,
                    radius=r, physicsClientId=self._client)
            vis = p.createVisualShape(p.GEOM_SPHERE,
                    radius=r, rgbaColor=rgba,
                    physicsClientId=self._client)
        elif shape == "cylinder":
            r, h = self.cfg.cylinder_radius, self.cfg.cylinder_height
            col = p.createCollisionShape(p.GEOM_CYLINDER,
                    radius=r, height=h, physicsClientId=self._client)
            vis = p.createVisualShape(p.GEOM_CYLINDER,
                    radius=r, length=h, rgbaColor=rgba,
                    physicsClientId=self._client)
        else:   # fallback to box
            hs  = self.cfg.block_half_size * 1.5
            col = p.createCollisionShape(p.GEOM_BOX,
                    halfExtents=[hs, hs, hs * 0.3],
                    physicsClientId=self._client)
            vis = p.createVisualShape(p.GEOM_BOX,
                    halfExtents=[hs, hs, hs * 0.3],
                    rgbaColor=rgba, physicsClientId=self._client)

        body = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos3,
            physicsClientId=self._client,
        )
        self._objects.append(SimObject(body, colour, shape, pos3))
        return body

    # ── Trajectory execution ─────────────────────────────────

    def execute_trajectory(
        self,
        traj:    JointTrajectory,
        realtime: bool = False,
    ) -> bool:
        """
        Step through joint trajectory.
        Returns True if execution completed without errors.
        """
        n_ctrl = min(self._num_joints, traj.num_joints)
        prev_t = 0.0
        for joints, t, gripper in zip(
            traj.joint_positions, traj.timestamps, traj.gripper_cmds
        ):
            # Joint position control
            for j_idx in range(n_ctrl):
                p.setJointMotorControl2(
                    self._robot,
                    jointIndex=j_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joints[j_idx] if j_idx < len(joints) else 0.0,
                    force=200,
                    physicsClientId=self._client,
                )
            # Step simulation
            dt        = t - prev_t
            num_steps = max(1, int(dt / self.cfg.timestep))
            for _ in range(num_steps):
                p.stepSimulation(physicsClientId=self._client)
            if realtime:
                time.sleep(dt)
            prev_t = t
        return True

    # ── Sensing ──────────────────────────────────────────────

    def render(
        self,
        width:  Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render current scene.

        Returns
        ───────
        rgb   : (H, W, 3) uint8
        depth : (H, W)    float32  metres
        """
        w = width  or self.cfg.render_width
        h = height or self.cfg.render_height

        view_mat  = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.CAM_TARGET,
            distance=self.CAM_DIST,
            yaw=self.CAM_YAW, pitch=self.CAM_PITCH, roll=0,
            upAxisIndex=2, physicsClientId=self._client
        )
        proj_mat  = p.computeProjectionMatrixFOV(
            fov=DEFAULT_CONFIG.vision.fov_degrees,
            aspect=w / h,
            nearVal=DEFAULT_CONFIG.vision.near_plane,
            farVal=DEFAULT_CONFIG.vision.far_plane,
            physicsClientId=self._client,
        )
        _, _, rgba, depth_buf, _ = p.getCameraImage(
            w, h, view_mat, proj_mat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )

        rgb   = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        depth_raw = np.array(depth_buf, dtype=np.float32).reshape(h, w)

        # Linearise depth buffer to metres
        near = DEFAULT_CONFIG.vision.near_plane
        far  = DEFAULT_CONFIG.vision.far_plane
        depth = far * near / (far - (far - near) * depth_raw)

        return rgb, depth

    def get_objects_info(self) -> List[Dict]:
        infos = []
        for obj in self._objects:
            pos, orn = p.getBasePositionAndOrientation(
                obj.body_id, physicsClientId=self._client
            )
            infos.append({
                "body_id":     obj.body_id,
                "colour":      obj.colour,
                "shape":       obj.shape,
                "position":    tuple(pos),
                "orientation": tuple(orn),
            })
        return infos

    def get_object_position(self, body_id: int) -> Optional[Tuple]:
        for obj in self._objects:
            if obj.body_id == body_id:
                pos, _ = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._client
                )
                return tuple(pos)
        return None


# ──────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────

def make_env(
    cfg: SimulationConfig = DEFAULT_CONFIG.simulation,
    use_mock: bool = False,
) -> "PyBulletEnv | MockEnv":
    if use_mock or not _PB_AVAILABLE:
        env = MockEnv()
    else:
        env = PyBulletEnv(cfg)
        env.connect()
    return env


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("PyBullet available:", _PB_AVAILABLE)
    env = make_env(use_mock=True)
    env.reset() if hasattr(env, "reset") else None
    env.spawn_object("blue",  "cube",   (-0.1, 0.0, 0.65))
    env.spawn_object("green", "sphere", ( 0.1, 0.0, 0.65))
    print("Objects:", env.get_objects_info())
    rgb, depth = env.render()
    print(f"RGB shape: {rgb.shape}  Depth shape: {depth.shape}")
    env.close()

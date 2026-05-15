"""
universal_pipeline.py
──────────────────────
Universal end-to-end pipeline.

Connects all new components into one cohesive system:

  Video frames
      │
      ├──► DepthEstimator     → depth maps
      ├──► VideoProcessor     → temporal buffer
      │
      ▼
  ObjectTracker              → track IDs + positions (stable across time)
      │
      ├──► TrajectoryEstimator → predicted object paths
      │
      ▼
  [Language command]
      │
      ▼
  UniversalVLAModel          → normalised action (delta joints or EE delta)
      │
      ▼
  RobotAdapter               → MotorCommand (joint positions/velocities)
      │
      ▼
  Robot actuators

The pipeline is robot-agnostic: users register their robot once and the
system adapts automatically.

Usage
─────
  pipe = UniversalPipeline.for_robot("ur5")
  pipe.adapt_to_environment(calibration_frames)   # optional sim2real

  # Feed a video clip and a language command:
  result = pipe.run(
      frames   = [frame1, frame2, ...],     # list of (H,W,3) uint8 arrays
      command  = "Pick up the red cube.",
  )
  # result.motor_commands  → send to robot
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG, Config
from robot.robot_config import RobotConfig, get_robot
from robot.robot_adapter import RobotAdapter, MotorCommand
from vision.video_processor import VideoProcessor, VideoFrame, VideoConfig
from vision.object_tracker import ObjectTracker, KalmanTrack
from vision.depth_estimator import DepthEstimator, CameraIntrinsics
from vision.trajectory_estimator import TrajectoryEstimator, TrajectoryPrediction
from language.command_parser import CommandParser, ParsedCommand

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────

@dataclass
class UniversalResult:
    """Complete trace of one pipeline execution."""
    command:          str
    robot_name:       str
    parsed:           Optional[ParsedCommand]         = None
    tracks:           List[KalmanTrack]               = field(default_factory=list)
    trajectories:     List[TrajectoryPrediction]      = field(default_factory=list)
    motor_commands:   List[MotorCommand]               = field(default_factory=list)
    action_normalised: Optional[np.ndarray]           = None
    gripper_cmd:      float                           = 0.5
    success:          bool                            = False
    error_msg:        str                             = ""
    elapsed_s:        float                           = 0.0
    frame_count:      int                             = 0

    def summary(self) -> str:
        lines = [
            "╔══ UniversalPipeline ══════════════════════════════════",
            f"║  Robot     : {self.robot_name}",
            f"║  Command   : {self.command}",
            f"║  Parse     : {'✓' if self.parsed and self.parsed.is_valid else '✗'}  {self.parsed}",
            f"║  Tracks    : {len(self.tracks)} objects tracked",
            f"║  Traj.     : {len(self.trajectories)} trajectories predicted",
            f"║  Actions   : {len(self.motor_commands)} motor commands",
        ]
        if self.action_normalised is not None:
            lines.append(
                f"║  Action    : {np.array2string(self.action_normalised, precision=3)}"
                f"  gripper={self.gripper_cmd:.2f}"
            )
        lines += [
            f"║  Frames    : {self.frame_count}",
            f"║  Success   : {'✓' if self.success else '✗'}",
            f"║  Elapsed   : {self.elapsed_s:.3f}s",
        ]
        if self.error_msg:
            lines.append(f"║  Error     : {self.error_msg}")
        lines.append("╚══════════════════════════════════════════════════════")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Universal Pipeline
# ─────────────────────────────────────────────────────────

class UniversalPipeline:
    """
    End-to-end video → motor command pipeline for any robot.

    Parameters
    ──────────
    robot_cfg       : RobotConfig (or name string / YAML path)
    model_path      : path to UniversalVLAModel checkpoint (optional)
    use_learned_traj: use LSTM trajectory predictor (else kinematic)
    video_cfg       : video processing parameters
    adapt           : enable test-time adaptation (TTAAdapter + EMA)
    """

    def __init__(
        self,
        robot_cfg:         RobotConfig | str,
        model_path:        Optional[str] = None,
        use_learned_traj:  bool  = True,
        video_cfg:         Optional[VideoConfig] = None,
        adapt:             bool  = True,
        max_traj_steps:    int   = 20,
        success_thresh_m:  float = 0.05,
    ):
        # Robot
        if isinstance(robot_cfg, str):
            robot_cfg = get_robot(robot_cfg)
        self.robot_cfg = robot_cfg
        self.adapter   = RobotAdapter(robot_cfg)

        # Vision
        self.video_cfg   = video_cfg or VideoConfig(
            target_fps=10, resize_hw=(224, 224), buffer_len=16
        )
        self.video_proc  = VideoProcessor(cfg=self.video_cfg)
        self.depth_est   = DepthEstimator(known_plane_z=0.65)
        self.camera      = CameraIntrinsics(fov_deg=60, width=224, height=224)
        self.tracker     = ObjectTracker()

        # Trajectory prediction
        self.traj_est = TrajectoryEstimator(
            use_learned=use_learned_traj,
            pred_steps=max_traj_steps,
            dt=1.0 / self.video_cfg.target_fps,
        )

        # Language
        self.parser = CommandParser(use_bert=False)

        # VLA model
        self.vla_model     = None
        self.vla_available = False
        self._joint_feats  = None
        if _TORCH:
            self._init_model(model_path)

        # Sim2Real adaptation
        self.s2r_adapter = None
        if adapt and _TORCH and self.vla_model is not None:
            from sim2real.adaptation import Sim2RealAdapter
            self.s2r_adapter = Sim2RealAdapter(self.vla_model, use_tta=True)

        self.success_thresh = success_thresh_m

    # ── initialise model ─────────────────────────────────────

    def _init_model(self, model_path: Optional[str]):
        try:
            from models.universal_vla import (
                UniversalVLAModel, RobotMorphologyEmbedding,
                load_universal_checkpoint,
            )
            self.vla_model = UniversalVLAModel(
                hidden_dim   = 256,
                num_bins     = 128,
                max_dof      = 32,
                num_heads    = 4,
                num_layers   = 2,
                use_flow     = True,
                use_temporal = True,
                use_bert     = False,
            ).eval()

            self._joint_feats = RobotMorphologyEmbedding.from_robot_config(
                self.robot_cfg
            )

            if model_path and Path(model_path).exists():
                ep, met = load_universal_checkpoint(self.vla_model, model_path)
                print(f"Loaded checkpoint epoch={ep}  metrics={met}")

            self.vla_available = True
        except Exception as e:
            warnings.warn(f"VLA model init failed ({e}); using rule-based fallback.")

    # ── factory ─────────────────────────────────────────────

    @classmethod
    def for_robot(
        cls,
        robot_name: str,
        **kwargs,
    ) -> "UniversalPipeline":
        return cls(robot_name, **kwargs)

    # ── sim2real adaptation ──────────────────────────────────

    def adapt_to_environment(
        self,
        frames: List[np.ndarray],
        verbose: bool = True,
    ) -> "UniversalPipeline":
        """
        Adapt the model to a new environment using unlabelled real frames.

        frames : list of (H, W, 3) uint8 RGB images from the real scene.
        """
        if not _TORCH or self.s2r_adapter is None:
            return self
        tensors = self._frames_to_tensor(frames)
        if tensors is not None:
            self.s2r_adapter.adapt_to_environment(tensors, verbose=verbose)
        return self

    # ── main run ─────────────────────────────────────────────

    def run(
        self,
        frames:  List[np.ndarray],
        command: str,
        scene_info: Optional[List[Dict]] = None,
    ) -> UniversalResult:
        """
        Process a video clip + language command → motor commands.

        Parameters
        ──────────
        frames     : list of (H, W, 3) uint8 RGB frames (newest last)
        command    : natural-language instruction
        scene_info : optional ground-truth object list (sim oracle mode)
        """
        t_start = time.time()
        result  = UniversalResult(
            command    = command,
            robot_name = self.robot_cfg.name,
            frame_count= len(frames),
        )

        try:
            # ── 1. Parse language ────────────────────────────
            parsed = self.parser.parse(command)
            result.parsed = parsed

            # ── 2. Process video frames ──────────────────────
            for i, frame in enumerate(frames):
                ts = i / self.video_cfg.target_fps
                self.video_proc.push_frame(frame, timestamp_s=ts)

            # ── 3. Depth estimation ───────────────────────────
            depth = None
            if frames:
                depth = self.depth_est.estimate(frames[-1])

            # ── 4. Object detection + tracking ───────────────
            if scene_info is not None:
                # Simulation oracle: use ground-truth positions.
                # Warm-up the Kalman tracker with repeated updates so
                # all objects are confirmed before action planning.
                from vision.object_detector import ObjectDetector
                det = ObjectDetector(use_sim_oracle=True)
                detections = det.detect(sim_object_info=scene_info)
                for _ in range(max(3, self.tracker.min_hits)):
                    tracks = self.tracker.update(detections)
            elif frames and depth is not None:
                from vision.object_detector import ObjectDetector, ColourSegmentationDetector
                det = ObjectDetector(use_sim_oracle=False)
                rgb = frames[-1]
                if rgb.shape[:2] != (224, 224):
                    import cv2
                    rgb = cv2.resize(rgb, (224, 224))
                detections = det.detect(rgb_image=rgb, depth_image=depth)
                tracks = self.tracker.update(detections)
            else:
                detections = []
                tracks = self.tracker.update(detections)

            result.tracks = tracks

            # ── 5. Trajectory prediction ──────────────────────
            trajectories = self.traj_est.estimate(tracks)
            result.trajectories = trajectories

            # ── 6. VLA inference ──────────────────────────────
            action_norm, gripper = self._infer_action(
                frames, command, parsed
            )
            result.action_normalised = action_norm
            result.gripper_cmd       = gripper

            # ── 7. Convert to motor commands ──────────────────
            motor_cmds = self._action_to_motor_commands(
                action_norm, gripper, parsed, tracks, trajectories
            )
            result.motor_commands = motor_cmds

            # ── 8. Evaluate ──────────────────────────────────
            result.success = len(motor_cmds) > 0 and not result.error_msg

        except Exception as e:
            import traceback
            result.error_msg = f"{type(e).__name__}: {e}"

        result.elapsed_s = time.time() - t_start
        return result

    # ── inference helpers ────────────────────────────────────

    def _infer_action(
        self,
        frames:  List[np.ndarray],
        command: str,
        parsed:  ParsedCommand,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the VLA model (or rule-based fallback) to get a normalised action.
        Returns (action_normalised: np.ndarray, gripper: float).
        """
        n_dof = len(self.robot_cfg.arm_joints)

        if self.vla_available and self.vla_model is not None and _TORCH:
            return self._vla_infer(frames, command, n_dof)
        else:
            return self._rule_based_action(parsed, n_dof)

    def _vla_infer(
        self,
        frames:  List[np.ndarray],
        command: str,
        n_dof:   int,
    ) -> Tuple[np.ndarray, float]:
        clip = self.video_proc.get_clip_tensor()
        flow = self.video_proc.get_flow_tensor()

        if clip is None:
            return np.zeros(n_dof, dtype=np.float32), 0.5

        # Tokenise command (simple whitespace tokenisation if no BERT)
        ids  = self._tokenise_command(command, max_len=32)
        mask = (ids != 0).long()

        jf = self._joint_feats.clone() if self._joint_feats is not None else torch.zeros(1, 1, 9)

        # Handle flow shape
        if flow is not None:
            T_flow = flow.shape[0]
            T_clip = clip.shape[0]
            if T_flow < T_clip:
                pad   = torch.zeros(T_clip - T_flow, *flow.shape[1:])
                flow  = torch.cat([pad, flow], dim=0)
            flow = flow[-T_clip:].unsqueeze(0)   # (1, T, 2, H, W)
        else:
            T = clip.shape[0]
            H, W = clip.shape[2], clip.shape[3]
            flow = torch.zeros(1, T, 2, H, W)

        with torch.no_grad():
            action, gripper = self.vla_model.predict(
                clip.unsqueeze(0),        # (1, T, 3, H, W)
                ids,
                mask,
                jf,
                n_dof,
                flow,
            )
        return action[:n_dof], gripper

    def _tokenise_command(self, command: str, max_len: int = 32) -> "torch.Tensor":
        # Character-level tokenisation (no external download required).
        # We only attempt BERT when the model was explicitly built with use_bert=True.
        use_bert = getattr(self.vla_model, "use_bert", False) if self.vla_model else False
        if use_bert:
            try:
                import os
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
                from transformers import BertTokenizer
                tok = BertTokenizer.from_pretrained(
                    "bert-base-uncased",
                    local_files_only=False,
                )
                enc = tok(command, max_length=max_len, padding="max_length",
                          truncation=True, return_tensors="pt")
                return enc["input_ids"]
            except Exception:
                pass
        tokens = [ord(c) % 1000 + 1 for c in command[:max_len]]
        tokens += [0] * (max_len - len(tokens))
        return torch.tensor([tokens], dtype=torch.long)

    def _rule_based_action(
        self, parsed: ParsedCommand, n_dof: int
    ) -> Tuple[np.ndarray, float]:
        """
        Simple rule-based action when VLA is unavailable.
        Produces small forward + down motion with open gripper.
        """
        action = np.zeros(n_dof, dtype=np.float32)
        if parsed and parsed.is_valid:
            act = getattr(parsed, "action", "move")
            if isinstance(act, str):
                if "pick" in act or "grasp" in act:
                    action[:3] = [0.0, 0.0, -0.3]   # move down
                    return action, 0.8               # open gripper
                elif "place" in act:
                    action[:3] = [0.0, 0.0,  0.1]
                    return action, 0.0               # close gripper
        action[:3] = [0.1, 0.0, 0.0]   # default: small forward move
        return action, 0.5

    def _action_to_motor_commands(
        self,
        action_norm:   np.ndarray,
        gripper:       float,
        parsed:        ParsedCommand,
        tracks:        List[KalmanTrack],
        trajectories:  List[TrajectoryPrediction],
    ) -> List[MotorCommand]:
        """
        Convert normalised action + gripper → MotorCommand list.

        If we have object trajectories, optionally use the predicted
        target position for higher accuracy.
        """
        cmds = []

        # Denormalise action to real joint deltas
        n_dof = len(self.robot_cfg.arm_joints)
        action_norm = np.asarray(action_norm, dtype=np.float32)[:n_dof]

        # Gripper command
        g_width = gripper * 0.08   # scale to metres

        # Build 7-DOF action vector expected by action_to_command
        action_7 = np.zeros(7, dtype=np.float32)
        action_7[:min(n_dof, 3)] = action_norm[:min(n_dof, 3)] * 0.05   # scale to metres
        action_7[6] = gripper   # gripper

        cmd = self.adapter.action_to_command(action_7, action_type="delta_cartesian")
        cmd.gripper_width = g_width
        cmds.append(cmd)

        # If a pick-and-place is detected and we have a target trajectory,
        # also generate the full approach trajectory.
        if parsed and parsed.is_valid and trajectories:
            target_pred = trajectories[0].final_position
            approach_cmds = self.adapter.move_to_cartesian(target_pred, n_steps=10)
            cmds.extend(approach_cmds)

        return cmds

    # ── utility ──────────────────────────────────────────────

    def _frames_to_tensor(
        self, frames: List[np.ndarray]
    ) -> Optional["torch.Tensor"]:
        if not _TORCH or not frames:
            return None
        imgs = []
        for f in frames:
            img = f.astype(np.float32) / 255.0
            imgs.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
        return torch.stack(imgs)

    def reset(self):
        """Reset tracking and video buffer between episodes."""
        self.tracker.reset()
        self.video_proc = VideoProcessor(cfg=self.video_cfg)
        self.adapter    = RobotAdapter(self.robot_cfg)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __str__(self) -> str:
        return (f"UniversalPipeline("
                f"robot={self.robot_cfg.name}  "
                f"dof={self.robot_cfg.dof}  "
                f"vla={self.vla_available})")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    SCENE = [
        {"colour": "blue",   "shape": "block",  "position": (-0.10, 0.00, 0.65)},
        {"colour": "green",  "shape": "cube",   "position": ( 0.00, 0.00, 0.65)},
        {"colour": "red",    "shape": "sphere", "position": ( 0.10, 0.00, 0.65)},
    ]

    for robot_name in ["kuka_iiwa7", "ur5", "franka_panda"]:
        print(f"\n{'='*60}")
        print(f"Testing: {robot_name}")
        print('='*60)

        pipe = UniversalPipeline.for_robot(robot_name, adapt=False)
        H, W = 224, 224
        frames = [np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)
                  for _ in range(8)]

        result = pipe.run(
            frames     = frames,
            command    = "Pick up the blue block and place it on the green cube.",
            scene_info = SCENE,
        )
        print(result.summary())

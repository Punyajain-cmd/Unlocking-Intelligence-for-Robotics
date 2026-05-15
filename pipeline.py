"""
pipeline.py
────────────
End-to-end pipeline orchestrator.

Connects every module:
  CommandParser → ObjectDetector → SceneGraph → ActionGenerator
  → MotionPlanner → SimulationEnv → Metrics

Usage
─────
  from pipeline import RoboLangPipeline

  pipe = RoboLangPipeline(use_mock=True)
  result = pipe.run("Move the blue block to the right of the green cube.")
  print(result.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG, Config
from language.command_parser import CommandParser, ParsedCommand
from vision.object_detector import ObjectDetector, DetectedObject
from vision.scene_graph import SceneGraph
from action.action_generator import ActionGenerator, ActionPlan
from action.motion_planner import MotionPlanner, JointTrajectory
from simulation.pybullet_env import make_env
from evaluation.metrics import EpisodeResult, ErrorCategory


# ──────────────────────────────────────────────────────────
# Pipeline result
# ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full trace of one pipeline execution."""
    command:        str
    parsed:         Optional[ParsedCommand]     = None
    scene_objects:  List[DetectedObject]        = field(default_factory=list)
    action_plan:    Optional[ActionPlan]        = None
    trajectory:     Optional[JointTrajectory]  = None
    final_obj_pos:  Optional[Tuple]             = None
    success:        bool                        = False
    error_msg:      str                         = ""
    elapsed_s:      float                       = 0.0

    def summary(self) -> str:
        lines = [
            f"╔══ PipelineResult ══════════════════════════════════════",
            f"║  Command   : {self.command}",
            f"║  Parse     : {'✓' if self.parsed and self.parsed.is_valid else '✗'}  "
            f"{self.parsed}",
            f"║  Objects   : {len(self.scene_objects)} detected",
            f"║  Plan      : {'✓' if self.action_plan and self.action_plan.success else '✗'}  "
            f"{self.action_plan.action_type if self.action_plan else 'N/A'}",
            f"║  Trajectory: "
            f"{self.trajectory.num_steps() if self.trajectory else 0} steps",
            f"║  Final pos : {self.final_obj_pos}",
            f"║  Success   : {'✓' if self.success else '✗'}",
            f"║  Elapsed   : {self.elapsed_s:.3f}s",
        ]
        if self.error_msg:
            lines.append(f"║  Error     : {self.error_msg}")
        lines.append("╚════════════════════════════════════════════════════")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────

class RoboLangPipeline:
    """
    Full NL → action pipeline.

    Parameters
    ──────────
    use_mock : bool
        If True, use MockEnv (no PyBullet required).
    use_sim_oracle : bool
        If True, use ground-truth object positions from the simulator
        instead of camera-based detection.
    cfg : Config
        Configuration object (uses defaults if None).
    """

    def __init__(
        self,
        use_mock:       bool   = True,
        use_sim_oracle: bool   = True,
        cfg:            Config = None,
        success_thresh: float  = 0.05,
    ):
        self.cfg    = cfg or DEFAULT_CONFIG
        self.thresh = success_thresh

        self.parser    = CommandParser(use_bert=False)
        self.detector  = ObjectDetector(use_sim_oracle=use_sim_oracle)
        self.generator = ActionGenerator(self.cfg.action)
        self.planner   = MotionPlanner(self.cfg.action)
        self.env       = make_env(
            cfg=self.cfg.simulation, use_mock=use_mock
        )

        # Active scene state
        self._scene_objects: List[Dict]          = []
        self._detected:      List[DetectedObject] = []

    # ── Scene setup ──────────────────────────────────────────

    def setup_scene(self, objects: List[Dict]) -> None:
        """
        Load objects into the simulator.

        objects : list of dicts  {colour, shape, position}
        """
        if hasattr(self.env, "reset"):
            self.env.reset()
        for obj in objects:
            self.env.spawn_object(
                colour   = obj["colour"],
                shape    = obj["shape"],
                position = obj["position"],
            )
        self._scene_objects = objects

    # ── Single command execution ─────────────────────────────

    def run(
        self,
        command:     str,
        scene_setup: Optional[List[Dict]] = None,
    ) -> PipelineResult:
        """
        Execute one natural-language command end-to-end.

        Parameters
        ──────────
        command     : NL instruction string
        scene_setup : optional list of objects to (re-)load before executing
        """
        t_start = time.time()
        result  = PipelineResult(command=command)

        try:
            # ── 0. Optional scene reset ──────────────────────
            if scene_setup is not None:
                self.setup_scene(scene_setup)

            # ── 1. Parse ─────────────────────────────────────
            parsed = self.parser.parse(command)
            result.parsed = parsed
            if not parsed.is_valid:
                result.error_msg = parsed.error_msg
                result.elapsed_s = time.time() - t_start
                return result

            # ── 2. Perceive ───────────────────────────────────
            sim_info = self.env.get_objects_info() if self._scene_objects else []
            if not sim_info:
                sim_info = self._scene_objects

            detected = self.detector.detect(sim_object_info=sim_info)
            result.scene_objects = detected

            if not detected:
                result.error_msg = "No objects detected in scene."
                result.elapsed_s = time.time() - t_start
                return result

            # ── 3. Build scene graph ──────────────────────────
            scene = SceneGraph().build(detected)

            # ── 4. Generate action plan ───────────────────────
            plan = self.generator.generate(parsed, scene)
            result.action_plan = plan
            if not plan.success:
                result.error_msg = plan.error_msg
                result.elapsed_s = time.time() - t_start
                return result

            # ── 5. Plan trajectory ────────────────────────────
            traj = self.planner.plan(plan)
            result.trajectory = traj

            # ── 6. Execute ────────────────────────────────────
            self.env.execute_trajectory(traj)

            # ── 7. Evaluate outcome ───────────────────────────
            final_pos = self._get_final_position(plan)
            result.final_obj_pos = final_pos

            if plan.target_pos and final_pos:
                err = float(np.linalg.norm(
                    np.array(final_pos) - np.array(plan.target_pos)
                ))
                result.success = err <= self.thresh
            else:
                result.success = True   # grasp-only: success if no error

        except Exception as e:
            result.error_msg = f"Pipeline error: {e}"

        result.elapsed_s = time.time() - t_start
        return result

    def run_batch(
        self,
        commands:    List[str],
        scene_setup: Optional[List[Dict]] = None,
    ) -> List[PipelineResult]:
        """Run multiple commands on the same scene."""
        results = []
        for cmd in commands:
            r = self.run(cmd, scene_setup=scene_setup if not results else None)
            results.append(r)
        return results

    # ── Helpers ──────────────────────────────────────────────

    def _get_final_position(
        self, plan: ActionPlan
    ) -> Optional[Tuple[float, float, float]]:
        """
        Query the simulated final position of the manipulated object.
        Falls back to plan.target_pos + noise for the mock env.
        """
        if plan.subject_obj is None:
            return None

        body_id = getattr(plan.subject_obj, "body_id", None)
        if body_id is not None:
            pos = self.env.get_object_position(body_id)
            if pos:
                return pos

        # Mock approximation: target with small noise
        if plan.target_pos:
            noise = np.random.normal(0, 0.015, 3)
            tp    = np.array(plan.target_pos)
            return tuple(float(v) for v in tp + noise)

        # Grasp-only
        sx, sy, sz = plan.subject_obj.centre_3d
        return (sx, sy, sz + 0.15)

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the RoboLang pipeline.")
    ap.add_argument("--command", type=str,
                    default="Move the blue block to the right of the green cube.")
    ap.add_argument("--render",   action="store_true")
    ap.add_argument("--use_mock", action="store_true", default=True)
    args = ap.parse_args()

    SCENE = [
        {"colour": "blue",  "shape": "block",  "position": (-0.10, 0.00, 0.65)},
        {"colour": "green", "shape": "cube",   "position": ( 0.00, 0.00, 0.65)},
        {"colour": "red",   "shape": "sphere", "position": ( 0.10, 0.00, 0.65)},
        {"colour": "yellow","shape": "block",  "position": ( 0.00,-0.10, 0.65)},
        {"colour": "cyan",  "shape": "cylinder","position":( 0.15, 0.10, 0.65)},
    ]

    with RoboLangPipeline(use_mock=args.use_mock) as pipe:
        pipe.setup_scene(SCENE)
        result = pipe.run(args.command)
        print(result.summary())

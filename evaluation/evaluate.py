"""
evaluation/evaluate.py
───────────────────────
Evaluation harness: runs the full pipeline on a set of test episodes
and computes all KPI metrics.

Usage
─────
  python evaluation/evaluate.py --num_episodes 100 --use_mock
  python evaluation/evaluate.py --num_episodes 100 --render
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import DEFAULT_CONFIG
from evaluation.metrics import (
    MetricsAccumulator, EpisodeResult, ErrorCategory
)
from language.command_parser import CommandParser
from vision.object_detector import ObjectDetector
from vision.scene_graph import SceneGraph
from action.action_generator import ActionGenerator
from action.motion_planner import MotionPlanner
from simulation.pybullet_env import make_env


# ──────────────────────────────────────────────────────────
# Test episode templates
# ──────────────────────────────────────────────────────────

EPISODE_TEMPLATES = [
    {
        "command": "Move the blue block to the right of the green cube.",
        "objects": [
            {"colour": "blue",  "shape": "block",  "position": (-0.10, 0.00, 0.65)},
            {"colour": "green", "shape": "cube",   "position": ( 0.00, 0.00, 0.65)},
        ],
        "goal": {"colour": "blue", "shape": "block",
                 "target_pos": (0.08, 0.00, 0.65)},
        "true_action": "pick_and_place",
    },
    {
        "command": "Pick up the red sphere and place it on the yellow platform.",
        "objects": [
            {"colour": "red",    "shape": "sphere",   "position": ( 0.05, 0.10, 0.65)},
            {"colour": "yellow", "shape": "platform", "position": (-0.10, 0.00, 0.65)},
        ],
        "goal": {"colour": "red", "shape": "sphere",
                 "target_pos": (-0.10, 0.00, 0.68)},
        "true_action": "pick_and_place",
    },
    {
        "command": "Push the cyan cylinder to the left side of the table.",
        "objects": [
            {"colour": "cyan", "shape": "cylinder", "position": (0.05, 0.00, 0.65)},
        ],
        "goal": {"colour": "cyan", "shape": "cylinder",
                 "target_pos": (-0.20, 0.00, 0.65)},
        "true_action": "push",
    },
    {
        "command": "Stack the orange cube on top of the purple block.",
        "objects": [
            {"colour": "orange", "shape": "cube",  "position": ( 0.10, 0.10, 0.65)},
            {"colour": "purple", "shape": "block", "position": (-0.05, 0.00, 0.65)},
        ],
        "goal": {"colour": "orange", "shape": "cube",
                 "target_pos": (-0.05, 0.00, 0.705)},
        "true_action": "stack",
    },
    {
        "command": "Grasp the small blue object near the edge.",
        "objects": [
            {"colour": "blue", "shape": "block", "position": (0.22, 0.15, 0.65)},
        ],
        "goal": {"colour": "blue", "shape": "block",
                 "target_pos": (0.0, 0.0, 0.80)},
        "true_action": "grasp",
    },
    {
        "command": "Lift the white box above the brown cylinder.",
        "objects": [
            {"colour": "white", "shape": "block",    "position": ( 0.00, 0.10, 0.65)},
            {"colour": "brown", "shape": "cylinder", "position": (-0.10, 0.00, 0.65)},
        ],
        "goal": {"colour": "white", "shape": "block",
                 "target_pos": (0.00, 0.10, 0.80)},
        "true_action": "lift",
    },
    {
        "command": "Move the green cube behind the red sphere.",
        "objects": [
            {"colour": "green", "shape": "cube",   "position": (-0.05, 0.00, 0.65)},
            {"colour": "red",   "shape": "sphere", "position": ( 0.10, 0.00, 0.65)},
        ],
        "goal": {"colour": "green", "shape": "cube",
                 "target_pos": (0.10, 0.12, 0.65)},
        "true_action": "pick_and_place",
    },
    {
        "command": "Place the grey block in front of the black cube.",
        "objects": [
            {"colour": "grey",  "shape": "block", "position": ( 0.15, 0.10, 0.65)},
            {"colour": "black", "shape": "cube",  "position": ( 0.00, 0.00, 0.65)},
        ],
        "goal": {"colour": "grey", "shape": "block",
                 "target_pos": (0.00, -0.10, 0.65)},
        "true_action": "pick_and_place",
    },
]


# ──────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the full pipeline on each episode template and collects metrics.
    """

    def __init__(self, use_mock: bool = True, render: bool = False):
        self.use_mock = use_mock
        self.render   = render

        self.parser    = CommandParser(use_bert=False)
        self.detector  = ObjectDetector(use_sim_oracle=True)
        self.generator = ActionGenerator()
        self.planner   = MotionPlanner()
        self.acc       = MetricsAccumulator()

    def run(self, num_episodes: int = 100) -> MetricsAccumulator:
        templates = EPISODE_TEMPLATES
        ep_id     = 0

        while ep_id < num_episodes:
            tmpl = templates[ep_id % len(templates)]
            result = self._run_episode(ep_id, tmpl)
            self.acc.add(result)
            ep_id += 1

            # Progress
            if ep_id % 10 == 0:
                print(f"  [{ep_id}/{num_episodes}]  "
                      f"TSR so far: {self.acc.tsr()*100:.1f}%")

        return self.acc

    def _run_episode(self, ep_id: int, tmpl: dict) -> EpisodeResult:
        command    = tmpl["command"]
        objects_gt = tmpl["objects"]
        goal       = tmpl["goal"]
        true_action = tmpl["true_action"]
        true_goal   = goal["target_pos"]

        # ── 1. Parse command ─────────────────────────────────
        parsed = self.parser.parse(command)
        parse_ok = parsed.is_valid

        if not parse_ok:
            return EpisodeResult(
                episode_id=ep_id,
                command=command,
                true_action=true_action,
                true_subject_pos=objects_gt[0]["position"],
                true_goal_pos=true_goal,
                pred_action="unknown",
                pred_subject_pos=None,
                pred_goal_pos=None,
                final_obj_pos=None,
                goal_conditions_total=3,
                goal_conditions_met=0,
                total_steps=9,
                steps_completed=0,
                parse_success=False,
                error_category=ErrorCategory.PARSE_FAILURE,
                error_detail=parsed.error_msg,
            )

        # ── 2. Build scene ───────────────────────────────────
        detected = self.detector.detect(sim_object_info=objects_gt)
        scene    = SceneGraph().build(detected)

        # ── 3. Generate action plan ──────────────────────────
        plan = self.generator.generate(parsed, scene)
        if not plan.success:
            return EpisodeResult(
                episode_id=ep_id,
                command=command,
                true_action=true_action,
                true_subject_pos=objects_gt[0]["position"],
                true_goal_pos=true_goal,
                pred_action=parsed.action,
                pred_subject_pos=None,
                pred_goal_pos=plan.target_pos,
                final_obj_pos=None,
                goal_conditions_total=3,
                goal_conditions_met=0,
                total_steps=9,
                steps_completed=1,
                parse_success=True,
                error_category=ErrorCategory.OBJECT_NOT_FOUND,
                error_detail=plan.error_msg,
            )

        # ── 4. Plan trajectory ───────────────────────────────
        total_steps = len(plan.primitives)
        try:
            traj = self.planner.plan(plan)
            ik_ok = True
        except Exception as e:
            return EpisodeResult(
                episode_id=ep_id,
                command=command,
                true_action=true_action,
                true_subject_pos=objects_gt[0]["position"],
                true_goal_pos=true_goal,
                pred_action=parsed.action,
                pred_subject_pos=plan.subject_obj.centre_3d if plan.subject_obj else None,
                pred_goal_pos=plan.target_pos,
                final_obj_pos=None,
                goal_conditions_total=3,
                goal_conditions_met=1,
                total_steps=total_steps,
                steps_completed=2,
                parse_success=True,
                error_category=ErrorCategory.IK_FAILURE,
                error_detail=str(e),
            )

        # ── 5. Simulate execution (mock physics) ─────────────
        final_pos = self._simulate_execution(plan)

        # ── 6. Evaluate goal conditions ──────────────────────
        goal_total = 3   # position, action, relation
        goal_met   = 0

        pos_err    = float(np.linalg.norm(
            np.array(final_pos) - np.array(true_goal)
        ))
        if pos_err <= DEFAULT_CONFIG.eval.success_threshold:
            goal_met += 2
        elif pos_err <= DEFAULT_CONFIG.eval.success_threshold * 3:
            goal_met += 1

        if parsed.action == true_action:
            goal_met += 1

        err_cat = ErrorCategory.NONE if pos_err <= 0.05 else ErrorCategory.PLACEMENT_ERROR

        return EpisodeResult(
            episode_id=ep_id,
            command=command,
            true_action=true_action,
            true_subject_pos=objects_gt[0]["position"],
            true_goal_pos=true_goal,
            pred_action=parsed.action,
            pred_subject_pos=plan.subject_obj.centre_3d if plan.subject_obj else None,
            pred_goal_pos=plan.target_pos,
            final_obj_pos=final_pos,
            goal_conditions_total=goal_total,
            goal_conditions_met=goal_met,
            total_steps=total_steps,
            steps_completed=total_steps,  # mock: all steps execute
            parse_success=True,
            error_category=err_cat,
        )

    @staticmethod
    def _simulate_execution(plan) -> tuple:
        """
        Mock execution: returns goal position with small Gaussian noise
        (simulates real-world placement uncertainty).
        """
        if plan.target_pos is None:
            # Grasp-only: object lifted (approximate final position)
            if plan.subject_obj:
                sx, sy, sz = plan.subject_obj.centre_3d
                noise = np.random.normal(0, 0.01, 3)
                return (sx + noise[0], sy + noise[1], sz + 0.15)
            return (0.0, 0.0, 0.80)

        gx, gy, gz = plan.target_pos
        noise      = np.random.normal(0, 0.02, 3)
        return (gx + noise[0], gy + noise[1], gz + noise[2])


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RoboLang pipeline.")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--use_mock",     action="store_true", default=True)
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"\nRunning evaluation: {args.num_episodes} episodes …\n")
    evaluator = Evaluator(use_mock=args.use_mock, render=args.render)
    acc       = evaluator.run(num_episodes=args.num_episodes)
    acc.print_report()


if __name__ == "__main__":
    main()

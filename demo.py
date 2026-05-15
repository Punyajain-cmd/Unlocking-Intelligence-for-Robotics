"""
demo.py
────────
Demonstration script — runs the complete RoboLang + Universal VLA system.

Part A (original): 8 NL manipulation scenarios on the rule-based pipeline.
Part B (new):      Universal VLA demo across 5 different robots with
                   video input → object tracking → motor commands.

Usage
─────
  python demo.py                       # run everything
  python demo.py --command "Move the blue block to the left."
  python demo.py --universal-only      # only the new universal pipeline
  python demo.py --classic-only        # only the original 8-scenario demo
  python demo.py --verbose             # detailed output
  python demo.py --robot shadow_hand   # test one specific robot
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from typing import List

import numpy as np

warnings.filterwarnings("ignore")

from pipeline import RoboLangPipeline, PipelineResult
from evaluation.metrics import MetricsAccumulator, EpisodeResult, ErrorCategory


# ──────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────

def _colour(text: str, code: str) -> str:
    try:
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text

GREEN  = lambda t: _colour(t, "32")
RED    = lambda t: _colour(t, "31")
YELLOW = lambda t: _colour(t, "33")
CYAN   = lambda t: _colour(t, "36")
BOLD   = lambda t: _colour(t, "1")
BAR    = "═" * 68


# ──────────────────────────────────────────────────────────
# Part A: Classic 8-scenario demo (original pipeline)
# ──────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    {
        "title":   "Scenario 1 – Relative placement (right of)",
        "command": "Move the blue block to the right of the green cube.",
        "scene": [
            {"colour": "blue",  "shape": "block", "position": (-0.10, 0.00, 0.65)},
            {"colour": "green", "shape": "cube",  "position": ( 0.00, 0.00, 0.65)},
        ],
        "expected_action": "pick_and_place",
        "goal_pos": (0.08, 0.00, 0.65),
    },
    {
        "title":   "Scenario 2 – Pick and place onto platform",
        "command": "Pick up the red sphere and place it on the yellow platform.",
        "scene": [
            {"colour": "red",    "shape": "sphere",   "position": ( 0.05, 0.10, 0.65)},
            {"colour": "yellow", "shape": "platform", "position": (-0.10, 0.00, 0.65)},
        ],
        "expected_action": "pick_and_place",
        "goal_pos": (-0.10, 0.00, 0.68),
    },
    {
        "title":   "Scenario 3 – Push to side",
        "command": "Push the cyan cylinder to the left side of the table.",
        "scene": [
            {"colour": "cyan", "shape": "cylinder", "position": (0.05, 0.00, 0.65)},
        ],
        "expected_action": "push",
        "goal_pos": (-0.20, 0.00, 0.65),
    },
    {
        "title":   "Scenario 4 – Stack operation",
        "command": "Stack the orange cube on top of the purple block.",
        "scene": [
            {"colour": "orange", "shape": "cube",  "position": ( 0.10, 0.10, 0.65)},
            {"colour": "purple", "shape": "block", "position": (-0.05, 0.00, 0.65)},
        ],
        "expected_action": "stack",
        "goal_pos": (-0.05, 0.00, 0.705),
    },
    {
        "title":   "Scenario 5 – Grasp edge object",
        "command": "Grasp the small blue object near the edge.",
        "scene": [
            {"colour": "blue", "shape": "block", "position": (0.22, 0.15, 0.65)},
        ],
        "expected_action": "grasp",
        "goal_pos": (0.22, 0.15, 0.80),
    },
    {
        "title":   "Scenario 6 – Lift above reference",
        "command": "Lift the white box above the brown cylinder.",
        "scene": [
            {"colour": "white", "shape": "block",    "position": ( 0.00, 0.10, 0.65)},
            {"colour": "brown", "shape": "cylinder", "position": (-0.10, 0.00, 0.65)},
        ],
        "expected_action": "lift",
        "goal_pos": (0.00, 0.10, 0.85),
    },
    {
        "title":   "Scenario 7 – Move behind reference",
        "command": "Move the green cube behind the red sphere.",
        "scene": [
            {"colour": "green", "shape": "cube",   "position": (-0.05, 0.00, 0.65)},
            {"colour": "red",   "shape": "sphere", "position": ( 0.10, 0.00, 0.65)},
        ],
        "expected_action": "pick_and_place",
        "goal_pos": (0.10, 0.12, 0.65),
    },
    {
        "title":   "Scenario 8 – Multi-object scene, selective manipulation",
        "command": "Transfer the grey block to the left of the black cube.",
        "scene": [
            {"colour": "grey",   "shape": "block",    "position": ( 0.15, 0.10, 0.65)},
            {"colour": "black",  "shape": "cube",     "position": ( 0.00, 0.00, 0.65)},
            {"colour": "orange", "shape": "sphere",   "position": (-0.10, 0.10, 0.65)},
            {"colour": "blue",   "shape": "cylinder", "position": ( 0.10,-0.10, 0.65)},
        ],
        "expected_action": "pick_and_place",
        "goal_pos": (-0.08, 0.00, 0.65),
    },
]


def _print_result(idx, scenario, result, verbose):
    title    = scenario["title"]
    cmd      = scenario["command"]
    exp_act  = scenario["expected_action"]
    goal_pos = scenario["goal_pos"]

    ok          = "✓" if result.success else "✗"
    ok_coloured = GREEN(ok) if result.success else RED(ok)

    print(f"\n{BAR}")
    print(BOLD(f"  [{idx}] {title}"))
    print(BAR)
    print(f"  Command    : {cmd}")
    print(f"  Parse      : {result.parsed}")

    if result.parsed and result.parsed.is_valid:
        pred_act = result.parsed.action
        act_ok   = "✓" if pred_act == exp_act else "✗"
        print(f"  Action     : "
              f"{GREEN(pred_act) if pred_act==exp_act else RED(pred_act)}"
              f"  (expected: {exp_act}) {act_ok}")

    if result.action_plan and result.action_plan.success:
        n_prims = len(result.action_plan.primitives)
        print(f"  Primitives : {n_prims} steps  "
              f"| goal={result.action_plan.target_pos}")
        if verbose and result.trajectory:
            print(f"\n  Trajectory ({result.trajectory.num_steps()} waypoints):")
            for j, p in enumerate(result.action_plan.primitives[:6]):
                print(f"    {j+1}. {p}")
            if n_prims > 6:
                print(f"    … (+{n_prims - 6} more)")

    if result.final_obj_pos and goal_pos:
        err = float(np.linalg.norm(
            np.array(result.final_obj_pos) - np.array(goal_pos)
        ))
        print(f"  Pos error  : {err*100:.1f} cm  "
              f"(goal={goal_pos}, final={result.final_obj_pos})")

    print(f"  Outcome    : {ok_coloured} "
          f"{'SUCCESS' if result.success else 'FAILURE'}"
          f"  (Δt={result.elapsed_s*1000:.0f} ms)")
    if result.error_msg:
        print(f"  Error      : {RED(result.error_msg)}")


def run_classic_demo(verbose: bool = False, seed: int = 42) -> List[PipelineResult]:
    np.random.seed(seed)

    print(f"\n{BAR}")
    print(BOLD("  RoboLang – Natural Language Robotic Manipulation Demo"))
    print(f"  Classic Pipeline: {len(DEMO_SCENARIOS)} scenarios")
    print(BAR)

    acc     = MetricsAccumulator()
    results = []

    with RoboLangPipeline(use_mock=True) as pipe:
        for i, scenario in enumerate(DEMO_SCENARIOS, start=1):
            result = pipe.run(
                command=scenario["command"],
                scene_setup=scenario["scene"],
            )
            results.append(result)
            _print_result(i, scenario, result, verbose)

            parse_ok    = result.parsed is not None and result.parsed.is_valid
            pred_action = result.parsed.action if parse_ok else "unknown"
            gpos        = scenario["goal_pos"]
            fpos        = result.final_obj_pos or (0.0, 0.0, 0.0)
            pos_err     = float(np.linalg.norm(np.array(fpos) - np.array(gpos)))
            gmet        = 2 if pos_err <= 0.05 else (1 if pos_err <= 0.15 else 0)
            if pred_action == scenario["expected_action"]:
                gmet += 1

            ep = EpisodeResult(
                episode_id=i,
                command=scenario["command"],
                true_action=scenario["expected_action"],
                true_subject_pos=scenario["scene"][0]["position"],
                true_goal_pos=gpos,
                pred_action=pred_action,
                pred_subject_pos=None,
                pred_goal_pos=result.action_plan.target_pos if result.action_plan else None,
                final_obj_pos=fpos,
                goal_conditions_total=3,
                goal_conditions_met=gmet,
                total_steps=len(result.action_plan.primitives) if result.action_plan else 1,
                steps_completed=len(result.action_plan.primitives) if result.action_plan and result.action_plan.success else 0,
                parse_success=parse_ok,
                error_category=ErrorCategory.NONE if result.success else ErrorCategory.PLACEMENT_ERROR,
                error_detail=result.error_msg,
            )
            acc.add(ep)

    print(f"\n{BAR}")
    print(BOLD("  CLASSIC PIPELINE — KPI SUMMARY"))
    print(BAR)
    n     = len(DEMO_SCENARIOS)
    n_ok  = sum(1 for r in results if r.success)
    print(f"  Scenarios run   : {n}")
    print(f"  Tasks succeeded : {GREEN(str(n_ok))} / {n}  ({n_ok/n*100:.0f}%)")
    print()

    r       = acc.report()
    targets = r["targets"]
    kpis    = [("TSR", "Task Success Rate"),
               ("GCA", "Goal Condition Accuracy"),
               ("CIA", "Command Interpretation Accuracy"),
               ("TCR", "Task Completion Rate"),
               ("EAC", "Error Analysis Coverage")]
    for k, name in kpis:
        val    = r[k]
        target = targets.get(k, 0)
        pct    = f"{val*100:5.1f}%"
        tgt    = f"(target ≥ {target*100:.0f}%)"
        status = GREEN("✓ PASS") if val >= target else YELLOW("~ BELOW")
        print(f"  {k:<5} {name:<35} {pct}  {tgt}  {status}")

    print(f"\n  Mean position error: {r['mean_position_error_m']*100:.1f} cm")
    print()
    print("  Error breakdown:")
    for cat, cnt in r["error_breakdown"].items():
        if cat != "NONE":
            print(f"    {cat:<25} {cnt:3d}")
    print(BAR + "\n")
    return results


# ──────────────────────────────────────────────────────────
# Part B: Universal VLA pipeline demo
# ──────────────────────────────────────────────────────────

UNIVERSAL_SCENARIOS = [
    {
        "robot":   "simple_2dof",
        "command": "Move the blue block to the left.",
        "scene": [
            {"colour": "blue",  "shape": "block", "position": (0.10, 0.00, 0.65)},
            {"colour": "green", "shape": "cube",  "position": (0.00, 0.10, 0.65)},
        ],
    },
    {
        "robot":   "kuka_iiwa7",
        "command": "Pick up the red sphere and place it on the yellow platform.",
        "scene": [
            {"colour": "red",    "shape": "sphere",   "position": ( 0.05, 0.10, 0.65)},
            {"colour": "yellow", "shape": "platform", "position": (-0.10, 0.00, 0.65)},
        ],
    },
    {
        "robot":   "ur5",
        "command": "Move the blue block to the right of the green cube.",
        "scene": [
            {"colour": "blue",  "shape": "block", "position": (-0.10, 0.00, 0.65)},
            {"colour": "green", "shape": "cube",  "position": ( 0.00, 0.00, 0.65)},
        ],
    },
    {
        "robot":   "franka_panda",
        "command": "Stack the orange cube on top of the purple block.",
        "scene": [
            {"colour": "orange", "shape": "cube",  "position": ( 0.10, 0.10, 0.65)},
            {"colour": "purple", "shape": "block", "position": (-0.05, 0.00, 0.65)},
        ],
    },
    {
        "robot":   "shadow_hand",
        "command": "Grasp the red sphere.",
        "scene": [
            {"colour": "red",  "shape": "sphere", "position": (0.05, 0.00, 0.65)},
            {"colour": "blue", "shape": "cube",   "position": (-0.05, 0.00, 0.65)},
        ],
    },
]


def _make_mock_frames(n: int = 8, h: int = 224, w: int = 224) -> List[np.ndarray]:
    """Synthetic video frames with coloured blobs."""
    frames = []
    colours_rgb = [(200,50,50), (50,100,200), (50,180,50), (200,180,50)]
    for i in range(n):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 160
        for j, (r, g, b) in enumerate(colours_rgb):
            cx = int(w * (0.20 + j * 0.20) + i * 0.8)
            cy = int(h * 0.5)
            cx = max(20, min(w-20, cx))
            y1, y2 = max(0, cy-18), min(h, cy+18)
            x1, x2 = max(0, cx-18), min(w, cx+18)
            frame[y1:y2, x1:x2] = (r, g, b)
        frames.append(frame)
    return frames


def run_universal_demo(
    scenarios: list = UNIVERSAL_SCENARIOS,
    verbose:   bool = False,
    robot_filter: str = "",
) -> None:
    from universal_pipeline import UniversalPipeline

    print(f"\n{BAR}")
    print(BOLD("  UNIVERSAL VLA PIPELINE — Video → Motor Commands"))
    print(f"  One model, any robot, any environment")
    print(BAR)

    passed = 0
    total  = 0

    for scenario in scenarios:
        robot_name = scenario["robot"]
        if robot_filter and robot_filter not in robot_name:
            continue

        total += 1
        cmd    = scenario["command"]
        scene  = scenario["scene"]

        print(f"\n{'─'*68}")
        print(BOLD(f"  Robot: {robot_name.upper()}"))
        print(f"  Command: {cmd}")

        try:
            pipe   = UniversalPipeline.for_robot(robot_name, adapt=False)
            frames = _make_mock_frames(8)
            result = pipe.run(frames=frames, command=cmd, scene_info=scene)

            # Metrics
            n_tracks   = len(result.tracks)
            n_trajs    = len(result.trajectories)
            n_cmds     = len(result.motor_commands)
            vla_ok     = result.action_normalised is not None

            print(f"  DOF        : {pipe.robot_cfg.dof}")
            print(f"  Frames     : {result.frame_count}")
            print(f"  Tracks     : {n_tracks} objects tracked")
            print(f"  Trajectory : {n_trajs} paths predicted")
            print(f"  VLA model  : {'✓' if vla_ok else '✗ (rule-based fallback)'}")
            if vla_ok:
                act_str = np.array2string(
                    result.action_normalised[:min(6, len(result.action_normalised))],
                    precision=3,
                )
                print(f"  Action     : {act_str}  gripper={result.gripper_cmd:.2f}")
            print(f"  Motor cmds : {n_cmds}")
            if n_cmds > 0:
                mc = result.motor_commands[0]
                print(f"  First cmd  : joints={mc.joint_positions[:min(4, len(mc.joint_positions))].round(3)}")

            ok = result.success
            status = GREEN("✓ PASS") if ok else YELLOW("~ PARTIAL")
            print(f"  Status     : {status}  ({result.elapsed_s*1000:.0f} ms)")
            if result.error_msg:
                print(f"  Note       : {result.error_msg[:80]}")
            if ok:
                passed += 1

            if verbose and result.tracks:
                print("\n  Tracked objects:")
                for t in result.tracks:
                    print(f"    {t}")
            if verbose and result.trajectories:
                print("\n  Predicted trajectories:")
                for tp in result.trajectories:
                    final = tuple(round(v,3) for v in tp.final_position)
                    print(f"    Track#{tp.track_id} [{tp.colour}] → {final}")

        except Exception as e:
            import traceback
            print(f"  {RED('ERROR')}: {e}")
            if verbose:
                traceback.print_exc()

    print(f"\n{BAR}")
    print(BOLD("  UNIVERSAL PIPELINE — SUMMARY"))
    print(BAR)
    if total > 0:
        pct = passed / total * 100
        col = GREEN if pct >= 80 else YELLOW
        print(f"  Robots tested : {total}")
        print(f"  Passed        : {col(str(passed))} / {total}  ({pct:.0f}%)")
        print()

    # Show component summary
    print("  Components integrated:")
    components = [
        ("VideoProcessor",         "Frame extraction + optical flow buffer"),
        ("DepthEstimator",         "Monocular depth (CNN + gradient fallback)"),
        ("ObjectTracker",          "SORT — Kalman + Hungarian assignment"),
        ("TrajectoryEstimator",    "Kalman kinematic + LSTM learned predictor"),
        ("RobotConfig",            "Universal robot schema (YAML-loadable)"),
        ("RobotKinematics",        "Universal FK/IK (Jacobian + CCD + ikpy)"),
        ("RobotAdapter",           "Any-DOF motor command generator"),
        ("TemporalBackbone",       "CNN + Transformer video encoder"),
        ("UniversalVLAModel",      "One model — any robot, any task"),
        ("DomainRandomizer",       "Visual + physics sim2real randomisation"),
        ("Sim2RealAdapter",        "EMA + AdaptiveBN + TTA at test time"),
        ("UniversalPipeline",      "Complete video → motor command orchestrator"),
    ]
    for name, desc in components:
        print(f"    {GREEN('✓')} {name:<28} {desc}")

    print()
    print("  Supported robot presets:")
    from robot.robot_config import list_presets, get_robot
    for pname in list_presets():
        r = get_robot(pname)
        print(f"    {CYAN(pname):<25} {r.dof:>2} DOF  {r.description}")

    print(f"\n{BAR}\n")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RoboLang Full Demo")
    ap.add_argument("--command",        type=str,  default=None)
    ap.add_argument("--verbose",        action="store_true")
    ap.add_argument("--seed",           type=int,  default=42)
    ap.add_argument("--universal-only", action="store_true",
                    help="Only run the universal pipeline demo")
    ap.add_argument("--classic-only",   action="store_true",
                    help="Only run the classic 8-scenario demo")
    ap.add_argument("--robot",          type=str,  default="",
                    help="Filter universal demo to one robot")
    args = ap.parse_args()

    if args.command:
        np.random.seed(args.seed)
        scene = [
            {"colour": "blue",   "shape": "block",    "position": (-0.10, 0.00, 0.65)},
            {"colour": "green",  "shape": "cube",     "position": ( 0.00, 0.00, 0.65)},
            {"colour": "red",    "shape": "sphere",   "position": ( 0.10, 0.00, 0.65)},
            {"colour": "yellow", "shape": "platform", "position": ( 0.00,-0.10, 0.65)},
            {"colour": "cyan",   "shape": "cylinder", "position": ( 0.15, 0.10, 0.65)},
            {"colour": "purple", "shape": "block",    "position": (-0.15, 0.10, 0.65)},
        ]
        with RoboLangPipeline(use_mock=True) as pipe:
            pipe.setup_scene(scene)
            result = pipe.run(args.command)
            print(result.summary())
        return

    if args.universal_only:
        run_universal_demo(verbose=args.verbose, robot_filter=args.robot)
        return

    if args.classic_only:
        run_classic_demo(verbose=args.verbose, seed=args.seed)
        return

    # Run both
    run_classic_demo(verbose=args.verbose, seed=args.seed)
    run_universal_demo(verbose=args.verbose, robot_filter=args.robot)


if __name__ == "__main__":
    main()

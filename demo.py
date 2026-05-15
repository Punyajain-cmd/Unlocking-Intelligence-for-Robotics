"""
demo.py
────────
Demonstration script – executes 8 natural language manipulation commands
and prints a formatted summary with KPI metrics.

Usage
─────
  python demo.py                   # run all 8 demo commands
  python demo.py --command "Move the blue block to the left of the red sphere."
  python demo.py --verbose         # per-step primitive listing
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import numpy as np

from pipeline import RoboLangPipeline, PipelineResult
from evaluation.metrics import MetricsAccumulator, EpisodeResult, ErrorCategory


# ──────────────────────────────────────────────────────────
# Demo scenarios (each has a scene + command + expected action)
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
            {"colour": "grey",   "shape": "block",   "position": ( 0.15, 0.10, 0.65)},
            {"colour": "black",  "shape": "cube",    "position": ( 0.00, 0.00, 0.65)},
            {"colour": "orange", "shape": "sphere",  "position": (-0.10, 0.10, 0.65)},
            {"colour": "blue",   "shape": "cylinder","position": ( 0.10,-0.10, 0.65)},
        ],
        "expected_action": "pick_and_place",
        "goal_pos": (-0.08, 0.00, 0.65),
    },
]


# ──────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────

BAR = "═" * 68


def _colour(text: str, code: str) -> str:
    """ANSI colour helper (skipped on Windows if not supported)."""
    try:
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text


GREEN  = lambda t: _colour(t, "32")
RED    = lambda t: _colour(t, "31")
YELLOW = lambda t: _colour(t, "33")
CYAN   = lambda t: _colour(t, "36")
BOLD   = lambda t: _colour(t, "1")


def _print_result(
    idx:      int,
    scenario: dict,
    result:   PipelineResult,
    verbose:  bool,
):
    title     = scenario["title"]
    cmd       = scenario["command"]
    exp_act   = scenario["expected_action"]
    goal_pos  = scenario["goal_pos"]

    ok        = "✓" if result.success else "✗"
    ok_coloured = GREEN(ok) if result.success else RED(ok)

    print(f"\n{BAR}")
    print(BOLD(f"  [{idx}] {title}"))
    print(BAR)
    print(f"  Command    : {cmd}")
    print(f"  Parse      : {result.parsed}")

    if result.parsed and result.parsed.is_valid:
        pred_act = result.parsed.action
        act_ok   = "✓" if pred_act == exp_act else "✗"
        print(f"  Action     : {GREEN(pred_act) if pred_act==exp_act else RED(pred_act)}"
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

    print(f"  Outcome    : {ok_coloured} {'SUCCESS' if result.success else 'FAILURE'}  "
          f"  (Δt={result.elapsed_s*1000:.0f} ms)")

    if result.error_msg:
        print(f"  Error      : {RED(result.error_msg)}")


# ──────────────────────────────────────────────────────────
# Main demo runner
# ──────────────────────────────────────────────────────────

def run_demo(
    scenarios: list = DEMO_SCENARIOS,
    verbose:   bool = False,
    seed:      int  = 42,
) -> List[PipelineResult]:
    np.random.seed(seed)

    print(f"\n{BAR}")
    print(BOLD("  RoboLang – Natural Language Robotic Manipulation Demo"))
    print(f"  Running {len(scenarios)} scenarios …")
    print(BAR)

    acc     = MetricsAccumulator()
    results = []

    with RoboLangPipeline(use_mock=True) as pipe:
        for i, scenario in enumerate(scenarios, start=1):
            result = pipe.run(
                command=scenario["command"],
                scene_setup=scenario["scene"],
            )
            results.append(result)
            _print_result(i, scenario, result, verbose)

            # Build episode result for metrics
            parse_ok   = result.parsed is not None and result.parsed.is_valid
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

    # ── Final KPI summary ─────────────────────────────────────
    print(f"\n{BAR}")
    print(BOLD("  DEMO RESULTS SUMMARY"))
    print(BAR)
    n     = len(scenarios)
    n_ok  = sum(1 for r in results if r.success)
    print(f"  Scenarios run   : {n}")
    print(f"  Tasks succeeded : {GREEN(str(n_ok))} / {n}  "
          f"({n_ok/n*100:.0f}%)")
    print()

    r = acc.report()
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

    print(f"\n  Mean position error: "
          f"{r['mean_position_error_m']*100:.1f} cm")
    print()
    print("  Error breakdown:")
    for cat, cnt in r["error_breakdown"].items():
        if cat != "NONE":
            print(f"    {cat:<25} {cnt:3d}")
    print(BAR + "\n")

    return results


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RoboLang demo.")
    ap.add_argument("--command",  type=str, default=None,
                    help="Single command to run (default: all 8 demos)")
    ap.add_argument("--verbose",  action="store_true")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    if args.command:
        # Single command mode
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
    else:
        run_demo(verbose=args.verbose, seed=args.seed)


if __name__ == "__main__":
    main()

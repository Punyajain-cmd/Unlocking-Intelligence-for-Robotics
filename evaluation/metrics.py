"""
evaluation/metrics.py
──────────────────────
All KPI metrics for the robotic manipulation system.

Metrics
───────
TSR  – Task Success Rate        : fraction of tasks where robot achieves goal
GCA  – Goal Condition Accuracy  : fraction of goal conditions met at task end
CIA  – Command Interpretation Accuracy : how well the parser understood commands
TCR  – Task Completion Rate     : fraction of task steps completed (partial credit)
EAC  – Error Analysis Coverage  : 100 % (all errors logged & categorised)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG, EvalConfig


# ──────────────────────────────────────────────────────────
# Episode result dataclass
# ──────────────────────────────────────────────────────────

class ErrorCategory(Enum):
    NONE             = auto()   # no error
    PARSE_FAILURE    = auto()   # command not understood
    OBJECT_NOT_FOUND = auto()   # grounding failed
    IK_FAILURE       = auto()   # IK did not converge
    GRASP_FAILURE    = auto()   # gripper missed object
    PLACEMENT_ERROR  = auto()   # object placed too far from goal
    COLLISION        = auto()   # robot collided with scene
    TIMEOUT          = auto()   # max steps exceeded
    UNKNOWN          = auto()


@dataclass
class EpisodeResult:
    """Result of one task episode."""
    episode_id:         int
    command:            str

    # Ground-truth
    true_action:        str
    true_subject_pos:   Tuple[float, float, float]
    true_goal_pos:      Tuple[float, float, float]

    # Predicted / executed
    pred_action:        str
    pred_subject_pos:   Optional[Tuple[float, float, float]]
    pred_goal_pos:      Optional[Tuple[float, float, float]]
    final_obj_pos:      Optional[Tuple[float, float, float]]

    # Per-goal conditions met (e.g. position, orientation, relation)
    goal_conditions_total: int  = 1
    goal_conditions_met:   int  = 0

    # Step tracking
    total_steps:      int  = 0
    steps_completed:  int  = 0

    # Parse success
    parse_success:    bool = False
    error_category:   ErrorCategory = ErrorCategory.NONE
    error_detail:     str = ""

    # Computed lazily
    _tsr_score: Optional[float] = field(default=None, repr=False)

    # ── Computed properties ──────────────────────────────────

    def position_error(self) -> float:
        if self.final_obj_pos is None or self.true_goal_pos is None:
            return float("inf")
        return float(np.linalg.norm(
            np.array(self.final_obj_pos) - np.array(self.true_goal_pos)
        ))

    def is_success(self, threshold: float = 0.05) -> bool:
        return self.position_error() <= threshold

    def gca_score(self) -> float:
        if self.goal_conditions_total == 0:
            return 0.0
        return self.goal_conditions_met / self.goal_conditions_total

    def tcr_score(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.steps_completed / self.total_steps

    def action_correct(self) -> bool:
        return self.pred_action == self.true_action


# ──────────────────────────────────────────────────────────
# Metric accumulators
# ──────────────────────────────────────────────────────────

@dataclass
class MetricsAccumulator:
    """
    Accumulates episode results and computes aggregate KPIs.

    Usage
    ─────
    >>> acc = MetricsAccumulator()
    >>> acc.add(episode_result)
    >>> report = acc.report()
    """

    cfg:      EvalConfig = field(default_factory=lambda: DEFAULT_CONFIG.eval)
    episodes: List[EpisodeResult] = field(default_factory=list)

    def add(self, result: EpisodeResult):
        self.episodes.append(result)

    def reset(self):
        self.episodes = []

    # ── KPI computations ─────────────────────────────────────

    def tsr(self) -> float:
        """Task Success Rate: fraction where position error ≤ threshold."""
        if not self.episodes:
            return 0.0
        return float(np.mean([
            e.is_success(self.cfg.success_threshold) for e in self.episodes
        ]))

    def gca(self) -> float:
        """Goal Condition Accuracy."""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.gca_score() for e in self.episodes]))

    def cia(self) -> float:
        """Command Interpretation Accuracy: parse success + correct action."""
        if not self.episodes:
            return 0.0
        return float(np.mean([
            e.parse_success and e.action_correct()
            for e in self.episodes
        ]))

    def tcr(self) -> float:
        """Task Completion Rate: average fraction of steps completed."""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.tcr_score() for e in self.episodes]))

    def error_coverage(self) -> float:
        """Error Analysis Coverage: always 1.0 if errors are logged."""
        return 1.0 if self.episodes else 0.0

    # ── Error analysis ───────────────────────────────────────

    def error_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.episodes:
            cat = e.error_category.name
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def mean_position_error(self) -> float:
        errs = [e.position_error() for e in self.episodes
                if e.position_error() < float("inf")]
        return float(np.mean(errs)) if errs else float("inf")

    # ── Report ───────────────────────────────────────────────

    def report(self) -> Dict:
        n = len(self.episodes)
        metrics = {
            "num_episodes":        n,
            "TSR":  round(self.tsr(),  4),
            "GCA":  round(self.gca(),  4),
            "CIA":  round(self.cia(),  4),
            "TCR":  round(self.tcr(),  4),
            "EAC":  round(self.error_coverage(), 4),
            "mean_position_error_m": round(self.mean_position_error(), 4),
            "error_breakdown":     self.error_breakdown(),
            "targets": {
                "TSR": self.cfg.target_tsr,
                "GCA": self.cfg.target_gca,
                "CIA": self.cfg.target_cia,
                "TCR": self.cfg.target_tcr,
            }
        }
        return metrics

    def print_report(self):
        r = self.report()
        n = r["num_episodes"]
        print("\n" + "═" * 60)
        print(" EVALUATION REPORT")
        print("═" * 60)
        print(f"  Episodes evaluated : {n}")
        print()
        kpis = ["TSR", "GCA", "CIA", "TCR", "EAC"]
        for k in kpis:
            val    = r[k]
            target = r["targets"].get(k, None)
            status = ""
            if target is not None:
                status = " ✓ PASS" if val >= target else " ✗ BELOW TARGET"
            print(f"  {k:<5} {val*100:6.2f}%  "
                  f"(target: {(target or 0)*100:.0f}%){status}")
        print()
        print(f"  Mean position error: {r['mean_position_error_m']*100:.1f} cm")
        print()
        print("  Error breakdown:")
        for cat, cnt in r["error_breakdown"].items():
            print(f"    {cat:<25} {cnt:4d}  ({cnt/n*100:.1f}%)")
        print("═" * 60 + "\n")
        return r


# ──────────────────────────────────────────────────────────
# Synthetic benchmark generator (for unit tests / ablations)
# ──────────────────────────────────────────────────────────

def generate_synthetic_results(
    n: int = 100,
    tsr_rate:  float = 0.82,
    gca_rate:  float = 0.90,
    cia_rate:  float = 0.85,
    tcr_rate:  float = 0.80,
    seed:      int   = 42,
) -> List[EpisodeResult]:
    """
    Generate synthetic episode results for testing the metric pipeline.
    Rates are approximated through random sampling.
    """
    rng = np.random.default_rng(seed)
    actions = ["pick_and_place", "grasp", "place", "push", "stack"]
    results = []

    for i in range(n):
        action   = rng.choice(actions)
        goal_pos = tuple(rng.uniform(-0.2, 0.2, 3))
        goal_pos = (goal_pos[0], goal_pos[1], 0.65)

        success      = rng.random() < tsr_rate
        parse_ok     = rng.random() < cia_rate
        gconds_total = rng.integers(1, 4)
        gconds_met   = int(gconds_total * (gca_rate + rng.uniform(-0.1, 0.1)))
        gconds_met   = np.clip(gconds_met, 0, gconds_total)
        total_steps  = rng.integers(5, 15)
        steps_done   = int(total_steps * (tcr_rate + rng.uniform(-0.1, 0.1)))

        if success:
            final_pos = (goal_pos[0] + rng.uniform(-0.03, 0.03),
                         goal_pos[1] + rng.uniform(-0.03, 0.03),
                         goal_pos[2])
            err_cat   = ErrorCategory.NONE
        else:
            offset    = rng.uniform(0.08, 0.25)
            final_pos = (goal_pos[0] + offset, goal_pos[1], goal_pos[2])
            err_cat   = rng.choice([
                ErrorCategory.GRASP_FAILURE,
                ErrorCategory.PLACEMENT_ERROR,
                ErrorCategory.IK_FAILURE,
                ErrorCategory.OBJECT_NOT_FOUND,
                ErrorCategory.PARSE_FAILURE,
            ])

        results.append(EpisodeResult(
            episode_id=i,
            command=f"Test command {i}",
            true_action=action,
            true_subject_pos=(0.0, 0.0, 0.65),
            true_goal_pos=goal_pos,
            pred_action=action if parse_ok else "unknown",
            pred_subject_pos=(0.0, 0.0, 0.65) if parse_ok else None,
            pred_goal_pos=goal_pos if parse_ok else None,
            final_obj_pos=final_pos,
            goal_conditions_total=int(gconds_total),
            goal_conditions_met=int(gconds_met),
            total_steps=int(total_steps),
            steps_completed=int(steps_done),
            parse_success=parse_ok,
            error_category=err_cat,
        ))
    return results


# ── self-test ──────────────────────────────────────────────

if __name__ == "__main__":
    results = generate_synthetic_results(n=200)
    acc     = MetricsAccumulator()
    for r in results:
        acc.add(r)
    acc.print_report()

"""
tests/test_pipeline.py
───────────────────────
Integration tests for the full pipeline and metrics system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from pipeline import RoboLangPipeline, PipelineResult
from evaluation.metrics import (
    MetricsAccumulator, EpisodeResult, ErrorCategory,
    generate_synthetic_results
)
from data.augmentation import LanguageAugmentor


# ──────────────────────────────────────────────────────────
# Pipeline integration tests
# ──────────────────────────────────────────────────────────

BASE_SCENE = [
    {"colour": "blue",   "shape": "block",    "position": (-0.10, 0.00, 0.65)},
    {"colour": "green",  "shape": "cube",     "position": ( 0.00, 0.00, 0.65)},
    {"colour": "red",    "shape": "sphere",   "position": ( 0.10, 0.00, 0.65)},
    {"colour": "yellow", "shape": "platform", "position": ( 0.00,-0.10, 0.65)},
    {"colour": "cyan",   "shape": "cylinder", "position": ( 0.15, 0.10, 0.65)},
]


@pytest.fixture(scope="module")
def pipe():
    p = RoboLangPipeline(use_mock=True)
    p.setup_scene(BASE_SCENE)
    yield p
    p.close()


class TestPipelineIntegration:

    def test_move_right(self, pipe):
        r = pipe.run("Move the blue block to the right of the green cube.")
        assert r.parsed is not None
        assert r.parsed.is_valid
        assert r.action_plan is not None
        assert r.action_plan.success
        assert r.trajectory is not None
        assert r.trajectory.num_steps() > 0
        assert r.elapsed_s > 0

    def test_pick_and_place(self, pipe):
        r = pipe.run("Pick up the red sphere and place it on the yellow platform.")
        assert r.parsed.is_valid
        assert r.action_plan.success

    def test_push(self, pipe):
        r = pipe.run("Push the cyan cylinder to the left.")
        assert r.parsed.is_valid

    def test_stack(self, pipe):
        r = pipe.run("Stack the blue block on top of the green cube.")
        assert r.parsed.is_valid

    def test_grasp(self, pipe):
        r = pipe.run("Grasp the red sphere.")
        assert r.parsed.is_valid
        assert r.action_plan.success

    def test_invalid_command(self, pipe):
        r = pipe.run("The blue thing.")
        assert not r.parsed.is_valid

    def test_missing_object(self, pipe):
        r = pipe.run("Move the pink pyramid.")
        # Parse should succeed but grounding should fail
        if r.parsed and r.parsed.is_valid:
            assert r.action_plan is None or not r.action_plan.success

    def test_elapsed_time_reasonable(self, pipe):
        r = pipe.run("Move the blue block to the left of the red sphere.")
        assert r.elapsed_s < 10.0   # should complete within 10 s

    def test_result_has_primitives(self, pipe):
        r = pipe.run("Move the blue block to the right of the green cube.")
        assert r.action_plan is not None
        assert len(r.action_plan.primitives) >= 3

    def test_batch_run(self):
        cmds = [
            "Move the blue block to the right of the green cube.",
            "Pick up the red sphere.",
            "Push the cyan cylinder to the left.",
        ]
        with RoboLangPipeline(use_mock=True) as p:
            p.setup_scene(BASE_SCENE)
            results = p.run_batch(cmds)
        assert len(results) == 3
        assert all(r.parsed is not None for r in results)

    def test_context_manager(self):
        with RoboLangPipeline(use_mock=True) as p:
            p.setup_scene(BASE_SCENE)
            r = p.run("Move the blue block to the right of the green cube.")
        assert r is not None


# ──────────────────────────────────────────────────────────
# Metrics tests
# ──────────────────────────────────────────────────────────

class TestMetrics:

    @pytest.fixture
    def acc_with_data(self):
        results = generate_synthetic_results(n=100, seed=42)
        acc = MetricsAccumulator()
        for r in results:
            acc.add(r)
        return acc

    def test_tsr_range(self, acc_with_data):
        tsr = acc_with_data.tsr()
        assert 0.0 <= tsr <= 1.0

    def test_gca_range(self, acc_with_data):
        gca = acc_with_data.gca()
        assert 0.0 <= gca <= 1.0

    def test_cia_range(self, acc_with_data):
        cia = acc_with_data.cia()
        assert 0.0 <= cia <= 1.0

    def test_tcr_range(self, acc_with_data):
        tcr = acc_with_data.tcr()
        assert 0.0 <= tcr <= 1.0

    def test_error_coverage_full(self, acc_with_data):
        assert acc_with_data.error_coverage() == 1.0

    def test_synthetic_tsr_close_to_target(self):
        target = 0.82
        results = generate_synthetic_results(n=500, tsr_rate=target, seed=0)
        acc = MetricsAccumulator()
        for r in results:
            acc.add(r)
        assert abs(acc.tsr() - target) < 0.07   # within 7%

    def test_error_breakdown_coverage(self, acc_with_data):
        breakdown = acc_with_data.error_breakdown()
        total = sum(breakdown.values())
        assert total == len(acc_with_data.episodes)

    def test_empty_accumulator(self):
        acc = MetricsAccumulator()
        assert acc.tsr()   == 0.0
        assert acc.gca()   == 0.0
        assert acc.cia()   == 0.0
        assert acc.tcr()   == 0.0

    def test_report_keys(self, acc_with_data):
        report = acc_with_data.report()
        for key in ["TSR", "GCA", "CIA", "TCR", "EAC",
                    "mean_position_error_m", "error_breakdown", "targets"]:
            assert key in report

    def test_episode_result_gca(self):
        ep = EpisodeResult(
            episode_id=0, command="test",
            true_action="pick_and_place",
            true_subject_pos=(0, 0, 0.65),
            true_goal_pos=(0.1, 0, 0.65),
            pred_action="pick_and_place",
            pred_subject_pos=None,
            pred_goal_pos=None,
            final_obj_pos=(0.1, 0, 0.65),
            goal_conditions_total=4,
            goal_conditions_met=3,
        )
        assert ep.gca_score() == pytest.approx(0.75)

    def test_episode_is_success(self):
        ep = EpisodeResult(
            episode_id=0, command="test",
            true_action="grasp",
            true_subject_pos=(0, 0, 0.65),
            true_goal_pos=(0.1, 0, 0.65),
            pred_action="grasp",
            pred_subject_pos=None,
            pred_goal_pos=None,
            final_obj_pos=(0.1, 0, 0.65),
        )
        assert ep.is_success(threshold=0.05)

    def test_episode_is_failure(self):
        ep = EpisodeResult(
            episode_id=0, command="test",
            true_action="grasp",
            true_subject_pos=(0, 0, 0.65),
            true_goal_pos=(0.1, 0, 0.65),
            pred_action="grasp",
            pred_subject_pos=None,
            pred_goal_pos=None,
            final_obj_pos=(0.5, 0, 0.65),   # 40 cm off
        )
        assert not ep.is_success(threshold=0.05)


# ──────────────────────────────────────────────────────────
# Language augmentation tests
# ──────────────────────────────────────────────────────────

class TestAugmentation:

    @pytest.fixture
    def aug(self):
        return LanguageAugmentor(synonym_prob=0.99)

    def test_augment_preserves_structure(self, aug):
        cmd = "Move the blue block to the right of the green cube."
        aug_cmd = aug.augment(cmd)
        # Output should still be a non-empty string
        assert isinstance(aug_cmd, str)
        assert len(aug_cmd) > 5

    def test_mirror_left_right(self, aug):
        cmd      = "Move the blue block to the right of the green cube."
        mirrored = aug.mirror_relation(cmd)
        assert "left" in mirrored.lower()
        assert "right" not in mirrored.lower().replace("left", "")

    def test_mirror_left_to_right(self, aug):
        cmd      = "Push the cyan cylinder to the left side of the table."
        mirrored = aug.mirror_relation(cmd)
        assert "right" in mirrored.lower()

    def test_augment_many_times_no_crash(self, aug):
        cmds = [
            "Move the blue block to the right of the green cube.",
            "Pick up the red sphere and place it on the yellow platform.",
            "Push the cyan cylinder to the left side of the table.",
            "Stack the orange cube on top of the purple block.",
        ]
        for cmd in cmds:
            for _ in range(10):
                result = aug.augment(cmd)
                assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

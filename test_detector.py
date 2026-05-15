"""
tests/test_detector.py
───────────────────────
Unit tests for the object detector, scene graph, and action generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from vision.object_detector import ObjectDetector, DetectedObject, _iou
from vision.scene_graph import SceneGraph, _spatial_relations
from action.action_generator import ActionGenerator
from language.command_parser import CommandParser


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def four_objects():
    return [
        DetectedObject(0, "blue",   "block",  centre_3d=(-0.10, 0.00, 0.65)),
        DetectedObject(1, "green",  "cube",   centre_3d=( 0.00, 0.00, 0.65)),
        DetectedObject(2, "red",    "sphere", centre_3d=( 0.10, 0.00, 0.65)),
        DetectedObject(3, "yellow", "block",  centre_3d=( 0.00,-0.10, 0.65)),
    ]


@pytest.fixture
def scene_graph(four_objects):
    sg = SceneGraph()
    sg.build(four_objects)
    return sg


@pytest.fixture
def sim_detector():
    return ObjectDetector(use_sim_oracle=True)


# ──────────────────────────────────────────────────────────
# DetectedObject tests
# ──────────────────────────────────────────────────────────

class TestDetectedObject:

    def test_matches_colour(self):
        obj = DetectedObject(0, "blue", "block")
        assert obj.matches("blue", None)
        assert not obj.matches("red", None)

    def test_matches_shape(self):
        obj = DetectedObject(0, "blue", "block")
        assert obj.matches(None, "block")
        assert not obj.matches(None, "sphere")

    def test_matches_generic_object(self):
        obj = DetectedObject(0, "blue", "cube")
        assert obj.matches("blue", "object")   # "object" matches any shape

    def test_matches_both(self):
        obj = DetectedObject(0, "blue", "block")
        assert obj.matches("blue", "block")
        assert not obj.matches("blue", "sphere")

    def test_str_repr(self):
        obj = DetectedObject(0, "blue", "block", centre_3d=(0.1, 0.2, 0.65))
        s = str(obj)
        assert "blue" in s
        assert "block" in s


# ──────────────────────────────────────────────────────────
# ObjectDetector tests
# ──────────────────────────────────────────────────────────

class TestObjectDetector:

    def test_sim_oracle_returns_objects(self, sim_detector):
        mock_info = [
            {"colour": "blue",  "shape": "block",  "position": (-0.10, 0.0, 0.65)},
            {"colour": "green", "shape": "cube",   "position": ( 0.10, 0.0, 0.65)},
        ]
        objs = sim_detector.detect(sim_object_info=mock_info)
        assert len(objs) == 2
        colours = {o.colour for o in objs}
        assert colours == {"blue", "green"}

    def test_find_object_by_colour(self, sim_detector):
        mock_info = [
            {"colour": "blue",  "shape": "block",  "position": (-0.1, 0.0, 0.65)},
            {"colour": "red",   "shape": "sphere", "position": ( 0.1, 0.0, 0.65)},
        ]
        objs = sim_detector.detect(sim_object_info=mock_info)
        found = sim_detector.find_object(objs, "blue", None)
        assert found is not None
        assert found.colour == "blue"

    def test_find_object_not_found(self, sim_detector):
        mock_info = [{"colour": "blue", "shape": "block", "position": (0, 0, 0.65)}]
        objs  = sim_detector.detect(sim_object_info=mock_info)
        found = sim_detector.find_object(objs, "cyan", "sphere")
        assert found is None

    def test_empty_scene(self, sim_detector):
        objs = sim_detector.detect(sim_object_info=[])
        assert objs == []

    def test_mock_fallback(self):
        det  = ObjectDetector(use_sim_oracle=False)
        objs = det._mock_detect()
        assert len(objs) >= 2

    def test_iou_identical_boxes(self):
        box = (10, 10, 50, 50)
        assert abs(_iou(box, box) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 10, 10)
        assert _iou(a, b) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────
# SceneGraph tests
# ──────────────────────────────────────────────────────────

class TestSceneGraph:

    def test_build_creates_nodes(self, scene_graph, four_objects):
        assert len(scene_graph.nodes) == 4

    def test_get_by_colour_shape(self, scene_graph):
        obj = scene_graph.get_by_colour_shape("blue", "block")
        assert obj is not None
        assert obj.colour == "blue"

    def test_get_by_colour_shape_miss(self, scene_graph):
        obj = scene_graph.get_by_colour_shape("pink", "pyramid")
        assert obj is None

    def test_right_of_relation(self, scene_graph):
        # red sphere is at x=+0.10, green cube at x=0.00
        # so red is to the right of green
        green = scene_graph.get_by_colour_shape("green", "cube")
        right_of_green = scene_graph.find_by_relation(green, "right_of")
        assert right_of_green is not None
        assert right_of_green.colour == "red"

    def test_left_of_relation(self, scene_graph):
        green = scene_graph.get_by_colour_shape("green", "cube")
        left_of_green = scene_graph.find_by_relation(green, "left_of")
        assert left_of_green is not None
        assert left_of_green.colour == "blue"

    def test_compute_target_right_of(self, scene_graph):
        green = scene_graph.get_by_colour_shape("green", "cube")
        goal  = scene_graph.compute_target_position(green, "right_of", offset_m=0.08)
        assert goal is not None
        gx, gy, gz = goal
        # Should be to the right (larger x) of green cube at (0.0, 0.0, 0.65)
        assert gx > 0.0

    def test_compute_target_on_top(self, scene_graph):
        green = scene_graph.get_by_colour_shape("green", "cube")
        goal  = scene_graph.compute_target_position(green, "on_top_of", offset_m=0.06)
        gx, gy, gz = goal
        assert gz > 0.65

    def test_spatial_tags_populated(self, four_objects):
        sg = SceneGraph().build(four_objects)
        # Some objects should have spatial tags
        all_tags = [tag for o in four_objects for tag in o.spatial_tags]
        assert len(all_tags) > 0

    def test_get_all(self, scene_graph, four_objects):
        all_objs = scene_graph.get_all()
        assert len(all_objs) == len(four_objects)

    def test_summary_not_empty(self, scene_graph):
        s = scene_graph.summary()
        assert "SceneGraph" in s
        assert len(s) > 10


class TestSpatialRelations:

    def test_right_of(self):
        rels = _spatial_relations((0.15, 0.0, 0.65), (0.0, 0.0, 0.65))
        assert "right_of" in rels

    def test_left_of(self):
        rels = _spatial_relations((-0.15, 0.0, 0.65), (0.0, 0.0, 0.65))
        assert "left_of" in rels

    def test_above(self):
        rels = _spatial_relations((0.0, 0.0, 0.80), (0.0, 0.0, 0.65))
        assert "above" in rels

    def test_near(self):
        rels = _spatial_relations((0.02, 0.02, 0.65), (0.0, 0.0, 0.65))
        assert "near" in rels


# ──────────────────────────────────────────────────────────
# ActionGenerator tests
# ──────────────────────────────────────────────────────────

class TestActionGenerator:

    @pytest.fixture
    def gen(self):
        return ActionGenerator()

    @pytest.fixture
    def parser(self):
        return CommandParser(use_bert=False)

    @pytest.fixture
    def scene_w_objects(self):
        objects = [
            DetectedObject(0, "blue",  "block",  centre_3d=(-0.10, 0.00, 0.65)),
            DetectedObject(1, "green", "cube",   centre_3d=( 0.00, 0.00, 0.65)),
            DetectedObject(2, "red",   "sphere", centre_3d=( 0.10, 0.00, 0.65)),
        ]
        return SceneGraph().build(objects)

    def test_pick_and_place_plan(self, gen, parser, scene_w_objects):
        cmd  = parser.parse("Move the blue block to the right of the green cube.")
        plan = gen.generate(cmd, scene_w_objects)
        assert plan.success
        assert plan.action_type == "pick_and_place"
        assert len(plan.primitives) >= 5
        assert plan.target_pos is not None

    def test_grasp_plan(self, gen, parser, scene_w_objects):
        cmd  = parser.parse("Pick up the red sphere.")
        plan = gen.generate(cmd, scene_w_objects)
        assert plan.success
        assert plan.subject_obj is not None
        assert plan.subject_obj.colour == "red"

    def test_missing_subject_returns_failure(self, gen, parser, scene_w_objects):
        cmd  = parser.parse("Move the cyan pyramid to the left.")
        plan = gen.generate(cmd, scene_w_objects)
        assert not plan.success
        assert plan.error_msg != ""

    def test_invalid_command_returns_failure(self, gen):
        from language.command_parser import ParsedCommand
        bad_cmd = ParsedCommand(raw="", action="unknown", action_raw="",
                                is_valid=False, error_msg="test error")
        plan = gen.generate(bad_cmd, SceneGraph())
        assert not plan.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

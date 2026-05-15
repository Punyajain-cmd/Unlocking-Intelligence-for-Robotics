"""
tests/test_parser.py
─────────────────────
Unit tests for the command parser and intent classifier.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from language.command_parser import CommandParser, ParsedCommand, ObjectRef
from language.intent_classifier import RuleIntentClassifier


PARSER = CommandParser(use_bert=False)
CLASSIFIER = RuleIntentClassifier()


class TestCommandParser:

    def test_basic_move_right_of(self):
        cmd = PARSER.parse("Move the blue block to the right of the green cube.")
        assert cmd.is_valid
        assert cmd.action == "pick_and_place"
        assert cmd.action_raw == "move"
        assert cmd.subject.colour == "blue"
        assert cmd.subject.shape  == "block"
        assert cmd.target.colour  == "green"
        assert cmd.target.shape   == "cube"
        assert cmd.relation       == "right_of"
        assert cmd.confidence     >= 0.9

    def test_pick_and_place_on(self):
        cmd = PARSER.parse("Pick up the red sphere and place it on the yellow platform.")
        assert cmd.is_valid
        assert cmd.subject.colour == "red"
        assert cmd.subject.shape  == "sphere"
        assert cmd.target.colour  == "yellow"

    def test_push_left(self):
        cmd = PARSER.parse("Push the cyan cylinder to the left side of the table.")
        assert cmd.is_valid
        assert cmd.action_raw == "push"
        assert cmd.action     == "push"
        assert cmd.subject.colour == "cyan"
        assert cmd.relation       == "left_of"

    def test_stack_on_top(self):
        cmd = PARSER.parse("Stack the orange cube on top of the purple block.")
        assert cmd.is_valid
        assert cmd.action     == "stack"
        assert cmd.subject.colour == "orange"
        assert cmd.target.colour  == "purple"
        assert cmd.relation       == "on_top_of"

    def test_grasp_only(self):
        cmd = PARSER.parse("Grasp the small blue object near the edge.")
        assert cmd.is_valid
        assert cmd.action     == "grasp"
        assert cmd.subject.colour == "blue"

    def test_lift(self):
        cmd = PARSER.parse("Lift the white box above the brown cylinder.")
        assert cmd.is_valid
        assert cmd.action_raw     == "lift"
        assert cmd.subject.colour == "white"
        assert cmd.target.colour  == "brown"

    def test_empty_command(self):
        cmd = PARSER.parse("")
        assert not cmd.is_valid

    def test_no_verb(self):
        cmd = PARSER.parse("The blue block.")
        assert not cmd.is_valid

    def test_confidence_range(self):
        cmd = PARSER.parse("Move the blue block to the right of the green cube.")
        assert 0.0 <= cmd.confidence <= 1.0

    def test_batch_parse(self):
        cmds = [
            "Move the blue block to the right of the green cube.",
            "Grasp the red sphere.",
        ]
        results = PARSER.parse_batch(cmds)
        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_transfer_synonym(self):
        cmd = PARSER.parse("Transfer the grey block to the left of the black cube.")
        assert cmd.is_valid
        assert cmd.action == "pick_and_place"
        assert cmd.subject.colour == "grey"
        assert cmd.relation       == "left_of"


class TestIntentClassifier:

    def test_pick_and_place(self):
        action, relation, ac, rc = CLASSIFIER.predict(
            "Move the blue block to the right of the green cube."
        )
        assert action == "pick_and_place"
        assert relation == "right_of"
        assert ac > 0.5
        assert rc > 0.5

    def test_push_relation(self):
        action, relation, _, _ = CLASSIFIER.predict(
            "Push the cyan cylinder to the left."
        )
        assert action == "push"
        assert relation in ("left_of", "left")

    def test_stack_on_top(self):
        action, relation, _, _ = CLASSIFIER.predict(
            "Stack the orange cube on top of the purple block."
        )
        assert action == "stack"
        assert relation == "on_top_of"


class TestObjectRef:

    def test_matches_colour_and_shape(self):
        ref = ObjectRef(colour="blue", shape="block")
        assert ref.matches("blue", "block")
        assert ref.matches("blue", None)
        assert ref.matches(None, "block")
        assert not ref.matches("red", "block")

    def test_is_valid(self):
        assert ObjectRef(colour="blue").is_valid()
        assert ObjectRef(shape="cube").is_valid()
        assert not ObjectRef().is_valid()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

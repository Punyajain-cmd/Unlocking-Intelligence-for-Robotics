from __future__ import annotations

import re
from dataclasses import dataclass


SUPPORTED_COLORS = ("red", "green", "blue", "yellow")
SUPPORTED_OBJECTS = ("block", "cube", "box")

RELATION_ALIASES = {
    "right of": "right_of",
    "to the right of": "right_of",
    "left of": "left_of",
    "to the left of": "left_of",
    "in front of": "in_front_of",
    "front of": "in_front_of",
    "behind": "behind",
    "next to": "next_to",
    "beside": "next_to",
    "on top of": "on_top_of",
}

ACTION_ALIASES = {
    "move": "move",
    "place": "move",
    "put": "move",
}


@dataclass(frozen=True)
class CommandIntent:
    raw_command: str
    action: str
    source_color: str
    source_object: str
    relation: str
    target_color: str
    target_object: str


class CommandParser:
    """Rule-based parser for manipulation commands.

    Expected command shape:
    "<action> the <source_color> <source_obj> ... <relation> the <target_color> <target_obj>"
    """

    def __init__(self) -> None:
        relation_tokens = sorted(RELATION_ALIASES.keys(), key=len, reverse=True)
        relation_pattern = "|".join(re.escape(token) for token in relation_tokens)
        color_pattern = "|".join(re.escape(color) for color in SUPPORTED_COLORS)
        object_pattern = "|".join(re.escape(obj) for obj in SUPPORTED_OBJECTS)
        action_pattern = "|".join(re.escape(action) for action in ACTION_ALIASES.keys())
        self._pattern = re.compile(
            rf"(?P<action>{action_pattern})\s+"
            rf"(?:the\s+)?(?P<src_color>{color_pattern})\s+(?P<src_obj>{object_pattern})\s+"
            rf"(?:to\s+)?(?:the\s+)?(?P<relation>{relation_pattern})\s+"
            rf"(?:the\s+)?(?P<tgt_color>{color_pattern})\s+(?P<tgt_obj>{object_pattern})",
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.strip().lower()
        return re.sub(r"[^\w\s]", "", lowered)

    @staticmethod
    def _normalize_object_name(name: str) -> str:
        if name == "box":
            return "block"
        return name

    def parse(self, command: str) -> CommandIntent:
        clean = self._normalize_text(command)
        match = self._pattern.search(clean)
        if not match:
            raise ValueError(
                "Could not parse command. Example supported form: "
                "'Move the blue block to the right of the green cube.'"
            )

        action = ACTION_ALIASES[match.group("action").lower()]
        relation = RELATION_ALIASES[match.group("relation").lower()]

        source_object = self._normalize_object_name(match.group("src_obj").lower())
        target_object = self._normalize_object_name(match.group("tgt_obj").lower())

        return CommandIntent(
            raw_command=command,
            action=action,
            source_color=match.group("src_color").lower(),
            source_object=source_object,
            relation=relation,
            target_color=match.group("tgt_color").lower(),
            target_object=target_object,
        )

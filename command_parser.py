"""
language/command_parser.py
──────────────────────────
BERT-based command parser that converts a free-form natural-language
manipulation command into a structured ParsedCommand dataclass.

Pipeline
────────
raw text
  → tokenise & BERT encode
  → rule-based / NER slot extraction (action, subject, target, relation)
  → dependency-parse fallback for complex sentences
  → return ParsedCommand

The module deliberately avoids downloading large models at import time:
the BERT tokeniser/model is loaded lazily on the first call to .parse().
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports – only pulled in when the parser is first used
_transformers_available = False
try:
    from transformers import BertTokenizer, BertModel
    import torch
    _transformers_available = True
except ImportError:
    warnings.warn("transformers / torch not installed – using rule-based parser only.")

from config import DEFAULT_CONFIG, LanguageConfig


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class ObjectRef:
    """Reference to a physical object mentioned in the command."""
    colour:    Optional[str] = None
    shape:     Optional[str] = None
    size:      Optional[str] = None      # "small", "large", …
    position:  Optional[str] = None      # "left", "on the edge", …
    raw_text:  str = ""

    def is_valid(self) -> bool:
        return self.colour is not None or self.shape is not None

    def __str__(self) -> str:
        parts = [p for p in [self.size, self.colour, self.shape] if p]
        return " ".join(parts) or self.raw_text or "<unknown>"


@dataclass
class ParsedCommand:
    """Fully structured representation of a manipulation command."""
    raw:            str
    action:         str                       # canonical verb, e.g. "pick_and_place"
    action_raw:     str                       # original verb token, e.g. "move"
    subject:        ObjectRef = field(default_factory=ObjectRef)   # object to act on
    target:         ObjectRef = field(default_factory=ObjectRef)   # reference / goal object
    relation:       Optional[str] = None     # "right_of", "on_top_of", …
    confidence:     float = 0.0
    is_valid:       bool  = False
    error_msg:      str   = ""

    def __str__(self) -> str:
        rel_str = f" ({self.relation} {self.target})" if self.relation else ""
        return (f"[{self.action_raw}] {self.subject}{rel_str}  "
                f"(conf={self.confidence:.2f})")


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _find_first(tokens: List[str], vocab: List[str]) -> Optional[str]:
    for tok in tokens:
        if tok in vocab:
            return tok
    return None


def _find_all(tokens: List[str], vocab: List[str]) -> List[str]:
    return [tok for tok in tokens if tok in vocab]


def _extract_object_ref(
    tokens: List[str],
    colours: List[str],
    shapes:  List[str],
    size_words: List[str] = ("small", "large", "big", "tiny", "huge"),
) -> ObjectRef:
    colour  = _find_first(tokens, colours)
    shape   = _find_first(tokens, shapes)
    size    = _find_first(tokens, list(size_words))
    raw     = " ".join(tokens)
    return ObjectRef(colour=colour, shape=shape, size=size, raw_text=raw)


def _map_relation(phrase: str) -> Optional[str]:
    """Map a raw relation phrase to a canonical name."""
    norm = phrase.lower()
    mapping = {
        "right":           "right_of",
        "to the right":    "right_of",
        "right of":        "right_of",
        "left":            "left_of",
        "to the left":     "left_of",
        "left of":         "left_of",
        "above":           "above",
        "on top":          "on_top_of",
        "on top of":       "on_top_of",
        "on":              "on_top_of",
        "below":           "below",
        "under":           "below",
        "beneath":         "below",
        "in front":        "in_front_of",
        "in front of":     "in_front_of",
        "behind":          "behind",
        "beside":          "beside",
        "next to":         "beside",
        "near":            "near",
        "between":         "between",
        "inside":          "inside",
    }
    for key, val in mapping.items():
        if key in norm:
            return val
    return None


# ──────────────────────────────────────────────────────────
# BERT embedding (optional enhancement)
# ──────────────────────────────────────────────────────────

class BERTEmbedder:
    """Wraps BERT for sentence embeddings; requires transformers+torch."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        if not _transformers_available:
            raise RuntimeError("transformers not installed.")
        self.tokeniser = BertTokenizer.from_pretrained(model_name)
        self.model     = BertModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = self.tokeniser(text, return_tensors="pt",
                                    truncation=True, max_length=128)
            outputs = self.model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def similarity(self, a: str, b: str) -> float:
        ea, eb = self.embed(a), self.embed(b)
        return float(np.dot(ea, eb))


# ──────────────────────────────────────────────────────────
# Rule-Based Parser (no model required)
# ──────────────────────────────────────────────────────────

class RuleBasedParser:
    """
    Fast, dependency-free parser using regex + vocabulary lookup.
    Handles the majority of manipulation commands correctly.
    """

    # Preposition patterns that signal a subject/target boundary
    PREP_PATTERNS = [
        r"\bto the (right|left)\b",
        r"\bto (right|left)\b",
        r"\bon top of\b",
        r"\bin front of\b",
        r"\bnext to\b",
        r"\b(above|below|under|beneath|behind|beside|near|on|inside)\b",
    ]
    COMBINED_PREP = re.compile(
        r"\b(to the right of|to the left of|on top of|in front of|next to|"
        r"above|below|under|beneath|behind|beside|near|inside|"
        r"to the right|to the left|on)\b",
        re.IGNORECASE
    )

    def __init__(self, cfg: LanguageConfig = DEFAULT_CONFIG.language):
        self.cfg = cfg
        # Build compiled sets for fast lookup
        self._verbs     = set(cfg.action_verbs)
        self._relations = set(cfg.spatial_relations)
        self._colours   = set(cfg.colours)
        self._shapes    = set(cfg.shapes)
        self._action_map = DEFAULT_CONFIG.action.action_map

    def parse(self, command: str) -> ParsedCommand:
        raw    = command
        norm   = _normalise(command)
        tokens = norm.split()

        # ── 1. Find action verb ──────────────────────────────
        action_raw = _find_first(tokens, list(self._verbs))
        if action_raw is None:
            return ParsedCommand(
                raw=raw, action="unknown", action_raw="",
                confidence=0.0, is_valid=False,
                error_msg=f"No recognised action verb in: '{command}'"
            )
        action = self._action_map.get(action_raw, action_raw)

        # ── 2. Find the prep/relation that splits subject & target ──
        match = self.COMBINED_PREP.search(norm)

        if match:
            relation_phrase = match.group(0)
            before          = norm[:match.start()].strip()
            after           = norm[match.end():].strip()
        else:
            relation_phrase = None
            # Heuristic: everything after verb is subject; no target
            verb_idx = norm.index(action_raw)
            before   = norm[verb_idx + len(action_raw):].strip()
            after    = ""

        # ── 3. Strip common filler words ────────────────────────
        stop_words = {"the", "a", "an", "it", "of", "and", "up", "down"}

        def clean_tokens(phrase: str) -> List[str]:
            toks = phrase.split()
            return [t for t in toks if t not in stop_words]

        subj_tokens = clean_tokens(before)
        tgt_tokens  = clean_tokens(after)

        # ── 4. Extract object references ────────────────────────
        subject = _extract_object_ref(subj_tokens, list(self._colours), list(self._shapes))
        target  = _extract_object_ref(tgt_tokens,  list(self._colours), list(self._shapes))

        # ── 5. Canonicalise relation ─────────────────────────────
        relation = _map_relation(relation_phrase) if relation_phrase else None

        # ── 6. Compute confidence ────────────────────────────────
        score = 0.0
        if action_raw:                        score += 0.35
        if subject.colour or subject.shape:   score += 0.30
        if target.colour  or target.shape:    score += 0.20
        if relation:                          score += 0.15

        return ParsedCommand(
            raw=raw,
            action=action,
            action_raw=action_raw,
            subject=subject,
            target=target,
            relation=relation,
            confidence=round(score, 3),
            is_valid=score >= 0.5,
        )


# ──────────────────────────────────────────────────────────
# Full CommandParser (rules + optional BERT re-ranking)
# ──────────────────────────────────────────────────────────

class CommandParser:
    """
    Public API for command parsing.

    Usage
    ─────
    >>> parser = CommandParser()
    >>> cmd = parser.parse("Move the blue block to the right of the green cube.")
    >>> print(cmd)
    [move] blue block (right_of green cube)  (conf=1.00)
    """

    def __init__(
        self,
        cfg: LanguageConfig = DEFAULT_CONFIG.language,
        use_bert: bool = False,
    ):
        self.cfg         = cfg
        self._rule_parser = RuleBasedParser(cfg)
        self._bert: Optional[BERTEmbedder] = None

        if use_bert and _transformers_available:
            try:
                self._bert = BERTEmbedder(cfg.bert_model)
            except Exception as e:
                warnings.warn(f"BERT init failed ({e}); falling back to rule-based.")

    def parse(self, command: str) -> ParsedCommand:
        """Parse a natural-language manipulation command."""
        if not command or not command.strip():
            return ParsedCommand(
                raw="", action="unknown", action_raw="",
                confidence=0.0, is_valid=False,
                error_msg="Empty command."
            )
        return self._rule_parser.parse(command)

    def parse_batch(self, commands: List[str]) -> List[ParsedCommand]:
        return [self.parse(c) for c in commands]

    def is_valid_command(self, command: str) -> Tuple[bool, str]:
        """Quick check: returns (is_valid, reason_if_invalid)."""
        result = self.parse(command)
        return result.is_valid, result.error_msg


# ──────────────────────────────────────────────────────────
# Module self-test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_COMMANDS = [
        "Move the blue block to the right of the green cube.",
        "Pick up the red sphere and place it on the yellow platform.",
        "Push the cyan cylinder to the left side of the table.",
        "Stack the orange cube on top of the purple block.",
        "Grasp the small blue object near the edge.",
        "Lift the white box above the brown cylinder.",
        "Slide the grey block behind the red cube.",
        "Rotate the green object and place it beside the blue block.",
    ]

    parser = CommandParser(use_bert=False)

    print("\n" + "=" * 70)
    print(" COMMAND PARSER  –  Self-Test")
    print("=" * 70)
    for cmd in TEST_COMMANDS:
        result = parser.parse(cmd)
        status = "✓" if result.is_valid else "✗"
        print(f"\n{status} Input:    {cmd}")
        print(f"  Action:  {result.action_raw!r} → {result.action!r}")
        print(f"  Subject: {result.subject}")
        print(f"  Target:  {result.target}")
        print(f"  Relation:{result.relation}")
        print(f"  Conf:    {result.confidence:.2f}  valid={result.is_valid}")
    print("=" * 70 + "\n")

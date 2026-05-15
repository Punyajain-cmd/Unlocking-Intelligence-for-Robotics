"""
language/intent_classifier.py
──────────────────────────────
Fine-tuneable classification heads that sit on top of BERT to
produce:
  • ActionType  – one of ~10 canonical manipulation primitives
  • RelationType – spatial relation between subject and target

Both heads share the same BERT encoder backbone (frozen by default).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertModel, BertTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn("torch / transformers not installed; IntentClassifier unavailable.")

from config import DEFAULT_CONFIG, LanguageConfig


# ──────────────────────────────────────────────────────────
# Label sets
# ──────────────────────────────────────────────────────────

ACTION_LABELS: List[str] = [
    "pick_and_place",
    "grasp",
    "place",
    "push",
    "pull",
    "lift",
    "stack",
    "rotate",
    "unknown",
]

RELATION_LABELS: List[str] = [
    "right_of",
    "left_of",
    "above",
    "on_top_of",
    "below",
    "in_front_of",
    "behind",
    "beside",
    "near",
    "between",
    "inside",
    "none",
]

ACTION2IDX:   Dict[str, int] = {a: i for i, a in enumerate(ACTION_LABELS)}
RELATION2IDX: Dict[str, int] = {r: i for i, r in enumerate(RELATION_LABELS)}
IDX2ACTION:   Dict[int, str] = {i: a for a, i in ACTION2IDX.items()}
IDX2RELATION: Dict[int, str] = {i: r for r, i in RELATION2IDX.items()}


# ──────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class IntentClassifier(nn.Module):
        """
        Dual-head classifier:
          BERT → [CLS] repr → Action head (9 classes)
                            → Relation head (12 classes)

        Parameters
        ──────────
        bert_model   : HuggingFace model name / path
        freeze_bert  : If True, BERT weights are frozen (faster training)
        dropout      : Dropout probability on the classification heads
        """

        def __init__(
            self,
            bert_model:  str   = "bert-base-uncased",
            freeze_bert: bool  = True,
            dropout:     float = 0.2,
        ):
            super().__init__()
            self.bert      = BertModel.from_pretrained(bert_model)
            hidden_size    = self.bert.config.hidden_size   # 768 for base

            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False

            self.dropout       = nn.Dropout(dropout)
            self.action_head   = nn.Linear(hidden_size, len(ACTION_LABELS))
            self.relation_head = nn.Linear(hidden_size, len(RELATION_LABELS))

        def forward(
            self,
            input_ids:      "torch.Tensor",
            attention_mask: "torch.Tensor",
            token_type_ids: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Returns
            ───────
            action_logits   : (B, n_actions)
            relation_logits : (B, n_relations)
            """
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            cls_repr = self.dropout(outputs.last_hidden_state[:, 0, :])
            return self.action_head(cls_repr), self.relation_head(cls_repr)

        def predict(
            self,
            tokeniser: "BertTokenizer",
            text: str,
            device: str = "cpu",
        ) -> Tuple[str, str, float, float]:
            """
            Convenience: run inference on a single string.

            Returns
            ───────
            (action_label, relation_label, action_conf, relation_conf)
            """
            self.eval()
            encoding = tokeniser(
                text, return_tensors="pt",
                truncation=True, max_length=128, padding=True
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                a_logits, r_logits = self(**encoding)

            a_probs = F.softmax(a_logits, dim=-1)[0].cpu().numpy()
            r_probs = F.softmax(r_logits, dim=-1)[0].cpu().numpy()

            a_idx = int(np.argmax(a_probs))
            r_idx = int(np.argmax(r_probs))

            return (
                IDX2ACTION[a_idx],
                IDX2RELATION[r_idx],
                float(a_probs[a_idx]),
                float(r_probs[r_idx]),
            )


# ──────────────────────────────────────────────────────────
# Rule-based fallback (no PyTorch)
# ──────────────────────────────────────────────────────────

class RuleIntentClassifier:
    """Deterministic rule-based version – works without torch."""

    ACTION_MAP = DEFAULT_CONFIG.action.action_map

    RELATION_KEYWORDS: Dict[str, List[str]] = {
        "right_of":   ["right of", "to the right", "right"],
        "left_of":    ["left of",  "to the left",  "left"],
        "on_top_of":  ["on top of", "on top", "on"],
        "above":      ["above"],
        "below":      ["below", "under", "beneath"],
        "in_front_of":["in front of", "in front"],
        "behind":     ["behind"],
        "beside":     ["beside", "next to"],
        "near":       ["near", "close to"],
        "between":    ["between"],
        "inside":     ["inside", "in"],
        "none":       [],
    }

    def predict(self, text: str) -> Tuple[str, str, float, float]:
        norm = text.lower()
        # Action
        action, a_conf = "unknown", 0.5
        for verb, canonical in self.ACTION_MAP.items():
            if verb in norm:
                action, a_conf = canonical, 0.9
                break
        # Relation
        relation, r_conf = "none", 0.5
        for rel, keywords in self.RELATION_KEYWORDS.items():
            for kw in keywords:
                if kw in norm:
                    relation, r_conf = rel, 0.85
                    break
            if relation != "none":
                break
        return action, relation, a_conf, r_conf


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def get_intent_classifier(
    use_neural: bool = False,
    bert_model: str  = "bert-base-uncased",
    checkpoint: Optional[str] = None,
):
    """
    Factory function: returns a neural or rule-based classifier.
    If torch is not available, always returns the rule-based version.
    """
    if use_neural and _TORCH_AVAILABLE:
        clf = IntentClassifier(bert_model=bert_model)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            clf.load_state_dict(state)
        return clf
    return RuleIntentClassifier()


# ──────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    clf = RuleIntentClassifier()
    TESTS = [
        "Move the blue block to the right of the green cube.",
        "Pick up the red sphere and place it on the yellow platform.",
        "Push the cyan cylinder to the left.",
        "Stack the orange cube on top of the purple block.",
        "Rotate the grey object.",
    ]
    print("\n" + "=" * 60)
    print(" INTENT CLASSIFIER – Self-Test")
    print("=" * 60)
    for t in TESTS:
        action, relation, ac, rc = clf.predict(t)
        print(f"\n  Input:    {t}")
        print(f"  Action:   {action!r}  (conf={ac:.2f})")
        print(f"  Relation: {relation!r} (conf={rc:.2f})")
    print("=" * 60 + "\n")

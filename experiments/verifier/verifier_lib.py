"""Claim verifiers compared in experiments/verifier: rule-based, NLI zero-shot, fine-tuned.

Every verifier maps (claim, evidence) to one of LABELS. The rule-based verifiers are
GroundedAI's own claim check (claims-v1 / claims-v2), with a fixed mapping of its statuses:
found -> supported, partial -> partial (claims-v2: a negation that differs -> contradicted),
not found -> nee. They never answer "contradicted" except through that v2 negation flag:
the lexical check was not designed to tell a contradiction from a missing fact.

The NLI zero-shot verifier uses a 3-way NLI cross-encoder (entailment / neutral /
contradiction). A rule fixed before any run turns it into four labels: entailment ->
supported; contradiction -> contradicted; neutral -> partial if the claim has several clauses
and at least one clause alone is entailed (and none contradicted), else nee.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

LABELS = ["supported", "partial", "contradicted", "nee"]
CLAUSE_RE = re.compile(
    r",\s*(?:and|but)\s+|;\s+|\s+and\s+(?=(?:it|this|they|the|must|should|shall|may|is|are)\b)"
)

Verifier = Callable[[Sequence[tuple[str, str]]], list[str]]


def load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def clauses(claim: str) -> list[str]:
    parts = [p.strip(" ,.;") for p in CLAUSE_RE.split(claim) if p and p.strip(" ,.;")]
    return [p[0].upper() + p[1:] + "." for p in parts if len(p.split()) >= 3]


def rule_verifier(version: str) -> Verifier:
    from groundedai.generation.claims import EvidenceSource, check_claims

    def run(pairs: Sequence[tuple[str, str]]) -> list[str]:
        out = []
        for claim, evidence in pairs:
            report = check_claims(claim, [EvidenceSource("S1", "Evidence", evidence)], "", version)
            statuses = [c.status for c in report.claims]
            flags = [f for c in report.claims for f in c.flags]
            if not statuses:
                out.append("nee")
            elif "negation_differs" in flags:
                out.append("contradicted")
            elif all(s in ("supported", "added_source", "wrong_source") for s in statuses):
                out.append("supported")
            elif any(
                s in ("supported", "added_source", "wrong_source", "partial") for s in statuses
            ):
                out.append("partial")
            else:
                out.append("nee")
        return out

    return run


class CrossEncoderClassifier:
    """A sequence-pair classifier (premise = evidence, hypothesis = claim) on the CPU."""

    def __init__(self, model_dir: str, max_length: int = 256, threads: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.max_length = max_length
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def probabilities(
        self, pairs: Sequence[tuple[str, str]], batch: int = 16
    ) -> list[dict[str, float]]:
        out = []
        for start in range(0, len(pairs), batch):
            chunk = pairs[start : start + batch]
            encoded = self.tokenizer(
                [e for _, e in chunk],
                [c for c, _ in chunk],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
            probs = logits.softmax(-1).tolist()
            out.extend({self.id2label[i]: p for i, p in enumerate(row)} for row in probs)
        return out


def nli_verifier(model_dir: str) -> Verifier:
    classifier = CrossEncoderClassifier(model_dir)

    def top(probs: dict[str, float]) -> str:
        return max(probs, key=probs.__getitem__)

    def run(pairs: Sequence[tuple[str, str]]) -> list[str]:
        whole = classifier.probabilities(pairs)
        out = []
        for (claim, evidence), probs in zip(pairs, whole, strict=True):
            label = top(probs)
            if label == "entailment":
                out.append("supported")
                continue
            if label == "contradiction":
                out.append("contradicted")
                continue
            parts = clauses(claim)
            if len(parts) >= 2:
                part_labels = [
                    top(p) for p in classifier.probabilities([(p, evidence) for p in parts])
                ]
                if "entailment" in part_labels and "contradiction" not in part_labels:
                    out.append("partial")
                    continue
            out.append("nee")
        return out

    return run


def finetuned_verifier(model_dir: str) -> Verifier:
    classifier = CrossEncoderClassifier(model_dir)

    def run(pairs: Sequence[tuple[str, str]]) -> list[str]:
        return [max(p, key=p.__getitem__) for p in classifier.probabilities(pairs)]

    return run


def metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, predicted, strict=True):
        confusion[g][p] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    # Binary view used by the application: is the claim fully supported or not?
    tp = sum(
        1 for g, p in zip(gold, predicted, strict=True) if g == "supported" and p == "supported"
    )
    fp = sum(
        1 for g, p in zip(gold, predicted, strict=True) if g != "supported" and p == "supported"
    )
    fn = sum(
        1 for g, p in zip(gold, predicted, strict=True) if g == "supported" and p != "supported"
    )
    tn = len(gold) - tp - fp - fn
    return {
        "n": len(gold),
        "accuracy": round(sum(g == p for g, p in zip(gold, predicted, strict=True)) / len(gold), 4),
        "macro_f1": round(sum(c["f1"] for c in per_class.values()) / len(LABELS), 4),
        "per_class": per_class,
        "confusion": confusion,
        "supported_vs_rest": {
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
            "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        },
    }


def evaluate(
    verifier: Verifier, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    started = time.perf_counter()
    predicted = verifier([(r["claim"], r["evidence"]) for r in rows])
    seconds = time.perf_counter() - started
    result = metrics([r["label"] for r in rows], predicted)
    result["ms_per_claim"] = round(1000 * seconds / len(rows), 2)
    return result, predicted

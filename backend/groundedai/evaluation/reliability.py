"""groundedai-reliability-v1: an independent evaluation suite for the answer checks.

A separate agent wrote the cases from ``evaluation/datasets/groundedai-reliability-v1/SPEC.md``
without access to this code. This module loads and validates the suite, runs one version of the
checks on every case, and computes the measures defined in the suite's README.

Nothing here prints or returns case text for the held-out test split: validation issues name a
case id and a field, and scores are counts, rates and intervals.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundedai.generation.citations import labels_in, parse_answer
from groundedai.generation.claims import MATCHED, EvidenceSource, check_claims
from groundedai.generation.context import AssembledContext, assemble_context
from groundedai.generation.disagreement import find_disagreements
from groundedai.retrieval.types import ChunkScores, RetrievedChunk

SUITE_NAME = "groundedai-reliability-v1"
SPLITS = ("dev", "test")
CATEGORIES = ("A", "B", "C", "D")
SCORING_VERSION = 1
# Set when the test split is frozen (after the label audit). Scoring the test split refuses any
# other content, so the held-out cases cannot change unnoticed.
FROZEN_TEST_SHA256: str | None = None

# Exact case counts per category, subtype and split: (dev, test).
COUNTS: dict[str, dict[str, tuple[int, int]]] = {
    "A": {
        "verbatim": (6, 14),
        "paraphrase": (15, 35),
        "number_form": (6, 14),
        "wrong_value": (9, 21),
        "wrong_subject": (6, 14),
        "negation": (6, 14),
        "absent": (6, 14),
    },
    "B": {
        "correct_single": (12, 28),
        "correct_two": (6, 14),
        "wrong_related": (9, 21),
        "wrong_unrelated": (3, 7),
        "one_of_two_wrong": (3, 7),
        "nonexistent_label": (3, 7),
    },
    "C": {
        "correct_value": (12, 28),
        "correct_other": (9, 21),
        "off_target": (9, 21),
        "unanswerable_answered": (9, 21),
        "correct_refusal": (9, 21),
    },
    "D": {
        "plain_conflict": (6, 14),
        "synonym_conflict": (6, 14),
        "range_shared_endpoint": (6, 14),
        "requirement_denial": (6, 14),
        "revision_date_only": (3, 7),
        "validity_dates": (6, 14),
        "agreement": (6, 14),
        "different_groups": (6, 14),
        "incomplete": (6, 14),
        "range_consistent": (6, 14),
        "dated_agreement": (3, 7),
    },
}
DOMAINS: dict[str, tuple[str, ...]] = {
    "dev": ("municipal waste collection", "language school", "marina berths", "childcare nursery"),
    "test": (
        "university exam rules",
        "veterinary clinic",
        "car-sharing service",
        "dental practice",
        "ferry operator",
        "laboratory safety",
        "pension fund member guide",
        "theatre ticketing",
    ),
}
VALUE_QUESTION_TYPES = (
    "amount",
    "duration",
    "date",
    "time",
    "count",
    "person_or_role",
    "code",
    "place",
)
OTHER_QUESTION_TYPES = ("yes_no", "procedure", "reason")

# Gold labels implied by each subtype.
A_SUPPORTED = {
    "verbatim": True,
    "paraphrase": True,
    "number_form": True,
    "wrong_value": False,
    "wrong_subject": False,
    "negation": False,
    "absent": False,
}
B_CORRECT = {
    "correct_single": True,
    "correct_two": True,
    "wrong_related": False,
    "wrong_unrelated": False,
    "one_of_two_wrong": False,
    "nonexistent_label": False,
}
C_KIND: dict[str, tuple[str, bool]] = {
    "correct_value": ("correct_answer", True),
    "correct_other": ("correct_answer", True),
    "off_target": ("off_target", True),
    "unanswerable_answered": ("unanswerable_answered", False),
    "correct_refusal": ("correct_refusal", False),
}
D_KIND = {
    "plain_conflict": "conflict",
    "synonym_conflict": "conflict",
    "range_shared_endpoint": "conflict",
    "requirement_denial": "conflict",
    "revision_date_only": "conflict",
    "validity_dates": "different_dates",
    "agreement": "none",
    "different_groups": "none",
    "incomplete": "none",
    "range_consistent": "none",
    "dated_agreement": "none",
}

PASSAGE_CHARS = (300, 800)  # the specification's range; outside it is a warning
PASSAGE_CHARS_HARD = (150, 1200)  # outside this it is an error
D_PASSAGE_CHARS = (100, 800)
D_PASSAGE_CHARS_HARD = (40, 1200)
_ID_RE = re.compile(r"^(?P<category>[ABCD])-(?P<split>[DT])-(?P<number>\d{3})$")
_PASSAGE_ID_RE = re.compile(r"^P-(?P<split>[DT])-\d{3,4}$")
_LABEL_RE = re.compile(r"^S([1-9]\d*)$")


# ------------------------------------------------------------------------- hashes
def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes with CRLF replaced by LF (the same on every platform)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def combined_sha256(files: Mapping[str, str]) -> str:
    """SHA-256 of ``<path>\\t<hash>\\n`` lines sorted by path."""
    text = "".join(f"{path}\t{files[path]}\n" for path in sorted(files))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


HASHED_DIRS = ("dev", "test", "documents", "notes")


def dataset_files(root: Path) -> dict[str, str]:
    """Hash of every generated file (cases, documents, notes), keyed by relative path."""
    files: dict[str, str] = {}
    for directory in HASHED_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = file_sha256(path)
    return files


def split_sha256(files: Mapping[str, str], split: str) -> str:
    """Combined hash of one split's cases and documents."""
    prefixes = (f"{split}/", f"documents/{split}/")
    return combined_sha256({p: h for p, h in files.items() if p.startswith(prefixes)})


def hash_record(root: Path) -> dict[str, Any]:
    files = dataset_files(root)
    return {
        "files": files,
        "combined_sha256": combined_sha256(files),
        "dev_sha256": split_sha256(files, "dev"),
        "test_sha256": split_sha256(files, "test"),
    }


# ------------------------------------------------------------------------- loading
@dataclass(frozen=True, slots=True)
class Passage:
    label: str
    document_id: str
    title: str
    text: str
    passage_id: str | None = None


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    category: str
    split: str
    subtype: str
    domain: str
    question: str
    passages: tuple[Passage, ...]
    data: Mapping[str, Any]

    @property
    def gold(self) -> Mapping[str, Any]:
        value = self.data.get("gold")
        return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True, slots=True)
class Documents:
    titles: dict[str, str]  # document_id -> title
    bodies: dict[str, str]  # document_id -> text without the title line
    passages: dict[str, tuple[str, str]]  # passage_id -> (document_id, text)


def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            rows.append((number, json.loads(line)))
    return rows


def load_documents(root: Path, split: str) -> Documents:
    base = root / "documents" / split
    titles: dict[str, str] = {}
    bodies: dict[str, str] = {}
    for path in sorted(base.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        titles[path.stem] = lines[0].strip().lstrip("#").strip() if lines else ""
        bodies[path.stem] = "\n".join(lines[1:])
    passages: dict[str, tuple[str, str]] = {}
    pool = base / "passages.jsonl"
    if pool.is_file():
        for _number, row in _read_jsonl(pool):
            passages[str(row["passage_id"])] = (str(row["document_id"]), str(row["text"]))
    return Documents(titles, bodies, passages)


def _case(raw: Mapping[str, Any], documents: Documents | None) -> Case:
    passages = []
    for item in raw.get("passages") or []:
        if "passage_id" in item and documents is not None:
            document_id, text = documents.passages[item["passage_id"]]
            title = documents.titles.get(document_id, "")
            passages.append(Passage(item["label"], document_id, title, text, item["passage_id"]))
        else:
            passages.append(
                Passage(item["label"], item["document_id"], item.get("title", ""), item["text"])
            )
    return Case(
        id=str(raw["id"]),
        category=str(raw["category"]),
        split=str(raw["split"]),
        subtype=str(raw["subtype"]),
        domain=str(raw.get("domain", "")),
        question=str(raw.get("question", "")),
        passages=tuple(passages),
        data=raw,
    )


def load_cases(root: Path, split: str, categories: Iterable[str] = CATEGORIES) -> list[Case]:
    documents = load_documents(root, split)
    cases = []
    for category in categories:
        path = root / split / f"{category}.jsonl"
        for _number, raw in _read_jsonl(path):
            cases.append(_case(raw, documents if category != "D" else None))
    return cases


# ---------------------------------------------------------------------- validation
@dataclass(frozen=True, slots=True)
class Issue:
    """A validation finding. ``where`` is a file or case id; no case text is ever included."""

    severity: str  # "error" or "warning"
    code: str
    where: str
    detail: str = ""

    def __str__(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.severity}: {self.code} at {self.where}{suffix}"


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _question_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _labels_ok(labels: Sequence[str]) -> bool:
    return list(labels) == [f"S{index}" for index in range(1, len(labels) + 1)]


def validate_suite(
    root: Path, counts_expected: Mapping[str, Mapping[str, tuple[int, int]]] = COUNTS
) -> tuple[list[Issue], dict[str, Any]]:
    """Check the suite against the specification. Returns issues and counts (no case text)."""
    issues: list[Issue] = []
    counts: dict[str, Any] = {split: defaultdict(Counter) for split in SPLITS}

    def add(severity: str, code: str, where: str, detail: str = "") -> None:
        issues.append(Issue(severity, code, where, detail))

    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        add("error", "missing_manifest", "manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            add("error", "bad_json", "manifest.json")

    questions: dict[str, str] = {}
    all_ids: set[str] = set()
    for split in SPLITS:
        documents = _validate_documents(root, split, add)
        for category in CATEGORIES:
            path = root / split / f"{category}.jsonl"
            if not path.is_file():
                add("error", "missing_file", f"{split}/{category}.jsonl")
                continue
            try:
                rows = _read_jsonl(path)
            except json.JSONDecodeError as exc:
                add("error", "bad_json", f"{split}/{category}.jsonl", f"line {exc.lineno}")
                continue
            for line, raw in rows:
                where = str(raw.get("id") or f"{split}/{category}.jsonl:{line}")
                if not isinstance(raw, dict):
                    add("error", "not_an_object", where)
                    continue
                _validate_case(raw, split, category, documents, add, where)
                if where in all_ids:
                    add("error", "duplicate_id", where)
                all_ids.add(where)
                key = _question_key(str(raw.get("question", "")))
                if key and key in questions:
                    add("error", "duplicate_question", where, f"same as {questions[key]}")
                questions.setdefault(key, where)
                counts[split][category][str(raw.get("subtype"))] += 1
    for split_index, split in enumerate(SPLITS):
        for category, subtypes in counts_expected.items():
            for subtype, expected in subtypes.items():
                found = counts[split][category][subtype]
                if found != expected[split_index]:
                    add(
                        "error",
                        "wrong_count",
                        f"{split}/{category}/{subtype}",
                        f"{found} instead of {expected[split_index]}",
                    )
    if manifest:
        _validate_manifest(root, manifest, add)
    summary = {
        split: {category: dict(counts[split][category]) for category in CATEGORIES}
        for split in SPLITS
    }
    return issues, summary


def _validate_documents(
    root: Path, split: str, add: Callable[[str, str, str, str], None]
) -> Documents:
    base = root / "documents" / split
    if not base.is_dir():
        add("error", "missing_documents", f"documents/{split}", "")
        return Documents({}, {}, {})
    documents = load_documents(root, split)
    if not documents.passages:
        add("error", "missing_passages", f"documents/{split}/passages.jsonl", "")
    prefix = "dev-" if split == "dev" else "test-"
    for document_id in documents.titles:
        if not document_id.startswith(prefix):
            add("error", "document_split", f"documents/{split}/{document_id}.md", "")
    seen_text: dict[str, str] = {}
    for passage_id, (document_id, text) in documents.passages.items():
        where = f"documents/{split}/passages.jsonl:{passage_id}"
        match = _PASSAGE_ID_RE.match(passage_id)
        if not match or match.group("split") != split[0].upper():
            add("error", "passage_id", where, "")
        if document_id not in documents.bodies:
            add("error", "passage_document", where, "unknown document")
            continue
        if _normalized(text) not in _normalized(documents.bodies[document_id]):
            add("error", "passage_not_verbatim", where, "")
        size = len(text)
        if not PASSAGE_CHARS_HARD[0] <= size <= PASSAGE_CHARS_HARD[1]:
            add("error", "passage_length", where, f"{size} characters")
        elif not PASSAGE_CHARS[0] <= size <= PASSAGE_CHARS[1]:
            add("warning", "passage_length", where, f"{size} characters")
        key = _normalized(text)
        if key in seen_text:
            add("error", "duplicate_passage", where, f"same as {seen_text[key]}")
        seen_text.setdefault(key, passage_id)
    return documents


def _validate_case(
    raw: Mapping[str, Any],
    split: str,
    category: str,
    documents: Documents,
    add: Callable[[str, str, str, str], None],
    where: str,
) -> None:
    match = _ID_RE.match(str(raw.get("id", "")))
    if not match or match.group("category") != category or match.group("split") != split[0].upper():
        add("error", "bad_id", where, "")
    for key, expected in (("category", category), ("split", split)):
        if raw.get(key) != expected:
            add("error", "wrong_field", where, key)
    subtype = str(raw.get("subtype"))
    if subtype not in COUNTS[category]:
        add("error", "unknown_subtype", where, "subtype")
        return
    if raw.get("domain") not in DOMAINS[split]:
        add("error", "wrong_domain", where, "domain")
    for key in ("question", "rationale"):
        if not str(raw.get(key) or "").strip():
            add("error", "empty_field", where, key)
    passages = raw.get("passages")
    if not isinstance(passages, list) or not passages:
        add("error", "missing_passages", where, "passages")
        return
    labels = [str(item.get("label")) for item in passages if isinstance(item, Mapping)]
    if not _labels_ok(labels):
        add("error", "passage_labels", where, "labels must be S1..Sn in order")
    gold = raw.get("gold")
    if not isinstance(gold, Mapping):
        add("error", "missing_gold", where, "gold")
        return
    if category in "ABC":
        if not 3 <= len(passages) <= 5:
            add("error", "passage_count", where, f"{len(passages)} passages")
        for item in passages:
            passage_id = str(item.get("passage_id"))
            if passage_id not in documents.passages:
                add("error", "unknown_passage", where, passage_id)
        _validate_abc(raw, category, subtype, labels, gold, add, where)
    else:
        _validate_d(passages, subtype, labels, gold, add, where)


def _validate_abc(
    raw: Mapping[str, Any],
    category: str,
    subtype: str,
    labels: list[str],
    gold: Mapping[str, Any],
    add: Callable[[str, str, str, str], None],
    where: str,
) -> None:
    known = set(labels)
    if category == "A":
        if not str(raw.get("statement") or "").strip():
            add("error", "empty_field", where, "statement")
        supporting = list(gold.get("supporting_labels") or [])
        if gold.get("supported") is not A_SUPPORTED[subtype]:
            add("error", "gold_subtype_mismatch", where, "supported")
        if not set(supporting) <= known:
            add("error", "gold_labels", where, "supporting_labels")
        if A_SUPPORTED[subtype] and not supporting:
            add("error", "gold_labels", where, "supported without supporting_labels")
        if not A_SUPPORTED[subtype] and supporting:
            add("error", "gold_labels", where, "unsupported with supporting_labels")
    elif category == "B":
        statement = str(raw.get("statement") or "")
        cited = [str(label) for label in raw.get("cited_labels") or []]
        if not statement.strip():
            add("error", "empty_field", where, "statement")
        if not cited or sorted(set(cited)) != sorted(set(labels_in(statement))):
            add("error", "cited_labels", where, "cited_labels must match the statement")
        if gold.get("citation_correct") is not B_CORRECT[subtype]:
            add("error", "gold_subtype_mismatch", where, "citation_correct")
        supporting = list(gold.get("supporting_labels") or [])
        if not supporting or not set(supporting) <= known:
            add("error", "gold_labels", where, "supporting_labels")
        outside = [label for label in cited if label not in known]
        if (subtype == "nonexistent_label") != bool(outside):
            add("error", "cited_labels", where, "label outside the passages")
        if subtype in ("correct_two", "one_of_two_wrong") and len(set(cited)) != 2:
            add("error", "cited_labels", where, "two labels expected")
    else:
        if not str(raw.get("answer") or "").strip():
            add("error", "empty_field", where, "answer")
        kind, answerable = C_KIND[subtype]
        if gold.get("answer_kind") != kind or gold.get("answerable") is not answerable:
            add("error", "gold_subtype_mismatch", where, "answer_kind/answerable")
        question_type = gold.get("question_type")
        allowed = {
            "correct_value": VALUE_QUESTION_TYPES,
            "correct_other": OTHER_QUESTION_TYPES,
        }.get(subtype, VALUE_QUESTION_TYPES + OTHER_QUESTION_TYPES)
        if question_type not in allowed:
            add("error", "question_type", where, "question_type")


def _validate_d(
    passages: list[Any],
    subtype: str,
    labels: list[str],
    gold: Mapping[str, Any],
    add: Callable[[str, str, str, str], None],
    where: str,
) -> None:
    if not 2 <= len(passages) <= 5:
        add("error", "passage_count", where, f"{len(passages)} passages")
    documents = {str(item.get("document_id")) for item in passages}
    if len(documents) < 2:
        add("error", "documents", where, "passages from at least two documents expected")
    by_label = {str(item.get("label")): str(item.get("document_id")) for item in passages}
    for item in passages:
        text = str(item.get("text") or "")
        if not str(item.get("title") or "").strip() or not text.strip():
            add("error", "empty_field", where, "passage title/text")
        if not D_PASSAGE_CHARS_HARD[0] <= len(text) <= D_PASSAGE_CHARS_HARD[1]:
            add("error", "passage_length", where, f"{len(text)} characters")
        elif not D_PASSAGE_CHARS[0] <= len(text) <= D_PASSAGE_CHARS[1]:
            add("warning", "passage_length", where, f"{len(text)} characters")
    kind = D_KIND[subtype]
    positive = kind != "none"
    if gold.get("kind") != kind or gold.get("disagreement") is not positive:
        add("error", "gold_subtype_mismatch", where, "kind/disagreement")
    pair = gold.get("pair")
    if positive:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or len(set(pair)) != 2
            or not set(pair) <= set(labels)
        ):
            add("error", "gold_pair", where, "two passage labels expected")
        elif by_label.get(pair[0]) == by_label.get(pair[1]):
            add("error", "gold_pair", where, "pair from one document")
    elif pair is not None:
        add("error", "gold_pair", where, "null expected")


def _validate_manifest(
    root: Path, manifest: Mapping[str, Any], add: Callable[[str, str, str, str], None]
) -> None:
    record = hash_record(root)
    listed = manifest.get("files") or {}
    for path, digest in record["files"].items():
        if listed.get(path) != digest:
            add("error", "hash_mismatch", path, "")
    for path in listed:
        if path not in record["files"]:
            add("error", "listed_file_missing", path, "")
    for key in ("combined_sha256", "dev_sha256", "test_sha256"):
        if key in manifest and manifest[key] != record[key]:
            add("error", "hash_mismatch", f"manifest.json:{key}", "")


# ------------------------------------------------------------------------ checks
@dataclass(frozen=True, slots=True)
class Checks:
    """One version of the answer checks, as the suite runs it."""

    name: str
    versions: Mapping[str, str | None]
    predict: Callable[[Case], dict[str, Any]]


def build_context(case: Case) -> AssembledContext:
    """The passages as the application would label and present them to its checks."""
    chunks = [
        RetrievedChunk(
            f"{case.id}:{passage.label}",
            passage.document_id,
            passage.title,
            index,
            passage.text,
            0,
            len(passage.text),
            None,
            None,
            ChunkScores(),
        )
        for index, passage in enumerate(case.passages)
    ]
    context = assemble_context(chunks, max_chars=1_000_000)
    if [block.label for block in context.sources] != [p.label for p in case.passages]:
        raise ValueError(f"{case.id}: passage labels changed while assembling the context")
    return context


def evidence(context: AssembledContext) -> list[EvidenceSource]:
    return [
        EvidenceSource(block.label, block.chunk.document_title, block.text)
        for block in context.sources
    ]


def _predict_v1(case: Case) -> dict[str, Any]:
    context = build_context(case)
    if case.category == "A":
        report = check_claims(str(case.data["statement"]), evidence(context), case.question)
        found = bool(report.claims) and all(c.status in MATCHED for c in report.claims)
        return {"flagged": not found}
    if case.category == "B":
        report = check_claims(str(case.data["statement"]), evidence(context), case.question)
        invalid = any(not reference.valid for reference in report.references)
        cited = [claim for claim in report.claims if claim.cited]
        correct = (
            not invalid and bool(cited) and all(claim.status == "supported" for claim in cited)
        )
        return {"judged_correct": correct}
    if case.category == "C":
        sources = {block.label: block.text for block in context.sources}
        parsed = parse_answer(str(case.data["answer"]), sources)
        # Version 1 has no check of whether an answer addresses the question.
        return {"flagged": False, "refusal": parsed.is_abstention}
    reports = find_disagreements(case.question, context.sources)
    return {"reports": [{"pair": sorted(d.labels), "kind": d.kind} for d in reports]}


CHECKS: dict[str, Checks] = {
    "v1": Checks(
        "v1",
        {
            "claim_check": "claims-v1",
            "conflict_check": "disagreement-v1",
            "relevance_check": None,
        },
        _predict_v1,
    ),
}


# ------------------------------------------------------------------------ scoring
def case_outcome(case: Case, prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Booleans the measures are computed from. ``correct`` feeds the paired comparison."""
    gold = case.gold
    outcome: dict[str, Any] = {"category": case.category, "subtype": case.subtype}
    if case.category == "A":
        positive = gold.get("supported") is False
        flagged = bool(prediction["flagged"])
        outcome.update(positive=positive, flagged=flagged, correct=flagged == positive)
    elif case.category == "B":
        gold_correct = gold.get("citation_correct") is True
        judged = bool(prediction["judged_correct"])
        outcome.update(
            gold_correct=gold_correct, judged_correct=judged, correct=judged == gold_correct
        )
    elif case.category == "C":
        refusal = bool(prediction["refusal"])
        if case.subtype == "correct_refusal":
            outcome.update(refusal_case=True, refusal=refusal, correct=refusal)
        else:
            positive = gold.get("answer_kind") in ("off_target", "unanswerable_answered")
            flagged = bool(prediction["flagged"]) or refusal
            outcome.update(
                refusal_case=False, positive=positive, flagged=flagged, correct=flagged == positive
            )
    else:
        positive = gold.get("disagreement") is True
        reports = list(prediction["reports"])
        pair = sorted(gold.get("pair") or [])
        matching = [report for report in reports if sorted(report["pair"]) == pair]
        detected = positive and bool(matching)
        outcome.update(
            positive=positive,
            reported=bool(reports),
            detected=detected,
            kind_right=detected and any(r["kind"] == gold.get("kind") for r in matching),
            correct=detected if positive else not reports,
        )
    return outcome


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes in n trials."""
    if n <= 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(fixed: int, broke: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = fixed + broke
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broke) + 1)) / (2.0**n)
    return min(1.0, 2 * tail)


@dataclass(frozen=True, slots=True)
class Target:
    target: float
    stretch: float
    higher_is_better: bool = True


# Targets to investigate, not results to expect (README, "Measures and targets").
TARGETS: dict[str, Target] = {
    "A.unsupported_recall": Target(0.80, 0.90),
    "A.unsupported_precision": Target(0.80, 0.90),
    "A.supported_false_flag_rate": Target(0.20, 0.10, higher_is_better=False),
    "A.paraphrase_acceptance": Target(0.80, 0.90),
    "B.correct_citation_precision": Target(0.80, 0.90),
    "B.correct_citation_recall": Target(0.80, 0.90),
    "B.wrong_citation_recall": Target(0.80, 0.90),
    "C.off_target_recall": Target(0.80, 0.90),
    "C.false_warning_rate": Target(0.20, 0.10, higher_is_better=False),
    "C.refusal_recognition": Target(0.80, 0.90),
    "D.disagreement_recall": Target(0.80, 0.90),
    "D.disagreement_precision": Target(0.80, 0.90),
    "D.false_alarm_rate": Target(0.20, 0.10, higher_is_better=False),
    "D.kind_accuracy": Target(0.80, 0.90),
}


def measure(name: str, k: int, n: int) -> dict[str, Any]:
    interval = wilson(k, n)
    record: dict[str, Any] = {
        "k": k,
        "n": n,
        "value": round(k / n, 4) if n else None,
        "ci95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
    }
    target = TARGETS.get(name)
    if target and n and interval:
        value = k / n
        if target.higher_is_better:
            met, stretch, established = (
                value >= target.target,
                value >= target.stretch,
                interval[0] >= target.target,
            )
        else:
            met, stretch, established = (
                value <= target.target,
                value <= target.stretch,
                interval[1] <= target.target,
            )
        record.update(
            target=target.target,
            stretch=target.stretch,
            direction="higher" if target.higher_is_better else "lower",
            met=met,
            stretch_met=stretch,
            established=established,
        )
    return record


def _f1(precision: Mapping[str, Any], recall: Mapping[str, Any]) -> float | None:
    p, r = precision.get("value"), recall.get("value")
    if p is None or r is None or p + r == 0:
        return None
    return round(2 * float(p) * float(r) / (float(p) + float(r)), 4)


def summarize(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The measures of each category (counts and rates only)."""
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        by_category[str(outcome["category"])].append(outcome)
    summary: dict[str, Any] = {}
    if rows := by_category.get("A"):
        pos = [r for r in rows if r["positive"]]
        neg = [r for r in rows if not r["positive"]]
        tp = sum(r["flagged"] for r in pos)
        fp = sum(r["flagged"] for r in neg)
        para = [r for r in neg if r["subtype"] == "paraphrase"]
        m: dict[str, Any] = {
            "unsupported_recall": measure("A.unsupported_recall", tp, len(pos)),
            "unsupported_precision": measure("A.unsupported_precision", tp, tp + fp),
            "supported_false_flag_rate": measure("A.supported_false_flag_rate", fp, len(neg)),
            "false_negative_rate": measure("A.false_negative_rate", len(pos) - tp, len(pos)),
            "supported_acceptance": measure("A.supported_acceptance", len(neg) - fp, len(neg)),
            "paraphrase_acceptance": measure(
                "A.paraphrase_acceptance", sum(not r["flagged"] for r in para), len(para)
            ),
        }
        m["f1"] = _f1(m["unsupported_precision"], m["unsupported_recall"])
        summary["A"] = m
    if rows := by_category.get("B"):
        gold_ok = [r for r in rows if r["gold_correct"]]
        gold_wrong = [r for r in rows if not r["gold_correct"]]
        judged_ok = [r for r in rows if r["judged_correct"]]
        judged_wrong = [r for r in rows if not r["judged_correct"]]
        both_ok = sum(r["gold_correct"] for r in judged_ok)
        both_wrong = sum(not r["gold_correct"] for r in judged_wrong)
        m = {
            "correct_citation_precision": measure(
                "B.correct_citation_precision", both_ok, len(judged_ok)
            ),
            "correct_citation_recall": measure("B.correct_citation_recall", both_ok, len(gold_ok)),
            "wrong_citation_precision": measure(
                "B.wrong_citation_precision", both_wrong, len(judged_wrong)
            ),
            "wrong_citation_recall": measure(
                "B.wrong_citation_recall", both_wrong, len(gold_wrong)
            ),
            "false_positive_rate": measure(
                "B.false_positive_rate", len(gold_ok) - both_ok, len(gold_ok)
            ),
            "false_negative_rate": measure(
                "B.false_negative_rate", len(gold_wrong) - both_wrong, len(gold_wrong)
            ),
            "accuracy": measure("B.accuracy", both_ok + both_wrong, len(rows)),
        }
        m["f1"] = _f1(m["wrong_citation_precision"], m["wrong_citation_recall"])
        summary["B"] = m
    if rows := by_category.get("C"):
        refusals = [r for r in rows if r["refusal_case"]]
        rest = [r for r in rows if not r["refusal_case"]]
        pos = [r for r in rest if r["positive"]]
        neg = [r for r in rest if not r["positive"]]
        tp = sum(r["flagged"] for r in pos)
        fp = sum(r["flagged"] for r in neg)
        m = {
            "off_target_recall": measure("C.off_target_recall", tp, len(pos)),
            "off_target_precision": measure("C.off_target_precision", tp, tp + fp),
            "false_warning_rate": measure("C.false_warning_rate", fp, len(neg)),
            "false_negative_rate": measure("C.false_negative_rate", len(pos) - tp, len(pos)),
            "refusal_recognition": measure(
                "C.refusal_recognition", sum(r["refusal"] for r in refusals), len(refusals)
            ),
        }
        m["f1"] = _f1(m["off_target_precision"], m["off_target_recall"])
        summary["C"] = m
    if rows := by_category.get("D"):
        pos = [r for r in rows if r["positive"]]
        neg = [r for r in rows if not r["positive"]]
        tp = sum(r["detected"] for r in pos)
        reported = sum(r["reported"] for r in rows)
        false_alarms = sum(r["reported"] for r in neg)
        m = {
            "disagreement_recall": measure("D.disagreement_recall", tp, len(pos)),
            "disagreement_precision": measure("D.disagreement_precision", tp, reported),
            "false_alarm_rate": measure("D.false_alarm_rate", false_alarms, len(neg)),
            "false_negative_rate": measure("D.false_negative_rate", len(pos) - tp, len(pos)),
            "kind_accuracy": measure("D.kind_accuracy", sum(r["kind_right"] for r in pos), tp),
        }
        m["f1"] = _f1(m["disagreement_precision"], m["disagreement_recall"])
        summary["D"] = m
    return summary


def by_subtype(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Share of correctly handled cases per subtype (descriptive; small n)."""
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for outcome in outcomes:
        groups[(str(outcome["category"]), str(outcome["subtype"]))].append(bool(outcome["correct"]))
    result: dict[str, dict[str, Any]] = defaultdict(dict)
    for (category, subtype), values in sorted(groups.items()):
        result[category][subtype] = measure(f"{category}.{subtype}", sum(values), len(values))
    return dict(result)


def paired(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Exact McNemar comparison of two runs on the same cases, per category."""
    result: dict[str, Any] = {}
    for category in CATEGORIES:
        ids = [
            case_id
            for case_id, outcome in before.items()
            if outcome["category"] == category and case_id in after
        ]
        fixed = sum(not before[i]["correct"] and after[i]["correct"] for i in ids)
        broke = sum(before[i]["correct"] and not after[i]["correct"] for i in ids)
        if ids:
            result[category] = {
                "cases": len(ids),
                "fixed": fixed,
                "broke": broke,
                "p_value": round(mcnemar_exact(fixed, broke), 6),
            }
    return result


def run_checks(
    cases: Sequence[Case], checks: Checks
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Predictions and outcomes keyed by case id."""
    predictions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for case in cases:
        prediction = checks.predict(case)
        predictions[case.id] = prediction
        outcomes[case.id] = case_outcome(case, prediction)
    return predictions, outcomes


def gold_label(case: Case) -> str:
    """The case's gold label as one word, as used in the label audit."""
    gold = case.gold
    if case.category == "A":
        return "supported" if gold.get("supported") else "not_supported"
    if case.category == "B":
        return "correct" if gold.get("citation_correct") else "wrong"
    if case.category == "C":
        return str(gold.get("answer_kind"))
    return str(gold.get("kind"))


AUDIT_LABELS: dict[str, tuple[tuple[str, str], ...]] = {
    "A": (
        ("supported", "Supported: one passage states every part of the statement"),
        ("not_supported", "Not supported: some part is wrong or stated nowhere"),
    ),
    "B": (
        ("correct", "Citation correct"),
        ("wrong", "Citation wrong"),
    ),
    "C": (
        ("correct_answer", "Correct answer: it gives what was asked"),
        (
            "off_target",
            "Off target: the passages contain the answer, but the answer gives other facts",
        ),
        (
            "unanswerable_answered",
            "Not answerable from the passages, but the answer gives other facts as if answering",
        ),
        ("correct_refusal", "Correct refusal: says the documents do not give it"),
    ),
    "D": (
        ("conflict", "Conflict"),
        ("different_dates", "Different values for different stated periods"),
        ("none", "No disagreement"),
    ),
}


def as_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

"""Versioned evaluation datasets.

Layout of a dataset directory::

    manifest.json      name, version, licence, corpus documents with SHA-256 hashes
    questions.jsonl    one question per line
    corpus/            the documents (text and PDF)

Relevance is annotated as *evidence quotes* (exact sentences from a corpus
document), not as chunk ids, so judgments stay valid when chunking changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "factual",
    "multi_chunk",
    "synthesis",
    "unanswerable",
    "out_of_scope",
    "ambiguous",
    "conflicting",
    "injection",
]
ExpectedBehavior = Literal["answer", "abstain", "answer_or_clarify", "flag_conflict"]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc: str
    quote: str = Field(min_length=10)


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Category
    question: str
    expected_behavior: ExpectedBehavior
    evidence: list[EvidenceItem] = Field(default_factory=list)
    answer_must_include: list[str] = Field(
        default_factory=list,
        description="Key facts; alternatives separated by '|' (e.g. '€140|140 euros').",
    )
    reference_answer: str | None = None
    forbidden_output: list[str] = Field(default_factory=list)
    notes: str | None = None

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    file: str
    title: str
    sha256: str
    description: str | None = None


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    license: str
    description: str
    created: str
    documents: list[CorpusDocument]


@dataclass(frozen=True)
class EvalDataset:
    root: Path
    manifest: DatasetManifest
    questions: list[EvalQuestion]
    fingerprint: str

    @property
    def documents(self) -> dict[str, CorpusDocument]:
        return {doc.key: doc for doc in self.manifest.documents}

    def document_path(self, key: str) -> Path:
        return self.root / self.documents[key].file

    def document_bytes(self, key: str) -> bytes:
        return self.document_path(key).read_bytes()

    def with_evidence(self) -> list[EvalQuestion]:
        return [question for question in self.questions if question.has_evidence]


class DatasetError(ValueError):
    pass


def load_dataset(root: Path, *, verify_hashes: bool = True) -> EvalDataset:
    manifest_path = root / "manifest.json"
    questions_path = root / "questions.jsonl"
    if not manifest_path.is_file() or not questions_path.is_file():
        raise DatasetError(f"{root} is not a dataset directory")
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    # The two metadata files are hashed with LF line endings: a manifest rewritten on Windows
    # (CRLF) must give the same fingerprint as a fresh clone. Documents are hashed as stored.
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes().replace(b"\r\n", b"\n"))
    digest.update(questions_path.read_bytes().replace(b"\r\n", b"\n"))

    keys = set()
    for document in manifest.documents:
        if document.key in keys:
            raise DatasetError(f"duplicate document key {document.key}")
        keys.add(document.key)
        path = (root / document.file).resolve()
        if root.resolve() not in path.parents:
            raise DatasetError(f"document path escapes dataset: {document.file}")
        if verify_hashes:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != document.sha256:
                raise DatasetError(f"hash mismatch for {document.file}")
        digest.update(path.read_bytes())

    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    for number, line in enumerate(questions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            question = EvalQuestion.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DatasetError(f"questions.jsonl line {number}: {exc}") from exc
        if question.id in seen_ids:
            raise DatasetError(f"duplicate question id {question.id}")
        seen_ids.add(question.id)
        for item in question.evidence:
            if item.doc not in keys:
                raise DatasetError(f"{question.id}: unknown document key {item.doc}")
        if question.expected_behavior in ("answer", "flag_conflict") and not question.evidence:
            raise DatasetError(f"{question.id}: answerable questions need evidence")
        questions.append(question)

    return EvalDataset(
        root=root, manifest=manifest, questions=questions, fingerprint=digest.hexdigest()[:16]
    )

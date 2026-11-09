"""Build claim-evidence pairs with four labels from training or validation PDFs, by rule.

    python experiments/verifier/make_verifier_data.py --split train \
        --out experiments/verifier/data/train.jsonl
    python experiments/verifier/make_verifier_data.py --split validation --out .../val.jsonl
    python experiments/verifier/make_verifier_data.py --split synthetic --out .../synthetic.jsonl

Labels: ``supported``, ``partial`` (part of the claim is stated, part is not), ``contradicted``,
``nee`` (not enough evidence). Each pair takes a sentence S of a GroundedAI chunk; the evidence
is a window of one to three sentences around S. Claims are made by rule (as in FactCC,
Kryscinski et al. 2020):

* supported: S itself, S shortened (a parenthetical or a trailing clause dropped), or S with
  wording swaps that keep the meaning ("SHALL" -> "must", "at least" -> "no less than").
* contradicted: a number in S changed, a negation added or removed, a modal changed in strength
  ("SHALL" -> "MAY"), or an acronym or capitalised term swapped for another from the document.
* partial: S (or its first clause) plus a detail the evidence does not state (a clause from
  another part of the document, or a templated extra requirement).
* nee: a sentence from another part of the same document that shares few words with the
  evidence.

Rule-made labels are noisy and the claims are stylised; the model is tested on hand-written
claims (evaluation/datasets/groundedai-claims-nist-v1), never on these. The split names the
documents used: train = experiments/finetune/corpus.json, validation = groundedai-nist-val-v1,
synthetic = the fictional documents of groundedai-synthetic-v1 (an out-of-domain check).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.core.text import content_tokens, split_sentences  # noqa: E402
from groundedai.ingestion.chunking import ChunkingConfig, chunk_pages  # noqa: E402
from groundedai.ingestion.parsers import parse_document  # noqa: E402

SEED = 20261109
NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
NEGATIONS = [
    (r"\bSHALL NOT\b", "SHALL"),
    (r"\bSHOULD NOT\b", "SHOULD"),
    (r"\bMUST NOT\b", "MUST"),
    (r"\bshall not\b", "shall"),
    (r"\bshould not\b", "should"),
    (r"\bmust not\b", "must"),
    (r"\bis not\b", "is"),
    (r"\bare not\b", "are"),
    (r"\bcannot\b", "can"),
    (r"\bdo not\b", "do"),
    (r"\bSHALL\b", "SHALL NOT"),
    (r"\bSHOULD\b", "SHOULD NOT"),
    (r"\bMUST\b", "MUST NOT"),
    (r"\bshall\b", "shall not"),
    (r"\bshould\b", "should not"),
    (r"\bmust\b", "must not"),
    (r"\bis\b", "is not"),
    (r"\bare\b", "are not"),
    (r"\bcan\b", "cannot"),
]
MODALS = [
    (r"\bSHALL\b", "MAY"),
    (r"\bMUST\b", "MAY"),
    (r"\bSHOULD NOT\b", "MAY"),
    (r"\bshall\b", "may"),
    (r"\bmust\b", "may"),
    (r"\brequired\b", "optional"),
]
KEEP_MEANING = [
    (r"\bSHALL\b", "must"),
    (r"\bSHOULD\b", "should"),
    (r"\bat least\b", "no less than"),
    (r"\bno more than\b", "at most"),
    (r"\bin order to\b", "to"),
    (r"\butilize\b", "use"),
]
EXTRAS = [
    "and it must be reviewed every {n} months",
    "and this applies only to systems with more than {n} users",
    "and it must be approved by the chief information officer",
    "and it must be recorded in a public registry within {n} days",
    "and organizations must report it to NIST each year",
    "and it requires a certificate from an accredited laboratory",
    "and the result must be kept for at least {n} years",
]
TERM_RE = re.compile(r"\b(?:[A-Z]{2,}[0-9]?|[A-Z][a-z]+(?: [A-Z][a-z]+)+)\b")


def documents(split: str) -> list[tuple[str, str, Path]]:
    if split == "train":
        manifest_path = ROOT / "experiments" / "finetune" / "corpus.json"
    elif split == "validation":
        manifest_path = (
            ROOT / "evaluation" / "datasets" / "groundedai-nist-val-v1" / "manifest.json"
        )
    elif split == "synthetic":
        manifest_path = (
            ROOT / "evaluation" / "datasets" / "groundedai-synthetic-v1" / "manifest.json"
        )
    else:
        raise SystemExit("test documents are never used to make training data")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [(d["key"], d["title"], manifest_path.parent / d["file"]) for d in manifest["documents"]]


def fact_sentences(text: str) -> list[str]:
    out = []
    for sentence in split_sentences(text):
        words = sentence.split()
        letters = sum(ch.isalpha() for ch in sentence)
        if 8 <= len(words) <= 45 and letters / max(1, len(sentence)) > 0.7 and ".." not in sentence:
            out.append(" ".join(words))
    return out


def first_change(sentence: str, rules: list[tuple[str, str]], rng: random.Random) -> str | None:
    options = [(pattern, new) for pattern, new in rules if re.search(pattern, sentence)]
    if not options:
        return None
    pattern, new = rng.choice(options)
    return re.sub(pattern, new, sentence, count=1)


def change_number(sentence: str, rng: random.Random) -> str | None:
    numbers = NUMBER_RE.findall(sentence)
    numbers = [n for n in numbers if not (len(n) == 4 and n.startswith(("19", "20")))] or numbers
    if not numbers:
        return None
    old = rng.choice(numbers)
    value = float(old)
    candidates = {value * 2, value + 1, max(0.0, value - 1), value * 10, value / 2}
    candidates.discard(value)
    new_value = rng.choice(sorted(candidates))
    new = str(int(new_value)) if new_value == int(new_value) else f"{new_value:g}"
    if new == old:
        return None
    return NUMBER_RE.sub(lambda m: new if m.group(1) == old else m.group(0), sentence, count=1)


def swap_term(sentence: str, terms: list[str], rng: random.Random) -> str | None:
    present = [t for t in TERM_RE.findall(sentence) if t in terms]
    if not present:
        return None
    old = rng.choice(present)
    others = [t for t in terms if t != old and (t.isupper() == old.isupper())]
    if not others:
        return None
    return sentence.replace(old, rng.choice(others), 1)


def shorten(sentence: str) -> str | None:
    without = re.sub(r"\s*\([^)]*\)", "", sentence)
    if without != sentence and len(without.split()) >= 6:
        return without
    head = re.split(r",\s+(?:which|including|such as|and|but)\b", sentence)[0]
    if head != sentence and len(head.split()) >= 6:
        return head.rstrip(",;") + "."
    return None


def overlap(a: str, b: str) -> float:
    left, right = set(content_tokens(a)), set(content_tokens(b))
    return len(left & right) / max(1, len(left))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["train", "validation", "synthetic"], required=True)
    parser.add_argument("--per-doc", type=int, default=120, help="source sentences per document")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(SEED)  # noqa: S311 - sampling, not security
    rows = []
    docs = documents(args.split)
    for key, _title, path in docs:
        parsed = parse_document(
            path.read_bytes(),
            path.suffix.lower(),
            max_pages=2000,
            max_chars=10_000_000,
            cleanup="v2",
        )
        chunks = chunk_pages(parsed.pages, ChunkingConfig(800, 120, "v2"))
        per_chunk = [(chunk, fact_sentences(chunk.text)) for chunk in chunks]
        per_chunk = [(c, s) for c, s in per_chunk if s]
        all_sentences = [(c.index, s) for c, ss in per_chunk for s in ss]
        text = " ".join(c.text for c in chunks)
        terms = sorted({t for t in TERM_RE.findall(text) if len(t) > 2})
        picks = rng.sample(per_chunk, min(args.per_doc, len(per_chunk)))
        for chunk, sentences in picks:
            sentence = rng.choice(sentences)
            all_in_chunk = split_sentences(chunk.text)
            position = next(
                (i for i, s in enumerate(all_in_chunk) if " ".join(s.split()) == sentence), 0
            )
            width = rng.choice([0, 1, 1, 2])
            start = max(0, position - rng.randint(0, width))
            evidence = " ".join(
                " ".join(s.split()) for s in all_in_chunk[start : start + width + 1]
            )
            if sentence not in evidence:
                evidence = sentence
            made: list[tuple[str, str, str]] = []
            kind = rng.random()
            if kind < 0.3:
                made.append((sentence, "supported", "verbatim"))
            else:
                alt = (
                    shorten(sentence) if kind < 0.65 else first_change(sentence, KEEP_MEANING, rng)
                )
                if alt and alt != sentence:
                    made.append((alt, "supported", "shortened" if kind < 0.65 else "reworded"))
                else:
                    made.append((sentence, "supported", "verbatim"))
            for maker, name in (
                (lambda s: change_number(s, rng), "number"),
                (lambda s: first_change(s, NEGATIONS, rng), "negation"),
                (lambda s: first_change(s, MODALS, rng), "modal"),
                (lambda s, t=terms: swap_term(s, t, rng), "entity"),
            ):
                changed = maker(sentence)
                if changed and changed != sentence:
                    made.append((changed, "contradicted", name))
                    break
            base = shorten(sentence) or sentence
            others = [
                s for i, s in all_sentences if i != chunk.index and overlap(s, evidence) < 0.2
            ]
            if rng.random() < 0.5 and others:
                clause = rng.choice(others).rstrip(".")
                clause = clause[0].lower() + clause[1:]
                made.append((base.rstrip(".") + ", and " + clause + ".", "partial", "added clause"))
            else:
                extra = rng.choice(EXTRAS).format(n=rng.choice([2, 3, 5, 6, 12, 30, 90, 100]))
                made.append((base.rstrip(".") + ", " + extra + ".", "partial", "added detail"))
            if others:
                made.append((rng.choice(others), "nee", "other passage"))
            for claim, label, kind_name in made:
                rows.append(
                    {
                        "split": args.split,
                        "doc": key,
                        "evidence": evidence,
                        "claim": claim,
                        "label": label,
                        "kind": kind_name,
                    }
                )
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
        newline="\n",
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(args.split, len(rows), counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

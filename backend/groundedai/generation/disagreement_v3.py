"""``disagreement-v3``: the disagreement check, aware of document versions.

Built on 2026-11-13 because no earlier version found the differences between the 2017 and 2025
editions of a NIST guideline. Traced on the four test questions (and not tuned on them), the
causes were in four places, each handled here:

* **Asked words.** An acronym such as "AAL2" was dropped as a name, and "reauthenticated" and
  "reauthentication" were different words, so no statement was "about the question". Here
  acronyms are asked words, word forms match with the extractive generator's prefix rule, and
  the documents' own title words do not count.
* **Context words.** A statement about "AL1" must not be compared with one about "AL2". A
  statement that names an acronym of the same family as the question's (same letters,
  another number) but not the question's own is left out.
* **Requirement strength.** Two editions can say the same thing with "SHOULD NOT" and "SHALL
  NOT". The normative words (SHALL, MUST, SHOULD, MAY and their negations; RFC 2119 style)
  are read as a stance and compared.
* **Pairs across editions.** For two documents that are editions of the same document
  (:mod:`groundedai.generation.versions`), each statement is compared only with the most
  similar statement of the other edition (shared words, at least :data:`ALIGNMENT`), so two
  different requirements of the same documents are not paired. All their values are compared:
  "09:00–19:00, 10:00–16:00" and "09:00–20:00, 10:00–16:00" differ although they share a range.
  The result is ``different_versions``: both statements, both editions, never a verdict on
  which one applies.

Pairs of documents that are not editions of each other are left to ``disagreement-v1``, the
default before this version, so v3 changes nothing for them except one thing: names in the
question ("Orchard Lane Library") now need every one of their words in a document, not one
word ("library" is in every library's rules). On 242 questions asked across five unrelated
NIST publications, v1 and v2 reported 18 and 23 disagreements, all false; v2's reading of
other pairs was therefore not taken over.

Retrieval is outside this module: when two editions are in scope and one has no passage among
the retrieved ones, the QA service adds its best passages (``ensure_editions_retrieved``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from groundedai.core.quantities import Quantity, read_quantities, read_ranges
from groundedai.core.text import find_normalized_span, split_sentences, tokenize
from groundedai.generation.claims import QUESTION_FILLERS, claim_key_v2, claim_words
from groundedai.generation.context import SourceBlock
from groundedai.generation.disagreement import (
    _AFFIRMED_REQUIREMENT_RE,
    _AFFIRMED_REQUIREMENT_V2_RE,
    _ANSWERED_NO_RE,
    _ASKS_FOR_DATE_RE,
    _NEGATED_REQUIREMENT_RE,
    _NEGATED_REQUIREMENT_V2_RE,
    _QUALIFIERS_V2,
    _QUESTION_SENTENCE_RE,
    _REQUIREMENT_WORDS,
    _REVISION_RE,
    _VALIDITY_RE,
    MAX_REPORTED,
    MIN_FOCUS_WORDS,
    Disagreement,
    Statement,
    _different_context,
    asked_kinds,
)
from groundedai.generation.equivalents import rewrite_phrases
from groundedai.generation.relevance import words_match
from groundedai.generation.safety import detect_injection
from groundedai.generation.versions import Edition, edition_of, same_family

ALIGNMENT = 0.3
_ACRONYM_RE = re.compile(r"\b(?=[A-Za-z]*\d|[A-Z]{2,})([A-Z][A-Za-z]*\d+[A-Za-z]*|[A-Z]{2,}\d*)\b")
_NAME_RUN_RE = re.compile(r"(?:\b[A-Z][a-z][\w'-]*(?:\s+|$))+")
_MODAL_RE = re.compile(
    r"\b(SHALL NOT|SHALL|MUST NOT|MUST|SHOULD NOT|SHOULD|MAY NOT|MAY|NOT RECOMMENDED|RECOMMENDED|"
    r"REQUIRED|OPTIONAL|shall not|shall|must not|must|should not|should)\b"
)
_STANCE = {
    "shall": "required", "must": "required", "required": "required",
    "shall not": "prohibited", "must not": "prohibited", "may not": "prohibited",
    "should": "recommended", "recommended": "recommended",
    "should not": "discouraged", "not recommended": "discouraged",
    "may": "permitted", "optional": "permitted",
}  # fmt: skip
_MODAL_WORDS = frozenset(
    claim_key_v2(w) for w in ["shall", "must", "should", "may", "required", "optional"]
)


_SECTION_LABEL_RE = re.compile(r"[A-Z]?(?:\.?\d+)*\.?")


def is_section_number(sentence: str, quantity: Quantity) -> bool:
    """A number that opens a heading glued to the sentence by text extraction: "4 Visits to
    your property Field staff ...", "A.2 Length Password length ..."."""
    prefix = sentence[: quantity.start].strip()
    following = sentence[quantity.start :].split(maxsplit=2)
    return (
        quantity.kind == "count"
        and (prefix == "" or bool(_SECTION_LABEL_RE.fullmatch(prefix)))
        and len(following) > 1
        and following[1][:1].isupper()
    )


def stance_of(modal: str | None) -> str | None:
    return _STANCE.get(modal.casefold()) if modal else None


def acronyms(text: str) -> set[str]:
    return {match.group(1) for match in _ACRONYM_RE.finditer(text)}


def _letters(acronym: str) -> str:
    return re.sub(r"\d+", "#", acronym.upper())


def question_names(question: str) -> list[set[str]]:
    """Runs of capitalised words after the first word, each as a set of keys ("Orchard Lane
    Library"). Acronyms are asked words, not names."""
    runs = []
    for match in _NAME_RUN_RE.finditer(question):
        if match.start() == 0 and len(match.group(0).split()) == 1:
            continue
        words = match.group(0).split()
        if match.start() == 0:
            words = words[1:]  # "What", "Who": the first word of the question
        keys = {claim_key_v2(token) for word in words for token in tokenize(word)}
        if keys:
            runs.append(keys)
    return runs


def asked_words(question: str, titles: Sequence[str]) -> set[str]:
    fillers = {claim_key_v2(word) for word in QUESTION_FILLERS}
    fillers |= {claim_key_v2(word) for word in _REQUIREMENT_WORDS.split()}
    names = set().union(*question_names(question)) if question_names(question) else set()
    title_words = {w for title in titles for w in claim_words(title, claim_key_v2)}
    found = claim_words(rewrite_phrases(question), claim_key_v2) - fillers - names - title_words
    found |= {claim_key_v2(token) for a in acronyms(question) for token in tokenize(a)}
    return found


def _focus(words: frozenset[str], asked: set[str]) -> int:
    return sum(1 for word in asked if any(words_match(word, other) for other in words))


def _statements(question: str, asked: set[str], sources: Sequence[SourceBlock]) -> list[Statement]:
    dates_are_values = bool(_ASKS_FOR_DATE_RE.search(question))
    kinds = asked_kinds(question)
    question_acronyms = acronyms(question)
    families = {_letters(a) for a in question_acronyms}
    out = []
    for block in sources:
        title_words = claim_words(block.chunk.document_title, claim_key_v2)
        sentences = split_sentences(block.text)
        for index, sentence in enumerate(sentences):
            if detect_injection(sentence):
                continue
            answered_no = False
            if _QUESTION_SENTENCE_RE.search(sentence):
                following = sentences[index + 1] if index + 1 < len(sentences) else ""
                if not _ANSWERED_NO_RE.match(following):
                    continue
                answered_no = True
            named = acronyms(sentence)
            if any(_letters(a) in families for a in named) and not named & question_acronyms:
                continue  # "At AL1, ..." when the question asks about AL2
            words = frozenset(claim_words(rewrite_phrases(sentence), claim_key_v2))
            if not words - title_words:
                continue
            found = [
                q
                for q in read_quantities(sentence).quantities
                if not is_section_number(sentence, q)
            ]
            values = tuple(
                q for q in found
                if (dates_are_values or q.kind not in ("date", "year"))
                and (kinds is None or q.kind in kinds)
            )  # fmt: skip
            denies = bool(
                _NEGATED_REQUIREMENT_RE.search(sentence)
                or _NEGATED_REQUIREMENT_V2_RE.search(sentence)
            )
            rest = _NEGATED_REQUIREMENT_V2_RE.sub(" ", _NEGATED_REQUIREMENT_RE.sub(" ", sentence))
            affirms = bool(
                _AFFIRMED_REQUIREMENT_RE.search(rest) or _AFFIRMED_REQUIREMENT_V2_RE.search(rest)
            )
            modal_match = _MODAL_RE.search(sentence)
            modal = modal_match.group(1) if modal_match else None
            if answered_no:
                denies, affirms, values = True, False, ()
            if not values and not denies and not affirms and not modal:
                continue
            kept = {id(v) for v in values}
            span = find_normalized_span(block.text, sentence) or (0, 0)
            out.append(
                Statement(
                    label=block.label,
                    document_id=block.chunk.document_id,
                    document_title=block.chunk.document_title,
                    sentence=sentence,
                    start=span[0],
                    end=span[1],
                    values=values,
                    denies=denies,
                    words=words,
                    focus=_focus(words, asked),
                    qualifiers={
                        group: words & names
                        for group, names in _QUALIFIERS_V2.items()
                        if words & names
                    },
                    dates=tuple(m.group(0) for m in _VALIDITY_RE.finditer(sentence)),
                    affirms=affirms,
                    ranges=tuple(
                        r
                        for r in read_ranges(sentence, found)
                        if id(r.low) in kept and id(r.high) in kept
                    ),
                    validity_dates=tuple(m.group(0) for m in _VALIDITY_RE.finditer(sentence)),
                    revision_dates=tuple(m.group(0) for m in _REVISION_RE.finditer(sentence)),
                    modal=modal,
                )
            )
    return out


def _about_the_question(statements: list[Statement], asked: set[str]) -> list[Statement]:
    if not statements:
        return []
    needed = min(MIN_FOCUS_WORDS, len(asked))
    best = max(s.focus for s in statements)
    return [s for s in statements if s.focus >= max(needed, best - 2, 1)]


def _content(statement: Statement) -> frozenset[str]:
    return frozenset(w for w in statement.words if w not in _MODAL_WORDS and not w.isdigit())


def aligned_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Shared words over all words, word forms matched ("reauthenticated" and
    "reauthentication" are one word), as everywhere else in this check."""
    if not a or not b:
        return 0.0
    shared = sum(1 for word in a if any(words_match(word, other) for other in b))
    return shared / (len(a) + len(b) - shared)


def _values_differ(a: Statement, b: Statement) -> str:
    """'agree', 'differ' or 'none': every value of each side against the other side."""

    def items(s: Statement) -> list[object]:
        ends = {id(end) for r in s.ranges for end in (r.low, r.high)}
        return [*s.ranges, *[v for v in s.values if id(v) not in ends]]

    def same(x: object, y: object) -> bool | None:
        from groundedai.core.quantities import Range

        if isinstance(x, Range) and isinstance(y, Range):
            return x.same_range(y) if x.comparable(y) else None
        if isinstance(x, Quantity) and isinstance(y, Quantity):
            return x.same_value(y) if x.comparable(y) else None
        return None

    left, right = items(a), items(b)
    compared = False
    for first, second in ((left, right), (right, left)):
        for x in first:
            results = [same(x, y) for y in second]
            if all(r is None for r in results):
                continue
            compared = True
            if not any(results):
                return "differ"
    return "agree" if compared else "none"


def _version_pair(
    a: Statement, b: Statement, editions: Mapping[str, Edition]
) -> Disagreement | None:
    if _different_context(a, b):
        return None
    relation = _values_differ(a, b)
    stances = stance_of(a.modal), stance_of(b.modal)
    reason = None
    if relation == "differ":
        reason = "values"
    elif stances[0] and stances[1] and stances[0] != stances[1]:
        reason = "requirement"  # "SHOULD" and "SHALL": the same values, a different strength
    elif relation == "none" and a.denies != b.denies:
        reason = "requirement"
    if reason is None:
        return None
    first, second = sorted((a, b), key=lambda s: int(s.label[1:]))
    first = replace(first, edition=editions[first.document_id].describe())
    second = replace(second, edition=editions[second.document_id].describe())
    return Disagreement("different_versions", first, second, show_dates=False, reason=reason)


def _named_sources(sources: Sequence[SourceBlock], question: str) -> list[SourceBlock]:
    """Passages of documents that contain every word of a name the question gives."""
    runs = question_names(question)
    if not runs:
        return list(sources)
    words: dict[str, set[str]] = {}
    for block in sources:
        words.setdefault(block.chunk.document_id, set()).update(
            claim_words(f"{block.chunk.document_title} {block.text}", claim_key_v2)
        )
    named = {doc for doc, found in words.items() if any(run <= found for run in runs)}
    if not named:
        return list(sources)
    return [block for block in sources if block.chunk.document_id in named]


def _edition_pairs(
    question: str, sources: Sequence[SourceBlock], known: Mapping[str, Edition]
) -> dict[tuple[str, str], Disagreement]:
    titles = sorted({b.chunk.document_title for b in sources})
    asked = asked_words(question, titles)
    if not asked:
        return {}
    candidates = _about_the_question(_statements(question, asked, sources), asked)
    by_document: dict[str, list[Statement]] = {}
    for statement in candidates:
        by_document.setdefault(statement.document_id, []).append(statement)
    found: dict[tuple[str, str], Disagreement] = {}
    documents = sorted(by_document)
    for i, first_doc in enumerate(documents):
        for second_doc in documents[i + 1 :]:
            if not same_family(known[first_doc], known[second_doc]):
                continue
            best: tuple[float, int, Disagreement] | None = None
            for a in by_document[first_doc]:
                scored = [
                    (aligned_similarity(_content(a), _content(b)), b)
                    for b in by_document[second_doc]
                ]
                score, b = max(scored, key=lambda item: item[0])
                if score < ALIGNMENT:
                    continue
                found_pair = _version_pair(a, b, known)
                if found_pair is not None:
                    rank = (score, a.focus + b.focus)
                    if best is None or rank > best[:2]:
                        best = (*rank, found_pair)
            if best is not None:
                found[(first_doc, second_doc)] = best[2]
    return found


def find_disagreements_v3(
    question: str,
    sources: Sequence[SourceBlock],
    editions: Mapping[str, Edition] | None = None,
) -> list[Disagreement]:
    """Edition pairs by the rules above; every other pair of documents by
    ``disagreement-v1`` (the default before), on the passages of documents that the
    question's names allow."""
    from groundedai.generation.disagreement import find_disagreements

    known = dict(editions or {})
    for block in sources:
        known.setdefault(
            block.chunk.document_id, edition_of(block.chunk.document_id, block.chunk.document_title)
        )
    named = _named_sources(sources, question)
    found = _edition_pairs(question, named, known)
    for disagreement in find_disagreements(question, named, "disagreement-v1"):
        pair = tuple(sorted((disagreement.first.document_id, disagreement.second.document_id)))
        if same_family(known[pair[0]], known[pair[1]]):
            continue  # edition pairs are judged above
        found.setdefault((pair[0], pair[1]), disagreement)
    ordered = sorted(
        found.values(), key=lambda d: (int(d.first.label[1:]), int(d.second.label[1:]))
    )
    return ordered[:MAX_REPORTED]

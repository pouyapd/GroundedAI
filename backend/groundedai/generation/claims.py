"""Claim-level check of a generated answer against the passages the model was given.

Every factual sentence of the answer is a *claim*. Each claim is compared with **every**
retrieved passage, not only with the passages the model cited, and gets one status:

``supported``
    The model cited a passage, and the cited passage(s) contain the claim's words and figures.
``wrong_source``
    The model cited a passage that does not support the claim, but another retrieved passage
    does. The evidence shown is the passage that does.
``added_source``
    The model cited nothing, and a retrieved passage supports the claim. The source was added
    by this check, not by the model.
``partial``
    Some of the claim's words are in a passage, not enough to call it supported.
``unsupported``
    No single retrieved passage contains the claim's figures and most of its words. (A compound
    claim is supported when each of its clauses is found in one passage; words are never
    pooled across passages.)

The comparison is lexical, as in :mod:`groundedai.generation.citations`: a claim is supported
when no figure is missing and at least 70% of its content words occur in the passage text (the
same thresholds as the sentence check). Words of the document title are left out of the claim
before the comparison, so a title cannot support a claim. It catches invented figures, wrong
numbers, claims about things the passages do not mention and citations that point to the wrong
passage. It does **not** understand meaning: a negation, a wrong subject ("contractors" for
"employees") or a correct paraphrase with other words can be misjudged.

Version 2 (``claims-v2``, chosen with :func:`check_claims`'s ``version``) keeps this method and
changes four things: :func:`claim_stem_v2` joins word forms that version 1 keeps apart
("changed"/"change", "applied"/"apply", "cancelled"/"cancel"); a limit word on a figure must
not contradict the evidence's ("at least 14" is not "at most 14"); a claim whose negation
differs from its evidence sentence is only ``partial``; and every cited label is checked on
its own, so a second citation that states nothing of the claim can be reported.

Version 2 was revised on 2026-10-19, after errors seen on the development split and the PDF
journey (version 1 is unchanged and reproduces every earlier result):

* **Support from sentences, not from a whole passage.** A claim is compared with the few
  sentences of one passage that carry it (:func:`_support_set`): the best sentence, plus up to
  two more only when each adds at least two of the claim's words, or a missing figure and one
  word. Words scattered over a passage no longer add up to support ("bring" in one sentence,
  "dog" in another), and a figure from the next sentence ("4" from the next room's ratio) does
  not complete a claim. The quotes are exactly those sentences, word for word. A question in
  an FAQ is never evidence on its own, only together with its answer.
* **Equivalent wordings** (:mod:`groundedai.generation.equivalents`): "do not have to pay",
  "at no cost" and "free of charge" read alike; fee / charge / cost and a few other word
  groups share a key. Negation is compared after this rewrite, part by part, for every quoted
  sentence; a condition ("if you have not paid") is not a negation, and an FAQ question
  answered "No" is.
* **Names and figures.** A name written in the claim ("the Minnow class") must be in the
  evidence or its document's title. Day numbers ("the 10th") are figures. A figure from the
  question is exempt only inside a condition ("if your train is 45 minutes late, ..."); a
  date without a year ("1 January") matches a dated claim when the evidence gives that year.
* **Unresolved labels.** A bracketed fragment that names no passage ("[Source A]", a title
  that is not one of the documents) is reported as an invalid citation, and its words are not
  part of the claim.

Citation forms other than ``[S1]`` (``[Source 1]``, ``(S1)``, "source 2 says", a bracketed
document title) are recognised here and counted as *variant* references, so that what the
model meant is not lost. The strict ``[S#]`` count used by the published metrics is unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from groundedai.core.quantities import (
    NUMBER_WORDS,
    Limit,
    Quantity,
    figure_present,
    limit_of,
    limits_compatible,
    read_quantities,
)
from groundedai.core.text import (
    STOPWORDS,
    find_normalized_span,
    light_stem,
    normalize_whitespace,
    split_sentences,
    tokenize,
)
from groundedai.generation.citations import (
    CITATION_RE,
    PARTIAL_RECALL,
    SUPPORTED_RECALL,
    is_source_note,
    labels_in,
    strip_citations,
)
from groundedai.generation.equivalents import keyed_stemmer, rewrite_phrases
from groundedai.generation.safety import detect_injection

ClaimStatus = Literal["supported", "wrong_source", "added_source", "partial", "unsupported"]
Verdict = Literal["verified", "partially_verified", "unverified", "not_applicable"]
ReferenceForm = Literal["standard", "variant", "title", "unresolved"]
ClaimCheckVersion = Literal["claims-v1", "claims-v2"]
CLAIM_CHECK_VERSIONS: tuple[ClaimCheckVersion, ...] = ("claims-v1", "claims-v2")

MATCHED: frozenset[str] = frozenset({"supported", "wrong_source", "added_source"})
MAX_EVIDENCE_SENTENCES = 2

_REPORTING_VERBS = (
    r"(?:also )?(?:states?|says?|mentions?|notes?|explains?|indicates?|confirms?|recommends?|"
    r"specifies|specify|shows?|adds?|lists?|describes?|reports?|suggests?|provides?|outlines?)"
)
# A label followed by one of these is the subject of its sentence ("[S2] states", "[S2] does not").
_SUBJECT_VERBS = rf"(?:{_REPORTING_VERBS}|(?:also )?(?:does|do|did) not)"
_VARIANT_BRACKET_RE = re.compile(
    r"\[\s*(?:Sources?|Src|Ref(?:erence)?)\s*:?\s*(?P<labels>S?\d+(?:\s*(?:,|;|and|&)\s*S?\d+)*)\s*\]"
    r"|\[\s*(?P<pair>S\d+(?:\s*(?:,|;)?\s*(?:and|&)\s*S\d+)+)\s*\]",
    re.IGNORECASE,
)
_VARIANT_PAREN_WORD_RE = re.compile(
    r"\(\s*Sources?\s*:?\s*(?P<labels>S?\d+(?:\s*(?:,|;|and|&)\s*S?\d+)*)\s*\)", re.IGNORECASE
)
_MENTION_RE = re.compile(r"\b[Ss]ources?\s+(?P<labels>S?\d+(?:\s*(?:,|and)\s*S?\d+)*)\b")
_TITLE_BRACKET_RE = re.compile(r"\[(?P<inner>[^\[\]\n]{3,200})\]")
# claims-v2: what is left in brackets after every resolvable reference became "[S#]".
_UNRESOLVED_RE = re.compile(r"\[(?!S\d+(?:,\s*S\d+)*\])[^\[\]\n]{1,200}\]")
_PREFIX_LABEL_RE = re.compile(r"(\[S\d+(?:,\s*S\d+)*\])\s*:\s*")
_HEADING_AFTER_SENTENCE_RE = re.compile(r"(?<=[.!?\]])[ \t]+(?=#{1,6}\s)")
_ADJACENT_MARKERS_RE = re.compile(r"\[(S\d+(?:,\s*S\d+)*)\]\s*\[(?=S\d)")
_OPENS_WITH_MARKER_RE = re.compile(r"^\s*\[S\d+(?:,\s*S\d+)*\]")
_MARKER_AT_START_RE = re.compile(
    r"(^|\n|(?<=[.!?])[ \t]+)\s*(\[S\d+(?:,\s*S\d+)*\])(?!\s*:)"
    rf"(?!\s*{_SUBJECT_VERBS}\b)\s*",
    re.IGNORECASE,
)
_LEADING_MARKERS_RE = re.compile(r"^((?:\s*\[S\d+(?:,\s*S\d+)*\])+)\s*(.*)$", re.DOTALL)
_ATTRIBUTION_PATTERNS = [
    re.compile(
        r"^\s*(?:according to|as (?:stated|mentioned|noted|described|specified|shown) (?:in|by)|"
        r"based on)\b[^,]{0,160},\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\b(?:the )?(?:sources?|documents?|passages?)\s*)?(?:\[S\d+(?:,\s*S\d+)*\]\s*)+"
        rf"{_REPORTING_VERBS}\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:the )?(?:sources?|documents?|passages?)\s+{_REPORTING_VERBS}\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r",?\s*as (?:stated|mentioned|noted|described|specified|shown) (?:in|by)"
        r"(?:\s+(?:the\s+)?[^.,;\[]{0,80})?\s*(?:\[S\d+(?:,\s*S\d+)*\])?",
        re.IGNORECASE,
    ),
    re.compile(rf"^\s*{_REPORTING_VERBS}\s+that\s+", re.IGNORECASE),
]
# Irregular forms that light stemming cannot join ("lost" / "lose", "paid" / "pay").
_IRREGULAR = {
    "lost": "lose",
    "paid": "pay",
    "made": "make",
    "took": "take",
    "taken": "take",
    "gave": "give",
    "given": "give",
    "bought": "buy",
    "sent": "send",
    "kept": "keep",
    "held": "hold",
    "left": "leave",
    "met": "meet",
    "ran": "run",
    "chose": "choose",
    "chosen": "choose",
    "went": "go",
    "gone": "go",
    "saw": "see",
    "seen": "see",
    "wrote": "write",
    "written": "write",
    "knew": "know",
    "known": "know",
    "shown": "show",
    "brought": "bring",
    "built": "build",
    "spent": "spend",
    "sold": "sell",
    "told": "tell",
    "found": "find",
    "began": "begin",
    "begun": "begin",
    "children": "child",
    "people": "person",
}
# Question words that only ask for the value ("how many", "how long"); left out when a
# figure-only claim ("€260") is read as the answer to the question.
_FILLER_TEXT = "much many long often far soon quickly large big old maximum minimum exact exactly"
QUESTION_FILLERS = frozenset(_FILLER_TEXT.split())
_NUMBER_TOKENS = frozenset(NUMBER_WORDS) | {"once", "twice", "thrice"}
# Words that talk about the sources or link sentences. They are left out of a claim's words
# (not out of the passages), so "[S2] provides more detailed information, stating that X" is
# judged on X.
_META_TEXT = (
    "yes according information detail detailed details stating state states stated mention "
    "mentions mentioned source sources document documents passage passages provided "
    "provides additionally also furthermore however therefore thus specifically "
    "explicitly based clearly following note notes noted allow allows allowed"
)
_META_WORDS = frozenset(_META_TEXT.split())


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """A passage as the model saw it: its label, its document title and its text."""

    label: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Reference:
    label: str
    form: ReferenceForm
    raw: str
    valid: bool


@dataclass(frozen=True, slots=True)
class EvidenceQuote:
    label: str
    quote: str
    start: int
    end: int


@dataclass(slots=True)
class Claim:
    text: str
    cited: list[str]
    status: ClaimStatus
    evidence: list[EvidenceQuote] = field(default_factory=list)
    missing_figures: list[str] = field(default_factory=list)
    word_overlap: float = 0.0
    flags: list[str] = field(default_factory=list)
    # claims-v2 only: labels written with the claim that are not among the passages, and for
    # each cited label whether its passage states at least part of the claim.
    invalid_citations: list[str] = field(default_factory=list)
    label_support: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class ClaimReport:
    claims: list[Claim]
    references: list[Reference]
    notes: int  # sentences about the sources themselves, not checked as claims
    version: ClaimCheckVersion = "claims-v1"

    @property
    def verdict(self) -> Verdict:
        if not self.claims:
            return "not_applicable"
        matched = sum(1 for claim in self.claims if claim.status in MATCHED)
        if matched == len(self.claims):
            return "verified"
        return "partially_verified" if matched else "unverified"

    def count(self, status: ClaimStatus) -> int:
        return sum(1 for claim in self.claims if claim.status == status)

    @property
    def variant_references(self) -> int:
        return sum(1 for reference in self.references if reference.form in ("variant", "title"))


# ------------------------------------------------------------------ references
def _labels_from(group: str, *, bare_numbers: bool) -> list[str]:
    labels = []
    for number in re.findall(r"S?(\d+)", group, flags=re.IGNORECASE):
        labels.append(f"S{int(number)}")
    if not bare_numbers:
        labels = [f"S{int(n)}" for n in re.findall(r"S(\d+)", group, flags=re.IGNORECASE)]
    return list(dict.fromkeys(labels))


def _title_key(text: str) -> tuple[str, ...]:
    return tuple(token for token in tokenize(text) if token not in STOPWORDS)


def resolve_references(text: str, sources: Sequence[EvidenceSource]) -> tuple[str, list[Reference]]:
    """Rewrite every recognisable reference as ``[S#]`` and list what was found.

    The returned text is only used for checking; the answer shown to the user is unchanged.
    """
    known = {source.label for source in sources}
    references: list[Reference] = []

    def canonical(labels: list[str], raw: str, form: ReferenceForm) -> str:
        for label in labels:
            references.append(Reference(label, form, raw, label in known))
        return "[" + ", ".join(labels) + "]" if labels else raw

    def standard(match: re.Match[str]) -> str:
        labels = labels_in(match.group(0))
        return canonical(labels, match.group(0), "standard")

    def variant_bracket(match: re.Match[str]) -> str:
        group = match.group("labels") or match.group("pair")
        return canonical(_labels_from(group, bare_numbers=True), match.group(0), "variant")

    def variant_paren(match: re.Match[str]) -> str:
        return canonical(_labels_from(match.group("labels"), bare_numbers=True), match.group(0),
                         "variant")  # fmt: skip

    def mention(match: re.Match[str]) -> str:
        labels = _labels_from(match.group("labels"), bare_numbers=True)
        return canonical(labels, match.group(0), "variant")

    titles: dict[tuple[str, ...], list[str]] = {}
    for source in sources:
        titles.setdefault(_title_key(source.title), []).append(source.label)

    def title(match: re.Match[str]) -> str:
        inner = match.group("inner")
        if CITATION_RE.fullmatch(match.group(0)):
            return match.group(0)
        labels = titles.get(_title_key(inner))
        if not labels:
            return match.group(0)
        return canonical(labels, match.group(0), "title")

    working = CITATION_RE.sub(standard, text)
    working = _VARIANT_BRACKET_RE.sub(variant_bracket, working)
    working = _VARIANT_PAREN_WORD_RE.sub(variant_paren, working)
    working = re.sub(
        r"\(\s*(?P<labels>S\d+(?:\s*(?:,|;|and|&)\s*S?\d+)*)\s*\)", variant_paren, working
    )
    working = _MENTION_RE.sub(mention, working)
    working = _TITLE_BRACKET_RE.sub(title, working)
    return working, references


# ---------------------------------------------------------------- sentences
def _assign_labels(text: str) -> list[tuple[str, list[str]]]:
    """Split checked text into sentences with the labels that belong to each.

    Labels written *before* text and followed by a colon ("[S1]: The fee is ...") apply to the
    following text up to the next such label. A label that opens a sentence followed by a
    reporting verb ("[S2] states that ...") belongs to that sentence. Otherwise a label that
    opens a sentence belongs to the previous one ("Fact one. [S1] Fact two. [S2]").
    """
    # A Markdown heading starts a new sentence even when it follows on the same line.
    text = _HEADING_AFTER_SENTENCE_RE.sub("\n\n", text)
    previous = None
    while previous != text:  # "[S1][S2]" -> "[S1, S2]"
        previous = text
        text = _ADJACENT_MARKERS_RE.sub(r"[\1, ", text)
    if _OPENS_WITH_MARKER_RE.match(text):
        # An answer that starts with a label writes labels *before* the text it supports;
        # read every label that starts a line or a sentence that way.
        text = _MARKER_AT_START_RE.sub(r"\1\2: ", text)
    pieces = _PREFIX_LABEL_RE.split(text)
    segments: list[tuple[list[str], str]] = [([], pieces[0])]
    for index in range(1, len(pieces), 2):
        segments.append((labels_in(pieces[index]), pieces[index + 1]))

    result: list[tuple[str, list[str]]] = []
    for prefix_labels, segment in segments:
        for sentence in split_sentences(segment):
            match = _LEADING_MARKERS_RE.match(sentence)
            if match and result and not prefix_labels:
                rest = match.group(2)
                if not re.match(_SUBJECT_VERBS, rest, flags=re.IGNORECASE):
                    previous_text, previous_labels = result[-1]
                    result[-1] = (
                        previous_text,
                        list(dict.fromkeys(previous_labels + labels_in(match.group(1)))),
                    )
                    if rest.strip():
                        result.append((rest, labels_in(rest)))
                    continue
            result.append((sentence, list(dict.fromkeys(prefix_labels + labels_in(sentence)))))
    return [(sentence, labels) for sentence, labels in result if sentence.strip()]


def _strip_attribution(sentence: str) -> str:
    text = sentence
    for pattern in _ATTRIBUTION_PATTERNS:
        text = pattern.sub(" ", text)
    return normalize_whitespace(strip_citations(text))


# ------------------------------------------------------------------ matching
def claim_stem(token: str) -> str:
    """:func:`light_stem` plus "-ly", doubled consonants and a few irregular forms (claims-v1)."""
    stem = light_stem(_IRREGULAR.get(token, token))
    if len(stem) > 5 and stem.endswith("ly"):
        stem = stem[:-2]
    if len(stem) > 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
        stem = stem[:-1]
    return stem


_VOWELS = frozenset("aeiou")


def _consonant(word: str, index: int) -> bool:
    letter = word[index]
    if letter in _VOWELS:
        return False
    return not (letter == "y" and index > 0 and _consonant(word, index - 1))


def _measure(stem: str) -> int:
    """Porter's measure: the number of vowel-consonant sequences in ``stem``."""
    pattern = "".join("c" if _consonant(stem, i) else "v" for i in range(len(stem)))
    return len(re.findall(r"v+c+", pattern))


def _has_vowel(stem: str) -> bool:
    return any(not _consonant(stem, i) for i in range(len(stem)))


def _ends_cvc(stem: str) -> bool:
    """Consonant-vowel-consonant at the end, the last not w, x or y ("hop", "plan")."""
    return (
        len(stem) >= 3
        and _consonant(stem, -3 % len(stem))
        and not _consonant(stem, -2 % len(stem))
        and _consonant(stem, -1 % len(stem))
        and stem[-1] not in "wxy"
    )


def _after_suffix(stem: str) -> str:
    """Porter's tidy-up after "-ed" or "-ing": "dat" -> "date", "plann" -> "plan"."""
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"
    if len(stem) > 1 and stem[-1] == stem[-2] and _consonant(stem, len(stem) - 1):
        if stem[-1] not in "lsz":
            return stem[:-1]
        return stem
    if _measure(stem) == 1 and _ends_cvc(stem):
        return stem + "e"
    return stem


def claim_stem_v2(token: str) -> str:
    """claims-v2: the inflection steps of the Porter stemmer, written out here.

    Joins forms that :func:`claim_stem` keeps apart: change/changed, require/required,
    update/updated, date/dated, base/based, use/used, apply/applied, empty/emptied,
    cancel/cancelled, monthly/month. Short words that only differ by a final "e" stay apart
    (plan/plane, hop/hope, rat/rate), as Porter's rules intend. Derived words ("cancellation",
    "requirement") are not reduced, and synonyms are not joined. :func:`light_stem` and
    :func:`claim_stem` are unchanged.
    """
    word = _IRREGULAR.get(token, token)
    if not word.isalpha() or len(word) <= 2:
        return word
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith(("ies", "ied")) and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]  # agreed -> agree; need, feed stay
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word = _after_suffix(word[:-2])
    elif word.endswith("ing") and _has_vowel(word[:-3]) and len(word) > 4:
        word = _after_suffix(word[:-3])
    elif word.endswith("es") and word[:-2].endswith(("ss", "x", "ch", "sh", "zz")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    if len(word) > 5 and word.endswith("ly"):
        word = word[:-2]
    if word.endswith("e"):
        stem = word[:-1]
        measure = _measure(stem)
        if measure > 1 or (measure == 1 and not _ends_cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


Stemmer = Callable[[str], str]
# claims-v2: claim_stem_v2, with the equivalent words of groundedai.generation.equivalents
# sharing one key (fee / charge / cost, annual / yearly, ...).
claim_key_v2: Stemmer = keyed_stemmer(claim_stem_v2)
STEMMERS: dict[ClaimCheckVersion, Stemmer] = {
    "claims-v1": claim_stem,
    "claims-v2": claim_key_v2,
}
# claims-v2 also leaves these out of a claim: "the answer is 12" talks about the answer, "4 pm"
# is a time (its figure is compared), and "every" says what "each" (a stopword) says.
_META_WORDS_V2 = _META_WORDS | {"answer", "am", "pm", "every"}


def _words(text: str, stem: Stemmer = claim_stem) -> set[str]:
    return {
        stem(token)
        for token in tokenize(text)
        if token not in STOPWORDS and not token.isdigit() and token not in _NUMBER_TOKENS
    }


def claim_words(
    text: str, stem: Stemmer = claim_stem, meta: frozenset[str] = _META_WORDS
) -> set[str]:
    return {
        stem(token)
        for token in tokenize(text)
        if token not in STOPWORDS
        and token not in meta
        and not token.isdigit()
        and token not in _NUMBER_TOKENS
    }


_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


def read_ordinals(text: str, figures: Sequence[Quantity]) -> list[Quantity]:
    """claims-v2: day numbers such as "the 5th" that are not part of a figure already read.

    An ordinal is only matched by the same ordinal, so "the 10th of each month" is not
    supported by "the 5th of each month".
    """
    taken = [(figure.start, figure.end) for figure in figures]
    return [
        Quantity(
            float(match.group(1)), "ordinal", "day", match.group(0), match.start(), match.end()
        )
        for match in _ORDINAL_RE.finditer(text)
        if not any(start < match.end() and match.start() < end for start, end in taken)
    ]


@dataclass(frozen=True, slots=True)
class _Passage:
    source: EvidenceSource
    words: frozenset[str]  # body and title
    figures: tuple[Quantity, ...]
    body_words: frozenset[str] = frozenset()
    title_words: frozenset[str] = frozenset()
    limits: tuple[Limit | None, ...] = ()  # claims-v2: the limit word on each figure
    # claims-v2: a whole passage carries its sentences as units; the sentences chosen to
    # support one claim carry their spans in the source text.
    units: tuple[_Passage, ...] = ()
    spans: tuple[tuple[int, int], ...] = ()


def _trusted_text(text: str) -> str:
    """The passage without sentences that give orders to an AI system (they are not facts)."""
    sentences = split_sentences(text)
    if not any(detect_injection(sentence) for sentence in sentences):
        return text
    return " ".join(sentence for sentence in sentences if not detect_injection(sentence))


@dataclass(frozen=True, slots=True)
class _Method:
    """What differs between versions of the check."""

    version: ClaimCheckVersion
    stem: Stemmer
    limits: bool  # compare limit words ("at least" / "at most") on figures
    # claims-v2: support comes from a few sentences of one passage, each adding to it (see
    # _support_set); equivalent phrases are rewritten before words are compared; ordinals are
    # figures.
    sentences: bool = False
    meta: frozenset[str] = _META_WORDS

    def rewrite(self, text: str) -> str:
        return rewrite_phrases(text) if self.sentences else text

    def figures(self, text: str) -> list[Quantity]:
        figures = read_quantities(text).quantities
        if self.sentences:
            figures = sorted(figures + read_ordinals(text, figures), key=lambda f: f.start)
        return figures


_METHODS: dict[ClaimCheckVersion, _Method] = {
    "claims-v1": _Method("claims-v1", claim_stem, limits=False),
    "claims-v2": _Method(
        "claims-v2", claim_key_v2, limits=True, sentences=True, meta=_META_WORDS_V2
    ),
}
_V1 = _METHODS["claims-v1"]


def _passage(
    source: EvidenceSource,
    text: str | None = None,
    method: _Method = _V1,
    span: tuple[int, int] | None = None,
) -> _Passage:
    body = _trusted_text(source.text if text is None else text)
    body_figures = method.figures(body)
    title_figures = method.figures(source.title)
    body_words = frozenset(_words(method.rewrite(body), method.stem))
    title_words = frozenset(_words(method.rewrite(source.title), method.stem))
    limits: tuple[Limit | None, ...] = ()
    if method.limits:
        limits = tuple(limit_of(body, f) for f in body_figures) + tuple(
            limit_of(source.title, f) for f in title_figures
        )
    units = _units(source, method) if method.sentences and text is None else ()
    return _Passage(
        source,
        body_words | title_words,
        tuple(body_figures + title_figures),
        body_words,
        title_words,
        limits,
        units,
        (span,) if span else (),
    )


_QUESTION_RE = re.compile(r"\?[\"')\]]*\s*$")
_ANSWER_NO_RE = re.compile(r"^\W*no\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _Sentence:
    text: str
    start: int
    end: int

    @property
    def question(self) -> bool:
        return bool(_QUESTION_RE.search(self.text))


def _source_sentences(source: EvidenceSource) -> list[_Sentence]:
    """The sentences of a passage with their spans, without instructions or a bare title."""
    title = _title_key(source.title)
    result = []
    offset = 0
    for sentence in split_sentences(source.text):
        span = find_normalized_span(source.text[offset:], sentence)
        if span is None:
            continue
        start, end = offset + span[0], offset + span[1]
        offset = end
        if detect_injection(sentence) or _title_key(sentence) == title:
            continue
        result.append(_Sentence(source.text[start:end], start, end))
    return result


def _units(source: EvidenceSource, method: _Method) -> tuple[_Passage, ...]:
    """claims-v2: the sentences of a passage that can be evidence, each a passage of its own.

    A question states nothing, so it is never evidence on its own; a question in an FAQ is
    kept together with the sentence that answers it ("Can I bring my dog? Dogs travel free.").
    """
    sentences = _source_sentences(source)
    units = []
    for index, sentence in enumerate(sentences):
        if not sentence.question:
            units.append(_passage(source, sentence.text, method, (sentence.start, sentence.end)))
        elif index + 1 < len(sentences) and not sentences[index + 1].question:
            end = sentences[index + 1].end
            units.append(_passage(source, source.text[sentence.start : end], method,
                                  (sentence.start, end)))  # fmt: skip
    return tuple(units)


MAX_SUPPORT_SENTENCES = 3
MIN_ADDED_WORDS = (
    2  # a further sentence must add this many of the claim's words (or a figure and one)
)


def _support_set(
    words: set[str],
    figures: Sequence[Quantity],
    passage: _Passage,
    names: frozenset[str] = frozenset(),
) -> _Passage:
    """claims-v2: the few sentences of ``passage`` that carry a claim, as one passage.

    The sentence with most of the claim's words and figures comes first. Another sentence is
    added only when it adds at least two more of the claim's words, or a missing figure with
    one more word, up to three sentences, and only until the claim is covered. So a claim
    spread over neighbouring sentences ("A card is ordered at reception. It costs €4.") is
    found, while one word or one figure from another sentence never completes it: "bring" in
    one sentence and "dog" in another do not support "you can bring a dog", and the "4" of the
    next room's ratio does not support "the Seedlings room has one adult to 4 children".
    """
    if not passage.units:
        return passage
    content = (words - passage.title_words) or words
    chosen: list[_Passage] = []
    covered: set[str] = set()
    found: set[int] = set()

    def gain(unit: _Passage) -> tuple[int, int]:
        added = len((content & unit.body_words) - covered)
        figures_added = sum(
            1
            for index, figure in enumerate(figures)
            if index not in found and figure_present(figure, list(unit.figures))
        )
        return added, figures_added

    def complete() -> bool:
        enough = len(covered) >= SUPPORTED_RECALL * len(content)
        return enough and len(found) == len(figures) and not (names - covered - passage.title_words)

    while len(chosen) < MAX_SUPPORT_SENTENCES and not (chosen and complete()):
        candidates = [unit for unit in passage.units if unit not in chosen]
        if not candidates:
            break
        # On a tie, the shorter sentence (an answer rather than the question with it).
        unit = max(candidates, key=lambda c: (sum(gain(c)), -sum(e - b for b, e in c.spans)))
        added, figures_added = gain(unit)
        if not chosen and not (added or figures_added):
            break
        if chosen and not (added >= MIN_ADDED_WORDS or (figures_added and added)):
            break
        chosen.append(unit)
        covered |= content & unit.body_words
        found |= {
            index
            for index, figure in enumerate(figures)
            if figure_present(figure, list(unit.figures))
        }
    body = frozenset().union(*(unit.body_words for unit in chosen))
    return _Passage(
        passage.source,
        body | passage.title_words,
        tuple(figure for unit in chosen for figure in unit.figures),
        body,
        passage.title_words,
        tuple(limit for unit in chosen for limit in unit.limits),
        (),
        tuple(sorted(span for unit in chosen for span in unit.spans)),
    )


def _sentence_windows(
    sources: Sequence[EvidenceSource], method: _Method = _V1
) -> tuple[list[_Passage], list[_Passage]]:
    """Each sentence, and each pair of neighbouring sentences, as a passage of its own.

    A figure-only claim ("€260") is checked against these, so its figure and the question's
    words must occur close together rather than anywhere in a long passage.
    """
    singles, pairs = [], []
    for source in sources:
        sentences = split_sentences(source.text)
        for index, sentence in enumerate(sentences):
            singles.append(_passage(source, sentence, method))
            if index + 1 < len(sentences):
                pairs.append(_passage(source, f"{sentence} {sentences[index + 1]}", method))
    return singles, pairs


@dataclass(frozen=True, slots=True)
class _Match:
    level: Literal["supported", "partial", "unsupported"]
    overlap: float
    missing: tuple[str, ...]

    @property
    def rank(self) -> tuple[int, int, float]:
        order = {"supported": 2, "partial": 1, "unsupported": 0}
        return (order[self.level], -len(self.missing), self.overlap)


Limits = Mapping[Quantity, Limit | None]


def _figure_found(figure: Quantity, limit: Limit | None, passages: Sequence[_Passage]) -> bool:
    """claims-v2: the figure is stated, and not with a contradicting limit word.

    A date given without a year ("1 January") states a dated claim ("1 January 2027") when the
    same evidence or its document's title gives that year ("... on Saturday 2 January 2027").
    """
    for passage in passages:
        years = {int(f.value // 10000) for f in passage.figures if f.kind == "date"}
        years |= {int(f.value) for f in passage.figures if f.kind == "year"}
        for index, candidate in enumerate(passage.figures):
            other = passage.limits[index] if index < len(passage.limits) else None
            found = figure_present(figure, [candidate]) or (
                figure.kind == candidate.kind == "date"
                and candidate.value < 10000 <= figure.value
                and figure.value % 10000 == candidate.value
                and int(figure.value // 10000) in years
            )
            if found and limits_compatible(limit, other):
                return True
    return False


def _match(
    words: set[str],
    figures: list[Quantity],
    passages: Sequence[_Passage],
    limits: Limits | None = None,
    names: frozenset[str] = frozenset(),
) -> _Match:
    """How well ``passages`` support a claim with these words and figures.

    Words of the document title only say which document a claim is about ("Orbis Fitness");
    they are left out of the overlap, which is measured on the passage text. Otherwise "Orbis
    Fitness membership is free" would pass on the strength of the name alone. With ``limits``
    (claims-v2), a figure also needs a compatible limit word in the passage. With ``names``
    (claims-v2), every name the claim gives ("the Minnow class") must be in the passage text
    or its document's title, or the claim is at most partly supported.
    """
    body: set[str] = set()
    title: set[str] = set()
    available_figures: list[Quantity] = []
    for passage in passages:
        body |= passage.body_words
        title |= passage.title_words
        available_figures.extend(passage.figures)
    content = (words - title) or words
    overlap = len(content & body) / len(content) if content else 1.0
    if limits is None:
        missing = tuple(f.text for f in figures if not figure_present(f, available_figures))
    else:
        missing = tuple(f.text for f in figures if not _figure_found(f, limits.get(f), passages))
    if missing or overlap < PARTIAL_RECALL:
        level: Literal["supported", "partial", "unsupported"] = "unsupported"
    elif overlap < SUPPORTED_RECALL or names - body - title:
        level = "partial"
    else:
        level = "supported"
    return _Match(level, round(overlap, 3), missing)


def _rank(match: _Match, passage: _Passage) -> tuple[int, int, float, int]:
    """Order of candidate evidence; on a tie, fewer sentences first (claims-v2)."""
    level, missing, overlap = match.rank
    return (level, missing, overlap, -len(passage.spans))


def _scored(
    words: set[str],
    figures: list[Quantity],
    passage: _Passage,
    limits: Limits | None,
    names: frozenset[str],
) -> tuple[_Match, _Passage]:
    candidate = _support_set(words, figures, passage, names)
    return _match(words, figures, [candidate], limits, names), candidate


def _best(
    words: set[str],
    figures: list[Quantity],
    passages: Sequence[_Passage],
    clauses: list[tuple[set[str], list[Quantity]]] | None = None,
    limits: Limits | None = None,
    names: frozenset[str] = frozenset(),
) -> tuple[_Match, list[_Passage]] | None:
    """The passage that best supports a claim, or one passage per clause of a compound one."""
    if not passages:
        return None
    scored = sorted(
        (_scored(words, figures, passage, limits, names) for passage in passages),
        key=lambda item: _rank(*item),
        reverse=True,
    )
    best = (scored[0][0], [scored[0][1]])
    if best[0].level == "supported" or not clauses or len(clauses) < 2:
        return best
    # A compound sentence ("Books ... 21 days, and DVDs ... 7 days") is supported when each of
    # its clauses is supported by one passage. Words are never pooled across passages: that
    # would let one document's subject and another document's fact "support" a claim.
    chosen: list[_Passage] = []
    overlaps = []
    for clause_words, clause_figures in clauses:
        clause_names = names & clause_words
        found = max(
            (_scored(clause_words, clause_figures, p, limits, clause_names) for p in passages),
            key=lambda item: _rank(*item),
        )
        if found[0].level != "supported":
            return best
        overlaps.append(found[0].overlap)
        if found[1] not in chosen:
            chosen.append(found[1])
    return _Match("supported", min(overlaps), ()), chosen


_CLAUSE_SPLIT_RE = re.compile(r";\s+|,\s+(?:and|but|while|whereas)\s+|\s+and\s+(?=(?:the|a|an)\s)")
MIN_CLAUSE_WORDS = 2


def clauses_of(
    text: str,
    figures: list[Quantity],
    stem: Stemmer = claim_stem,
    meta: frozenset[str] = _META_WORDS,
    rewrite: Callable[[str], str] | None = None,
) -> list[tuple[set[str], list[Quantity]]]:
    """The independent clauses of a sentence, each with its words and its own figures."""
    parts = [part for part in _CLAUSE_SPLIT_RE.split(text) if part.strip()]
    if len(parts) < 2:
        return []
    result = []
    offset = 0
    for part in parts:
        start = text.find(part, offset)
        end = start + len(part)
        offset = end
        part_words = claim_words(rewrite(part) if rewrite else part, stem, meta)
        part_figures = [f for f in figures if start <= f.start < end]
        if len(part_words) < MIN_CLAUSE_WORDS and not part_figures:
            return []  # a fragment, not a clause: judge the sentence as a whole
        result.append((part_words, part_figures))
    return result


def _evidence(
    claim_words: set[str], figures: list[Quantity], passage: _Passage, stem: Stemmer = claim_stem
) -> EvidenceQuote | None:
    """The sentence(s) of the passage that carry the claim, with their character offsets."""
    title = _title_key(passage.source.title)
    sentences = [
        sentence
        for sentence in split_sentences(passage.source.text)
        if _title_key(sentence) != title and not detect_injection(sentence)
    ] or [passage.source.text]
    # As in _match, words of the document title do not count: a sentence that only names the
    # document ("Harbourline Ferries runs ...") must not be quoted for "you can bring a dog".
    content = (claim_words - passage.title_words) or claim_words
    scored = []
    for sentence in sentences:
        sentence_figures = read_quantities(sentence).quantities
        hits = sum(1 for f in figures if figure_present(f, sentence_figures))
        overlap = len(content & _words(sentence, stem))
        scored.append((hits, overlap, sentence))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not scored[0][0] and not scored[0][1]:
        return None
    chosen = [scored[0][2]]
    covered = scored[0][0]
    for hits, _overlap, sentence in scored[1:]:
        if covered >= len(figures) or len(chosen) >= MAX_EVIDENCE_SENTENCES:
            break
        if hits:
            chosen.append(sentence)
            covered += hits
    spans = [find_normalized_span(passage.source.text, sentence) for sentence in chosen]
    found = [span for span in spans if span is not None]
    if not found:
        return EvidenceQuote(passage.source.label, chosen[0], 0, 0)
    start = min(span[0] for span in found)
    end = max(span[1] for span in found)
    return EvidenceQuote(passage.source.label, passage.source.text[start:end], start, end)


def _quotes(
    claim_words: set[str],
    figures: list[Quantity],
    passages: Sequence[_Passage],
    stem: Stemmer = claim_stem,
) -> list[EvidenceQuote]:
    quotes: list[EvidenceQuote] = []
    for passage in passages:
        if passage.spans:
            # claims-v2: one quote per sentence that supports the claim, word for word.
            text = passage.source.text
            quotes.extend(
                EvidenceQuote(passage.source.label, text[start:end], start, end)
                for start, end in passage.spans
            )
            continue
        if passage.units:
            continue  # claims-v2: no sentence of this passage carries the claim
        quote = _evidence(claim_words, figures, passage, stem)
        if quote is not None:
            quotes.append(quote)
    return quotes


_NEGATION_RE = re.compile(r"\b(?:not|never|cannot|no\s+longer)\b|n't\b", re.IGNORECASE)
# Parts of a sentence that can carry their own negation ("Towels are not provided, so ...").
_SEGMENT_SPLIT_RE = re.compile(
    r"[,;:]\s+|\s+(?:but|and|so|while|whereas|although|though|however)\s+", re.IGNORECASE
)


# A condition ("if you have not paid") is not a statement that something is not so.
_CONDITION_RE = re.compile(r"^\W*(?:if|unless|when|whether)\b", re.IGNORECASE)


def _stated_parts(text: str) -> list[str]:
    """claims-v2: the parts of a sentence that can carry their own negation, without
    conditions."""
    parts = _SEGMENT_SPLIT_RE.split(rewrite_phrases(text))
    return [part for part in parts if part.strip() and not _CONDITION_RE.match(part)]


def _in_condition(text: str, given: Quantity, figures: Sequence[Quantity]) -> bool:
    """claims-v2: the claim states the question's figure ``given`` inside a condition."""
    offset = 0
    for part in _SEGMENT_SPLIT_RE.split(text):
        start = text.find(part, offset)
        offset = start + len(part)
        if _CONDITION_RE.match(part) and any(
            start <= figure.start < offset and figure_present(figure, [given]) for figure in figures
        ):
            return True
    return False


def negated(text: str) -> bool:
    """Whether a sentence negates its verb ("is not", "cannot", "never", "don't")."""
    return bool(_NEGATION_RE.search(text))


def _negation_differs(claim: Claim, words: set[str], stem: Stemmer) -> bool:
    """claims-v2: the claim negates what its evidence affirms, or the reverse.

    Every sentence of the evidence is compared: the part of the claim and the part of the
    sentence that share most words must both be negated or both not. A negated part elsewhere
    in a compound claim ("X is not Y; Z is W") or in the evidence sentence does not count,
    nor does a condition ("if you have not paid, ..."). Equivalent phrases are rewritten
    first, so "do not have to pay" and "free" read alike. A question in an FAQ followed by
    "No" counts as denied ("Do I need a card? No.").
    """
    del words  # the comparison is per part
    claim_parts = _stated_parts(claim.text)
    if not claim_parts:
        return False
    for quote in claim.evidence:
        sentences = split_sentences(quote.quote)
        for index, sentence in enumerate(sentences):
            flip = False
            if _QUESTION_RE.search(sentence):
                following = sentences[index + 1] if index + 1 < len(sentences) else ""
                flip = bool(_ANSWER_NO_RE.match(following))
            parts = _stated_parts(sentence)
            if not parts:
                continue
            shared, claim_part, part = max(
                (
                    (len(_words(c, stem) & _words(s, stem)), c, s)
                    for c in claim_parts
                    for s in parts
                ),
                key=lambda item: item[0],
            )
            if shared and negated(claim_part) != (negated(part) != flip):
                return True
    return False


_NAME_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:[-'][A-Za-z]+)*")


def _names(text: str, stem: Stemmer) -> frozenset[str]:
    """claims-v2: words written with a capital inside a sentence ("the Minnow class").

    The first word of the claim, and of any sentence or quotation in it, is left out.
    """
    names: set[str] = set()
    for match in _NAME_RE.finditer(text):
        before = text[: match.start()].rstrip()
        if not before or before.endswith((".", "!", "?", ":", '"', "(", "“")):
            continue
        names.update(stem(token) for token in tokenize(match.group(0)) if token not in STOPWORDS)
    return frozenset(names)


# --------------------------------------------------------------------- check
def check_claims(
    answer: str,
    sources: Sequence[EvidenceSource],
    question: str | None = None,
    version: ClaimCheckVersion = "claims-v1",
) -> ClaimReport:
    """Check every factual sentence of ``answer`` against every passage in ``sources``.

    Figures that the question itself states ("a train that arrives 45 minutes late") need not
    appear in a passage, as long as the claim also states a figure of its own that does
    ("you can claim 25%"); a claim whose only figure comes from the question still needs it.

    ``version`` chooses the method; ``claims-v1`` reproduces every earlier result.
    """
    method = _METHODS[version]
    stem = method.stem
    working, references = resolve_references(answer, sources)
    passages = {source.label: _passage(source, method=method) for source in sources}
    pool = list(passages.values())
    question_figures = method.figures(question) if question else []
    windows: tuple[list[_Passage], list[_Passage]] | None = None
    claims: list[Claim] = []
    notes = 0
    for sentence, labels in _assign_labels(working):
        if is_source_note(sentence):
            notes += 1
            continue
        unresolved: list[str] = []
        if method.sentences:
            # A bracketed fragment that names no passage ("[Source A]", a title that is not one
            # of the documents) is not a citation, and its words are not part of the claim.
            unresolved = [m.group(0) for m in _UNRESOLVED_RE.finditer(sentence)]
            sentence = _UNRESOLVED_RE.sub(" ", sentence)
            references.extend(Reference(raw, "unresolved", raw, False) for raw in unresolved)
        text = _strip_attribution(sentence)
        figures = method.figures(text)
        if text.endswith(":") and not figures:
            notes += 1  # "Data retention periods are as follows:"
            continue
        words = claim_words(method.rewrite(text), stem, method.meta)
        if not words and not figures:
            notes += 1
            continue
        if any(_title_key(text) == _title_key(source.title) for source in sources):
            notes += 1  # a document title used as a heading
            continue
        given = question_figures
        if method.sentences:
            # claims-v2: a figure from the question is exempt only as a premise ("if your
            # train is 45 minutes late, you can claim 25%"), not when the claim asserts it
            # ("the Seedlings room has one adult for every 4 children").
            given = [f for f in question_figures if _in_condition(text, f, figures)]
        own = [f for f in figures if not figure_present(f, given)]
        required = own if own else figures
        cited = [label for label in labels if label in passages]
        display = text[:1].upper() + text[1:]
        limits = {f: limit_of(text, f) for f in required} if method.limits else None
        if not words and question:
            # A figure on its own ("€260") answers the question: it must be in the sentence(s)
            # that match the question best, not merely somewhere in a passage.
            if windows is None:
                windows = (
                    ([unit for passage in pool for unit in passage.units], [])
                    if method.sentences
                    else _sentence_windows(sources, method)
                )
            words = claim_words(method.rewrite(question), stem, method.meta) - {
                stem(word) for word in QUESTION_FILLERS
            }
            claim = _judge(
                display, words, required, cited, _answer_windows(words, required, *windows), None,
                method, limits,
            )  # fmt: skip
        else:
            clauses = clauses_of(text, required, stem, method.meta, method.rewrite)
            names = _names(text, stem) & words if method.sentences else frozenset()
            claim = _judge(display, words, required, cited, pool, clauses, method, limits, names)
        if method.version == "claims-v2":
            claim.invalid_citations = [label for label in labels if label not in passages]
            claim.invalid_citations += unresolved
            if claim.status in MATCHED and _negation_differs(claim, words, stem):
                claim.status = "partial"
                claim.flags.append("negation_differs")
        if detect_injection(text):
            # Text that gives orders to an AI system is not a fact, even when a passage
            # contains it word for word.
            claim.status = "unsupported"
            claim.flags.append("instruction_text")
        claims.append(claim)
    return ClaimReport(claims=claims, references=references, notes=notes, version=version)


def _answer_windows(
    words: set[str],
    figures: list[Quantity],
    singles: Sequence[_Passage],
    pairs: Sequence[_Passage],
) -> list[_Passage]:
    """The sentences that match the question best among those stating a comparable figure.

    "€260" for "How much does a *hot* desk cost?" is then checked against the hot-desk
    sentence, not against the fixed-desk sentence that also states €260. Single sentences
    are tried first; two neighbouring sentences only when no single sentence states such a
    figure. A bare number ("28") is compared with figures of any kind.
    """

    def compatible(window: _Passage) -> bool:
        return any(
            f.kind == "number" or f.comparable(other) for f in figures for other in window.figures
        )

    for windows in (singles, pairs):
        candidates = [window for window in windows if compatible(window)]
        if candidates:
            top = max(len(words & window.words) for window in candidates)
            return [window for window in candidates if len(words & window.words) == top]
    return list(singles)


def _missing(
    figures: list[Quantity], best: tuple[_Match, list[_Passage]] | None
) -> tuple[str, ...]:
    """Figures not found even in the best passage (used when no cited passage qualifies)."""
    return best[0].missing if best else tuple(f.text for f in figures)


def _label_support(
    words: set[str],
    figures: list[Quantity],
    cited: list[str],
    pool: Sequence[_Passage],
    clauses: list[tuple[set[str], list[Quantity]]] | None,
    limits: Limits | None,
    names: frozenset[str] = frozenset(),
) -> dict[str, bool]:
    """claims-v2: for each cited label, whether its passage states at least part of the claim.

    A passage states part of a claim when it supports the whole claim, or one clause of a
    compound claim ("X is 5 [S1], and Y is 7 [S3]"). A second label whose passage states
    nothing of the claim is then visible, even though the first label supports it.
    """
    support: dict[str, bool] = {}
    for label in cited:
        passages = [passage for passage in pool if passage.source.label == label]
        if not passages:
            support[label] = False
            continue
        whole = _best(words, figures, passages, None, limits, names)
        stated = bool(whole and whole[0].level == "supported")
        for clause_words, clause_figures in clauses or []:
            if stated:
                break
            part = _best(clause_words, clause_figures, passages, None, limits, names & clause_words)
            stated = bool(part and part[0].level == "supported")
        support[label] = stated
    return support


def _judge(
    display: str,
    words: set[str],
    figures: list[Quantity],
    cited: list[str],
    pool: Sequence[_Passage],
    clauses: list[tuple[set[str], list[Quantity]]] | None = None,
    method: _Method = _V1,
    limits: Limits | None = None,
    names: frozenset[str] = frozenset(),
) -> Claim:
    """Status of one claim. ``pool`` holds whole passages, or sentence windows of them."""
    stem = method.stem
    best = _best(words, figures, pool, clauses, limits, names)
    if cited:
        cited_pool = [passage for passage in pool if passage.source.label in cited]
        found = _best(words, figures, cited_pool, clauses, limits, names)
        cited_match = found[0] if found else _Match("unsupported", 0.0, _missing(figures, best))
        cited_passages = found[1] if found else []
        support = (
            _label_support(words, figures, cited, pool, clauses, limits, names)
            if method.version == "claims-v2"
            else {}
        )
        if cited_match.level == "supported":
            return Claim(
                display, cited, "supported", _quotes(words, figures, cited_passages, stem), [],
                cited_match.overlap, label_support=support,
            )  # fmt: skip
        if best and best[0].level == "supported":
            return Claim(
                display, cited, "wrong_source", _quotes(words, figures, best[1], stem), [],
                best[0].overlap, label_support=support,
            )  # fmt: skip
        status: ClaimStatus = "partial" if cited_match.level == "partial" else "unsupported"
        evidence = (
            _quotes(words, figures, cited_passages[:1], stem)
            if cited_match.overlap >= PARTIAL_RECALL
            else []
        )
        return Claim(
            display, cited, status, evidence, list(cited_match.missing), cited_match.overlap,
            label_support=support,
        )  # fmt: skip
    if best is None:
        return Claim(display, [], "unsupported", [], [f.text for f in figures], 0.0)
    match, chosen = best
    if match.level == "supported":
        return Claim(
            display, [], "added_source", _quotes(words, figures, chosen, stem), [], match.overlap
        )
    status = "partial" if match.level == "partial" else "unsupported"
    evidence = _quotes(words, figures, chosen[:1], stem) if match.overlap >= PARTIAL_RECALL else []
    return Claim(display, [], status, evidence, list(match.missing), match.overlap)

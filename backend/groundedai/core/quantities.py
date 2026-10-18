"""Numbers with units, for comparing what answers and passages state.

Used by the claim check (does a passage contain the figure an answer states?) and the
disagreement check (do two passages give different values for the same thing?). It is kept
apart from :func:`groundedai.core.text.extract_numbers`, which published baselines depend on.

What it reads:

* digits with thousands separators and decimals (``10,000``, ``0.30``) and number words
  from zero to twenty and the tens (``three``, ``twelve``, ``forty``);
* currencies (``€45``, ``45 euros``, ``EUR 45``, ``£10``, ``$120``, ``20 cents``) and percentages;
* durations from seconds to years. "Working days" and "business days" are a unit of their own,
  not convertible to calendar days;
* frequencies (``every 3 months``, ``twice a year``, ``daily``), read as an interval;
* times of day (``06:30``, ``7:00``, ``8 am``), dates (``1 March 2026``, ``January 2026``) and
  years (``2024``);
* sizes, weights, lengths, speeds, temperatures, power, voltage and current;
* counts (a number followed by a word: ``twelve people``, ``2 sessions``).

Identifiers such as ``SX-5``, ``E07``, ``IP65`` or ``SEV1`` are *not* figures; they are
returned separately so a check can still require them to appear in the evidence.

It is a pattern matcher, not a parser. :func:`read_quantities` reads a range as two values
with the shared unit, and anything it does not recognise is a plain number. Two helpers used
only by version 2 of the checks leave its output unchanged: :func:`read_ranges` links the two
ends of a range ("09:00–17:00", "3 to 5 days", "between 7 am and 4 pm"), and
:func:`limit_of` reads the limit word attached to a figure ("at least 14", "up to 12").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from groundedai.core.text import STOPWORDS

QuantityKind = Literal[
    "money",
    "percent",
    "duration",
    "time_of_day",
    "date",
    "year",
    "size",
    "weight",
    "length",
    "speed",
    "temperature",
    "power",
    "voltage",
    "current",
    "count",
    "number",
]

NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
MULTIPLIER_WORDS: dict[str, int] = {"once": 1, "twice": 2, "thrice": 3}
MONTHS: dict[str, int] = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}

_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_WORDS = r"(?:" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")"
_NUM = rf"(?:{_DIGITS}|{_WORDS})"
_MONTH = r"(?:" + "|".join(MONTHS) + r")"

# unit spelling -> (kind, canonical unit, factor to the canonical unit). Abbreviations that are
# also ordinary words or letters ("A", "m", "g", "W", "V") only match in their exact case.
_UNITS: list[tuple[str, QuantityKind, str, float]] = [
    (r"working[ -]days?|business[ -]days?", "duration", "working_day", 1.0),
    (r"seconds?|secs?", "duration", "second", 1.0),
    (r"minutes?|mins?", "duration", "second", 60.0),
    (r"hours?|hrs?", "duration", "second", 3600.0),
    (r"days?|nights?", "duration", "second", 86400.0),
    (r"weeks?", "duration", "second", 604800.0),
    (r"months?", "duration", "month", 1.0),
    (r"years?|yrs?", "duration", "month", 12.0),
    (r"terabytes?|(?-i:TB)", "size", "MB", 1_000_000.0),
    (r"gigabytes?|(?-i:GB)", "size", "MB", 1000.0),
    (r"megabytes?|(?-i:MB)", "size", "MB", 1.0),
    (r"kilobytes?|(?-i:KB|kB)", "size", "MB", 0.001),
    (r"km/h|kilometres per hour|kilometers per hour", "speed", "km/h", 1.0),
    (r"mph|miles per hour", "speed", "mph", 1.0),
    (r"kilograms?|(?-i:kg)", "weight", "kg", 1.0),
    (r"grams?|(?-i:g)", "weight", "kg", 0.001),
    (r"tonnes?", "weight", "kg", 1000.0),
    (r"kilometres?|kilometers?|(?-i:km)", "length", "m", 1000.0),
    (r"centimetres?|centimeters?|(?-i:cm)", "length", "m", 0.01),
    (r"millimetres?|millimeters?|(?-i:mm)", "length", "m", 0.001),
    (r"metres?|meters?|(?-i:m)", "length", "m", 1.0),
    (r"°\s?C|degrees Celsius|degrees C", "temperature", "C", 1.0),
    (r"°\s?F|degrees Fahrenheit|degrees F", "temperature", "F", 1.0),
    (r"kilowatts?|(?-i:kW)", "power", "W", 1000.0),
    (r"megawatts?|(?-i:MW)", "power", "W", 1_000_000.0),
    (r"watts?|(?-i:W)", "power", "W", 1.0),
    (r"kilovolts?|(?-i:kV)", "voltage", "V", 1000.0),
    (r"volts?|(?-i:V)", "voltage", "V", 1.0),
    (r"milliamps?|(?-i:mA)", "current", "A", 0.001),
    (r"amps?|amperes?|(?-i:A)", "current", "A", 1.0),
]
_UNIT_PATTERN = "|".join(f"(?:{pattern})" for pattern, *_ in _UNITS)
_UNIT_LOOKUP = [
    (re.compile(rf"^(?:{pattern})$", re.I), kind, unit, factor)
    for pattern, kind, unit, factor in _UNITS
]
_PERIOD = r"(?:second|minute|hour|day|night|week|month|year)"
_PERIOD_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "night": 86400.0,
    "week": 604800.0,
}
_PERIOD_MONTHS = {"month": 1.0, "year": 12.0}
_ADVERB_PERIODS = {
    "hourly": "hour",
    "daily": "day",
    "nightly": "night",
    "weekly": "week",
    "monthly": "month",
    "yearly": "year",
    "annually": "year",
}

_CURRENCY_SYMBOL = {"€": "EUR", "£": "GBP", "$": "USD"}
_CURRENCY_WORD = {
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "cent": "cent",
    "cents": "cent",
    "pence": "pence",
    "p": "pence",
}

_DATE_RE = re.compile(
    rf"\b(?:(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+)?(?P<month>{_MONTH})(?:\s+(?P<year>(?:19|20)\d{{2}}))?\b"
    rf"|\b(?P<iso>(?:19|20)\d{{2}}-\d{{2}}-\d{{2}})\b",
    re.I,
)
_TIME_RE = re.compile(
    r"(?<![€£$\d.,])\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\s*(?P<ampm>am|pm|a\.m\.|p\.m\.)?"
    r"(?![\d:]|\.\d)"
    r"|(?<![€£$\d.,])\b(?P<hd>[01]?\d|2[0-3])\.(?P<md>[0-5]\d)\s*(?P<ampmd>am|pm|a\.m\.|p\.m\.)"
    r"|(?<![€£$\d.,])\b(?P<h2>1[0-2]|0?[1-9])\s*(?P<ampm2>am|pm|a\.m\.|p\.m\.)(?!\w)"
    r"|\b(?P<word>noon|midday|midnight)\b",
    re.I,
)
_MONEY_RE = re.compile(
    rf"(?P<sym>[€£$])\s?(?P<v1>{_DIGITS})"
    rf"|\b(?P<code>EUR|GBP|USD)\s?(?P<v2>{_DIGITS})"
    rf"|\b(?P<v3>{_NUM})\s?(?:-\s?)?(?P<word>euros?|EUR|pounds?|GBP|dollars?|USD|cents?|pence)\b",
    re.I,
)
_PERCENT_RE = re.compile(rf"\b(?P<v>{_NUM})\s?(?:%|percent\b|per cent\b)", re.I)
_RANGE_RE = re.compile(
    rf"\b(?P<a>{_NUM})\s*(?:to|-|–|and)\s*(?P<b>{_NUM})\s*(?P<unit>{_UNIT_PATTERN})(?![\w/])",
    re.I,
)
_FREQUENCY_RE = re.compile(
    rf"\b(?P<mult>once|twice|thrice|(?P<n>{_NUM})\s+times)\s+(?:a|an|per|each|every)\s+(?P<period>{_PERIOD})\b"
    rf"|\bevery\s+(?:(?P<n2>{_NUM})\s+)?(?P<unit>{_PERIOD})s?\b"
    r"|\b(?P<adverb>hourly|daily|nightly|weekly|monthly|yearly|annually)\b",
    re.I,
)
_REPETITION_RE = re.compile(
    rf"\b(?:(?P<n>{_NUM})\s+times|(?P<mult>once|twice|thrice))\b(?!-)", re.I
)
_UNIT_RE = re.compile(
    rf"\b(?P<v>{_NUM})\s?(?:-\s?)?(?P<unit>{_UNIT_PATTERN})(?![\w/])",
    re.I,
)
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z]+[-_]?\d+[A-Za-z\d]*(?:[-_][A-Za-z\d]+)*|\d+[A-Za-z]+[A-Za-z\d]*)\b"
)
_YEAR_RE = re.compile(r"(?<![\w.,])(?:19|20)\d{2}(?![\w.,]\d)")
_COUNT_RE = re.compile(rf"\b(?P<v>{_NUM})\s+(?P<noun>[A-Za-z][A-Za-z'-]*)", re.I)
_PLAIN_RE = re.compile(rf"(?<![\w.]){_DIGITS}(?![\w])")
_PER_RE = re.compile(
    rf"^\s*(?:(?:per|a|an|each)\s+|/\s*)(?P<per>{_PERIOD}|[A-Za-z]+)\b",
    re.I,
)
_COUNT_STOP_TEXT = (
    "a an the and or of to in on at by for with from per than is are was were be been "
    "times time percent cent"
)
_COUNT_STOP = frozenset(_COUNT_STOP_TEXT.split())


@dataclass(frozen=True, slots=True)
class Quantity:
    """A value with its kind and a canonical unit, plus where it was found."""

    value: float
    kind: QuantityKind
    unit: str
    text: str
    start: int
    end: int
    per: str | None = None
    interval: bool = False  # "every 3 months", "twice a year", "daily"
    amount: float | None = None  # the number as written, before unit conversion

    @property
    def written(self) -> float:
        return self.value if self.amount is None else self.amount

    def comparable(self, other: Quantity) -> bool:
        if self.kind != other.kind:
            # a bare number can stand for a count and vice versa ("15 visitors", "15")
            return {self.kind, other.kind} <= {"count", "number"}
        return self.unit == other.unit

    def same_value(self, other: Quantity) -> bool:
        if not self.comparable(other):
            return False
        if self.unit == "month":  # month/year conversions are approximate
            return abs(self.value - other.value) <= 0.02 * max(abs(self.value), abs(other.value))
        return abs(self.value - other.value) <= 1e-9 * max(1.0, abs(self.value))


@dataclass(slots=True)
class Reading:
    quantities: list[Quantity] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)


def _number(text: str) -> float | None:
    lowered = text.strip().lower()
    if lowered in NUMBER_WORDS:
        return float(NUMBER_WORDS[lowered])
    if lowered in MULTIPLIER_WORDS:
        return float(MULTIPLIER_WORDS[lowered])
    try:
        return float(lowered.replace(",", ""))
    except ValueError:
        return None


def _unit(spelling: str) -> tuple[QuantityKind, str, float] | None:
    for pattern, kind, unit, factor in _UNIT_LOOKUP:
        if pattern.match(spelling.strip()):
            return kind, unit, factor
    return None


class _Scanner:
    def __init__(self, text: str):
        self.text = text
        self.taken = [False] * len(text)
        self.found: list[Quantity] = []
        self.identifiers: list[str] = []

    def free(self, start: int, end: int) -> bool:
        return not any(self.taken[start:end])

    def take(self, start: int, end: int) -> None:
        for index in range(start, end):
            self.taken[index] = True

    def add(self, quantity: Quantity) -> None:
        self.found.append(quantity)

    def per_after(self, end: int) -> tuple[str | None, int]:
        match = _PER_RE.match(self.text[end : end + 40])
        if not match:
            return None, end
        word = match.group("per").lower().rstrip("s")
        return word, end + match.end()


def _scan_dates(scanner: _Scanner) -> None:
    for match in _DATE_RE.finditer(scanner.text):
        start, end = match.span()
        if not scanner.free(start, end):
            continue
        if match.group("iso"):
            year, month, day = (int(part) for part in match.group("iso").split("-"))
        else:
            if not match.group("day") and not match.group("year"):
                continue  # a bare month name ("in March") is not a date figure
            # "may" is also a verb: require a day or a year next to it
            month = MONTHS[match.group("month").lower()]
            year = int(match.group("year")) if match.group("year") else 0
            day = int(match.group("day")) if match.group("day") else 0
        value = year * 10000 + month * 100 + day
        scanner.add(Quantity(float(value), "date", "date", match.group(0), start, end))
        scanner.take(start, end)


def _scan_times(scanner: _Scanner) -> None:
    for match in _TIME_RE.finditer(scanner.text):
        start, end = match.span()
        if not scanner.free(start, end):
            continue
        if match.group("word"):
            minutes = 0 if match.group("word").lower() == "midnight" else 720
        else:
            hour = int(match.group("h") or match.group("hd") or match.group("h2"))
            minute = int(match.group("m") or match.group("md") or 0)
            ampm = match.group("ampm") or match.group("ampmd") or match.group("ampm2") or ""
            ampm = ampm.lower().replace(".", "")
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            minutes = hour * 60 + minute
        text = match.group(0).rstrip()
        end = start + len(text)
        scanner.add(Quantity(float(minutes), "time_of_day", "minute", text, start, end))
        scanner.take(start, end)


def _scan_money(scanner: _Scanner) -> None:
    for match in _MONEY_RE.finditer(scanner.text):
        start, end = match.span()
        if not scanner.free(start, end):
            continue
        if match.group("sym"):
            currency, raw = _CURRENCY_SYMBOL[match.group("sym")], match.group("v1")
        elif match.group("code"):
            currency, raw = match.group("code").upper(), match.group("v2")
        else:
            currency, raw = _CURRENCY_WORD[match.group("word").lower()], match.group("v3")
        value = _number(raw)
        if value is None:
            continue
        per, stop = scanner.per_after(end)
        scanner.add(Quantity(value, "money", currency, scanner.text[start:end], start, end, per))
        scanner.take(start, stop)


def _scan_percent(scanner: _Scanner) -> None:
    for match in _PERCENT_RE.finditer(scanner.text):
        start, end = match.span()
        value = _number(match.group("v"))
        if value is None or not scanner.free(start, end):
            continue
        scanner.add(Quantity(value, "percent", "%", match.group(0), start, end))
        scanner.take(start, end)


def _scan_ranges(scanner: _Scanner) -> None:
    for match in _RANGE_RE.finditer(scanner.text):
        start, end = match.span()
        unit = _unit(match.group("unit"))
        low, high = _number(match.group("a")), _number(match.group("b"))
        if unit is None or low is None or high is None or not scanner.free(start, end):
            continue
        kind, canonical, factor = unit
        per, stop = scanner.per_after(end)
        for value, (part_start, part_end) in ((low, match.span("a")), (high, match.span("b"))):
            scanner.add(
                Quantity(
                    value * factor,
                    kind,
                    canonical,
                    f"{scanner.text[part_start:part_end]} {match.group('unit')}",
                    part_start,
                    part_end,
                    per,
                    amount=value,
                )
            )
        scanner.take(start, stop)


def _period_value(period: str, count: float) -> tuple[float, str]:
    period = period.lower().rstrip("s")
    if period in _PERIOD_MONTHS:
        return count * _PERIOD_MONTHS[period], "month"
    return count * _PERIOD_SECONDS[period], "second"


def _scan_frequencies(scanner: _Scanner) -> None:
    for match in _FREQUENCY_RE.finditer(scanner.text):
        start, end = match.span()
        if not scanner.free(start, end):
            continue
        if match.group("adverb"):
            following = re.match(r"\s+([A-Za-z]+)", scanner.text[end:])
            if following and following.group(1).lower() not in STOPWORDS:
                continue  # an adjective ("daily allowance", "nightly limit"), not a frequency
            amount = 1.0
            value, unit = _period_value(_ADVERB_PERIODS[match.group("adverb").lower()], 1.0)
        elif match.group("unit"):
            count = _number(match.group("n2")) if match.group("n2") else 1.0
            if count is None:
                continue
            amount = count
            value, unit = _period_value(match.group("unit"), count)
        else:
            times = (
                _number(match.group("n"))
                if match.group("n")
                else float(MULTIPLIER_WORDS[match.group("mult").lower()])
            )
            if not times:
                continue
            amount = times
            length, unit = _period_value(match.group("period"), 1.0)
            value = length / times  # twice a year = every 6 months
        scanner.add(
            Quantity(
                value, "duration", unit, match.group(0), start, end, interval=True, amount=amount
            )
        )
        scanner.take(start, end)


def _scan_repetitions(scanner: _Scanner) -> None:
    """ "three times", "twice": how often something may happen, without a period."""
    for match in _REPETITION_RE.finditer(scanner.text):
        start, end = match.span()
        if not scanner.free(start, end):
            continue
        value = _number(match.group("n") or match.group("mult"))
        if value is None:
            continue
        scanner.add(Quantity(value, "count", "times", match.group(0), start, end))
        scanner.take(start, end)


def _scan_units(scanner: _Scanner) -> None:
    for match in _UNIT_RE.finditer(scanner.text):
        start, end = match.span()
        if not scanner.free(start, end):
            continue
        unit = _unit(match.group("unit"))
        value = _number(match.group("v"))
        if unit is None or value is None:
            continue
        if match.group("v").lower() == "one" and unit[0] != "duration":
            continue  # "one m..." is rarely a measurement
        kind, canonical, factor = unit
        per, stop = scanner.per_after(end)
        scanner.add(
            Quantity(value * factor, kind, canonical, match.group(0), start, end, per, amount=value)
        )
        scanner.take(start, stop)


def _scan_identifiers(scanner: _Scanner) -> None:
    for match in _IDENTIFIER_RE.finditer(scanner.text):
        start, end = match.span()
        if scanner.free(start, end):
            scanner.identifiers.append(match.group(0))
            scanner.take(start, end)


def _scan_years(scanner: _Scanner) -> None:
    for match in _YEAR_RE.finditer(scanner.text):
        start, end = match.span()
        following = scanner.text[end : end + 12].lstrip()
        if not scanner.free(start, end) or (
            re.match(r"[A-Za-z]{2,}", following)
            and not re.match(
                r"(?:edition|tariff|version|price|prices|rules|policy|guide)\b", following, re.I
            )
        ):
            continue
        scanner.add(Quantity(float(match.group(0)), "year", "year", match.group(0), start, end))
        scanner.take(start, end)


def _scan_counts(scanner: _Scanner) -> None:
    for match in _COUNT_RE.finditer(scanner.text):
        start, end = match.span()
        value_start, value_end = match.span("v")
        if not scanner.free(value_start, value_end):
            continue
        noun = match.group("noun").lower()
        raw = match.group("v")
        before = scanner.text[max(0, start - 1) : start]
        if noun in _COUNT_STOP or before == "-" or scanner.text[value_end : value_end + 1] == "-":
            continue
        if raw.lower() == "one" and not noun.endswith(("s",)) and noun not in {"person"}:
            continue  # "one of", "one way": keep only clear counts
        value = _number(raw)
        if value is None:
            continue
        per, _stop = scanner.per_after(end)
        scanner.add(Quantity(value, "count", "count", match.group(0), start, end, per))
        scanner.take(value_start, value_end)


def _scan_plain(scanner: _Scanner) -> None:
    for match in _PLAIN_RE.finditer(scanner.text):
        start, end = match.span()
        if not scanner.free(start, end):
            continue
        value = _number(match.group(0))
        if value is None:
            continue
        scanner.add(Quantity(value, "number", "", match.group(0), start, end))
        scanner.take(start, end)


def read_quantities(text: str) -> Reading:
    """All figures in ``text`` in reading order, and the identifiers that are not figures."""
    scanner = _Scanner(text)
    for step in (
        _scan_dates,
        _scan_times,
        _scan_money,
        _scan_percent,
        _scan_ranges,
        _scan_frequencies,
        _scan_repetitions,
        _scan_units,
        _scan_identifiers,
        _scan_years,
        _scan_counts,
        _scan_plain,
    ):
        step(scanner)
    return Reading(
        quantities=sorted(scanner.found, key=lambda quantity: quantity.start),
        identifiers=scanner.identifiers,
    )


@dataclass(frozen=True, slots=True)
class Range:
    """Two figures of one kind that are the ends of one range ("09:00–17:00")."""

    low: Quantity
    high: Quantity
    text: str

    def comparable(self, other: Range | Quantity) -> bool:
        return self.low.comparable(other.low if isinstance(other, Range) else other)

    def same_range(self, other: Range) -> bool:
        return self.low.same_value(other.low) and self.high.same_value(other.high)

    def has_end(self, value: Quantity) -> bool:
        return value.same_value(self.low) or value.same_value(self.high)

    def contains(self, value: Quantity) -> bool:
        low, high = sorted((self.low.value, self.high.value))
        return self.comparable(value) and low <= value.value <= high


# What may stand between the two ends of a range. "and" only after "between".
_RANGE_JOIN_RE = re.compile(r"^\s*(?:-|–|—|to|until|till|through)\s*$", re.I)
_BETWEEN_JOIN_RE = re.compile(r"^\s*and\s*$", re.I)
_BETWEEN_BEFORE_RE = re.compile(r"\bbetween\s+$", re.I)
_RANGE_KINDS = frozenset(
    {"money", "percent", "duration", "time_of_day", "date", "size", "weight", "length", "count",
     "number", "speed", "temperature", "power", "voltage", "current"}
)  # fmt: skip


def _range_end(first: Quantity, second: Quantity) -> bool:
    if first.kind not in _RANGE_KINDS or first.interval or second.interval:
        return False
    if first.kind == "duration" and second.kind == "duration":
        return True  # "18 months to 3 years": the units may differ
    return first.comparable(second) and first.kind == second.kind


def read_ranges(text: str, quantities: list[Quantity] | None = None) -> list[Range]:
    """Ranges in ``text``: two neighbouring figures of one kind joined by "–", "to", "until"
    or "through", or by "and" after "between". Used by version 2 of the checks, which compare
    a range as one value; :func:`read_quantities` itself still returns the two ends."""
    found = read_quantities(text).quantities if quantities is None else quantities
    ranges: list[Range] = []
    index = 0
    while index + 1 < len(found):
        first, second = found[index], found[index + 1]
        between = text[first.end : second.start]
        joined = bool(_RANGE_JOIN_RE.match(between)) or (
            bool(_BETWEEN_JOIN_RE.match(between))
            and bool(_BETWEEN_BEFORE_RE.search(text[max(0, first.start - 12) : first.start]))
        )
        if joined and _range_end(first, second):
            ranges.append(Range(first, second, text[first.start : second.end]))
            index += 2
            continue
        index += 1
    return ranges


Limit = Literal["at_least", "at_most", "more_than", "less_than", "exactly"]

# A limit word right before a figure ("at least 14 days", "a maximum of 8 students") or right
# after it ("14 days or more"). Words of one meaning are one limit: "minimum" = "at least".
_LIMIT_BEFORE: list[tuple[re.Pattern[str], Limit]] = [
    (
        re.compile(
            r"(?:\bat\s+least|\b(?:a\s+)?minimum(?:\s+of)?|\bmin\.|\bno\s+(?:fewer|less)\s+than"
            r"|\bnot\s+(?:fewer|less)\s+than)\s*$",
            re.I,
        ),
        "at_least",
    ),
    (
        re.compile(
            r"(?:\bat\s+(?:the\s+)?most|\b(?:a\s+)?maximum(?:\s+of)?|\bmax\.|\bup\s+to"
            r"|\bno\s+(?:more|longer)\s+than|\bnot\s+more\s+than|\bwithin"
            r"|\b(?:not|never)\s+exceed(?:ing)?|\b(?:must|may|can|should|will|does|do)\s+not\s+"
            r"exceed|\bcannot\s+exceed)\s*(?:the\s+)?$",
            re.I,
        ),
        "at_most",
    ),
    (
        re.compile(
            r"(?:\bmore\s+than|\bover|\babove|\bgreater\s+than|\bin\s+excess\s+of|\bexceeding"
            r"|\blonger\s+than)\s*$",
            re.I,
        ),
        "more_than",
    ),
    (
        re.compile(r"(?:\b(?:less|fewer|shorter)\s+than|\bunder|\bbelow)\s*$", re.I),
        "less_than",
    ),
    (re.compile(r"(?:\bexactly|\bprecisely)\s*$", re.I), "exactly"),
]
_LIMIT_AFTER: list[tuple[re.Pattern[str], Limit]] = [
    (
        re.compile(r"^\s*(?:or\s+(?:more|longer|above|over|later)|minimum|at\s+least)\b", re.I),
        "at_least",
    ),
    (
        re.compile(
            r"^\s*(?:or\s+(?:less|fewer|under|shorter|earlier)|maximum|at\s+(?:the\s+)?most)\b",
            re.I,
        ),
        "at_most",
    ),
]


def limit_of(text: str, figure: Quantity) -> Limit | None:
    """The limit word attached to ``figure`` in ``text``, if any.

    Only words right next to the figure count (a currency sign, "a", "an" or "the" may stand
    between), so "the minimum charge is for 8 metres" attaches nothing to "8 metres".
    """
    before = re.sub(r"(?:\b(?:a|an|the)\s+|[€£$]\s*)$", "", text[max(0, figure.start - 40) :
                                                              figure.start].rstrip())  # fmt: skip
    for pattern, limit in _LIMIT_BEFORE:
        if pattern.search(before + " "):
            return limit
    after = text[figure.end : figure.end + 24]
    for pattern, limit in _LIMIT_AFTER:
        if pattern.match(after):
            return limit
    return None


def limits_compatible(first: Limit | None, second: Limit | None) -> bool:
    """Two limits on the same value agree unless both are stated and differ ("at least 14" vs
    "at most 14"). A limit stated on one side only is accepted."""
    return first is None or second is None or first == second


def figure_present(figure: Quantity, candidates: list[Quantity]) -> bool:
    """Whether ``figure`` is stated by one of ``candidates`` (lenient about the unit wording).

    Working days count as days here, and a bare number or count matches any figure with the
    same value: the claim check asks "is this number in the passage?", not "is it the same
    property?".
    """
    for candidate in candidates:
        if figure.same_value(candidate):
            return True
        if figure.kind in ("count", "number") and abs(candidate.written - figure.written) < 1e-9:
            return True
        if figure.kind == candidate.kind == "duration" and {figure.unit, candidate.unit} == {
            "working_day",
            "second",
        }:
            days = figure.value if figure.unit == "working_day" else figure.value / 86400.0
            other = candidate.value if candidate.unit == "working_day" else candidate.value / 86400
            if abs(days - other) < 1e-9:
                return True
    return False

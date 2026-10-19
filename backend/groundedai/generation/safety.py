"""Heuristic detection of instruction-like text inside retrieved documents.

This is a signal, not a guarantee: pattern lists catch common phrasings and
miss paraphrases. Flagged sources are still passed to the model as data (they
may hold legitimate facts), and the response carries a warning.
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "ignore_instructions": (
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
            r"(instructions?|prompts?|rules|guidelines|messages)\b"
        ),
        "reveal_prompt": (
            r"\b(reveal|print|show|repeat|output)\b[^.\n]{0,30}\b"
            r"(system prompt|hidden prompt|instructions)\b"
        ),
        "role_change": r"\byou are now\b|\bact as (an?|the)\b|\bpretend to be\b",
        "fake_system_header": r"\b(system|admin)\s+(notice|override|message|prompt)\b\s*:",
        "answer_override": (
            r"\b(answer|respond|reply)\b[^.\n]{0,25}\b(every|all|any)\b[^.\n]{0,15}"
            r"\b(questions?|requests?)\b[^.\n]{0,10}\bwith\b"
        ),
        "suppress_citations": r"\bdo not (cite|mention|reference)\b[^.\n]{0,20}\bsources?\b",
        "chat_markup": r"</?\s*(system|assistant|instructions?)\s*>|<\|im_(start|end)\|>",
    }.items()
}


def detect_injection(text: str) -> list[str]:
    """Names of the injection patterns found in ``text`` (empty if none)."""
    return [name for name, pattern in _PATTERNS.items() if pattern.search(text)]


def looks_like_instruction(text: str) -> bool:
    return bool(detect_injection(text))

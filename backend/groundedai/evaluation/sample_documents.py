"""Two small invented PDFs with known content, for the end-to-end checks and the README demo.

Every fact sits on a known page, so tests can check that an answer cites the right
document and page. ``scripts/make_sample_pdfs.py`` writes them to ``examples/``.
"""

from __future__ import annotations

from dataclasses import dataclass

from groundedai.evaluation.pdf_fixture import make_pdf


@dataclass(frozen=True, slots=True)
class SampleDocument:
    filename: str
    title: str
    pages: tuple[str, ...]

    def pdf(self) -> bytes:
        return make_pdf(list(self.pages), title=self.title)


FERRY_GUIDE = SampleDocument(
    filename="harbourline-ferry-guide.pdf",
    title="Harbourline Ferries Passenger Guide",
    pages=(
        "Harbourline Ferries - Passenger Guide 2026\n"
        "Harbourline Ferries runs passenger boats between Kessport and the island of Mirrow. "
        "This guide explains tickets, luggage and pets.",
        "Tickets\n"
        "A single adult ticket from Kessport to Mirrow costs 14 euros. Children under 5 travel "
        "free. Tickets bought on board cost 3 euros more than tickets bought online.",
        "Luggage and pets\n"
        "Each passenger may bring two bags of up to 20 kg each at no extra charge. Bicycles cost "
        "6 euros per trip. Dogs are allowed on the open deck if they are kept on a lead.",
    ),
)

LIBRARY_RULES = SampleDocument(
    filename="orchard-library-rules.pdf",
    title="Orchard Lane Library Card Rules",
    pages=(
        "Orchard Lane Library - Card Rules\n"
        "A library card is free for residents. Members may borrow up to 12 items at a time. "
        "The loan period for books is 21 days.",
    ),
)

SAMPLES = (FERRY_GUIDE, LIBRARY_RULES)

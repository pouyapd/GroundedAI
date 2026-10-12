"""Rebuild derived files of the synthetic evaluation dataset.

* renders ``source/solvane-sx5-manual.txt`` into ``corpus/solvane-sx5-manual.pdf``
  (pages are separated by ``===PAGE===``);
* rewrites ``manifest.json`` with the SHA-256 hash of every corpus file.

Usage (from the repository root):
    python evaluation/scripts/build_dataset.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from groundedai.evaluation.pdf_fixture import make_pdf

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "groundedai-synthetic-v1"

DOCUMENTS = [
    (
        "aldervale-travel-policy",
        "corpus/aldervale-travel-policy.txt",
        "Aldervale Systems Travel and Expense Policy",
        "Company travel policy; hotel limits deliberately conflict with the finance FAQ.",
    ),
    (
        "aldervale-finance-faq",
        "corpus/aldervale-finance-faq.txt",
        "Aldervale Systems Finance FAQ",
        "FAQ with a hotel limit that contradicts the travel policy.",
    ),
    (
        "aldervale-remote-work",
        "corpus/aldervale-remote-work.md",
        "Aldervale Remote Work Guidelines",
        "Markdown document; contains outdated password advice that conflicts with the "
        "security standard.",
    ),
    (
        "aldervale-leave-policy",
        "corpus/aldervale-leave-policy.txt",
        "Aldervale Systems Leave and Holiday Policy",
        "Leave rules; says nothing about contractors.",
    ),
    (
        "aldervale-it-security-standard",
        "corpus/aldervale-it-security-standard.txt",
        "Aldervale Systems IT Security Standard",
        "Password, device and phishing rules.",
    ),
    (
        "aldervale-supplier-onboarding",
        "corpus/aldervale-supplier-onboarding.txt",
        "Aldervale Procurement: Supplier Onboarding Notes",
        "Contains a planted prompt-injection paragraph.",
    ),
    (
        "solvane-sx5-manual",
        "corpus/solvane-sx5-manual.pdf",
        "Solvane SX-5 Hybrid Inverter - Installation and Service Manual",
        "Four-page PDF generated from source/solvane-sx5-manual.txt; error codes test exact "
        "lexical matching.",
    ),
    (
        "brindlewick-water-faq",
        "corpus/brindlewick-water-faq.txt",
        "Brindlewick Water District - Customer Questions and Answers",
        "Utility customer FAQ.",
    ),
    (
        "brindlewick-cycling-club",
        "corpus/brindlewick-cycling-club.txt",
        "Brindlewick Cycling Club - Ride Rules for Members",
        "Shares the town name with other documents (ambiguity tests).",
    ),
    (
        "brindlewick-greenhouse-guide",
        "corpus/brindlewick-greenhouse-guide.txt",
        "Brindlewick Community Greenhouse - Volunteer Guide",
        "Contains a planted prompt-injection sentence.",
    ),
    (
        "brindlewick-makerspace-policy",
        "corpus/brindlewick-makerspace-policy.txt",
        "Brindlewick Library Makerspace - Equipment Policy",
        "Equipment rules.",
    ),
    (
        "orion-incident-runbook",
        "corpus/orion-incident-runbook.txt",
        "Orion ML Platform - Incident Response Runbook",
        "Incident severities and procedures.",
    ),
    (
        "orion-data-retention",
        "corpus/orion-data-retention.txt",
        "Orion ML Platform - Data Retention Standard",
        "Retention periods (many similar numbers).",
    ),
    (
        "tessera-api-limits",
        "corpus/tessera-api-limits.txt",
        "Tessera Public API - Rate Limits and Errors",
        "API quotas and error handling.",
    ),
]


def build_pdf() -> None:
    source = (DATASET / "source" / "solvane-sx5-manual.txt").read_text(encoding="utf-8")
    pages = [page.strip() for page in source.split("===PAGE===")]
    title = "Solvane SX-5 Hybrid Inverter - Installation and Service Manual"
    (DATASET / "corpus" / "solvane-sx5-manual.pdf").write_bytes(make_pdf(pages, title=title))


def build_manifest() -> None:
    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"] = [
        {
            "key": key,
            "file": file,
            "title": title,
            "sha256": hashlib.sha256((DATASET / file).read_bytes()).hexdigest(),
            "description": description,
        }
        for key, file, title, description in DOCUMENTS
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    build_pdf()
    build_manifest()
    print(f"Rebuilt PDF and manifest in {DATASET}")

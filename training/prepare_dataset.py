"""Build the LoRA training and validation data for GroundedAI's answer model.

    python training/prepare_dataset.py --out training/data

Every example is made with GroundedAI's own pipeline: the question is asked through the real QA
service (text clean-up v2, hybrid retrieval, top 5, the answer-v2 prompt), the exact chat
messages the model would receive are recorded, and the target reply is built from the
passages that were actually retrieved. Targets are grounded by construction: an answer is a
sentence copied from a retrieved passage of the right document, cited with that passage's label;
anything else is the application's abstention format.

Two sources:

* **Fictional documents** written from templates (`synthetic` below): eight kinds of
  organisation (ferry company, library, gym, parking permits, water utility, college, clinic,
  museum), each document stating a random subset of facts with random values. Split by
  organisation: training organisations and held-out validation organisations (the clinic and
  museum kinds appear only in validation). All training documents share one index and questions
  name the organisation, so the retrieved passages include other organisations' documents with
  other values (irrelevant passages to ignore).
* **NIST training PDFs** (`experiments/finetune/corpus.json`, five publications that are not in
  any validation or test set) with the model-written questions of `experiments/finetune/data`;
  their validation counterpart uses the NIST validation PDFs.

Kinds of example (`kind`):

| kind | question | target |
|---|---|---|
| answer | one fact the document states | the sentence, cited |
| multi | two facts the document states | both sentences, each cited |
| unanswerable | a fact the document does not state | abstention |
| insufficient | a related fact not stated (senior fare, only adult fare given) | abstention |
| partial | two facts, one stated | the stated one, cited, then an abstention |
| detail_bait | a stated fact plus a detail no document states | the fact, then an abstention |
| irrelevant | only another organisation's document in scope | abstention |
| conflict | a fact two editions state differently | both sentences, cited, "disagree" |
| nist_answer, nist_other_document | NIST, own or another document | sentence, abstention |

The protected test sets (`groundedai-nist-v1`, `groundedai-claims-nist-v1`) are never read. The
held-out fictional organisations are also written as a GroundedAI evaluation dataset
(`synthetic-eval/`) for `training/evaluate_finetuned.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from groundedai.core.config import Settings  # noqa: E402
from groundedai.core.text import normalize_whitespace, split_sentences  # noqa: E402
from groundedai.generation.prompts import ABSTAIN_MARKER  # noqa: E402
from groundedai.generation.providers import GenerationProvider  # noqa: E402
from groundedai.schemas.query import QueryRequest  # noqa: E402
from groundedai.services.container import Services, build_services  # noqa: E402

SEED = 20261116
FINETUNE = ROOT / "experiments" / "finetune"
PROTECTED = ("groundedai-nist-v1", "groundedai-claims-nist-v1")


# ----------------------------------------------------------------------------- capture
class Capture(BaseChatModel):
    """Records the prompt messages; replies with an abstention (nothing is generated)."""

    last: ClassVar[list[dict[str, str]]] = []

    @property
    def _llm_type(self) -> str:
        return "capture"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        Capture.last = [{"role": roles[m.type], "content": str(m.content)} for m in messages]
        reply = AIMessage(content=f"{ABSTAIN_MARKER}: captured.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def open_index(workdir: Path) -> Services:
    settings = Settings(
        _env_file=None, data_dir=workdir, text_cleanup="v2", rate_limit_enabled=False
    )
    provider = GenerationProvider(
        name="capture", model="capture", is_llm=True, chat_model=Capture()
    )
    return build_services(settings, generation_provider=provider)


def ask(
    services: Services, question: str, scope: list[str] | None
) -> tuple[list[dict[str, str]], Any]:
    response = services.qa.answer(QueryRequest(question=question, document_ids=scope))
    return list(Capture.last), response


def cited_sentence(response: Any, document_id: str, value_sentence: str) -> tuple[str, str] | None:
    """The retrieved sentence that states ``value_sentence`` in ``document_id``, with its label."""
    wanted = normalize_whitespace(value_sentence).rstrip(" .").lower()
    for source in response.sources:
        if source.document_id != document_id:
            continue
        for sentence in split_sentences(source.text):
            if normalize_whitespace(sentence).rstrip(" .").lower() == wanted:
                return normalize_whitespace(sentence).rstrip(" ."), source.label
    return None


def abstain(what: str) -> str:
    return f"{ABSTAIN_MARKER}: The sources do not state {what}."


# ------------------------------------------------------------------- fictional documents
@dataclass(frozen=True)
class Attr:
    key: str
    what: str  # "the price of a single adult ticket"
    questions: tuple[str, ...]  # with {org}
    sentence: str  # with {v}
    values: Callable[[random.Random], str]


@dataclass(frozen=True)
class Kind:
    name: str
    doc: str
    attrs: tuple[Attr, ...]
    related: tuple[tuple[str, str], ...]  # (stated, not stated) pairs for "insufficient"
    baits: tuple[str, ...]  # details no document states, for "detail_bait"
    filler: tuple[str, ...]


def money(low: int, high: int, unit: str = "euros") -> Callable[[random.Random], str]:
    return lambda rng: f"{rng.randint(low, high)} {unit}"


def count(low: int, high: int, unit: str) -> Callable[[random.Random], str]:
    return lambda rng: f"{rng.randint(low, high)} {unit}"


def clock(hours: list[int]) -> Callable[[random.Random], str]:
    return lambda rng: f"{rng.choice(hours):02d}:{rng.choice(['00', '30'])}"


KINDS: tuple[Kind, ...] = (
    Kind(
        "ferry",
        "Passenger Guide",
        (
            Attr(
                "adult",
                "the price of a single adult ticket",
                (
                    "How much does a single adult ticket cost with {org}?",
                    "What is the adult single fare on {org} ferries?",
                ),
                "A single adult ticket costs {v}.",
                money(8, 30),
            ),
            Attr(
                "child",
                "the fare for children",
                (
                    "How much do children pay on {org} ferries?",
                    "What is the child fare with {org}?",
                ),
                "Children aged 5 to 15 pay {v}.",
                money(3, 12),
            ),
            Attr(
                "bike",
                "the charge for taking a bicycle",
                (
                    "How much does it cost to take a bicycle on {org} ferries?",
                    "What does {org} charge for a bike?",
                ),
                "A bicycle can be taken on board for {v}.",
                money(2, 9),
            ),
            Attr(
                "checkin",
                "how early passengers must check in",
                (
                    "How early must passengers check in with {org}?",
                    "When does check-in close for {org} sailings?",
                ),
                "Passengers must check in at least {v} before departure.",
                count(15, 60, "minutes"),
            ),
            Attr(
                "luggage",
                "the luggage allowance",
                (
                    "How much luggage may a passenger bring on {org} ferries?",
                    "What is the luggage allowance with {org}?",
                ),
                "Each passenger may bring up to {v} of luggage.",
                count(15, 40, "kg"),
            ),
            Attr(
                "refund",
                "the refund deadline",
                (
                    "Until when can {org} tickets be refunded?",
                    "What is the refund deadline for {org} tickets?",
                ),
                "Tickets can be refunded up to {v} before departure.",
                count(2, 48, "hours"),
            ),
            Attr(
                "first",
                "the time of the first departure",
                (
                    "When does the first {org} ferry leave?",
                    "What time is the first sailing with {org}?",
                ),
                "The first ferry leaves at {v} every day.",
                clock([5, 6, 7]),
            ),
        ),
        (("adult", "the fare for seniors"), ("bike", "the charge for an electric scooter")),
        ("a free hot meal", "a reserved window seat", "free Wi-Fi"),
        (
            "Our crews are trained in first aid.",
            "Timetables can change in bad weather.",
            "Please keep your ticket until you leave the ship.",
        ),
    ),
    Kind(
        "library",
        "Lending Rules",
        (
            Attr(
                "loan",
                "the loan period for books",
                (
                    "How long can books be borrowed from {org}?",
                    "What is the loan period for books at {org}?",
                ),
                "Books may be borrowed for {v}.",
                count(14, 35, "days"),
            ),
            Attr(
                "items",
                "how many items a member may borrow",
                (
                    "How many items can a member of {org} borrow?",
                    "What is the borrowing limit at {org}?",
                ),
                "Members may borrow up to {v} at a time.",
                count(5, 20, "items"),
            ),
            Attr(
                "latefee",
                "the late fee",
                ("What is the late fee at {org}?", "How much does {org} charge for a late return?"),
                "The late fee is {v} per item per day.",
                lambda rng: f"{rng.randint(10, 50)} cents",
            ),
            Attr(
                "renew",
                "how many times an item can be renewed",
                (
                    "How many times can items be renewed at {org}?",
                    "What is the renewal limit at {org}?",
                ),
                "An item can be renewed {v}.",
                lambda rng: rng.choice(["twice", "three times", "four times", "once"]),
            ),
            Attr(
                "card",
                "the fee for a replacement card",
                (
                    "How much does a replacement card cost at {org}?",
                    "What is the fee for a new library card at {org}?",
                ),
                "A replacement card costs {v}.",
                money(2, 8),
            ),
            Attr(
                "opens",
                "the weekday opening time",
                (
                    "What time does {org} open on weekdays?",
                    "When does {org} open from Monday to Friday?",
                ),
                "On weekdays the library opens at {v}.",
                clock([8, 9, 10]),
            ),
            Attr(
                "print",
                "the price of printing a page",
                (
                    "How much does printing a page cost at {org}?",
                    "What does {org} charge for printing?",
                ),
                "Printing costs {v} per page.",
                lambda rng: f"{rng.randint(5, 25)} cents",
            ),
        ),
        (("loan", "the loan period for DVDs"), ("opens", "the Sunday opening time")),
        ("free coffee", "home delivery", "a reading room booking"),
        (
            "Quiet study areas are on the first floor.",
            "Members can suggest new books at the desk.",
            "Please handle rare books with care.",
        ),
    ),
    Kind(
        "gym",
        "Membership Terms",
        (
            Attr(
                "monthly",
                "the monthly membership fee",
                (
                    "How much is the monthly membership at {org}?",
                    "What is the monthly fee at {org}?",
                ),
                "The monthly membership fee is {v}.",
                money(20, 60),
            ),
            Attr(
                "joining",
                "the joining fee",
                (
                    "Is there a joining fee at {org}, and how much is it?",
                    "How much is the joining fee at {org}?",
                ),
                "New members pay a one-time joining fee of {v}.",
                money(10, 50),
            ),
            Attr(
                "notice",
                "the cancellation notice period",
                (
                    "How much notice is needed to cancel at {org}?",
                    "What is the cancellation notice period at {org}?",
                ),
                "Memberships can be cancelled with {v} notice.",
                count(14, 60, "days"),
            ),
            Attr(
                "guest",
                "the price of a guest pass",
                (
                    "How much is a guest pass at {org}?",
                    "What does a day pass for a guest cost at {org}?",
                ),
                "A guest pass costs {v} per visit.",
                money(5, 20),
            ),
            Attr(
                "opens",
                "the weekday opening time",
                ("What time does {org} open on weekdays?", "When does {org} open in the morning?"),
                "On weekdays the gym opens at {v}.",
                clock([5, 6, 7]),
            ),
            Attr(
                "booking",
                "how far ahead classes can be booked",
                (
                    "How far ahead can classes be booked at {org}?",
                    "When does class booking open at {org}?",
                ),
                "Classes can be booked up to {v} in advance.",
                count(2, 14, "days"),
            ),
        ),
        (("monthly", "the annual membership fee"), ("guest", "the price of a student pass")),
        ("a free personal trainer session", "free towels", "sauna access"),
        (
            "Please wipe equipment after use.",
            "Lockers are emptied every night.",
            "Staff can show you how to use the machines.",
        ),
    ),
    Kind(
        "parking",
        "Residents' Parking Permits",
        (
            Attr(
                "annual",
                "the price of an annual permit",
                (
                    "How much is an annual parking permit from {org}?",
                    "What does a yearly resident permit cost from {org}?",
                ),
                "An annual resident permit costs {v}.",
                money(40, 160),
            ),
            Attr(
                "second",
                "the price of a permit for a second car",
                (
                    "How much is a permit for a second car from {org}?",
                    "What does {org} charge for a second vehicle?",
                ),
                "A permit for a second car costs {v}.",
                money(60, 220),
            ),
            Attr(
                "visitor",
                "how many visitor permits a household gets",
                (
                    "How many visitor permits does a household get from {org}?",
                    "What is the yearly visitor permit allowance from {org}?",
                ),
                "Each household may receive {v} per year.",
                count(20, 80, "visitor permits"),
            ),
            Attr(
                "processing",
                "how long an application takes",
                (
                    "How long does {org} take to process a permit application?",
                    "How quickly are permit applications handled by {org}?",
                ),
                "Applications are processed within {v}.",
                count(3, 15, "working days"),
            ),
            Attr(
                "vehicles",
                "the number of permits per household",
                (
                    "How many permits can one household have from {org}?",
                    "What is the maximum number of permits per household from {org}?",
                ),
                "A household may hold at most {v}.",
                count(2, 4, "permits"),
            ),
        ),
        (("annual", "the price of a monthly permit"), ("visitor", "the price of a visitor permit")),
        ("free parking on Sundays", "a free electric charging session", "a reserved bay"),
        (
            "Permits must be displayed in the windscreen.",
            "Changes of address must be reported.",
            "Enforcement officers patrol every day.",
        ),
    ),
    Kind(
        "water",
        "Customer Service Standards",
        (
            Attr(
                "leak",
                "how quickly reported leaks are attended",
                (
                    "How quickly does {org} attend reported leaks?",
                    "What is the response time for leaks at {org}?",
                ),
                "We attend reported leaks within {v}.",
                count(2, 8, "hours"),
            ),
            Attr(
                "meter",
                "how often meters are read",
                (
                    "How often does {org} read water meters?",
                    "What is the meter reading interval at {org}?",
                ),
                "Meters are read every {v}.",
                lambda rng: rng.choice(["3 months", "6 months", "12 months"]),
            ),
            Attr(
                "reconnect",
                "the reconnection fee",
                (
                    "How much is the reconnection fee at {org}?",
                    "What does {org} charge to reconnect a supply?",
                ),
                "The reconnection fee is {v}.",
                money(25, 90, "pounds"),
            ),
            Attr(
                "complaint",
                "how quickly complaints are answered",
                (
                    "How quickly does {org} answer complaints?",
                    "What is the reply time for complaints at {org}?",
                ),
                "Written complaints are answered within {v}.",
                count(5, 20, "working days"),
            ),
            Attr(
                "compensation",
                "the compensation for a missed appointment",
                (
                    "What compensation does {org} pay for a missed appointment?",
                    "How much does {org} pay if it misses an appointment?",
                ),
                "If we miss an appointment we pay {v} in compensation.",
                money(10, 40, "pounds"),
            ),
        ),
        (("leak", "how quickly burst mains are repaired"), ("reconnect", "the disconnection fee")),
        ("a free water-saving kit", "a free plumber visit", "bottled water deliveries"),
        (
            "We test water quality every day.",
            "Our call centre can arrange a visit.",
            "Please report leaks as soon as you see them.",
        ),
    ),
    Kind(
        "college",
        "Admissions Information",
        (
            Attr(
                "deadline",
                "the application deadline",
                (
                    "When is the application deadline at {org}?",
                    "By what date must applications reach {org}?",
                ),
                "Applications must arrive by {v}.",
                lambda rng: f"{rng.randint(1, 28)} {rng.choice(['March', 'April', 'May', 'June'])}",
            ),
            Attr(
                "tuition",
                "the yearly tuition fee",
                (
                    "How much is the yearly tuition at {org}?",
                    "What are the annual tuition fees at {org}?",
                ),
                "The tuition fee is {v} per year.",
                money(2000, 9000),
            ),
            Attr(
                "deposit",
                "the deposit to accept a place",
                (
                    "How much is the deposit to accept a place at {org}?",
                    "What deposit does {org} ask for?",
                ),
                "A deposit of {v} secures a place.",
                money(100, 500),
            ),
            Attr(
                "classsize",
                "the maximum class size",
                ("How large are classes at {org}?", "What is the maximum class size at {org}?"),
                "Classes have at most {v}.",
                count(12, 30, "students"),
            ),
            Attr(
                "attendance",
                "the attendance requirement",
                (
                    "What attendance does {org} require?",
                    "How much attendance is required at {org}?",
                ),
                "Students must attend at least {v} of classes.",
                lambda rng: f"{rng.choice([70, 75, 80, 85, 90])}%",
            ),
        ),
        (
            ("tuition", "the tuition fee for international students"),
            ("deadline", "the deadline for late applications"),
        ),
        ("a free laptop", "guaranteed housing", "a scholarship for every student"),
        (
            "Open days are held every term.",
            "The library is open to all students.",
            "Advisers can help with course choices.",
        ),
    ),
    Kind(
        "clinic",
        "Patient Information",
        (
            Attr(
                "booking",
                "how far ahead appointments can be booked",
                (
                    "How far ahead can appointments be booked at {org}?",
                    "When can patients book appointments at {org}?",
                ),
                "Appointments can be booked up to {v} ahead.",
                count(2, 6, "weeks"),
            ),
            Attr(
                "cancel",
                "the cancellation notice for appointments",
                (
                    "How much notice does {org} need to cancel an appointment?",
                    "When must patients cancel appointments at {org}?",
                ),
                "Please cancel at least {v} before your appointment.",
                count(12, 48, "hours"),
            ),
            Attr(
                "missed",
                "the fee for a missed appointment",
                (
                    "Is there a fee for a missed appointment at {org}?",
                    "How much does {org} charge for a missed appointment?",
                ),
                "A missed appointment costs {v}.",
                money(10, 40),
            ),
            Attr(
                "repeat",
                "how long a repeat prescription takes",
                (
                    "How long does a repeat prescription take at {org}?",
                    "When is a repeat prescription ready at {org}?",
                ),
                "Repeat prescriptions are ready within {v}.",
                count(2, 5, "working days"),
            ),
            Attr(
                "phone",
                "the time the phone line opens",
                (
                    "When does the {org} phone line open?",
                    "What time can patients start calling {org}?",
                ),
                "The phone line opens at {v}.",
                clock([7, 8, 9]),
            ),
        ),
        (("missed", "the fee for a home visit"), ("phone", "the time the phone line closes")),
        ("free parking", "a same-day doctor guarantee", "free flu vaccines for everyone"),
        (
            "Please bring a list of your medicines.",
            "Interpreters can be arranged.",
            "The waiting room has step-free access.",
        ),
    ),
    Kind(
        "museum",
        "Visitor Guide",
        (
            Attr(
                "ticket",
                "the price of an adult ticket",
                (
                    "How much is an adult ticket at {org}?",
                    "What does entry cost for an adult at {org}?",
                ),
                "An adult ticket costs {v}.",
                money(8, 25),
            ),
            Attr(
                "tour",
                "the maximum size of a guided tour",
                (
                    "How many people can join a guided tour at {org}?",
                    "What is the group limit for guided tours at {org}?",
                ),
                "Guided tours are limited to {v}.",
                count(10, 25, "people"),
            ),
            Attr(
                "lastentry",
                "the time of last entry",
                ("When is the last entry at {org}?", "What time is last admission at {org}?"),
                "Last entry is at {v}.",
                clock([15, 16, 17]),
            ),
            Attr(
                "locker",
                "the locker deposit",
                (
                    "How much is the locker deposit at {org}?",
                    "What deposit is needed for a locker at {org}?",
                ),
                "Lockers need a deposit of {v}.",
                money(1, 5),
            ),
            Attr(
                "childfree",
                "the age under which children enter free",
                ("Up to what age do children enter {org} free?", "Do young children pay at {org}?"),
                "Children under {v} enter free.",
                lambda rng: str(rng.randint(5, 16)),
            ),
        ),
        (("ticket", "the price of a family ticket"), ("tour", "the price of a guided tour")),
        ("a free audio guide", "a free coffee", "free parking"),
        (
            "Photography without flash is allowed.",
            "Large bags are not allowed in the galleries.",
            "The shop sells books and prints.",
        ),
    ),
)
VALIDATION_ONLY_KINDS = ("clinic", "museum")
NAMES = (
    "Tollbrook",
    "Calloway",
    "Dunmore",
    "Elsworth",
    "Fernhollow",
    "Garnetbridge",
    "Halverton",
    "Ivymere",
    "Juniper Point",
    "Vexhall",
    "Merrowdale",
    "Netherby",
    "Oakhaven",
    "Pembury",
    "Sablewood",
    "Redwick",
    "Saltmere",
    "Thornbury",
    "Upton Vale",
    "Wexcombe",
    "Yarrowby",
    "Amberleigh",
    "Bramblecote",
    "Cindermoor",
    "Drystone",
    "Eastwold",
    "Foxhollow",
    "Glenbrook",
    "Hollinford",
    "Kingsmere",
    "Lindenfield",
    "Marlowe",
    "Northcliff",
    "Orrinton",
    "Pinewold",
    "Rosedown",
    "Stonecroft",
    "Tamberly",
    "Westerleigh",
    "Willowmere",
    "Ashcombe Ridge",
    "Bellhaven",
    "Corrindale",
    "Dovecote",
    "Emberley",
    "Fairwyn",
    "Greyhallow",
    "Heathfold",
)
SUFFIX = {
    "ferry": "Ferries",
    "library": "Library",
    "gym": "Fitness",
    "parking": "Council",
    "water": "Water",
    "college": "College",
    "clinic": "Health Centre",
    "museum": "Museum",
}


@dataclass
class Document:
    key: str
    kind: Kind
    org: str
    title: str
    split: str
    values: dict[str, str]
    edition: str | None = None
    text: str = ""
    sentences: dict[str, str] = field(default_factory=dict)


def render(document: Document, rng: random.Random) -> None:
    attrs = [attr for attr in document.kind.attrs if attr.key in document.values]
    lines = [document.title, ""]
    if document.edition:
        lines += [f"This edition was published in {document.edition}.", ""]
    lines += [
        f"1. About {document.org}",
        "",
        f"{document.org} serves the local area. " + rng.choice(document.kind.filler),
        "",
    ]
    order = attrs[:]
    rng.shuffle(order)
    for number, chunk in enumerate([order[i : i + 2] for i in range(0, len(order), 2)], start=2):
        body = []
        for attr in chunk:
            sentence = attr.sentence.format(v=document.values[attr.key])
            document.sentences[attr.key] = sentence
            body.append(sentence)
        body.append(rng.choice(document.kind.filler))
        lines += [
            f"{number}. {chunk[0].what[0].upper() + chunk[0].what[1:]}",
            "",
            " ".join(body),
            "",
        ]
    lines += ["Contact", "", f"Questions can be sent to the {document.org} help desk."]
    document.text = "\n".join(lines) + "\n"


def make_documents(rng: random.Random) -> list[Document]:
    names = list(NAMES)
    rng.shuffle(names)
    docs: list[Document] = []
    for kind in KINDS:
        orgs = 2 if kind.name in VALIDATION_ONLY_KINDS else 7  # 5 training + 2 validation
        for index in range(orgs):
            split = "validation" if kind.name in VALIDATION_ONLY_KINDS or index >= 5 else "train"
            org = f"{names.pop()} {SUFFIX[kind.name]}"
            keep = [attr for attr in kind.attrs if rng.random() < 0.7]
            if len(keep) < 3:
                keep = list(kind.attrs[:3])
            values = {attr.key: attr.values(rng) for attr in keep}
            key = f"{kind.name}-{org.split()[0].lower()}"
            doc = Document(key, kind, org, f"{org} - {kind.doc}", split, values)
            docs.append(doc)
            if index in (0, 5):  # one organisation per split also has a newer edition
                changed = rng.choice(keep)
                new_values = dict(values)
                while new_values[changed.key] == values[changed.key]:
                    new_values[changed.key] = changed.values(rng)
                doc.edition, doc.title = "2022", f"{org} - {kind.doc} (Edition 2022)"
                docs.append(
                    Document(
                        f"{key}-2025",
                        kind,
                        org,
                        f"{org} - {kind.doc} (Edition 2025)",
                        split,
                        new_values,
                        edition="2025",
                    )
                )
    for doc in docs:
        render(doc, rng)
    return docs


def two_part(a: Attr, b: Attr, org: str, rng: random.Random) -> str:
    """Two of the attributes' own questions in one ("How much ...? Also, how many ...?")."""
    second = rng.choice(b.questions).format(org=org)
    return f"{rng.choice(a.questions).format(org=org)} Also, {second[0].lower()}{second[1:]}"


def synthetic_questions(docs: list[Document], rng: random.Random) -> list[dict[str, Any]]:
    """Question specs (not yet asked)."""
    specs: list[dict[str, Any]] = []
    by_org: dict[str, list[Document]] = {}
    for doc in docs:
        by_org.setdefault(doc.org, []).append(doc)
    for org, editions in by_org.items():
        doc = editions[0]
        kind = doc.kind
        stated = [attr for attr in kind.attrs if attr.key in doc.values]
        absent = [attr for attr in kind.attrs if attr.key not in doc.values]
        base = {"org": org, "doc": doc.key, "split": doc.split, "kind_of_org": kind.name}
        if len(editions) == 2:
            new = editions[1]
            changed = next(a for a in stated if new.values.get(a.key) != doc.values[a.key])
            specs.append(
                {
                    **base,
                    "kind": "conflict",
                    "attr": changed.key,
                    "question": rng.choice(changed.questions).format(org=org),
                    "editions": [doc.key, new.key],
                }
            )
            stated = [a for a in stated if a.key != changed.key]
        for attr in stated:
            for template in attr.questions:
                specs.append(
                    {
                        **base,
                        "kind": "answer",
                        "attr": attr.key,
                        "question": template.format(org=org),
                    }
                )
        if len(stated) >= 2:
            a, b = rng.sample(stated, 2)
            specs.append(
                {
                    **base,
                    "kind": "multi",
                    "attrs": [a.key, b.key],
                    "question": two_part(a, b, org, rng),
                }
            )
        for attr in absent:
            specs.append(
                {
                    **base,
                    "kind": "unanswerable",
                    "attr": attr.key,
                    "question": rng.choice(attr.questions).format(org=org),
                    "what": f"{attr.what} for {org}",
                }
            )
        for have, missing in kind.related:
            if have in doc.values:
                specs.append(
                    {
                        **base,
                        "kind": "insufficient",
                        "question": f"What is {missing} at {org}?",
                        "what": f"{missing} for {org}",
                    }
                )
        if stated and absent:
            a, b = rng.choice(stated), rng.choice(absent)
            specs.append(
                {
                    **base,
                    "kind": "partial",
                    "attrs": [a.key],
                    "question": two_part(a, b, org, rng),
                    "what": f"{b.what} for {org}",
                }
            )
        if stated:
            a = rng.choice(stated)
            bait = rng.choice(kind.baits)
            first = rng.choice(a.questions).format(org=org)
            specs.append(
                {
                    **base,
                    "kind": "detail_bait",
                    "attrs": [a.key],
                    "question": f"{first} Also, does {org} offer {bait}?",
                    "what": f"whether {org} offers {bait}",
                }
            )
            others = [d for d in docs if d.kind is kind and d.org != org and d.split == doc.split]
            if others:
                specs.append(
                    {
                        **base,
                        "kind": "irrelevant",
                        "attr": a.key,
                        "question": rng.choice(
                            next(x for x in kind.attrs if x.key == a.key).questions
                        ).format(org=org),
                        "scope_doc": rng.choice(others).key,
                        "what": f"{a.what} for {org}",
                    }
                )
    return specs


def build_synthetic(
    docs: list[Document], specs: list[dict[str, Any]], split: str, workdir: Path
) -> tuple[list[dict[str, Any]], Counter[str]]:
    services = open_index(workdir)
    dropped: Counter[str] = Counter()
    ids: dict[str, str] = {}
    for doc in docs:
        if doc.split != split:
            continue
        row = services.ingestion.ingest_bytes(
            f"{doc.key}.txt", "text/plain", doc.text.encode("utf-8"), title=doc.title
        )
        ids[doc.key] = row.id
    by_key = {doc.key: doc for doc in docs}
    examples: list[dict[str, Any]] = []
    try:
        for spec in specs:
            if spec["split"] != split:
                continue
            doc = by_key[spec["doc"]]
            scope = [ids[spec["scope_doc"]]] if spec["kind"] == "irrelevant" else None
            messages, response = ask(services, spec["question"], scope)
            kind = spec["kind"]
            target: str | None
            if kind in ("unanswerable", "insufficient", "irrelevant"):
                target = abstain(spec["what"])
            elif kind == "conflict":
                parts = []
                for key in spec["editions"]:
                    edition = by_key[key]
                    found = cited_sentence(response, ids[key], edition.sentences[spec["attr"]])
                    if found is None:
                        break
                    parts.append(f"The {edition.edition} edition says: {found[0]} [{found[1]}].")
                target = "The sources disagree. " + " ".join(parts) if len(parts) == 2 else None
            else:
                attrs = spec.get("attrs") or [spec["attr"]]
                parts = []
                for key in attrs:
                    found = cited_sentence(response, ids[doc.key], doc.sentences[key])
                    if found is None:
                        break
                    parts.append(f"{found[0]} [{found[1]}].")
                target = " ".join(parts) if len(parts) == len(attrs) else None
                if target and kind in ("partial", "detail_bait"):
                    target += " " + abstain(spec["what"])
            if target is None:
                dropped[kind] += 1  # the passage with the answer was not retrieved
                continue
            examples.append(
                {
                    "id": "syn-"
                    + hashlib.sha256((split + spec["question"] + str(scope)).encode()).hexdigest()[
                        :12
                    ],
                    "source": "synthetic",
                    "split": split,
                    "kind": kind,
                    "org": spec["org"],
                    "doc": spec["doc"],
                    "question": spec["question"],
                    "messages": messages,
                    "target": target,
                }
            )
    finally:
        services.close()
    return examples, dropped


# ---------------------------------------------------------------------------------- NIST
def build_nist(
    queries_file: Path, corpus_file: Path, split: str, workdir: Path, rng: random.Random
) -> tuple[list[dict[str, Any]], Counter[str]]:
    corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
    for name in PROTECTED:
        if name in str(corpus_file):
            raise SystemExit(f"refusing to read a protected dataset: {corpus_file}")
    services = open_index(workdir)
    dropped: Counter[str] = Counter()
    ids: dict[str, str] = {}
    titles: dict[str, str] = {}
    for doc in corpus["documents"]:
        path = corpus_file.parent / doc["file"]
        if not path.is_file():
            raise SystemExit(f"missing {path}: run python scripts/fetch_nist_corpus.py first")
        row = services.ingestion.ingest_bytes(
            path.name, None, path.read_bytes(), title=doc["title"]
        )
        ids[doc["key"]] = row.id
        titles[doc["key"]] = doc["title"]
    queries = [
        json.loads(line)
        for line in queries_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples: list[dict[str, Any]] = []
    try:
        for query in queries:
            asks = [(query["doc"], "nist_answer")]
            if rng.random() < 1 / 3:
                asks.append(
                    (rng.choice([k for k in ids if k != query["doc"]]), "nist_other_document")
                )
            for doc_key, kind in asks:
                messages, response = ask(services, query["question"], [ids[doc_key]])
                wanted = normalize_whitespace(query["answer"]).lower().strip(" .\"'")
                target = None
                for source in response.sources:
                    for sentence in split_sentences(source.text):
                        if wanted and wanted in normalize_whitespace(sentence).lower():
                            target = (
                                f"{normalize_whitespace(sentence).rstrip(' .')} [{source.label}]."
                            )
                            break
                    if target:
                        break
                if kind == "nist_other_document":
                    if target is not None:
                        dropped[kind] += 1  # the other document happens to contain it
                        continue
                    target = abstain("this")
                elif target is None:
                    dropped[kind] += 1  # the answer was not retrieved
                    continue
                examples.append(
                    {
                        "id": "nist-"
                        + hashlib.sha256(
                            (split + query["question"] + doc_key).encode()
                        ).hexdigest()[:12],
                        "source": "nist",
                        "split": split,
                        "kind": kind,
                        "doc": doc_key,
                        "question": query["question"],
                        "messages": messages,
                        "target": target,
                    }
                )
    finally:
        services.close()
    return examples, dropped


# ------------------------------------------------------------------------- evaluation set
def write_eval_dataset(docs: list[Document], specs: list[dict[str, Any]], out: Path) -> int:
    """The held-out organisations as a GroundedAI evaluation dataset (for rag_eval)."""
    corpus = out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    val_docs = [doc for doc in docs if doc.split == "validation"]
    by_key = {doc.key: doc for doc in docs}
    manifest = {
        "name": "groundedai-finetune-synthetic-eval",
        "version": "1.0.0",
        "license": "MIT; fictional documents generated by training/prepare_dataset.py",
        "description": "Held-out fictional organisations (validation split of "
        "training/prepare_dataset.py). Used to compare the base and the fine-tuned answer model; "
        "never trained on.",
        "created": "2026-11-16",
        "documents": [],
    }
    for doc in val_docs:
        path = corpus / f"{doc.key}.txt"
        path.write_bytes(doc.text.encode("utf-8"))
        manifest["documents"].append(
            {
                "key": doc.key,
                "file": f"corpus/{doc.key}.txt",
                "title": doc.title,
                "sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                "split": "validation",
            }
        )
    category = {
        "answer": "factual",
        "multi": "multi_chunk",
        "unanswerable": "unanswerable",
        "insufficient": "unanswerable",
        "irrelevant": "out_of_scope",
        "partial": "factual",
        "detail_bait": "factual",
        "conflict": "conflicting",
    }
    rows = []
    for number, spec in enumerate(s for s in specs if s["split"] == "validation"):
        doc = by_key[spec["doc"]]
        kind = spec["kind"]
        row: dict[str, Any] = {
            "id": f"fe{number:03d}",
            "category": category[kind],
            "question": spec["question"],
            "tags": [f"kind:{kind}", f"org-kind:{doc.kind.name}"],
            "scope": [spec["scope_doc"]] if kind == "irrelevant" else [],
        }
        if kind in ("unanswerable", "insufficient", "irrelevant"):
            row.update(expected_behavior="abstain", answer_must_include=[], evidence=[])
        elif kind == "conflict":
            editions = [by_key[k] for k in spec["editions"]]
            row.update(
                expected_behavior="flag_conflict",
                answer_must_include=[e.values[spec["attr"]] for e in editions],
                evidence=[{"doc": e.key, "quote": e.sentences[spec["attr"]]} for e in editions],
            )
        else:
            attrs = spec.get("attrs") or [spec["attr"]]
            row.update(
                expected_behavior="answer",
                answer_must_include=[doc.values[a] for a in attrs],
                evidence=[{"doc": doc.key, "quote": doc.sentences[a]} for a in attrs],
            )
        rows.append(row)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "questions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    return len(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--no-nist", action="store_true", help="fictional documents only")
    args = parser.parse_args()
    rng = random.Random(SEED)  # noqa: S311 - data generation, not security
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    docs = make_documents(rng)
    specs = synthetic_questions(docs, rng)
    docs_dir = out / "synthetic-documents"
    docs_dir.mkdir(exist_ok=True)
    for doc in docs:
        (docs_dir / f"{doc.split}-{doc.key}.txt").write_bytes(doc.text.encode("utf-8"))
    work = Path(tempfile.mkdtemp(prefix="groundedai-sft-"))
    report: dict[str, Any] = {"seed": SEED, "dropped": {}}
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split in ("train", "validation"):
        examples, dropped = build_synthetic(docs, specs, split, work / f"syn-{split}")
        splits[split] += examples
        report["dropped"][f"synthetic-{split}"] = dict(dropped)
    if not args.no_nist:
        data = FINETUNE / "data"
        nist_train, dropped = build_nist(
            data / "train_queries.jsonl",
            FINETUNE / "corpus.json",
            "train",
            work / "nist-train",
            rng,
        )
        splits["train"] += nist_train
        report["dropped"]["nist-train"] = dict(dropped)
        val_manifest = ROOT / "evaluation" / "datasets" / "groundedai-nist-val-v1" / "manifest.json"
        nist_val, dropped = build_nist(
            data / "val_queries.jsonl", val_manifest, "validation", work / "nist-val", rng
        )
        splits["validation"] += nist_val
        report["dropped"]["nist-validation"] = dict(dropped)
    for name in splits:
        random.Random(SEED).shuffle(splits[name])  # noqa: S311
    report["sha256"] = {
        "train.jsonl": write_jsonl(out / "train.jsonl", splits["train"]),
        "validation.jsonl": write_jsonl(out / "validation.jsonl", splits["validation"]),
    }
    report["counts"] = {
        name: dict(Counter(r["kind"] for r in rows)) for name, rows in splits.items()
    }
    report["totals"] = {name: len(rows) for name, rows in splits.items()}
    report["organisations"] = {
        split: sorted({d.org for d in docs if d.split == split})
        for split in ("train", "validation")
    }
    report["eval_questions"] = write_eval_dataset(docs, specs, out / "synthetic-eval")
    (out / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {k: report[k] for k in ("totals", "counts", "dropped", "eval_questions")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

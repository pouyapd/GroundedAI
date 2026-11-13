"""Supervised fine-tuning examples for the answer model, built with GroundedAI's own pipeline.

    python experiments/finetune/make_sft_data.py --queries data/train_queries.jsonl \
        --corpus corpus.json --out data/sft_train.jsonl

Each generated question is asked through the real QA service over the training PDFs (text
clean-up v2, hybrid retrieval, top 5, answer-v2 prompt). The exact chat messages the model
would receive are recorded, and the target reply is built from the retrieved passages:

* if a passage contains the answer phrase: the sentence of that passage that holds it, cited
  with that passage's label ("<sentence> [S3].") - grounded by construction;
* if no retrieved passage contains it: "INSUFFICIENT_EVIDENCE: the sources do not state this."

A third of the questions are also asked against a different training document, which gives
questions the sources cannot answer (target: the abstention). Nothing here touches the
validation or test documents.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from groundedai.core.config import Settings  # noqa: E402
from groundedai.core.text import normalize_whitespace, split_sentences  # noqa: E402
from groundedai.generation.prompts import ABSTAIN_MARKER  # noqa: E402
from groundedai.generation.providers import GenerationProvider  # noqa: E402
from groundedai.schemas.query import QueryRequest  # noqa: E402
from groundedai.services.container import build_services  # noqa: E402

SEED = 20261109
ABSTAIN = f"{ABSTAIN_MARKER}: the sources do not state this."


class Capture(BaseChatModel):
    """Records the prompt messages and replies with an abstention (nothing is generated)."""

    last: ClassVar[list[dict[str, str]]] = []

    @property
    def _llm_type(self) -> str:
        return "capture"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        Capture.last = [{"role": roles[m.type], "content": str(m.content)} for m in messages]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=ABSTAIN))])


def target_for(sources: list[Any], answer: str) -> tuple[str, str | None]:
    wanted = normalize_whitespace(answer).lower().strip(" .\"'")
    for source in sources:
        text = normalize_whitespace(source.text)
        if wanted and wanted in text.lower():
            for sentence in split_sentences(source.text):
                if wanted in normalize_whitespace(sentence).lower():
                    body = normalize_whitespace(sentence).rstrip(" .")
                    return f"{body} [{source.label}].", source.label
    return ABSTAIN, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).parent
    resolve = lambda p: p if p.is_absolute() else here / p  # noqa: E731
    rng = random.Random(SEED)  # noqa: S311 - sampling, not security
    queries = [
        json.loads(line)
        for line in resolve(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus_path = resolve(args.corpus)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="sft-"))
    settings = Settings(_env_file=None, data_dir=tmp, text_cleanup="v2", rate_limit_enabled=False)
    provider = GenerationProvider(
        name="capture", model="capture", is_llm=True, chat_model=Capture()
    )
    services = build_services(settings, generation_provider=provider)
    ids = {}
    for doc in corpus["documents"]:
        path = corpus_path.parent / doc["file"]
        row = services.ingestion.ingest_bytes(
            path.name, None, path.read_bytes(), title=doc["title"]
        )
        ids[doc["key"]] = row.id
    rows = []
    for query in queries:
        asks = [(query["doc"], "own document")]
        if rng.random() < 1 / 3:
            other = rng.choice([key for key in ids if key != query["doc"]])
            asks.append((other, "other document"))
        for doc_key, how in asks:
            response = services.qa.answer(
                QueryRequest(question=query["question"], document_ids=[ids[doc_key]])
            )
            target, label = target_for(response.sources, query["answer"])
            if how == "other document" and label is not None:
                continue  # the other document happens to contain the answer: skip
            rows.append(
                {
                    "doc": query["doc"],
                    "asked_in": doc_key,
                    "question": query["question"],
                    "messages": Capture.last,
                    "target": target,
                    "gold_label": label,
                    "kind": "answer" if label else "abstain",
                }
            )
    services.close()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
        newline="\n",
    )
    print(
        len(rows),
        "examples;",
        sum(r["kind"] == "answer" for r in rows),
        "answers,",
        sum(r["kind"] == "abstain" for r in rows),
        "abstentions",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

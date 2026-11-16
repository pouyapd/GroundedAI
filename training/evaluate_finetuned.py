"""Evaluate the base and the fine-tuned answer model through GroundedAI's own pipeline.

    python training/evaluate_finetuned.py --dataset training/data/synthetic-eval \
        --label base --out base.json
    python training/evaluate_finetuned.py --dataset training/data/synthetic-eval \
        --adapter runs/run1/adapter --label lora --out lora.json
    python scripts/compare_rag_reports.py base.json lora.json

Both runs use the same index, retrieval, prompt (answer-v2), checks and decoding: GroundedAI's
evaluation runner (`groundedai.evaluation.rag_eval`) with a local Transformers model on the GPU
(fp16; the CPU in fp32 works for testing), greedy decoding, at most ``max_new_tokens`` (384)
new tokens. ``--adapter`` loads the LoRA adapter on top of the same base model (PEFT);
``--merge`` merges it first, to check that a merged model answers the same.

Added to each question's row, as measurements (they do not change any status):

* ``relevance``: asked words the passages do not mention, and the answer's coverage of asked words;
* ``added_detail_claims``: statements with two or more consecutive words found in no passage
  (the optional added-detail rule, run here only to count);
* ``format_valid``: the reply is either an abstention in the application's format, or an
  answer whose every sentence-level citation names a provided source and which has at least one.

The summary adds these, and correctness and refusals by kind of question (``kind:`` tags of the
synthetic set, or the dataset's categories). The protected test set is run once, after model
selection; see docs/FINETUNING.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import ConfigDict  # noqa: E402

ROLES = {"system": "system", "human": "user", "ai": "assistant"}


class TransformersChat(BaseChatModel):
    """A local Transformers causal LM behind LangChain's chat interface (greedy decoding)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any
    tokenizer: Any
    max_new_tokens: int = 384
    device: str = "cpu"
    generated_tokens: int = 0

    @property
    def _llm_type(self) -> str:
        return "transformers-local"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        import torch

        chat = [{"role": ROLES[m.type], "content": str(m.content)} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = output[0][inputs["input_ids"].shape[1] :]
        self.generated_tokens += int(new.shape[0])
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def load_model(
    base: str, adapter: Path | None, merge: bool
) -> tuple[Any, Any, str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter.parent / "tokenizer")
        if adapter and (adapter.parent / "tokenizer").is_dir()
        else base
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    info: dict[str, Any] = {
        "base": base,
        "adapter": str(adapter) if adapter else None,
        "merged": False,
    }
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        info["adapter_loaded"] = True
        if merge:
            model = model.merge_and_unload()
            info["merged"] = True
    model.to(device).eval()
    info["load_seconds"] = round(time.perf_counter() - started, 1)
    info["device"] = device
    info["dtype"] = "float16" if device == "cuda" else "float32"
    return model, tokenizer, device, info


def extra_measures(response: Any, question: Any, claim_check: str) -> dict[str, Any]:
    from groundedai.generation.claims import EvidenceSource, check_claims
    from groundedai.generation.prompts import ABSTAIN_MARKER

    row: dict[str, Any] = {}
    if response.relevance is not None:
        row["relevance"] = {
            "unmatched": response.relevance.unmatched,
            "answer_coverage": response.relevance.answer_coverage,
        }
    output = (response.model_output or "").strip()
    if response.status == "answered" and response.answer:
        sources = [EvidenceSource(s.label, s.document_title, s.text) for s in response.sources]
        report = check_claims(
            response.answer, sources, question.question, claim_check, detail_check=True
        )  # type: ignore[arg-type]
        row["added_detail_claims"] = sum(
            1 for claim in report.claims if "added_detail" in claim.flags
        )
        row["format_valid"] = (
            bool(response.citation_check.valid) and not response.citation_check.invalid
        )
    elif response.status == "insufficient_evidence":
        row["added_detail_claims"] = 0
        row["format_valid"] = ABSTAIN_MARKER in output
    else:
        row["added_detail_claims"] = 0
        row["format_valid"] = False  # invalid_output or error
    return row


def by_kind(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        kinds = [t.split(":", 1)[1] for t in row.get("tags", []) if t.startswith("kind:")]
        groups[kinds[0] if kinds else row["category"]].append(row)
    out = {}
    for name, items in sorted(groups.items()):
        answer = [r for r in items if r["expected_behavior"] in ("answer", "flag_conflict")]
        refuse = [r for r in items if r["expected_behavior"] == "abstain"]
        out[name] = {
            "questions": len(items),
            "correct": sum(bool(r.get("correct")) for r in answer) if answer else None,
            "should_answer": len(answer),
            "refused": sum(bool(r.get("abstained")) for r in items),
            "should_refuse": len(refuse),
            "refused_correctly": sum(bool(r.get("abstained")) for r in refuse) if refuse else None,
        }
    return out


def extra_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in rows if r.get("status") == "answered"]
    coverage = [
        r["relevance"]["answer_coverage"]
        for r in answered
        if r.get("relevance") and r["relevance"]["answer_coverage"] is not None
    ]
    return {
        "format_valid": sum(bool(r.get("format_valid")) for r in rows),
        "format_checked": sum(1 for r in rows if "format_valid" in r),
        "answers_with_added_details": sum(1 for r in answered if r.get("added_detail_claims")),
        "added_detail_claims": sum(r.get("added_detail_claims", 0) for r in rows),
        "answered": len(answered),
        "mean_answer_coverage_of_asked_words": round(sum(coverage) / len(coverage), 4)
        if coverage
        else None,
        "by_kind": by_kind(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from groundedai.core.config import Settings
    from groundedai.evaluation import rag_eval
    from groundedai.evaluation.dataset import load_dataset
    from groundedai.evaluation.environment import code_version, environment_info
    from groundedai.evaluation.index import build_index
    from groundedai.generation.providers import GenerationProvider
    from groundedai.ingestion.chunking import ChunkingConfig
    from groundedai.retrieval.embeddings import build_embedding_provider

    workdir = Path(tempfile.mkdtemp(prefix="ft-eval-"))
    settings = Settings(
        _env_file=None,
        data_dir=workdir,
        rate_limit_enabled=False,
        generation_timeout_s=600.0,
        max_concurrent_generations=1,
        generation_queue_timeout_s=600.0,
    )
    dataset = load_dataset(args.dataset)
    corpus = build_index(
        dataset,
        build_embedding_provider(settings),
        ChunkingConfig(settings.chunk_size, settings.chunk_overlap, settings.text_cleanup),
        workdir,
        dataset.manifest.name,
    )
    model, tokenizer, device, model_info = load_model(args.base, args.adapter, args.merge)
    chat = TransformersChat(
        model=model, tokenizer=tokenizer, max_new_tokens=settings.max_new_tokens, device=device
    )
    provider = GenerationProvider(
        name="transformers-local", model=args.label, is_llm=True, chat_model=chat
    )

    original = rag_eval.score_question

    def score(question: Any, response: Any, *rest: Any) -> dict[str, Any]:
        row = original(question, response, *rest)
        if response is not None:
            row.update(extra_measures(response, question, settings.claim_check))
        return row

    rag_eval.score_question = score  # type: ignore[assignment]
    started = time.perf_counter()
    try:
        report = rag_eval.run_rag_eval(
            dataset, corpus, provider, settings, progress=print, limit=args.limit
        )
    finally:
        rag_eval.score_question = original  # type: ignore[assignment]
    seconds = time.perf_counter() - started
    report.update(
        label=args.label,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        generator="transformers-local",
        model=model_info,
        decoding={
            "do_sample": False,
            "max_new_tokens": settings.max_new_tokens,
            "chat_template": "the tokenizer's",
        },
        settings={
            name: getattr(settings, name)
            for name in (
                "text_cleanup",
                "retrieval_mode",
                "top_k",
                "prompt_version",
                "claim_check",
                "claim_detail_check",
                "conflict_check",
                "extractive_relevance",
                "embedding_provider",
                "embedding_model",
                "chunk_size",
                "chunk_overlap",
            )
        },
        extra=extra_summary(report["per_question"]),
        cost={"seconds": round(seconds, 1), "generated_tokens": chat.generated_tokens},
        environment=environment_info(),
        code=code_version(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "answers": summary.get("answers"),
                "abstention": summary.get("abstention"),
                "extra": {k: v for k, v in report["extra"].items() if k != "by_kind"},
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

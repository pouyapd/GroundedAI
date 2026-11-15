"""A deterministic extractive "chat model" for tests and the no-model demo mode.

It is NOT a language model. It reads the sources from the prompt, picks the
sentences with the highest word overlap with the question and returns them
verbatim with citations, or abstains when overlap is low. It implements
LangChain's ``BaseChatModel`` so the real prompt -> model -> parser chain is
exercised end to end without downloading anything. API responses mark its
output with ``is_llm: false`` and the UI labels it as demo output.
"""

from __future__ import annotations

import html
import re
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from groundedai.core.text import content_tokens, split_sentences
from groundedai.generation.prompts import ABSTAIN_MARKER
from groundedai.generation.relevance import question_relevance, shared_prefix, words_match
from groundedai.generation.safety import looks_like_instruction

_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
_SOURCE_RE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_SOURCE_TITLE_RE = re.compile(r'<source id="S\d+" title="([^"]*)"')

EXTRACTIVE_MODEL_NAME = "extractive-demo-v1"
EXTRACTIVE_PREFIX_MODEL_NAME = "extractive-demo-v2-prefix"
NOT_ANSWERED = "The sources do not appear to answer this question."

__all__ = ["ExtractiveDemoChatModel", "extractive_model_name", "shared_prefix", "words_match"]


def extractive_model_name(matching: str, relevance: str = "v1") -> str:
    name = EXTRACTIVE_MODEL_NAME if matching == "exact" else EXTRACTIVE_PREFIX_MODEL_NAME
    return name if relevance == "v1" else f"{name}-relevance-v2"


class ExtractiveDemoChatModel(BaseChatModel):
    min_overlap: float = 0.5
    max_sentences: int = 3
    # "exact": a question word counts only when the sentence has the same word (v1, published
    # results); "prefix": also when it has another form of it (see words_match).
    matching: Literal["exact", "prefix"] = "exact"
    # "v1": every question word counts (published results); "v2": title words do not count, so
    # a sentence must cover half of what the question asks about beyond the document's name.
    relevance: Literal["v1", "v2"] = "v1"

    @property
    def _llm_type(self) -> str:
        return "groundedai-extractive-demo"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "min_overlap": self.min_overlap,
            "max_sentences": self.max_sentences,
            "matching": self.matching,
            "relevance": self.relevance,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(message.content) for message in messages)
        question_match = _QUESTION_RE.search(prompt)
        question = question_match.group(1) if question_match else ""
        sources = _SOURCE_RE.findall(prompt)
        titles = [html.unescape(title) for title in _SOURCE_TITLE_RE.findall(prompt)]
        answer = self.compose(question, sources, titles)
        name = extractive_model_name(self.matching, self.relevance)
        message = AIMessage(content=answer, response_metadata={"model_name": name})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def compose(
        self, question: str, sources: list[tuple[str, str]], titles: list[str] | None = None
    ) -> str:
        question_tokens = set(content_tokens(question))
        if self.relevance == "v2" and question_tokens and sources:
            relevance = question_relevance(question, titles or [], [body for _, body in sources])
            if relevance.unmatched:
                missing = ", ".join(f'"{word}"' for word in relevance.unmatched)
                return (
                    f"{ABSTAIN_MARKER}: The sources do not mention {missing}, which the question "
                    "asks about."
                )
            # Scored on every question word except title words; the generic words left out
            # above only decide whether the passages mention what is asked.
            question_tokens = set(relevance.scored) or question_tokens
        if not question_tokens or not sources:
            return f"{ABSTAIN_MARKER}: No relevant sources were provided for this question."

        candidates: list[tuple[float, int, int, str, str]] = []
        for source_index, (label, body) in enumerate(sources):
            for sentence_index, sentence in enumerate(split_sentences(body)):
                if looks_like_instruction(sentence):
                    continue  # never repeat instruction-like text from documents
                overlap = self._overlap(question_tokens, set(content_tokens(sentence)))
                if overlap > 0:
                    candidates.append((overlap, source_index, sentence_index, label, sentence))
        if not candidates:
            return f"{ABSTAIN_MARKER}: The sources do not appear to answer this question."

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        best = candidates[0][0]
        if best < self.min_overlap:
            return f"{ABSTAIN_MARKER}: The sources do not appear to answer this question."

        threshold = max(self.min_overlap, 0.6 * best)
        eligible = [item for item in candidates if item[0] >= threshold]
        selected: list[tuple[float, int, int, str, str]] = []
        used_labels: set[str] = set()
        # First prefer one sentence per source, then fill with the remaining best ones.
        for item in eligible:
            if len(selected) < self.max_sentences and item[3] not in used_labels:
                selected.append(item)
                used_labels.add(item[3])
        for item in eligible:
            if len(selected) >= self.max_sentences:
                break
            if item not in selected:
                selected.append(item)

        parts = []
        seen_text: set[str] = set()
        for _, _, _, label, sentence in selected:
            key = sentence.casefold()
            if key in seen_text:
                continue
            seen_text.add(key)
            body = sentence.rstrip()
            ending = body[-1] if body and body[-1] in ".!?" else "."
            body = body[:-1] if body and body[-1] in ".!?" else body
            parts.append(f"{body} [{label}]{ending}")
        return " ".join(parts)

    def _overlap(self, question_tokens: set[str], sentence_tokens: set[str]) -> float:
        if self.matching == "exact":
            return len(question_tokens & sentence_tokens) / len(question_tokens)
        matched = sum(
            1
            for word in question_tokens
            if word in sentence_tokens or any(words_match(word, other) for other in sentence_tokens)
        )
        return matched / len(question_tokens)

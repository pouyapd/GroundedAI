"""A deterministic extractive "chat model" for tests and the no-model demo mode.

It is NOT a language model. It reads the sources from the prompt, picks the
sentences with the highest word overlap with the question and returns them
verbatim with citations, or abstains when overlap is low. It implements
LangChain's ``BaseChatModel`` so the real prompt -> model -> parser chain is
exercised end to end without downloading anything. API responses mark its
output with ``is_llm: false`` and the UI labels it as demo output.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from groundedai.core.text import content_tokens, split_sentences
from groundedai.generation.prompts import ABSTAIN_MARKER
from groundedai.generation.safety import looks_like_instruction

_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
_SOURCE_RE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)

EXTRACTIVE_MODEL_NAME = "extractive-demo-v1"
EXTRACTIVE_PREFIX_MODEL_NAME = "extractive-demo-v2-prefix"
MIN_SHARED_PREFIX = 5


def shared_prefix(a: str, b: str) -> int:
    length = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        length += 1
    return length


def words_match(a: str, b: str) -> bool:
    """Same word, or two forms of one word: a shared prefix of at least five letters that
    leaves at most three letters of the shorter word ("reauthenticated" / "reauthentication",
    "minutes" / "minute"). A heuristic; it also joins some unrelated words."""
    if a == b:
        return True
    shared = shared_prefix(a, b)
    return shared >= max(MIN_SHARED_PREFIX, min(len(a), len(b)) - 3)


class ExtractiveDemoChatModel(BaseChatModel):
    min_overlap: float = 0.5
    max_sentences: int = 3
    # "exact": a question word counts only when the sentence has the same word (v1, published
    # results); "prefix": also when it has another form of it (see words_match).
    matching: Literal["exact", "prefix"] = "exact"

    @property
    def _llm_type(self) -> str:
        return "groundedai-extractive-demo"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "min_overlap": self.min_overlap,
            "max_sentences": self.max_sentences,
            "matching": self.matching,
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
        answer = self.compose(question, sources)
        name = EXTRACTIVE_MODEL_NAME if self.matching == "exact" else EXTRACTIVE_PREFIX_MODEL_NAME
        message = AIMessage(content=answer, response_metadata={"model_name": name})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def compose(self, question: str, sources: list[tuple[str, str]]) -> str:
        question_tokens = set(content_tokens(question))
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

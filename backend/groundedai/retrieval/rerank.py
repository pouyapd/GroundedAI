"""Optional second-stage reranking."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Protocol

from groundedai.core.errors import ProviderUnavailable
from groundedai.retrieval.embeddings import configure_hf_offline


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder, loaded lazily from the local cache."""

    def __init__(self, model_name: str, *, offline: bool = True, max_length: int = 512):
        self.name = f"cross-encoder:{model_name}"
        self.model_name = model_name
        self.offline = offline
        self.max_length = max_length
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                configure_hf_offline(self.offline)
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise ProviderUnavailable(
                        "Reranking needs the optional ML extra: pip install -e 'backend[ml]'"
                    ) from exc
                try:
                    self._model = CrossEncoder(
                        self.model_name,
                        device="cpu",
                        max_length=self.max_length,
                        local_files_only=self.offline,
                    )
                except OSError as exc:
                    raise ProviderUnavailable(
                        f"Reranker model '{self.model_name}' is not in the local cache. "
                        "Download it with: python scripts/download_models.py "
                        f"--reranker {self.model_name}"
                    ) from exc
            return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        scores = model.predict([(query, passage) for passage in passages], show_progress_bar=False)
        return [float(value) for value in scores]

"""Embedding providers behind LangChain's ``Embeddings`` interface.

* ``hashing``: a deterministic feature-hashing embedder. It needs no model and
  is used by the default test suite and the offline demo. It captures word
  overlap only; it is **not** a semantic model, and benchmark reports label it
  as a baseline.
* ``sentence-transformers``: local models via ``langchain-huggingface``
  (optional ``[ml]`` extra). Loads from the local Hugging Face cache only unless
  ``GROUNDEDAI_HF_OFFLINE=false``.
* ``ollama``: embeddings served by a local Ollama runtime.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from langchain_core.embeddings import Embeddings

from groundedai.core.config import Settings
from groundedai.core.errors import ProviderUnavailable
from groundedai.core.net import http_client_options
from groundedai.core.text import content_tokens

HASHING_VERSION = 1


class HashingEmbeddings(Embeddings):
    """Signed feature hashing of unigrams and bigrams, L2-normalised.

    Uses BLAKE2b rather than Python's ``hash()``, which is salted per process
    and would make vectors differ between runs.
    """

    def __init__(self, dim: int = 384):
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = content_tokens(text)
        features = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, count in Counter(features).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
            vector[value % self.dim] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return [float(x) for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class EmbeddingProvider:
    """An ``Embeddings`` implementation plus the metadata stored with each document."""

    provider: str
    model: str
    dim: int
    embeddings: Embeddings
    batch_size: int = 32

    @property
    def model_id(self) -> str:
        """Identifier stored on documents; dense search only compares like with like."""
        return f"{self.provider}:{self.model}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors = np.asarray(self.embeddings.embed_documents(batch), dtype=np.float32)
            batches.append(vectors)
        matrix = np.vstack(batches)
        self._check_dim(matrix.shape[1])
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(self.embeddings.embed_query(text), dtype=np.float32)
        self._check_dim(vector.shape[0])
        return vector

    def _check_dim(self, actual: int) -> None:
        if actual != self.dim:
            raise ProviderUnavailable(
                f"Embedding model returned {actual}-dimensional vectors but "
                f"GROUNDEDAI_EMBEDDING_DIM is {self.dim}. Update the setting (and re-run "
                "migrations for PostgreSQL) to match the model."
            )


def configure_hf_offline(offline: bool) -> None:
    """Stop Hugging Face libraries from downloading anything implicitly."""
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    name = settings.embedding_provider
    if name == "hashing":
        return EmbeddingProvider(
            provider="hashing",
            model=f"v{HASHING_VERSION}-{settings.embedding_dim}",
            dim=settings.embedding_dim,
            embeddings=HashingEmbeddings(settings.embedding_dim),
            batch_size=settings.embedding_batch_size,
        )
    if name == "sentence-transformers":
        return _sentence_transformers_provider(settings)
    if name == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return EmbeddingProvider(
            provider="ollama",
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            embeddings=OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
                client_kwargs=http_client_options(
                    settings.ollama_base_url, settings.generation_timeout_s
                ),
            ),
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(f"unknown embedding provider: {name}")


def _sentence_transformers_provider(settings: Settings) -> EmbeddingProvider:
    configure_hf_offline(settings.hf_offline)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ProviderUnavailable(
            "The sentence-transformers provider needs the optional ML extra: "
            "pip install -e 'backend[ml]'"
        ) from exc
    query_kwargs: dict[str, object] = {"normalize_embeddings": True}
    if settings.embedding_query_prefix:
        query_kwargs["prompt"] = settings.embedding_query_prefix
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu", "local_files_only": settings.hf_offline},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.embedding_batch_size,
            },
            query_encode_kwargs=query_kwargs,
        )
    except OSError as exc:
        raise ProviderUnavailable(
            f"Embedding model '{settings.embedding_model}' is not in the local Hugging Face "
            "cache. Download it explicitly with: python scripts/download_models.py "
            f"--embedding {settings.embedding_model}"
        ) from exc
    return EmbeddingProvider(
        provider="sentence-transformers",
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        embeddings=embeddings,
        batch_size=settings.embedding_batch_size,
    )

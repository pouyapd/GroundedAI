import numpy as np
import pytest

from groundedai.core.errors import ProviderUnavailable
from groundedai.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddings,
    build_embedding_provider,
)


def cosine(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_hashing_embeddings_are_deterministic_across_instances() -> None:
    # Regression guard: Python's built-in hash() is salted per process.
    first = HashingEmbeddings(64).embed_query("hotel limit per night")
    second = HashingEmbeddings(64).embed_query("hotel limit per night")
    assert first == second


def test_hashing_embeddings_are_unit_length() -> None:
    vector = HashingEmbeddings(128).embed_query("The DC input voltage range is 150 to 850 V.")
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_word_overlap_increases_similarity() -> None:
    embedder = HashingEmbeddings(384)
    query = embedder.embed_query("hotel limit per night")
    related = embedder.embed_documents(["The hotel limit is 140 euros per night."])[0]
    unrelated = embedder.embed_documents(["Tomatoes grow best in warm greenhouses."])[0]
    assert cosine(query, related) > cosine(query, unrelated)


def test_empty_text_gives_zero_vector() -> None:
    vector = HashingEmbeddings(16).embed_query("the of and")
    assert not any(vector)


def test_dimension_must_be_reasonable() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        HashingEmbeddings(4)


def test_provider_batches_and_checks_dimension() -> None:
    provider = EmbeddingProvider("hashing", "v1-32", 32, HashingEmbeddings(32), batch_size=2)
    matrix = provider.embed_documents([f"text {i}" for i in range(5)])
    assert matrix.shape == (5, 32)
    assert matrix.dtype == np.float32
    assert provider.embed_documents([]).shape == (0, 32)


def test_provider_rejects_wrong_dimension() -> None:
    provider = EmbeddingProvider("hashing", "v1", 64, HashingEmbeddings(32))
    with pytest.raises(ProviderUnavailable, match="dimensional"):
        provider.embed_query("text")


def test_build_hashing_provider_from_settings(settings) -> None:  # type: ignore[no-untyped-def]
    provider = build_embedding_provider(settings)
    assert provider.model_id == "hashing:v1-384"
    assert provider.embed_query("x").shape == (384,)

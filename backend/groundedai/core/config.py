"""Application configuration, loaded from environment variables (prefix ``GROUNDEDAI_``).

All settings have safe local defaults: the application starts with the
deterministic hashing embedder and the extractive demo generator, so it runs
without any model download, API key or external service.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

EmbeddingProviderName = Literal["hashing", "sentence-transformers", "ollama"]
GenerationProviderName = Literal["extractive-demo", "ollama", "huggingface-local"]
RerankerName = Literal["none", "cross-encoder"]
RetrievalMode = Literal["lexical", "dense", "hybrid"]
FusionMethod = Literal["rrf", "weighted"]

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[1]


def default_base_dir() -> Path:
    """Repository root for a source checkout, otherwise the working directory."""
    if (_REPO_ROOT / "backend" / "pyproject.toml").is_file():
        return _REPO_ROOT
    return Path.cwd()


class Environment(enum.StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROUNDEDAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- runtime ---------------------------------------------------------
    env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_json: bool = True
    log_query_text: bool = Field(
        default=False, description="Log question text. Off by default to avoid logging user data."
    )

    # --- storage ---------------------------------------------------------
    data_dir: Path = Field(default_factory=lambda: default_base_dir() / "data")
    database_url: str | None = Field(
        default=None, description="SQLAlchemy URL. Defaults to SQLite in data_dir."
    )
    auto_migrate: bool = True
    frontend_dist: Path | None = None

    # --- embeddings ------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "hashing"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, ge=8, le=4096)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_query_prefix: str = ""
    hf_offline: bool = Field(
        default=True,
        description="Never download Hugging Face models implicitly; use the local cache only.",
    )

    # --- chunking --------------------------------------------------------
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)

    # --- retrieval -------------------------------------------------------
    retrieval_mode: RetrievalMode = "hybrid"
    fusion: FusionMethod = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    hybrid_alpha: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight of the dense signal in fusion."
    )
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=200)
    reranker: RerankerName = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=200)

    # --- generation ------------------------------------------------------
    generation_provider: GenerationProviderName = "extractive-demo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    hf_generation_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    generation_timeout_s: float = Field(default=120.0, gt=0)
    max_new_tokens: int = Field(default=384, ge=16, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_context_chars: int = Field(default=6000, ge=500, le=200_000)
    max_concurrent_generations: int = Field(default=2, ge=1, le=64)
    generation_queue_timeout_s: float = Field(default=5.0, ge=0)

    # --- input limits ----------------------------------------------------
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_pdf_pages: int = Field(default=300, ge=1)
    max_document_chars: int = Field(default=2_000_000, ge=1000)
    max_question_chars: int = Field(default=1000, ge=10, le=10_000)

    # --- security --------------------------------------------------------
    api_key: SecretStr | None = None
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    rate_limit_enabled: bool = True
    rate_limit_query_per_minute: int = Field(default=30, ge=1)
    rate_limit_upload_per_minute: int = Field(default=10, ge=1)
    rate_limit_eval_per_hour: int = Field(default=6, ge=1)
    trust_forwarded_for: bool = Field(
        default=False,
        description="Use X-Forwarded-For for client identity. Enable only behind a trusted proxy.",
    )

    # --- evaluation ------------------------------------------------------
    evaluation_dir: Path = Field(default_factory=lambda: default_base_dir() / "evaluation")
    evaluation_dataset: str = "groundedai-synthetic-v1"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.env is Environment.PRODUCTION:
            key = self.api_key.get_secret_value() if self.api_key else ""
            if len(key) < 24:
                raise ValueError(
                    "GROUNDEDAI_API_KEY (at least 24 characters) is required when "
                    "GROUNDEDAI_ENV=production"
                )
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS origins are not allowed in production")
        return self

    # --- derived values ----------------------------------------------------
    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'groundedai.db').as_posix()}"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def auth_enabled(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())

    @property
    def dataset_dir(self) -> Path:
        return self.evaluation_dir / "datasets" / self.evaluation_dataset

    @property
    def published_reports_dir(self) -> Path:
        return self.evaluation_dir / "reports" / "published"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

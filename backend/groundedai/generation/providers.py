"""Generation providers (LangChain chat models) and their health checks.

There is no silent fallback: if the configured provider cannot be used, the
caller gets :class:`~groundedai.core.errors.ProviderUnavailable` with setup
guidance, never an answer from a different model.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from groundedai.core.config import Settings
from groundedai.core.errors import GroundedAIError, ProviderTimeout, ProviderUnavailable
from groundedai.core.net import http_client_options
from groundedai.generation.extractive import EXTRACTIVE_MODEL_NAME, ExtractiveDemoChatModel
from groundedai.retrieval.embeddings import configure_hf_offline

OLLAMA_NUM_CTX = 4096


@dataclass
class GenerationProvider:
    name: str
    model: str
    is_llm: bool
    chat_model: BaseChatModel


@dataclass
class ProviderStatus:
    name: str
    model: str
    is_llm: bool
    available: bool
    detail: str
    checked_at: float


def build_generation_provider(settings: Settings) -> GenerationProvider:
    name = settings.generation_provider
    if name == "extractive-demo":
        return GenerationProvider(
            name=name,
            model=EXTRACTIVE_MODEL_NAME,
            is_llm=False,
            chat_model=ExtractiveDemoChatModel(),
        )
    if name == "ollama":
        from langchain_ollama import ChatOllama

        chat = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.max_new_tokens,
            num_ctx=OLLAMA_NUM_CTX,
            seed=0,
            client_kwargs=http_client_options(
                settings.ollama_base_url, settings.generation_timeout_s
            ),
        )
        return GenerationProvider(
            name=name, model=settings.ollama_model, is_llm=True, chat_model=chat
        )
    if name == "huggingface-local":
        return _huggingface_local(settings)
    raise ValueError(f"unknown generation provider: {name}")


def _huggingface_local(settings: Settings) -> GenerationProvider:
    configure_hf_offline(settings.hf_offline)
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    except ImportError as exc:
        raise ProviderUnavailable(
            "The huggingface-local provider needs the optional ML extra: "
            "pip install -e 'backend[ml]'"
        ) from exc
    try:
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=settings.hf_generation_model,
            task="text-generation",
            device=-1,
            model_kwargs={"local_files_only": settings.hf_offline},
            pipeline_kwargs={
                "max_new_tokens": settings.max_new_tokens,
                "do_sample": False,
                "return_full_text": False,
            },
        )
        chat = ChatHuggingFace(llm=pipeline, model_id=settings.hf_generation_model)
    except OSError as exc:
        raise ProviderUnavailable(
            f"Model '{settings.hf_generation_model}' is not in the local Hugging Face cache. "
            "Download it explicitly with: python scripts/download_models.py "
            f"--generation {settings.hf_generation_model}"
        ) from exc
    return GenerationProvider(
        name="huggingface-local",
        model=settings.hf_generation_model,
        is_llm=True,
        chat_model=chat,
    )


def translate_provider_error(exc: BaseException, settings: Settings) -> GroundedAIError | None:
    """Map transport/runtime errors from model clients to API errors."""
    if isinstance(exc, GroundedAIError):
        return exc
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return ProviderTimeout(
            f"The model did not answer within {settings.generation_timeout_s:.0f} s."
        )
    if isinstance(exc, httpx.ConnectError | ConnectionError):
        return ProviderUnavailable(
            f"Cannot reach Ollama at {settings.ollama_base_url}. Start it with 'ollama serve' "
            f"and pull the model with 'ollama pull {settings.ollama_model}'."
        )
    status_code = getattr(exc, "status_code", None)
    if type(exc).__name__ == "ResponseError" and status_code in (502, 503, 504):
        return ProviderUnavailable(
            f"Ollama at {settings.ollama_base_url} answered HTTP {status_code}. If you use an "
            "HTTP proxy, make sure requests to the Ollama host bypass it (NO_PROXY)."
        )
    if type(exc).__name__ == "ResponseError" and status_code == 404:
        return ProviderUnavailable(
            f"Ollama does not have the model '{settings.ollama_model}'. "
            f"Pull it with 'ollama pull {settings.ollama_model}'."
        )
    return None


def check_generation_provider(settings: Settings, timeout_s: float = 2.0) -> ProviderStatus:
    """Cheap availability check that never loads or downloads a model."""
    name = settings.generation_provider
    now = time.time()
    if name == "extractive-demo":
        return ProviderStatus(
            name,
            EXTRACTIVE_MODEL_NAME,
            False,
            True,
            "Deterministic extractive demo (not an LLM).",
            now,
        )
    if name == "ollama":
        available, detail = _check_ollama(settings, timeout_s)
        return ProviderStatus(name, settings.ollama_model, True, available, detail, now)
    available, detail = _check_hf_cache(settings.hf_generation_model)
    return ProviderStatus(name, settings.hf_generation_model, True, available, detail, now)


def _check_ollama(settings: Settings, timeout_s: float) -> tuple[bool, str]:
    try:
        options = http_client_options(settings.ollama_base_url, timeout_s)
        with httpx.Client(**options) as client:  # type: ignore[arg-type]
            response = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except (httpx.HTTPError, ValueError):
        return False, (
            f"Ollama is not reachable at {settings.ollama_base_url}. Install Ollama, run "
            f"'ollama serve', then 'ollama pull {settings.ollama_model}'."
        )
    names = {str(item.get("name", "")) for item in payload.get("models", [])}
    wanted = settings.ollama_model
    if wanted in names or f"{wanted}:latest" in names:
        return True, "Ollama is running and the model is available."
    return False, f"Ollama is running but '{wanted}' is not pulled. Run 'ollama pull {wanted}'."


def _check_hf_cache(model_id: str) -> tuple[bool, str]:
    if importlib.util.find_spec("transformers") is None:
        return False, "Install the optional ML extra: pip install -e 'backend[ml]'."
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_id, "config.json")
    if isinstance(cached, str):
        return True, "Model found in the local Hugging Face cache."
    return False, (
        f"'{model_id}' is not in the local Hugging Face cache. Download it with "
        f"'python scripts/download_models.py --generation {model_id}'."
    )

"""Grounded question answering and retrieval inspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from groundedai.api.deps import ServicesDep, rate_limit, require_api_key
from groundedai.schemas.query import AnswerResponse, QueryRequest, RetrieveRequest, RetrieveResponse

router = APIRouter(tags=["question answering"], dependencies=[Depends(require_api_key)])

_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"description": "Missing or invalid API key"},
    404: {"description": "Unknown document id in the filter"},
    422: {"description": "Invalid request"},
    429: {"description": "Rate limit exceeded"},
    502: {"description": "The model provider returned an error"},
    503: {"description": "Model provider unavailable or server busy"},
    504: {"description": "The model provider timed out"},
}


@router.post(
    "/query",
    response_model=AnswerResponse,
    summary="Answer a question from the indexed documents, with validated citations",
    responses=_ERRORS,
    dependencies=[Depends(rate_limit("query"))],
)
def query(body: QueryRequest, services: ServicesDep) -> AnswerResponse:
    return services.qa.answer(body)


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Run retrieval only and return ranked chunks with score diagnostics",
    responses={k: v for k, v in _ERRORS.items() if k in (401, 404, 422, 429, 503)},
    dependencies=[Depends(rate_limit("query"))],
)
def retrieve(body: RetrieveRequest, services: ServicesDep) -> RetrieveResponse:
    return services.qa.retrieve(body)

"""LangChain ``BaseRetriever`` adapter for the retrieval engine.

Lets GroundedAI's hybrid retrieval be composed into any LangChain (LCEL)
pipeline. Component scores travel in ``Document.metadata``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sqlalchemy.orm import Session, sessionmaker

from groundedai.persistence.db import session_scope
from groundedai.retrieval.engine import RetrievalEngine
from groundedai.retrieval.types import RetrievalRequest, RetrievedChunk


def chunk_to_document(chunk: RetrievedChunk) -> Document:
    return Document(
        page_content=chunk.text,
        id=chunk.chunk_id,
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "rank": chunk.rank,
            "scores": {
                "lexical": chunk.scores.lexical,
                "lexical_rank": chunk.scores.lexical_rank,
                "dense": chunk.scores.dense,
                "dense_rank": chunk.scores.dense_rank,
                "fused": chunk.scores.fused,
                "rerank": chunk.scores.rerank,
            },
        },
    )


class GroundedRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: RetrievalEngine
    session_factory: sessionmaker[Session]
    request_template: RetrievalRequest = RetrievalRequest(query="")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        request = replace(self.request_template, query=query)
        with session_scope(self.session_factory) as session:
            result = self.engine.retrieve(session, request)
        return [chunk_to_document(chunk) for chunk in result.chunks]

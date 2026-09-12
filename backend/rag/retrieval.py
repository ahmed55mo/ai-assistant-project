from __future__ import annotations

from typing import Mapping

from backend.rag.errors import (
    EmbeddingError,
    InvalidRetrievalRequest,
    RetrievalError,
    VectorStoreError,
)
from backend.rag.interfaces import (
    EmbeddingService,
    RetrievalRequest,
    RetrievalResult,
    VectorStore,
)


class RetrievalService:
    """Embed and retrieve document chunks within an explicit ownership scope."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        self._validate_request(request)
        try:
            query_embedding = self.embedding_service.embed_text(request.query)
        except EmbeddingError as exc:
            raise RetrievalError("The retrieval query could not be embedded.") from exc
        except Exception as exc:
            raise RetrievalError("The retrieval query could not be embedded.") from exc

        filters = {
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
        }
        if request.document_id is not None:
            filters["document_id"] = request.document_id
        try:
            results = self.vector_store.search(
                query_embedding,
                top_k=request.top_k,
                filters=filters,
            )
        except (VectorStoreError, InvalidRetrievalRequest):
            raise
        except Exception as exc:
            raise RetrievalError("The document search could not be completed.") from exc

        if request.score_threshold is None:
            return results
        return [
            result
            for result in results
            if result.similarity_score >= request.score_threshold
        ]

    @staticmethod
    def _validate_request(request: RetrievalRequest) -> None:
        if not isinstance(request.query, str) or not request.query.strip():
            raise InvalidRetrievalRequest("query must not be empty.")
        for name in ("user_id", "conversation_id"):
            value = getattr(request, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidRetrievalRequest(f"{name} must not be empty.")
        if request.document_id is not None and (
            not isinstance(request.document_id, str) or not request.document_id.strip()
        ):
            raise InvalidRetrievalRequest("document_id must not be empty.")
        if isinstance(request.top_k, bool) or not isinstance(request.top_k, int):
            raise InvalidRetrievalRequest("top_k must be a positive integer.")
        if request.top_k <= 0:
            raise InvalidRetrievalRequest("top_k must be a positive integer.")
        if request.score_threshold is not None:
            if isinstance(request.score_threshold, bool) or not isinstance(
                request.score_threshold, (int, float)
            ):
                raise InvalidRetrievalRequest(
                    "score_threshold must be between 0 and 1."
                )
            if not 0 <= request.score_threshold <= 1:
                raise InvalidRetrievalRequest(
                    "score_threshold must be between 0 and 1."
                )

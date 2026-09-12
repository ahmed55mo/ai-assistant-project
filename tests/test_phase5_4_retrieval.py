from __future__ import annotations

from unittest.mock import Mock

import pytest

from backend.documents.chunking import DocumentChunk
from backend.rag.embeddings import LocalEmbeddingService
from backend.rag.errors import (
    EmbeddingError,
    InvalidRetrievalRequest,
    RetrievalError,
    VectorStoreError,
)
from backend.rag.interfaces import RetrievalRequest
from backend.rag.retrieval import RetrievalService
from backend.rag.vector_store import FaissVectorStore


def _chunk(
    chunk_id: str,
    document_id: str,
    user_id: str,
    conversation_id: str,
    text: str,
    index: int = 0,
    page_number: int | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        user_id=user_id,
        conversation_id=conversation_id,
        chunk_index=index,
        text=text,
        page_number=page_number,
        filename=f"{document_id}.pdf",
    )


def _service() -> tuple[RetrievalService, LocalEmbeddingService]:
    embedding = LocalEmbeddingService(dimension=32)
    store = FaissVectorStore(embedding.dimension)
    chunks = [
        _chunk(
            "a-1", "doc-a", "user-a", "conversation-a",
            "annual leave policy and holiday allowance", 0, 3
        ),
        _chunk(
            "a-2", "doc-a", "user-a", "conversation-a",
            "calendar event scheduling policy", 1, 4
        ),
        _chunk(
            "b-1", "doc-b", "user-b", "conversation-b",
            "annual leave policy for another user", 0, 1
        ),
        _chunk(
            "a-other", "doc-other", "user-a", "conversation-b",
            "annual leave policy in another conversation", 0, 2
        ),
    ]
    store.add(chunks, embedding.embed_texts([chunk.text for chunk in chunks]))
    return RetrievalService(embedding, store), embedding


def test_valid_request_returns_scoped_relevant_result_and_metadata() -> None:
    service, _ = _service()

    results = service.retrieve(
        RetrievalRequest(
            query="annual leave policy and holiday allowance",
            user_id="user-a",
            conversation_id="conversation-a",
        )
    )

    assert results
    result = results[0]
    assert result.chunk_id == "a-1"
    assert result.document_id == "doc-a"
    assert result.user_id == "user-a"
    assert result.conversation_id == "conversation-a"
    assert result.chunk_index == 0
    assert result.filename == "doc-a.pdf"
    assert result.page_number == 3
    assert result.similarity_score == result.score


@pytest.mark.parametrize(
    "retrieval_request",
    [
        RetrievalRequest("", "user", "conversation"),
        RetrievalRequest("   ", "user", "conversation"),
        RetrievalRequest("query", "", "conversation"),
        RetrievalRequest("query", "   ", "conversation"),
        RetrievalRequest("query", "user", "conversation", top_k=0),
        RetrievalRequest("query", "user", "conversation", top_k=-1),
        RetrievalRequest("query", "user", "conversation", score_threshold=-0.1),
        RetrievalRequest("query", "user", "conversation", score_threshold=1.1),
    ],
)
def test_invalid_requests_raise_controlled_error(
    retrieval_request: RetrievalRequest,
) -> None:
    service, _ = _service()

    with pytest.raises(InvalidRetrievalRequest):
        service.retrieve(retrieval_request)


def test_top_k_and_deterministic_order_are_preserved() -> None:
    service, _ = _service()

    first = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-a",
            top_k=2,
        )
    )
    second = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-a",
            top_k=2,
        )
    )

    assert len(first) == 2
    assert [result.chunk_id for result in first] == [
        result.chunk_id for result in second
    ]
    assert all(first[index].similarity_score >= first[index + 1].similarity_score for index in range(len(first) - 1))


def test_document_filter_restricts_results() -> None:
    service, _ = _service()

    results = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-a",
            document_id="doc-a",
            top_k=10,
        )
    )

    assert results
    assert all(result.document_id == "doc-a" for result in results)


def test_user_and_conversation_isolation_are_always_explicit() -> None:
    service, _ = _service()

    user_results = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-a",
            top_k=10,
        )
    )
    conversation_results = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-b",
            top_k=10,
        )
    )

    assert user_results
    assert all(
        result.user_id == "user-a"
        and result.conversation_id == "conversation-a"
        for result in user_results
    )
    assert conversation_results
    assert all(
        result.user_id == "user-a"
        and result.conversation_id == "conversation-b"
        for result in conversation_results
    )


def test_same_document_id_under_wrong_scope_does_not_leak() -> None:
    service, _ = _service()

    results = service.retrieve(
        RetrievalRequest(
            query="annual leave",
            user_id="user-b",
            conversation_id="conversation-b",
            document_id="doc-a",
        )
    )

    assert results == []


def test_threshold_filters_results_and_can_return_empty() -> None:
    service, _ = _service()

    unfiltered = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-a",
            top_k=10,
        )
    )
    filtered = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-a",
            top_k=10,
            score_threshold=unfiltered[0].similarity_score,
        )
    )
    empty = service.retrieve(
        RetrievalRequest(
            query="annual leave policy",
            user_id="user-a",
            conversation_id="conversation-a",
            score_threshold=1.0,
        )
    )

    assert filtered
    assert all(result.similarity_score >= filtered[0].similarity_score for result in filtered)
    assert empty == []


def test_empty_store_and_no_match_are_normal_empty_results() -> None:
    embedding = LocalEmbeddingService(dimension=16)
    service = RetrievalService(embedding, FaissVectorStore(embedding.dimension))

    assert service.retrieve(
        RetrievalRequest("query", "user", "conversation")
    ) == []

    populated_service, _ = _service()
    assert populated_service.retrieve(
        RetrievalRequest("query", "missing", "conversation")
    ) == []


def test_query_is_embedded_with_existing_embedding_service() -> None:
    embedding = Mock()
    embedding.embed_text.return_value = [1.0, 0.0]
    store = Mock()
    store.search.return_value = []
    service = RetrievalService(embedding, store)
    request = RetrievalRequest("query", "user", "conversation")

    assert service.retrieve(request) == []
    embedding.embed_text.assert_called_once_with("query")
    store.search.assert_called_once_with(
        [1.0, 0.0],
        top_k=5,
        filters={"user_id": "user", "conversation_id": "conversation"},
    )


def test_embedding_failure_is_controlled() -> None:
    embedding = Mock()
    embedding.embed_text.side_effect = EmbeddingError("provider failure")
    service = RetrievalService(embedding, Mock())

    with pytest.raises(RetrievalError, match="could not be embedded"):
        service.retrieve(RetrievalRequest("query", "user", "conversation"))


def test_vector_store_failure_is_controlled() -> None:
    embedding = LocalEmbeddingService(dimension=8)
    store = Mock()
    store.search.side_effect = VectorStoreError("store failure")
    service = RetrievalService(embedding, store)

    with pytest.raises(VectorStoreError):
        service.retrieve(RetrievalRequest("query", "user", "conversation"))

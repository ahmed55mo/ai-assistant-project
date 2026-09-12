from __future__ import annotations

from unittest.mock import Mock

import pytest

from backend.documents.chunking import DocumentChunk
from backend.documents.errors import InvalidDocumentError
from backend.documents.ingestion import DocumentIngestionService
from backend.documents.models import DocumentStatus
from backend.rag.embeddings import (
    EmbeddingConfig,
    LocalEmbeddingService,
    build_embedding_service,
)
from backend.rag.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    InvalidVectorStoreRequest,
    VectorStoreError,
)
from backend.rag.vector_store import FaissVectorStore


def _chunk(
    chunk_id: str,
    document_id: str,
    user_id: str,
    conversation_id: str,
    text: str,
    index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        user_id=user_id,
        conversation_id=conversation_id,
        chunk_index=index,
        text=text,
        page_number=None,
        filename=f"{document_id}.txt",
    )


def test_local_embeddings_are_deterministic_and_have_configured_dimension() -> None:
    service = LocalEmbeddingService(dimension=16)
    first = service.embed_text("annual leave policy")
    second = service.embed_text("annual leave policy")

    assert first == second
    assert len(first) == 16
    assert first != service.embed_text("calendar policy")


def test_embedding_batch_and_configuration() -> None:
    service = LocalEmbeddingService(dimension=8)
    batch = service.embed_texts(["one", "two"])

    assert len(batch) == 2
    assert all(len(vector) == 8 for vector in batch)
    assert build_embedding_service(EmbeddingConfig(provider="local", dimension=8)).dimension == 8


def test_embedding_rejects_empty_text() -> None:
    with pytest.raises(EmbeddingError):
        LocalEmbeddingService().embed_text(" \n")


def test_faiss_store_adds_searches_and_preserves_metadata() -> None:
    embedding = LocalEmbeddingService(dimension=16)
    store = FaissVectorStore(embedding.dimension)
    chunks = [
        _chunk("one", "doc-a", "user-a", "conversation-a", "annual leave policy"),
        _chunk("two", "doc-b", "user-b", "conversation-b", "telegram message"),
    ]
    store.add(chunks, embedding.embed_texts([chunk.text for chunk in chunks]))

    results = store.search(
        embedding.embed_text("annual leave"),
        top_k=1,
        filters={"user_id": "user-a", "conversation_id": "conversation-a"},
    )

    assert len(results) == 1
    assert results[0].chunk == chunks[0]
    assert results[0].score is not None


def test_faiss_store_uses_cosine_similarity_and_deterministic_order() -> None:
    store = FaissVectorStore(2)
    chunks = [
        _chunk("first", "doc", "user", "conversation", "first"),
        _chunk("second", "doc", "user", "conversation", "second", index=1),
    ]
    store.add(chunks, [[1.0, 0.0], [1.0, 0.0]])

    results = store.search([1.0, 0.0], top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["first", "second"]
    assert [result.score for result in results] == [1.0, 1.0]


def test_faiss_store_empty_top_k_and_dimension_validation() -> None:
    store = FaissVectorStore(3)
    assert store.search([1.0, 0.0, 0.0]) == []
    with pytest.raises(InvalidVectorStoreRequest):
        store.search([1.0, 0.0, 0.0], top_k=0)
    with pytest.raises(EmbeddingDimensionError):
        store.add([_chunk("one", "doc", "user", "conversation", "text")], [[1.0, 0.0]])
    with pytest.raises(EmbeddingDimensionError):
        store.search([1.0, 0.0])


def test_faiss_store_explicit_isolation_filters() -> None:
    embedding = LocalEmbeddingService(dimension=16)
    store = FaissVectorStore(embedding.dimension)
    chunks = [
        _chunk("a", "doc-a", "user-a", "conversation-a", "shared policy"),
        _chunk("b", "doc-b", "user-b", "conversation-b", "shared policy", index=1),
    ]
    store.add(chunks, embedding.embed_texts([chunk.text for chunk in chunks]))

    assert [item.chunk.chunk_id for item in store.search(
        embedding.embed_text("shared policy"), filters={"user_id": "user-a"}
    )] == ["a"]
    assert [item.chunk.chunk_id for item in store.search(
        embedding.embed_text("shared policy"), filters={"conversation_id": "conversation-b"}
    )] == ["b"]
    assert [item.chunk.chunk_id for item in store.search(
        embedding.embed_text("shared policy"), filters={"document_id": "doc-a"}
    )] == ["a"]
    with pytest.raises(InvalidVectorStoreRequest):
        store.search(embedding.embed_text("shared policy"), filters={"tenant_id": "x"})


def test_faiss_store_delete_document_and_clear() -> None:
    store = FaissVectorStore(2)
    chunks = [
        _chunk("a", "doc-a", "user", "conversation", "a"),
        _chunk("b", "doc-b", "user", "conversation", "b", index=1),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    store.delete_document("doc-a")
    assert [item.chunk.chunk_id for item in store.search([0.0, 1.0])] == ["b"]
    store.clear()
    assert store.search([0.0, 1.0]) == []


def test_ingestion_embeds_chunks_and_reingestion_replaces_vectors() -> None:
    service = DocumentIngestionService()
    first = service.ingest(
        user_id="user",
        conversation_id="conversation",
        filename="policy.txt",
        content_type="text/plain",
        content=b"annual leave policy",
        document_id="doc-fixed",
    )
    service.ingest(
        user_id="user",
        conversation_id="conversation",
        filename="policy.txt",
        content_type="text/plain",
        content=b"updated annual leave policy",
        document_id="doc-fixed",
    )

    results = service.vector_store.search(
        service.embedding_service.embed_text("updated annual leave policy"),
        filters={"document_id": first.document_id},
        top_k=10,
    )
    assert first.status is DocumentStatus.READY
    assert len(results) == len(service.get_chunks(first.document_id))
    assert len(results) == 1


def test_embedding_failure_marks_document_failed() -> None:
    embedding = Mock()
    embedding.dimension = 8
    embedding.embed_texts.side_effect = EmbeddingError("embedding failed")
    service = DocumentIngestionService(embedding_service=embedding)

    with pytest.raises(EmbeddingError):
        service.ingest(
            user_id="user",
            conversation_id="conversation",
            filename="policy.txt",
            content_type="text/plain",
            content=b"policy",
        )

    failed = next(iter(service._metadata.values()))
    assert failed.status is DocumentStatus.FAILED
    assert failed.error == "embedding failed"


def test_vector_store_failure_marks_document_failed() -> None:
    vector_store = Mock()
    vector_store.delete_document.return_value = None
    vector_store.add.side_effect = VectorStoreError("vector store failed")
    vector_store.dimension = 8
    embedding = LocalEmbeddingService(dimension=8)
    service = DocumentIngestionService(
        embedding_service=embedding,
        vector_store=vector_store,
    )

    with pytest.raises(VectorStoreError):
        service.ingest(
            user_id="user",
            conversation_id="conversation",
            filename="policy.txt",
            content_type="text/plain",
            content=b"policy",
        )

    failed = next(iter(service._metadata.values()))
    assert failed.status is DocumentStatus.FAILED

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from backend import main
from backend.documents.ingestion import DocumentIngestionService
from backend.rag.interfaces import RetrievalRequest
from backend.rag.retrieval import RetrievalService
from backend.services.memory_service import ConversationMemory


def _setup(monkeypatch):
    memory = ConversationMemory()
    conversation_id = memory.create_conversation()
    ingestion = DocumentIngestionService()
    monkeypatch.setattr(main, "conversation_memory", memory)
    monkeypatch.setattr(main, "document_ingestion", ingestion)
    return memory, ingestion, conversation_id


def _ingest(
    ingestion: DocumentIngestionService,
    conversation_id: str,
    *,
    user_id: str = "user-1",
    filename: str = "policy.txt",
    content: bytes = b"Annual leave is twenty days.",
    document_id: str | None = None,
):
    return ingestion.ingest(
        user_id=user_id,
        conversation_id=conversation_id,
        filename=filename,
        content_type="text/plain",
        content=content,
        document_id=document_id,
    )


def test_list_documents_returns_scoped_frontend_metadata(monkeypatch) -> None:
    _, ingestion, conversation_id = _setup(monkeypatch)
    metadata = _ingest(ingestion, conversation_id)

    response = TestClient(main.app).get(
        "/api/documents",
        params={"conversation_id": conversation_id, "user_id": "user-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {
                "document_id": metadata.document_id,
                "filename": "policy.txt",
                "content_type": "text/plain",
                "status": "ready",
                "conversation_id": conversation_id,
                "user_id": "user-1",
                "chunk_count": 1,
            }
        ]
    }


def test_list_documents_returns_empty_for_other_scope(monkeypatch) -> None:
    _, ingestion, conversation_id = _setup(monkeypatch)
    _ingest(ingestion, conversation_id, user_id="user-1")

    response = TestClient(main.app).get(
        "/api/documents",
        params={"conversation_id": conversation_id, "user_id": "user-2"},
    )

    assert response.status_code == 200
    assert response.json() == {"documents": []}


def test_get_document_metadata_and_not_found_cases(monkeypatch) -> None:
    memory, ingestion, conversation_id = _setup(monkeypatch)
    metadata = _ingest(ingestion, conversation_id)
    other_conversation = memory.create_conversation()
    client = TestClient(main.app)

    found = client.get(
        f"/api/documents/{metadata.document_id}",
        params={"conversation_id": conversation_id, "user_id": "user-1"},
    )
    wrong_conversation = client.get(
        f"/api/documents/{metadata.document_id}",
        params={"conversation_id": other_conversation, "user_id": "user-1"},
    )
    wrong_user = client.get(
        f"/api/documents/{metadata.document_id}",
        params={"conversation_id": conversation_id, "user_id": "user-2"},
    )
    unknown = client.get(
        "/api/documents/missing",
        params={"conversation_id": conversation_id, "user_id": "user-1"},
    )

    assert found.status_code == 200
    assert found.json()["document_id"] == metadata.document_id
    assert found.json()["chunk_count"] == 1
    assert wrong_conversation.status_code == 404
    assert wrong_user.status_code == 404
    assert unknown.status_code == 404


def test_delete_document_removes_metadata_vectors_and_only_target(monkeypatch) -> None:
    _, ingestion, conversation_id = _setup(monkeypatch)
    first = _ingest(ingestion, conversation_id, filename="first.txt")
    second = _ingest(
        ingestion,
        conversation_id,
        filename="second.txt",
        content=b"Calendar scheduling details.",
    )
    retrieval = RetrievalService(ingestion.embedding_service, ingestion.vector_store)
    client = TestClient(main.app)

    response = client.delete(
        f"/api/documents/{first.document_id}",
        params={"conversation_id": conversation_id, "user_id": "user-1"},
    )

    assert response.status_code == 204
    assert client.get(
        "/api/documents",
        params={"conversation_id": conversation_id, "user_id": "user-1"},
    ).json()["documents"] == [
        {
            "document_id": second.document_id,
            "filename": "second.txt",
            "content_type": "text/plain",
            "status": "ready",
            "conversation_id": conversation_id,
            "user_id": "user-1",
            "chunk_count": 1,
        }
    ]
    with pytest.raises(KeyError):
        ingestion.get_content(first.document_id)
    with pytest.raises(KeyError):
        ingestion.get_chunks(first.document_id)
    assert retrieval.retrieve(
        RetrievalRequest(
            query="Annual leave",
            user_id="user-1",
            conversation_id=conversation_id,
            document_id=first.document_id,
        )
    ) == []
    assert retrieval.retrieve(
        RetrievalRequest(
            query="Calendar scheduling",
            user_id="user-1",
            conversation_id=conversation_id,
            document_id=second.document_id,
        )
    )


def test_delete_wrong_scope_unknown_and_repeat_are_safe(monkeypatch) -> None:
    memory, ingestion, conversation_id = _setup(monkeypatch)
    metadata = _ingest(ingestion, conversation_id)
    other_conversation = memory.create_conversation()
    client = TestClient(main.app)

    wrong_scope = client.delete(
        f"/api/documents/{metadata.document_id}",
        params={"conversation_id": other_conversation, "user_id": "user-1"},
    )
    deleted = client.delete(
        f"/api/documents/{metadata.document_id}",
        params={"conversation_id": conversation_id, "user_id": "user-1"},
    )
    repeated = client.delete(
        f"/api/documents/{metadata.document_id}",
        params={"conversation_id": conversation_id, "user_id": "user-1"},
    )

    assert wrong_scope.status_code == 404
    assert deleted.status_code == 204
    assert repeated.status_code == 404


def test_document_management_requires_conversation_id(monkeypatch) -> None:
    _setup(monkeypatch)

    response = TestClient(main.app).get("/api/documents")

    assert response.status_code == 422

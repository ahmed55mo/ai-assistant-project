from __future__ import annotations

import io

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend import main
from backend.documents.errors import (
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentTypeError,
)
from backend.documents.ingestion import DocumentIngestionService
from backend.documents.models import DocumentStatus


def _pdf() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(output)
    return output.getvalue()


def _docx() -> bytes:
    output = io.BytesIO()
    Document().save(output)
    return output.getvalue()


@pytest.mark.parametrize(
    ("filename", "content_type", "content_factory"),
    [
        ("policy.pdf", "application/pdf", _pdf),
        ("notes.txt", "text/plain", lambda: b"Annual leave is twenty days.\n"),
        (
            "handbook.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _docx,
        ),
    ],
)
def test_supported_documents_become_ready(
    filename: str, content_type: str, content_factory
) -> None:
    service = DocumentIngestionService()
    content = content_factory()

    metadata = service.ingest(
        user_id="user-1",
        conversation_id="conversation-1",
        filename=filename,
        content_type=content_type,
        content=content,
    )

    assert metadata.status is DocumentStatus.READY
    assert metadata.filename == filename
    assert metadata.size_bytes == len(content)
    assert service.get_content(metadata.document_id) == content


def test_unsupported_type_is_failed_and_not_stored_as_ready() -> None:
    service = DocumentIngestionService()

    with pytest.raises(UnsupportedDocumentTypeError):
        service.ingest(
            user_id="user-1",
            conversation_id="conversation-1",
            filename="notes.csv",
            content_type="text/csv",
            content=b"a,b",
        )


def test_oversized_document_is_failed() -> None:
    service = DocumentIngestionService(max_bytes=4)

    with pytest.raises(DocumentTooLargeError):
        service.ingest(
            user_id="user-1",
            conversation_id="conversation-1",
            filename="notes.txt",
            content_type="text/plain",
            content=b"12345",
        )


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("broken.pdf", "application/pdf", b"not a pdf"),
        ("broken.txt", "text/plain", b"\xff\xfe"),
        ("broken.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"not a zip"),
        ("wrong.txt", "application/pdf", _pdf()),
    ],
)
def test_invalid_documents_are_rejected(
    filename: str, content_type: str, content: bytes
) -> None:
    with pytest.raises(InvalidDocumentError):
        DocumentIngestionService().ingest(
            user_id="user-1",
            conversation_id="conversation-1",
            filename=filename,
            content_type=content_type,
            content=content,
        )


def test_metadata_contains_isolation_identifiers() -> None:
    metadata = DocumentIngestionService().ingest(
        user_id="user-7",
        conversation_id="conversation-9",
        filename="policy.txt",
        content_type="text/plain",
        content=b"policy",
    )

    assert metadata.document_id
    assert metadata.user_id == "user-7"
    assert metadata.conversation_id == "conversation-9"
    assert metadata.status is DocumentStatus.READY


def test_lifecycle_exposes_processing_and_failed_states() -> None:
    service = DocumentIngestionService()
    original_validate = __import__(
        "backend.documents.ingestion", fromlist=["_validate_document"]
    )._validate_document
    captured: list[DocumentStatus] = []

    def fail_after_processing(*args: object) -> None:
        captured.append(DocumentStatus.PROCESSING)
        raise InvalidDocumentError("invalid")

    module = __import__("backend.documents.ingestion", fromlist=["_validate_document"])
    module._validate_document = fail_after_processing
    try:
        with pytest.raises(InvalidDocumentError):
            service.ingest(
                user_id="user-1",
                conversation_id="conversation-1",
                filename="bad.txt",
                content_type="text/plain",
                content=b"bad",
            )
    finally:
        module._validate_document = original_validate

    assert captured == [DocumentStatus.PROCESSING]
    failed = next(iter(service._metadata.values()))
    assert failed.status is DocumentStatus.FAILED


def test_upload_api_returns_structured_ready_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = main.ConversationMemory()
    conversation_id = memory.create_conversation()
    monkeypatch.setattr(main, "conversation_memory", memory)
    monkeypatch.setattr(main, "document_ingestion", DocumentIngestionService())

    response = TestClient(main.app).post(
        "/api/documents/upload",
        data={"conversation_id": conversation_id, "user_id": "user-1"},
        files={"file": ("notes.txt", b"document text", "text/plain")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "notes.txt"
    assert payload["content_type"] == "text/plain"
    assert payload["status"] == "ready"
    assert payload["conversation_id"] == conversation_id
    assert payload["document_id"]


def test_upload_api_rejects_unknown_conversation() -> None:
    response = TestClient(main.app).post(
        "/api/documents/upload",
        data={"conversation_id": "missing"},
        files={"file": ("notes.txt", b"text", "text/plain")},
    )

    assert response.status_code == 404

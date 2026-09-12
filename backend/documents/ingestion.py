from __future__ import annotations

import io
import zipfile
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO
from uuid import uuid4

from backend.documents.chunking import DocumentChunk, TextChunkingService
from backend.documents.errors import (
    DocumentNotFoundError,
    DocumentStorageError,
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentTypeError,
)
from backend.documents.models import (
    DocumentMetadata,
    DocumentStatus,
    SupportedDocumentType,
)
from backend.documents.extraction import DocumentTextExtractor, normalize_segments
from backend.documents.repository import DocumentRepository
from backend.rag.embeddings import build_embedding_service
from backend.rag.errors import RAGFoundationError


DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
_CONTENT_TYPES = {item.value for item in SupportedDocumentType}


class DocumentIngestionService:
    """Validate and retain document metadata/content in process-local memory."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        extractor: DocumentTextExtractor | None = None,
        chunker: TextChunkingService | None = None,
        repository: DocumentRepository | None = None,
        embedding_service: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive.")
        self.max_bytes = max_bytes
        self.extractor = extractor or DocumentTextExtractor()
        self.chunker = chunker or TextChunkingService()
        self.repository = repository or DocumentRepository()
        self.embedding_service = embedding_service or build_embedding_service()
        if vector_store is None:
            from backend.rag.vector_store import FaissVectorStore

            vector_store = FaissVectorStore(self.embedding_service.dimension)
        self.vector_store = vector_store
        # Compatibility views for the Phase 5.1 in-memory service surface.
        self._metadata = self.repository._metadata
        self._content = self.repository._content
        self._lock = Lock()

    def ingest(
        self,
        *,
        user_id: str,
        conversation_id: str,
        filename: str,
        content_type: str | None,
        content: bytes | BinaryIO,
        document_id: str | None = None,
    ) -> DocumentMetadata:
        document_id = document_id or str(uuid4())
        safe_filename = Path(filename or "").name.strip()
        initial_size = len(content) if isinstance(content, bytes) else 0
        metadata = DocumentMetadata.create(
            document_id=document_id,
            user_id=user_id.strip(),
            conversation_id=conversation_id.strip(),
            filename=safe_filename,
            content_type=(content_type or "").split(";", 1)[0].strip().lower(),
            size_bytes=initial_size,
            status=DocumentStatus.PROCESSING,
        )
        with self._lock:
            self._metadata[document_id] = metadata

        try:
            data = content if isinstance(content, bytes) else _read_limited(content, self.max_bytes)
            if len(data) > self.max_bytes:
                raise DocumentTooLargeError(
                    f"Document exceeds the {self.max_bytes} byte limit."
                )
            _validate_document(safe_filename, metadata.content_type, data)
            segments = normalize_segments(
                self.extractor.extract(data, metadata.content_type)
            )
            chunks = self.chunker.chunk(metadata, segments)
            self.vector_store.delete_document(document_id)
            embeddings = self.embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            self.vector_store.add(chunks, embeddings)
            ready = metadata.model_copy(
                update={"size_bytes": len(data), "status": DocumentStatus.READY}
            )
            self.repository.save(ready, data, chunks)
            return ready
        except (
            DocumentTooLargeError,
            UnsupportedDocumentTypeError,
            InvalidDocumentError,
            RAGFoundationError,
        ) as exc:
            failed = metadata.model_copy(
                update={
                    "size_bytes": min(initial_size, self.max_bytes),
                    "status": DocumentStatus.FAILED,
                    "error": str(exc),
                }
            )
            self.repository.save(failed, b"", [])
            raise

    def get_metadata(self, document_id: str) -> DocumentMetadata:
        return self.repository.get_metadata(document_id)

    def get_content(self, document_id: str) -> bytes:
        return self.repository.get_content(document_id)

    def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        return self.repository.get_chunks(document_id)

    def list_documents(
        self, *, user_id: str, conversation_id: str
    ) -> list[tuple[DocumentMetadata, int]]:
        metadata = self.repository.list_metadata(user_id, conversation_id)
        return [
            (item, self.repository.get_chunk_count(item.document_id))
            for item in metadata
        ]

    def get_document(
        self, *, document_id: str, user_id: str, conversation_id: str
    ) -> tuple[DocumentMetadata, int]:
        if not document_id.strip():
            raise DocumentNotFoundError("Document was not found.")
        metadata = self.repository.get_metadata_scoped(
            document_id, user_id, conversation_id
        )
        if metadata is None:
            raise DocumentNotFoundError("Document was not found.")
        return metadata, self.repository.get_chunk_count(document_id)

    def delete_document(
        self, *, document_id: str, user_id: str, conversation_id: str
    ) -> None:
        metadata, _ = self.get_document(
            document_id=document_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        try:
            self.vector_store.delete_document(metadata.document_id)
            self.repository.delete(metadata.document_id)
        except RAGFoundationError as exc:
            raise DocumentStorageError("Document could not be deleted.") from exc


def _read_limited(stream: BinaryIO, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return b"".join(chunks) + chunk
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_document(filename: str, content_type: str, data: bytes) -> None:
    if content_type not in _CONTENT_TYPES:
        raise UnsupportedDocumentTypeError(
            "Only PDF, TXT, and DOCX documents are supported."
        )
    suffix = Path(filename).suffix.lower()
    expected_suffix = {
        SupportedDocumentType.PDF: ".pdf",
        SupportedDocumentType.TXT: ".txt",
        SupportedDocumentType.DOCX: ".docx",
    }[SupportedDocumentType(content_type)]
    if suffix != expected_suffix:
        raise InvalidDocumentError("The filename extension does not match the content type.")
    if not data:
        raise InvalidDocumentError("The document is empty.")
    if content_type == SupportedDocumentType.PDF:
        if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-1024:]:
            raise InvalidDocumentError("The PDF document is invalid or incomplete.")
    elif content_type == SupportedDocumentType.TXT:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("The text document is not valid UTF-8.") from exc
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise InvalidDocumentError("The DOCX document is missing required parts.")
                if archive.testzip() is not None:
                    raise InvalidDocumentError("The DOCX document is corrupted.")
        except zipfile.BadZipFile as exc:
            raise InvalidDocumentError("The DOCX document is corrupted.") from exc

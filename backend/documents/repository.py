from __future__ import annotations

from threading import Lock

from backend.documents.chunking import DocumentChunk
from backend.documents.models import DocumentMetadata


class DocumentRepository:
    """Process-local repository replaceable by persistent storage later."""

    def __init__(self) -> None:
        self._metadata: dict[str, DocumentMetadata] = {}
        self._content: dict[str, bytes] = {}
        self._chunks: dict[str, list[DocumentChunk]] = {}
        self._lock = Lock()

    def save(
        self, metadata: DocumentMetadata, content: bytes, chunks: list[DocumentChunk]
    ) -> None:
        with self._lock:
            self._metadata[metadata.document_id] = metadata
            self._content[metadata.document_id] = content
            self._chunks[metadata.document_id] = list(chunks)

    def get_metadata(self, document_id: str) -> DocumentMetadata:
        with self._lock:
            return self._metadata[document_id]

    def list_metadata(self, user_id: str, conversation_id: str) -> list[DocumentMetadata]:
        with self._lock:
            return [
                metadata.model_copy()
                for metadata in self._metadata.values()
                if metadata.user_id == user_id
                and metadata.conversation_id == conversation_id
            ]

    def get_metadata_scoped(
        self, document_id: str, user_id: str, conversation_id: str
    ) -> DocumentMetadata | None:
        with self._lock:
            metadata = self._metadata.get(document_id)
            if metadata is None:
                return None
            if (
                metadata.user_id != user_id
                or metadata.conversation_id != conversation_id
            ):
                return None
            return metadata.model_copy()

    def delete(self, document_id: str) -> None:
        with self._lock:
            self._metadata.pop(document_id, None)
            self._content.pop(document_id, None)
            self._chunks.pop(document_id, None)

    def get_chunk_count(self, document_id: str) -> int:
        with self._lock:
            return len(self._chunks.get(document_id, []))

    def get_content(self, document_id: str) -> bytes:
        with self._lock:
            return self._content[document_id]

    def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        with self._lock:
            return list(self._chunks[document_id])

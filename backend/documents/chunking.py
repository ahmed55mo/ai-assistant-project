from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from backend.documents.extraction import ExtractedSegment
from backend.documents.models import DocumentMetadata


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    user_id: str
    conversation_id: str
    chunk_index: int
    text: str
    page_number: int | None
    filename: str


class TextChunkingService:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, metadata: DocumentMetadata, segments: list[ExtractedSegment]
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if end < len(text):
                    boundary = max(
                        text.rfind("\n", start, end),
                        text.rfind(" ", start, end),
                    )
                    if boundary > start:
                        end = boundary
                piece = text[start:end].strip()
                if piece:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid4()),
                            document_id=metadata.document_id,
                            user_id=metadata.user_id,
                            conversation_id=metadata.conversation_id,
                            chunk_index=len(chunks),
                            text=piece,
                            page_number=segment.page_number,
                            filename=metadata.filename,
                        )
                    )
                if end >= len(text):
                    break
                start = max(end - self.chunk_overlap, start + 1)
        return chunks

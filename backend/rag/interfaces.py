from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from backend.documents.chunking import DocumentChunk


Chunk = DocumentChunk


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    user_id: str
    conversation_id: str
    document_id: str | None = None
    top_k: int = 5
    score_threshold: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    similarity_score: float

    @property
    def score(self) -> float:
        """Compatibility alias for vector-store score consumers."""
        return self.similarity_score

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def user_id(self) -> str:
        return self.chunk.user_id

    @property
    def conversation_id(self) -> str:
        return self.chunk.conversation_id

    @property
    def chunk_index(self) -> int:
        return self.chunk.chunk_index

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def filename(self) -> str:
        return self.chunk.filename

    @property
    def page_number(self) -> int | None:
        return self.chunk.page_number


@dataclass(frozen=True)
class CitationSource:
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str


class EmbeddingService(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievalResult]:
        ...

    def delete_document(self, document_id: str) -> None:
        ...

    def clear(self) -> None:
        ...


class Retriever(Protocol):
    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        ...


class RAGService(Protocol):
    def answer_with_context(self, request: RetrievalRequest) -> str:
        ...

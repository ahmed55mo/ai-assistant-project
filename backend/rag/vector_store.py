from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Sequence

import faiss
import numpy as np

from backend.documents.chunking import DocumentChunk
from backend.rag.errors import (
    EmbeddingDimensionError,
    InvalidVectorStoreRequest,
    VectorStoreError,
)
from backend.rag.interfaces import RetrievalResult


@dataclass(frozen=True)
class _StoredVector:
    chunk: DocumentChunk
    vector: tuple[float, ...]


class FaissVectorStore:
    """Process-local cosine-similarity store backed by FAISS inner product search."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("Vector-store dimension must be positive.")
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._records: list[_StoredVector] = []
        self._lock = RLock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise InvalidVectorStoreRequest(
                "The number of chunks must match the number of embeddings."
            )
        if not chunks:
            return
        matrix = self._validated_matrix(embeddings)
        with self._lock:
            try:
                self._index.add(matrix)
                self._records.extend(
                    _StoredVector(chunk=chunk, vector=tuple(vector))
                    for chunk, vector in zip(chunks, matrix.tolist())
                )
            except Exception as exc:
                self._rebuild_locked()
                raise VectorStoreError("Vectors could not be added.") from exc

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise InvalidVectorStoreRequest("top_k must be positive.")
        normalized_filters = self._validate_filters(filters)
        query = self._validated_matrix([query_embedding])
        with self._lock:
            if not self._records:
                return []
            candidate_count = len(self._records)
            scores, indexes = self._index.search(query, candidate_count)
            ranked: list[tuple[float, int, RetrievalResult]] = []
            for score, index in zip(scores[0], indexes[0]):
                if index < 0:
                    continue
                record = self._records[index]
                if not _matches(record.chunk, normalized_filters):
                    continue
                ranked.append(
                    (
                        float(score),
                        int(index),
                        RetrievalResult(
                            chunk=record.chunk,
                            similarity_score=float(score),
                        ),
                    )
                )
            ranked.sort(key=lambda item: (-item[0], item[1]))
            return [item[2] for item in ranked[:top_k]]

    def delete_document(self, document_id: str) -> None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise InvalidVectorStoreRequest("document_id must not be empty.")
        with self._lock:
            self._records = [
                record
                for record in self._records
                if record.chunk.document_id != document_id
            ]
            self._rebuild_locked()

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._rebuild_locked()

    def _validated_matrix(self, vectors: Sequence[Sequence[float]]) -> np.ndarray:
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbeddingDimensionError("Vectors must be numeric.") from exc
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise EmbeddingDimensionError(
                f"Vectors must have dimension {self._dimension}."
            )
        if not np.isfinite(matrix).all():
            raise EmbeddingDimensionError("Vectors must contain finite values.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingDimensionError("Vectors must not be zero-length.")
        return matrix / norms

    def _rebuild_locked(self) -> None:
        self._index = faiss.IndexFlatIP(self._dimension)
        if self._records:
            self._index.add(np.asarray([record.vector for record in self._records], dtype=np.float32))

    @staticmethod
    def _validate_filters(filters: Mapping[str, str] | None) -> dict[str, str]:
        if filters is None:
            return {}
        allowed = {"user_id", "conversation_id", "document_id"}
        unknown = set(filters) - allowed
        if unknown:
            raise InvalidVectorStoreRequest("Unsupported metadata filter.")
        normalized: dict[str, str] = {}
        for key, value in filters.items():
            if not isinstance(value, str) or not value.strip():
                raise InvalidVectorStoreRequest("Metadata filters must not be empty.")
            normalized[key] = value
        return normalized


def _matches(chunk: DocumentChunk, filters: Mapping[str, str]) -> bool:
    return all(getattr(chunk, key) == value for key, value in filters.items())

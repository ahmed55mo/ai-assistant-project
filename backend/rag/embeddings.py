from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Sequence

from backend.rag.errors import EmbeddingError


DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_EMBEDDING_DIMENSION = 64


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    dimension: int = DEFAULT_EMBEDDING_DIMENSION

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig":
        provider = os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER).strip().lower()
        raw_dimension = os.getenv(
            "EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)
        )
        try:
            dimension = int(raw_dimension)
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSION must be an integer.") from exc
        if not provider:
            raise ValueError("EMBEDDING_PROVIDER must not be empty.")
        if dimension < 1:
            raise ValueError("EMBEDDING_DIMENSION must be positive.")
        return cls(provider=provider, dimension=dimension)


class LocalEmbeddingService:
    """Deterministic local hashed embeddings for development and tests."""

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive.")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Text to embed must not be empty.")
        values = [0.0] * self._dimension
        tokens = text.casefold().split()
        for index, token in enumerate(tokens):
            digest = hashlib.blake2b(
                f"{index}:{token}".encode("utf-8"),
                digest_size=16,
            ).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            values[bucket] += sign * (1.0 + digest[5] / 255.0)
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise EmbeddingError("Text could not produce an embedding.")
        return [value / norm for value in values]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def build_embedding_service(config: EmbeddingConfig | None = None) -> LocalEmbeddingService:
    resolved = config or EmbeddingConfig.from_environment()
    if resolved.provider != DEFAULT_EMBEDDING_PROVIDER:
        raise ValueError(f"Unsupported embedding provider: {resolved.provider}.")
    return LocalEmbeddingService(dimension=resolved.dimension)

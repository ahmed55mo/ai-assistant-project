from __future__ import annotations


class RAGFoundationError(RuntimeError):
    """Base error for embedding and vector-store foundation operations."""


class EmbeddingError(RAGFoundationError):
    """Raised when text cannot be embedded."""


class RetrievalError(RAGFoundationError):
    """Raised when retrieval cannot be completed."""


class InvalidRetrievalRequest(RetrievalError):
    """Raised when a retrieval request is invalid."""


class KnowledgeSearchError(RAGFoundationError):
    """Raised when the LangGraph knowledge-search adapter fails."""


class EmbeddingDimensionError(RAGFoundationError):
    """Raised when an embedding has an unexpected dimension."""


class VectorStoreError(RAGFoundationError):
    """Raised when a vector-store operation cannot be completed."""


class InvalidVectorStoreRequest(VectorStoreError):
    """Raised when vector-store input or metadata scope is invalid."""

"""Embedding and vector-store foundation for future retrieval phases."""

from backend.rag.embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_PROVIDER,
    EmbeddingConfig,
    LocalEmbeddingService,
    build_embedding_service,
)
from backend.rag.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    InvalidVectorStoreRequest,
    InvalidRetrievalRequest,
    KnowledgeSearchError,
    RAGFoundationError,
    RetrievalError,
    VectorStoreError,
)
from backend.rag.interfaces import (
    Chunk,
    CitationSource,
    EmbeddingService,
    RAGService,
    Retriever,
    RetrievalRequest,
    RetrievalResult,
    VectorStore,
)
from backend.rag.vector_store import FaissVectorStore
from backend.rag.retrieval import RetrievalService
from backend.rag.tool import (
    build_knowledge_search_tool,
    reset_trusted_scope,
    set_trusted_scope,
)

__all__ = [
    "Chunk",
    "CitationSource",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_PROVIDER",
    "EmbeddingConfig",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingService",
    "FaissVectorStore",
    "InvalidVectorStoreRequest",
    "InvalidRetrievalRequest",
    "KnowledgeSearchError",
    "LocalEmbeddingService",
    "RAGService",
    "RAGFoundationError",
    "RetrievalError",
    "RetrievalService",
    "build_knowledge_search_tool",
    "reset_trusted_scope",
    "set_trusted_scope",
    "Retriever",
    "RetrievalRequest",
    "RetrievalResult",
    "VectorStore",
    "VectorStoreError",
    "build_embedding_service",
]

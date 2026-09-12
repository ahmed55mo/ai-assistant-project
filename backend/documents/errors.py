from __future__ import annotations


class DocumentError(RuntimeError):
    """Base error for document ingestion."""


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when a document format is not supported."""


class DocumentTooLargeError(DocumentError):
    """Raised when a document exceeds the configured size limit."""


class InvalidDocumentError(DocumentError):
    """Raised when a document cannot be safely validated."""


class DocumentNotFoundError(DocumentError):
    """Raised when a document is absent or outside the requested scope."""


class DocumentStorageError(DocumentError):
    """Raised when document storage cannot complete an operation."""

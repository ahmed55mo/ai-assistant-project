from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class SupportedDocumentType(StrEnum):
    PDF = "application/pdf"
    TXT = "text/plain"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    user_id: str
    conversation_id: str
    filename: str
    content_type: str
    size_bytes: int = Field(ge=0)
    status: DocumentStatus
    created_at: datetime
    error: str | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        user_id: str,
        conversation_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        status: DocumentStatus,
        error: str | None = None,
    ) -> "DocumentMetadata":
        return cls(
            document_id=document_id,
            user_id=user_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            status=status,
            created_at=datetime.now(timezone.utc),
            error=error,
        )


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    status: Literal["processing", "ready", "failed"]
    conversation_id: str


class DocumentManagementResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    status: Literal["processing", "ready", "failed"]
    conversation_id: str
    user_id: str
    chunk_count: int = Field(ge=0)


class DocumentListResponse(BaseModel):
    documents: list[DocumentManagementResponse]

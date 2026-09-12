from __future__ import annotations

import io
from dataclasses import dataclass

from docx import Document
from pypdf import PdfReader

from backend.documents.errors import InvalidDocumentError
from backend.documents.models import SupportedDocumentType


@dataclass(frozen=True)
class ExtractedSegment:
    text: str
    page_number: int | None = None


class DocumentTextExtractor:
    """Extract text while retaining source boundaries where the format provides them."""

    def extract(self, content: bytes, content_type: str) -> list[ExtractedSegment]:
        try:
            if content_type == SupportedDocumentType.PDF:
                return self._pdf(content)
            if content_type == SupportedDocumentType.TXT:
                return [ExtractedSegment(content.decode("utf-8"))]
            if content_type == SupportedDocumentType.DOCX:
                return self._docx(content)
        except InvalidDocumentError:
            raise
        except (UnicodeDecodeError, OSError, ValueError, RuntimeError) as exc:
            raise InvalidDocumentError("The document text could not be extracted.") from exc
        raise InvalidDocumentError("Unsupported document type for text extraction.")

    @staticmethod
    def _pdf(content: bytes) -> list[ExtractedSegment]:
        try:
            reader = PdfReader(io.BytesIO(content), strict=False)
            segments = [
                ExtractedSegment(page.extract_text() or "", page_number=index)
                for index, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise InvalidDocumentError("The PDF text could not be extracted.") from exc
        if not segments:
            raise InvalidDocumentError("The PDF document has no pages.")
        return segments

    @staticmethod
    def _docx(content: bytes) -> list[ExtractedSegment]:
        try:
            document = Document(io.BytesIO(content))
        except Exception as exc:
            raise InvalidDocumentError("The DOCX text could not be extracted.") from exc
        return [ExtractedSegment(paragraph.text) for paragraph in document.paragraphs]


def normalize_segments(segments: list[ExtractedSegment]) -> list[ExtractedSegment]:
    normalized: list[ExtractedSegment] = []
    for segment in segments:
        lines = [" ".join(line.split()) for line in segment.text.splitlines()]
        text = "\n".join(line for line in lines if line).strip()
        if text:
            normalized.append(
                ExtractedSegment(text=text, page_number=segment.page_number)
            )
    return normalized

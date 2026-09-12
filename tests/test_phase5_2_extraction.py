from __future__ import annotations

import io
from unittest.mock import Mock

import pytest
from docx import Document
from pypdf import PdfWriter

from backend.documents.chunking import TextChunkingService
from backend.documents.errors import InvalidDocumentError
from backend.documents.extraction import (
    DocumentTextExtractor,
    ExtractedSegment,
    normalize_segments,
)
from backend.documents.ingestion import DocumentIngestionService
from backend.documents.models import DocumentMetadata, DocumentStatus


def _docx(paragraphs: list[str]) -> bytes:
    output = io.BytesIO()
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(output)
    return output.getvalue()


def _metadata() -> DocumentMetadata:
    return DocumentMetadata.create(
        document_id="doc-1",
        user_id="user-1",
        conversation_id="conversation-1",
        filename="policy.txt",
        content_type="text/plain",
        size_bytes=100,
        status=DocumentStatus.PROCESSING,
    )


def test_pdf_text_extraction_is_page_ordered_with_page_metadata() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    output = io.BytesIO()
    writer.write(output)
    reader = Mock()
    first, second = Mock(), Mock()
    first.extract_text.return_value = "page one"
    second.extract_text.return_value = "page two"
    reader.pages = [first, second]
    extractor = DocumentTextExtractor()
    extractor_reader = Mock()
    extractor_reader.return_value = reader

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("backend.documents.extraction.PdfReader", extractor_reader)
        segments = extractor.extract(output.getvalue(), "application/pdf")

    assert [(item.text, item.page_number) for item in segments] == [
        ("page one", 1),
        ("page two", 2),
    ]


def test_txt_extraction_decodes_utf8_and_rejects_invalid_bytes() -> None:
    extractor = DocumentTextExtractor()
    assert extractor.extract("hello\nworld".encode(), "text/plain")[0].text == "hello\nworld"
    with pytest.raises(InvalidDocumentError):
        extractor.extract(b"\xff", "text/plain")


def test_docx_extraction_preserves_paragraph_boundaries() -> None:
    segments = DocumentTextExtractor().extract(
        _docx(["First paragraph", "Second paragraph"]),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert [segment.text for segment in segments] == ["First paragraph", "Second paragraph"]


def test_empty_extraction_is_safe() -> None:
    assert normalize_segments([ExtractedSegment(" \n \t")]) == []
    assert DocumentTextExtractor().extract(
        _docx([]),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ) == []


def test_normalization_preserves_paragraphs_and_removes_excess_whitespace() -> None:
    result = normalize_segments(
        [ExtractedSegment("  First   line \n\n second\tline  \n\n\n")]
    )
    assert result[0].text == "First line\nsecond line"


def test_chunk_creation_order_size_and_metadata() -> None:
    service = TextChunkingService(chunk_size=12, chunk_overlap=3)
    chunks = service.chunk(
        _metadata(),
        [ExtractedSegment("one two three four five", page_number=2)],
    )

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(0 < len(chunk.text) <= 12 for chunk in chunks)
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.user_id == "user-1" for chunk in chunks)
    assert all(chunk.conversation_id == "conversation-1" for chunk in chunks)
    assert all(chunk.filename == "policy.txt" for chunk in chunks)


def test_chunk_overlap_is_configurable_and_deterministic() -> None:
    service = TextChunkingService(chunk_size=10, chunk_overlap=4)
    segments = [ExtractedSegment("abcdefghij klmnopqrst")]
    first = service.chunk(_metadata(), segments)
    second = service.chunk(_metadata(), segments)

    assert [item.text for item in first] == [item.text for item in second]
    assert first[1].text[:4] in first[0].text


def test_ingestion_extracts_chunks_and_marks_failure() -> None:
    service = DocumentIngestionService()
    metadata = service.ingest(
        user_id="user-1",
        conversation_id="conversation-1",
        filename="policy.txt",
        content_type="text/plain",
        content=b"Annual leave policy.",
    )
    assert metadata.status is DocumentStatus.READY
    chunks = service.get_chunks(metadata.document_id)
    assert chunks[0].document_id == metadata.document_id

    failing_extractor = Mock()
    failing_extractor.extract.side_effect = InvalidDocumentError("extraction failed")
    failing = DocumentIngestionService(extractor=failing_extractor)
    with pytest.raises(InvalidDocumentError):
        failing.ingest(
            user_id="user-1",
            conversation_id="conversation-1",
            filename="policy.txt",
            content_type="text/plain",
            content=b"Annual leave policy.",
        )
    failed = next(iter(failing._metadata.values()))
    assert failed.status is DocumentStatus.FAILED

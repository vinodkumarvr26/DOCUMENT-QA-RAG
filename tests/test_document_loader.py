"""Offline tests for PDF and text document loading."""

from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from document_loader import (
    DocumentNotFoundError,
    EmptyDocumentError,
    EncryptedPDFError,
    InvalidDocumentError,
    UnsupportedDocumentError,
    load_document,
    load_pdf,
    load_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "samples"
SYNTHETIC_PDF = SAMPLES / "synthetic_two_page.pdf"
SAMPLE_TEXT = SAMPLES / "sample.txt"


def test_pdf_loads_each_page_as_separate_document() -> None:
    documents = load_pdf(SYNTHETIC_PDF)

    assert len(documents) == 2
    assert "Synthetic first page" in documents[0].page_content
    assert "Synthetic second page" in documents[1].page_content


def test_pdf_page_indexes_are_zero_based() -> None:
    documents = load_pdf(SYNTHETIC_PDF)

    assert documents[0].metadata["page_index"] == 0
    assert documents[1].metadata["page_index"] == 1


def test_pdf_source_and_page_metadata_are_preserved() -> None:
    documents = load_pdf(SYNTHETIC_PDF)

    assert documents[0].metadata["source"] == SYNTHETIC_PDF.name
    assert documents[1].metadata["source"] == SYNTHETIC_PDF.name
    assert documents[0].metadata["page_label"] == "A-1"
    assert documents[1].metadata["page_label"] == "A-2"
    assert documents[0].metadata["total_pages"] == 2
    assert documents[1].metadata["total_pages"] == 2


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"", "Could not read PDF"),
        (b"not a PDF", "Could not read PDF"),
    ],
)
def test_empty_or_invalid_pdf_has_meaningful_error(
    tmp_path: Path, contents: bytes, message: str
) -> None:
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_bytes(contents)

    with pytest.raises(InvalidDocumentError, match=message):
        load_pdf(pdf_path)


def test_pdf_with_no_extractable_text_is_rejected(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_path = tmp_path / "blank.pdf"
    with pdf_path.open("wb") as file:
        writer.write(file)

    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        load_pdf(pdf_path)


def test_encrypted_pdf_is_rejected_clearly(tmp_path: Path) -> None:
    writer = PdfWriter(clone_from=str(SYNTHETIC_PDF))
    writer.encrypt("synthetic-test-password")
    pdf_path = tmp_path / "encrypted.pdf"
    with pdf_path.open("wb") as file:
        writer.write(file)

    with pytest.raises(EncryptedPDFError, match="Encrypted PDF"):
        load_pdf(pdf_path)


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "document.md"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError, match="expected '.pdf' or '.txt'"):
        load_document(path)


def test_text_content_and_source_metadata_are_preserved() -> None:
    documents = load_text(SAMPLE_TEXT)

    assert len(documents) == 1
    assert "small synthetic sample document" in documents[0].page_content
    assert documents[0].metadata == {"source": SAMPLE_TEXT.name}


def test_utf8_text_is_read_correctly(tmp_path: Path) -> None:
    path = tmp_path / "utf8.txt"
    expected = "Café — résumé"
    path.write_text(expected, encoding="utf-8")

    assert load_text(path)[0].page_content == expected


def test_utf8_bom_is_removed(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfText after the UTF-8 BOM.")

    assert load_text(path)[0].page_content == "Text after the UTF-8 BOM."


def test_missing_text_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentNotFoundError, match="does not exist"):
        load_text(tmp_path / "missing.txt")


@pytest.mark.parametrize("contents", [b"", b" \t\r\n"])
def test_empty_text_file_is_rejected(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(contents)

    with pytest.raises(EmptyDocumentError, match="contains no text"):
        load_text(path)


def test_loader_classes_implement_base_loader_interface() -> None:
    from document_loader import PDFDocumentLoader, TextDocumentLoader

    assert len(PDFDocumentLoader(SYNTHETIC_PDF).load()) == 2
    assert len(TextDocumentLoader(SAMPLE_TEXT).load()) == 1


def test_fixture_pdf_has_two_pages() -> None:
    reader = PdfReader(SYNTHETIC_PDF)

    assert len(reader.pages) == 2

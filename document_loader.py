"""Load supported PDF and UTF-8 text files as LangChain documents."""

from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError


class DocumentLoadingError(Exception):
    """Base exception for supported-document loading failures."""


class DocumentNotFoundError(DocumentLoadingError):
    """Raised when a requested document does not exist or is not a file."""


class UnsupportedDocumentError(DocumentLoadingError):
    """Raised when a document's file extension is not supported."""


class EmptyDocumentError(DocumentLoadingError):
    """Raised when a document contains no extractable text."""


class InvalidDocumentError(DocumentLoadingError):
    """Raised when a document is malformed or cannot be decoded."""


class EncryptedPDFError(DocumentLoadingError):
    """Raised when a PDF is encrypted and cannot be read without a password."""


def _validate_file(path: str | Path, supported_extension: str) -> Path:
    """Resolve and validate a path for one of the supported document formats."""
    file_path = Path(path)
    if file_path.suffix.lower() != supported_extension:
        raise UnsupportedDocumentError(
            f"Unsupported file extension {file_path.suffix or '(none)'!r}; "
            f"expected {supported_extension!r}."
        )
    if not file_path.exists() or not file_path.is_file():
        raise DocumentNotFoundError(f"Document file does not exist: {file_path}")
    return file_path


class PDFDocumentLoader(BaseLoader):
    """Load a PDF as one LangChain document per page.

    Page indexes in metadata are zero-based. Scanned pages without an
    extractable text layer are not OCR-processed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = _validate_file(path, ".pdf")

    def lazy_load(self) -> Iterator[Document]:
        """Yield each PDF page with its source and page metadata."""
        try:
            with self.path.open("rb") as pdf_file:
                reader = PdfReader(pdf_file, strict=True)
                if reader.is_encrypted:
                    raise EncryptedPDFError(
                        f"Encrypted PDF is not supported: {self.path.name}"
                    )

                pages = reader.pages
                total_pages = len(pages)
                if total_pages == 0:
                    raise EmptyDocumentError(
                        f"PDF contains no pages: {self.path.name}"
                    )

                try:
                    page_labels = reader.page_labels
                except (PdfReadError, ValueError):
                    page_labels = []

                page_texts: list[str] = []
                for page_index, page in enumerate(pages):
                    try:
                        page_texts.append(page.extract_text() or "")
                    except (PdfReadError, FileNotDecryptedError, ValueError) as exc:
                        raise InvalidDocumentError(
                            f"Could not extract text from page {page_index} "
                            f"of {self.path.name}: {exc}"
                        ) from exc

        except EncryptedPDFError:
            raise
        except EmptyDocumentError:
            raise
        except (PdfReadError, FileNotDecryptedError, OSError, ValueError) as exc:
            raise InvalidDocumentError(
                f"Could not read PDF {self.path.name}: {exc}"
            ) from exc

        if not any(text.strip() for text in page_texts):
            raise EmptyDocumentError(
                f"PDF contains no extractable text: {self.path.name}"
            )

        for page_index, page_text in enumerate(page_texts):
            metadata: dict[str, str | int] = {
                "source": self.path.name,
                "page_index": page_index,
                "total_pages": total_pages,
            }
            if page_index < len(page_labels):
                metadata["page_label"] = page_labels[page_index]
            yield Document(page_content=page_text, metadata=metadata)


class TextDocumentLoader(BaseLoader):
    """Load one UTF-8 text file as a LangChain document."""

    def __init__(self, path: str | Path) -> None:
        self.path = _validate_file(path, ".txt")

    def lazy_load(self) -> Iterator[Document]:
        """Yield the text document with its original filename."""
        try:
            content = self.path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise InvalidDocumentError(
                f"Could not read UTF-8 text file {self.path.name}: {exc}"
            ) from exc

        if not content.strip():
            raise EmptyDocumentError(
                f"Text file contains no text: {self.path.name}"
            )
        yield Document(page_content=content, metadata={"source": self.path.name})


def load_pdf(path: str | Path) -> list[Document]:
    """Load every page of a PDF into a separate LangChain document."""
    return PDFDocumentLoader(path).load()


def load_text(path: str | Path) -> list[Document]:
    """Load a UTF-8 text file into a single LangChain document."""
    return TextDocumentLoader(path).load()


def load_document(path: str | Path) -> list[Document]:
    """Load a supported PDF or text file, dispatching by its extension."""
    file_path = Path(path)
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return load_pdf(file_path)
    if extension == ".txt":
        return load_text(file_path)
    raise UnsupportedDocumentError(
        f"Unsupported file extension {extension or '(none)'!r}; "
        "expected '.pdf' or '.txt'."
    )

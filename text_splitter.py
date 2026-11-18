"""Recursively split LangChain documents while retaining their metadata."""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings


def _validate_chunk_configuration(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk sizes before constructing a splitter."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int):
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def create_text_splitter(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a configured recursive splitter, defaulting to project settings."""
    settings = get_settings()
    resolved_chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    resolved_chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    _validate_chunk_configuration(resolved_chunk_size, resolved_chunk_overlap)

    return RecursiveCharacterTextSplitter(
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split non-empty documents and retain their original metadata.

    ``chunk_index`` starts at zero for each input document. The splitter adds
    ``start_index`` offsets where it can locate each chunk in its source text.
    """
    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence of LangChain Document objects.")
    if any(not isinstance(document, Document) for document in documents):
        raise TypeError("every item in documents must be a LangChain Document.")

    chunks: list[Document] = []
    for document in documents:
        if not document.page_content.strip():
            continue
        document_chunks = splitter.split_documents([document])
        for chunk_index, chunk in enumerate(document_chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunks.append(chunk)
    return chunks

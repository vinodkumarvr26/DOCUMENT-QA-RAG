"""Offline tests for recursive document chunking."""

import pytest
from langchain_core.documents import Document

from config import get_settings
from text_splitter import create_text_splitter, split_documents


def test_short_document_produces_one_appropriate_chunk() -> None:
    document = Document(
        page_content="A short synthetic passage.",
        metadata={"source": "short.txt"},
    )

    chunks = split_documents([document], chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].page_content == document.page_content
    assert chunks[0].metadata["source"] == "short.txt"


def test_long_document_produces_multiple_chunks_within_size() -> None:
    document = Document(
        page_content=" ".join(f"token{i}" for i in range(60)),
        metadata={"source": "long.txt"},
    )

    chunks = split_documents([document], chunk_size=40, chunk_overlap=8)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 40 for chunk in chunks)


def test_overlap_repeats_context_between_adjacent_chunks() -> None:
    document = Document(
        page_content=" ".join(f"word{i}" for i in range(30)),
        metadata={"source": "overlap.txt"},
    )

    chunks = split_documents([document], chunk_size=30, chunk_overlap=10)

    assert len(chunks) > 1
    assert chunks[0].page_content.split()[-1] in chunks[1].page_content


def test_configured_defaults_are_read_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "48")
    monkeypatch.setenv("CHUNK_OVERLAP", "9")
    splitter = create_text_splitter()

    assert splitter._chunk_size == 48
    assert splitter._chunk_overlap == 9
    assert get_settings().chunk_size == 48
    assert get_settings().chunk_overlap == 9


def test_page_and_source_metadata_are_preserved() -> None:
    document = Document(
        page_content="Page evidence. " * 20,
        metadata={
            "source": "report.pdf",
            "page_index": 1,
            "page_label": "A-2",
            "total_pages": 2,
        },
    )

    chunks = split_documents([document], chunk_size=50, chunk_overlap=10)

    assert chunks
    assert all(chunk.metadata["source"] == "report.pdf" for chunk in chunks)
    assert all(chunk.metadata["page_index"] == 1 for chunk in chunks)
    assert all(chunk.metadata["page_label"] == "A-2" for chunk in chunks)
    assert all(chunk.metadata["total_pages"] == 2 for chunk in chunks)


def test_source_filenames_are_preserved() -> None:
    documents = [
        Document(page_content="alpha " * 20, metadata={"source": "a.txt"}),
        Document(page_content="beta " * 20, metadata={"source": "b.txt"}),
    ]

    chunks = split_documents(documents, chunk_size=30, chunk_overlap=5)

    assert {chunk.metadata["source"] for chunk in chunks} == {"a.txt", "b.txt"}


def test_chunk_indexes_start_at_zero_for_each_input_document() -> None:
    documents = [
        Document(page_content="alpha " * 20, metadata={"source": "a.txt"}),
        Document(page_content="beta " * 20, metadata={"source": "b.txt"}),
    ]

    chunks = split_documents(documents, chunk_size=30, chunk_overlap=5)
    first_source_indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
        if chunk.metadata["source"] == "a.txt"
    ]
    second_source_indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
        if chunk.metadata["source"] == "b.txt"
    ]

    assert first_source_indexes == list(range(len(first_source_indexes)))
    assert second_source_indexes == list(range(len(second_source_indexes)))


def test_start_indexes_are_assigned_where_possible() -> None:
    document = Document(
        page_content=" ".join(f"piece{i}" for i in range(30)),
        metadata={"source": "offsets.txt"},
    )

    chunks = split_documents([document], chunk_size=32, chunk_overlap=8)

    assert len(chunks) > 1
    assert chunks[0].metadata["start_index"] == 0
    assert all(isinstance(chunk.metadata["start_index"], int) for chunk in chunks)
    assert all(chunk.metadata["start_index"] >= 0 for chunk in chunks)


def test_empty_documents_are_skipped_safely() -> None:
    documents = [
        Document(page_content="", metadata={"source": "empty.txt"}),
        Document(page_content=" \n\t ", metadata={"source": "whitespace.txt"}),
    ]

    assert split_documents(documents) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be greater than zero"),
        (20, -1, "chunk_overlap must be non-negative"),
        (20, 20, "chunk_overlap must be smaller than chunk_size"),
        (20, 21, "chunk_overlap must be smaller than chunk_size"),
    ],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size: int, chunk_overlap: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_zero_overlap_is_valid() -> None:
    chunks = split_documents(
        [Document(page_content="Some text. " * 20)],
        chunk_size=30,
        chunk_overlap=0,
    )

    assert len(chunks) > 1

"""Offline tests for DocuQuery UI helpers and indexing integration."""

from pathlib import Path
from typing import Any
import uuid

import pytest
from langchain_core.documents import Document

import app
from document_loader import EmptyDocumentError
from rag_chain import EmptyQuestionError, MissingAPIKeyError, RAGQuotaError


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_session_collection_names_are_stable_and_distinct() -> None:
    first_session = "12345678-1234-5678-1234-567812345678"
    other_session = "87654321-4321-8765-4321-876543218765"

    first = app.collection_name_for_session(first_session)

    assert first == app.collection_name_for_session(first_session)
    assert first != app.collection_name_for_session(other_session)
    assert first.startswith("docuquery_")
    assert len(first) <= 63
    assert all(character.islower() or character.isdigit() or character == "_" for character in first)


def test_session_state_is_isolated_and_stable_across_initialization(
    tmp_path: Path,
) -> None:
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    app.initialize_session_state(
        first,
        storage_root=tmp_path,
        session_id_factory=lambda: uuid.UUID(
            "12345678-1234-5678-1234-567812345678"
        ),
    )
    app.initialize_session_state(
        second,
        storage_root=tmp_path,
        session_id_factory=lambda: uuid.UUID(
            "87654321-4321-8765-4321-876543218765"
        ),
    )
    original_collection = first["collection_name"]

    app.initialize_session_state(first, storage_root=tmp_path)

    assert first["collection_name"] == original_collection
    assert first["collection_name"] != second["collection_name"]
    assert first["upload_directory"] != second["upload_directory"]
    assert Path(first["upload_directory"]).is_dir()


def test_upload_validation_supports_pdf_and_txt() -> None:
    assert app.validate_uploaded_file(
        FakeUploadedFile("notes.TXT", b"Synthetic text.")
    ) == ("notes.TXT", b"Synthetic text.")
    assert app.validate_uploaded_file(
        FakeUploadedFile("scan.pdf", b"%PDF synthetic")
    )[0] == "scan.pdf"


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("unsupported.md", b"text", "only PDF and TXT"),
        ("empty.txt", b"", "empty"),
    ],
)
def test_upload_validation_rejects_unsupported_and_empty_files(
    name: str,
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(app.UploadValidationError, match=message):
        app.validate_uploaded_file(FakeUploadedFile(name, content))


def test_safe_filename_removes_paths_and_unsafe_characters() -> None:
    assert app.safe_filename(r"..\folder\report:final.txt") == "report_final.txt"


def test_upload_is_saved_under_generated_path_and_original_name_is_kept(
    tmp_path: Path,
) -> None:
    uploaded = FakeUploadedFile(r"..\private\report.txt", b"Synthetic safe content.")

    saved = app.save_uploaded_file(uploaded, tmp_path)

    assert saved.original_filename == "report.txt"
    assert saved.path.parent == tmp_path.resolve()
    assert saved.path.name != saved.original_filename
    assert saved.path.read_bytes() == b"Synthetic safe content."
    assert saved.document_id


def test_duplicate_filenames_have_distinct_stable_document_ids(
    tmp_path: Path,
) -> None:
    first = app.save_uploaded_file(
        FakeUploadedFile("duplicate.txt", b"First synthetic content."),
        tmp_path,
        duplicate_index=0,
    )
    second = app.save_uploaded_file(
        FakeUploadedFile("duplicate.txt", b"Second synthetic content."),
        tmp_path,
        duplicate_index=1,
    )
    updated_first = app.save_uploaded_file(
        FakeUploadedFile("duplicate.txt", b"Updated synthetic content."),
        tmp_path,
        duplicate_index=0,
    )

    assert first.document_id != second.document_id
    assert first.document_id == updated_first.document_id
    assert first.path != updated_first.path


def test_indexing_uses_existing_load_split_store_add_count_sequence(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    uploaded = FakeUploadedFile("synthetic.txt", b"synthetic upload")
    store = object()

    def loader(path: str | Path) -> list[Document]:
        calls.append("load")
        assert Path(path).read_bytes() == b"synthetic upload"
        return [Document(page_content="Synthetic document content.")]

    def splitter(documents: list[Document]) -> list[Document]:
        calls.append("split")
        assert documents[0].metadata["source"] == "synthetic.txt"
        assert documents[0].metadata["original_filename"] == "synthetic.txt"
        assert documents[0].metadata["document_id"]
        return list(documents)

    def store_factory(collection_name: str) -> object:
        calls.append("create")
        assert collection_name == "docuquery_test"
        return store

    def add_documents(target: object, documents: list[Document]) -> list[str]:
        calls.append("add")
        assert target is store
        assert documents[0].page_content == "Synthetic document content."
        return ["synthetic-id"]

    def count_documents(target: object) -> int:
        calls.append("count")
        assert target is store
        return 1

    result = app.index_uploaded_files(
        [uploaded],
        "docuquery_test",
        tmp_path,
        loader_fn=loader,
        splitter_fn=splitter,
        store_factory=store_factory,
        add_fn=add_documents,
        count_fn=count_documents,
    )

    assert calls == ["load", "split", "create", "add", "count"]
    assert result.chunks_indexed == 1
    assert result.collection_count == 1
    assert result.files_indexed == [
        {
            "source": "synthetic.txt",
            "document_id": result.files_indexed[0]["document_id"],
        }
    ]


def test_indexing_duplicate_names_batches_documents_without_mixing(
    tmp_path: Path,
) -> None:
    received: list[Document] = []

    def loader(path: str | Path) -> list[Document]:
        return [Document(page_content=Path(path).read_text(encoding="utf-8"))]

    def splitter(documents: list[Document]) -> list[Document]:
        return list(documents)

    result = app.index_uploaded_files(
        [
            FakeUploadedFile("same.txt", b"First document."),
            FakeUploadedFile("same.txt", b"Second document."),
        ],
        "docuquery_duplicates",
        tmp_path,
        loader_fn=loader,
        splitter_fn=splitter,
        store_factory=lambda _name: object(),
        add_fn=lambda _store, docs: received.extend(docs) or ["1", "2"],
        count_fn=lambda _store: 2,
    )

    assert len(received) == 2
    assert {document.page_content for document in received} == {
        "First document.",
        "Second document.",
    }
    assert {document.metadata["source"] for document in received} == {"same.txt"}
    assert len({document.metadata["document_id"] for document in received}) == 2
    assert len(result.files_indexed) == 2


def test_indexing_reports_individual_invalid_file_and_indexes_valid_files(
    tmp_path: Path,
) -> None:
    def loader(path: str | Path) -> list[Document]:
        if Path(path).suffix == ".pdf":
            raise EmptyDocumentError("synthetic empty PDF")
        return [Document(page_content="Valid synthetic text.")]

    result = app.index_uploaded_files(
        [
            FakeUploadedFile("bad.pdf", b"%PDF fake"),
            FakeUploadedFile("good.txt", b"Synthetic content."),
        ],
        "docuquery_partial",
        tmp_path,
        loader_fn=loader,
        splitter_fn=lambda documents: list(documents),
        store_factory=lambda _name: object(),
        add_fn=lambda _store, documents: ["id"] * len(documents),
        count_fn=lambda _store: 1,
    )

    assert result.chunks_indexed == 1
    assert result.files_indexed[0]["source"] == "good.txt"
    assert result.errors == [
        {
            "filename": "bad.pdf",
            "message": "bad.pdf: no extractable text was found.",
        }
    ]


def test_indexing_without_files_does_not_create_vector_store(
    tmp_path: Path,
) -> None:
    def unexpected_store(_name: str) -> Any:
        raise AssertionError("store must not be created without valid documents")

    result = app.index_uploaded_files(
        [FakeUploadedFile("empty.txt", b"")],
        "docuquery_empty",
        tmp_path,
        store_factory=unexpected_store,
    )

    assert result.chunks_indexed == 0
    assert result.errors


def test_indexing_rejects_empty_collection_after_chunk_creation(
    tmp_path: Path,
) -> None:
    with pytest.raises(app.IndexingError, match="No document chunks were found"):
        app.index_uploaded_files(
            [FakeUploadedFile("synthetic.txt", b"synthetic upload")],
            "docuquery_empty_result",
            tmp_path,
            loader_fn=lambda _path: [Document(page_content="Synthetic content.")],
            splitter_fn=lambda documents: list(documents),
            store_factory=lambda _name: object(),
            add_fn=lambda _store, _documents: ["synthetic-id"],
            count_fn=lambda _store: 0,
        )


def test_refresh_indexed_chunk_count_recovers_actual_collection_state() -> None:
    state: dict[str, Any] = {
        "collection_name": "docuquery_session",
        "indexed_chunk_count": 0,
        "indexed_files": [{"source": "synthetic.txt", "document_id": "doc-1"}],
    }
    store = object()

    count = app.refresh_indexed_chunk_count(
        state,
        store_factory=lambda collection_name: (
            store if collection_name == "docuquery_session" else None
        ),
        count_fn=lambda candidate: 3 if candidate is store else 0,
    )

    assert count == 3
    assert state["indexed_chunk_count"] == 3


def test_refresh_empty_collection_clears_stale_index_and_chat() -> None:
    state: dict[str, Any] = {
        "collection_name": "docuquery_session",
        "indexed_chunk_count": 4,
        "indexed_files": [{"source": "synthetic.txt", "document_id": "doc-1"}],
        "chat_history": [{"question": "Old?", "answer": "Old.", "sources": []}],
    }

    count = app.refresh_indexed_chunk_count(
        state,
        store_factory=lambda _name: object(),
        count_fn=lambda _store: 0,
    )

    assert count == 0
    assert state["indexed_chunk_count"] == 0
    assert state["indexed_files"] == []
    assert state["chat_history"] == []


def test_fresh_unindexed_state_does_not_open_a_vector_collection() -> None:
    state: dict[str, Any] = {
        "collection_name": "docuquery_session",
        "indexed_chunk_count": 0,
        "indexed_files": [],
    }

    def unexpected_store(_name: str) -> Any:
        raise AssertionError("fresh unindexed session must not open a collection")

    assert app.refresh_indexed_chunk_count(
        state,
        store_factory=unexpected_store,
    ) == 0


def test_recording_success_enables_chat_from_actual_collection_count() -> None:
    state: dict[str, Any] = {"indexed_chunk_count": 0, "indexed_files": []}
    result = app.IndexResult(
        chunks_indexed=2,
        collection_count=2,
        files_indexed=[
            {"source": "synthetic.txt", "document_id": "doc-1"},
        ],
        errors=[],
    )

    assert app.record_indexing_result(state, result)
    assert state["indexed_chunk_count"] == 2
    assert not app.chat_input_is_disabled(state["indexed_chunk_count"])


def test_zero_count_keeps_chat_disabled_after_failed_indexing() -> None:
    state: dict[str, Any] = {"indexed_chunk_count": 0, "indexed_files": []}
    result = app.IndexResult(
        chunks_indexed=1,
        collection_count=0,
        files_indexed=[{"source": "synthetic.txt", "document_id": "doc-1"}],
        errors=[],
    )

    assert not app.record_indexing_result(state, result)
    assert app.chat_input_is_disabled(state["indexed_chunk_count"])


def test_chat_history_append_and_clear() -> None:
    state: dict[str, Any] = {"chat_history": []}
    sources = [{"source_id": "S1", "source": "synthetic.txt"}]

    app.append_chat_turn(state, "Synthetic question?", "Synthetic answer.", sources, 1)
    sources[0]["source"] = "mutated.txt"
    app.clear_chat_history(state)

    assert state["chat_history"] == []


def test_indexed_file_merge_replaces_document_id_not_duplicate_filename() -> None:
    existing = [
        {"source": "duplicate.txt", "document_id": "copy-1"},
        {"source": "duplicate.txt", "document_id": "copy-2"},
    ]
    updated = [
        {"source": "duplicate.txt", "document_id": "copy-1"},
        {"source": "new.txt", "document_id": "new-file"},
    ]

    merged = app.merge_indexed_files(existing, updated)

    assert merged == [
        {"source": "duplicate.txt", "document_id": "copy-2"},
        {"source": "duplicate.txt", "document_id": "copy-1"},
        {"source": "new.txt", "document_id": "new-file"},
    ]


def test_chat_turn_stores_plain_response_data() -> None:
    state: dict[str, Any] = {}
    app.append_chat_turn(
        state,
        "Synthetic question?",
        "Synthetic answer.",
        [{"source_id": "S1", "source": "synthetic.txt"}],
        1,
    )

    assert state["chat_history"] == [
        {
            "question": "Synthetic question?",
            "answer": "Synthetic answer.",
            "sources": [{"source_id": "S1", "source": "synthetic.txt"}],
            "retrieved_count": 1,
        }
    ]


def test_citation_display_metadata_omits_txt_pages() -> None:
    details = app.citation_display_metadata(
        {
            "source_id": "S1",
            "source": "synthetic.txt",
            "page": 99,
            "chunk_indexes": [0, 1],
        }
    )

    assert details == {
        "source_id": "S1",
        "source": "synthetic.txt",
        "chunk_indexes": [0, 1],
    }


def test_citation_display_metadata_keeps_pdf_page_and_chunks() -> None:
    details = app.citation_display_metadata(
        {
            "source_id": "S2",
            "source": "synthetic.pdf",
            "page": 4,
            "page_label": "A-4",
            "chunk_indexes": [2, 3],
        }
    )

    assert details["page"] == 4
    assert details["page_label"] == "A-4"
    assert details["chunk_indexes"] == [2, 3]


def test_not_found_answer_suppresses_citation_display() -> None:
    sources = [{"source_id": "S1", "source": "synthetic.pdf", "page": 2}]

    assert app.citations_for_answer_display(
        app.NOT_FOUND_ANSWER,
        sources,
    ) == []


def test_supported_answer_keeps_citation_display() -> None:
    sources = [{"source_id": "S1", "source": "synthetic.pdf", "page": 2}]

    assert app.citations_for_answer_display(
        "Supported synthetic answer. [S1]",
        sources,
    ) == sources


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MissingAPIKeyError("secret key text"), "GEMINI_API_KEY"),
        (EmptyQuestionError("raw prompt contents"), "Enter a question"),
        (RAGQuotaError("private quota response"), "quota or rate limit"),
    ],
)
def test_errors_are_mapped_without_exposing_raw_contents(
    error: Exception,
    expected: str,
) -> None:
    message = app.safe_error_message(error)

    assert expected.casefold() in message.casefold()
    assert "secret key text" not in message
    assert "raw prompt contents" not in message
    assert "private quota response" not in message


def test_reset_clears_only_session_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state: dict[str, Any] = {}
    app.initialize_session_state(state, storage_root=tmp_path)
    upload_directory = Path(state["upload_directory"])
    (upload_directory / "sample.txt").write_text("synthetic", encoding="utf-8")
    cleared: list[str] = []
    monkeypatch.setattr(app, "clear_collection", cleared.append)

    app.reset_session_controls(state, storage_root=tmp_path)

    assert cleared and cleared[0].startswith("docuquery_")
    assert not upload_directory.exists()
    assert "session_id" not in state
    assert state == {}

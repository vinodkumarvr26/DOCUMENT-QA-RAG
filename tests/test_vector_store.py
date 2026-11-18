"""Offline persistence and retrieval tests using deterministic fake vectors."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from tests.fakes import DeterministicFakeEmbeddings
from vector_store import (
    VectorStoreConfigurationError,
    add_documents_to_store,
    clear_collection,
    create_vector_store,
    get_collection_count,
    get_vector_store,
    list_collections,
    similarity_search,
)


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbeddings:
    return DeterministicFakeEmbeddings(dimensions=768)


def _document(
    text: str,
    source: str,
    *,
    page_index: int = 0,
    page_label: str = "1",
    chunk_index: int = 0,
    start_index: int = 0,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "page_index": page_index,
            "page_label": page_label,
            "chunk_index": chunk_index,
            "start_index": start_index,
        },
    )


def test_create_local_collection_and_directory(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    persistence = tmp_path / "created" / "chroma"

    store = create_vector_store(
        "local_docs",
        embedding_function=fake_embeddings,
        persist_directory=persistence,
    )

    assert persistence.is_dir()
    assert store._collection.name == "local_docs"
    assert list_collections(persist_directory=persistence) == ["local_docs"]


def test_add_multiple_documents_and_preserve_metadata_and_original_text(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "metadata_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    documents = [
        _document(
            "Synthetic Python language reference.",
            "language.pdf",
            page_index=1,
            page_label="A-2",
            chunk_index=3,
            start_index=45,
        ),
        _document(
            "Synthetic astronomy reference about stars.",
            "stars.txt",
        ),
    ]

    add_documents_to_store(store, documents)

    stored = store.get(include=["documents", "metadatas"])
    assert get_collection_count(store) == 2
    rows = {
        metadata["source"]: (text, metadata)
        for text, metadata in zip(
            stored["documents"], stored["metadatas"], strict=True
        )
    }
    assert rows["language.pdf"][0] == "Synthetic Python language reference."
    assert rows["language.pdf"][1]["page_index"] == 1
    assert rows["language.pdf"][1]["page_label"] == "A-2"
    assert rows["language.pdf"][1]["chunk_index"] == 3
    assert rows["language.pdf"][1]["start_index"] == 45
    assert rows["stars.txt"][0] == "Synthetic astronomy reference about stars."


def test_similarity_search_uses_fake_embeddings_and_configurable_top_k(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "search_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    add_documents_to_store(
        store,
        [
            _document(
                "Python programming language syntax and code.",
                "python.txt",
            ),
            _document(
                "Ocean currents and marine biology.",
                "ocean.txt",
            ),
            _document(
                "Python software development and programming.",
                "software.txt",
            ),
        ],
    )

    results = similarity_search(store, "Python programming", top_k=2)

    assert len(results) == 2
    assert results[0].metadata["source"] in {"python.txt", "software.txt"}
    assert all(result.metadata["source"] != "ocean.txt" for result in results)


def test_collection_count_and_persistence_after_reopen(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    first_store = create_vector_store(
        "persistent_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    add_documents_to_store(
        first_store,
        [_document("Synthetic persistence example.", "persist.txt")],
    )
    expected_count = get_collection_count(first_store)

    reopened = get_vector_store(
        "persistent_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )

    assert expected_count == 1
    assert get_collection_count(reopened) == 1
    assert similarity_search(
        reopened, "persistence example", top_k=1
    )[0].page_content == "Synthetic persistence example."


def test_reindexing_identical_content_does_not_duplicate_vectors(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "dedupe_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    document = _document("Stable synthetic chunk.", "stable.txt", chunk_index=2)

    first_ids = add_documents_to_store(store, [document])
    second_ids = add_documents_to_store(store, [document])

    assert first_ids == second_ids
    assert get_collection_count(store) == 1


def test_updating_source_replaces_its_old_chunks_only(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "update_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    add_documents_to_store(
        store,
        [
            _document("Old synthetic source text.", "replace.txt"),
            _document("Keep this unrelated source.", "keep.txt"),
        ],
    )

    add_documents_to_store(
        store,
        [_document("Updated synthetic source text.", "replace.txt")],
    )

    assert get_collection_count(store) == 2
    stored = store.get(include=["documents", "metadatas"])
    rows = {
        metadata["source"]: text
        for text, metadata in zip(
            stored["documents"], stored["metadatas"], strict=True
        )
    }
    assert rows == {
        "replace.txt": "Updated synthetic source text.",
        "keep.txt": "Keep this unrelated source.",
    }


def test_duplicate_source_names_can_be_updated_independently(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "duplicate_sources",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    add_documents_to_store(
        store,
        [
            Document(
                page_content="First original duplicate.",
                metadata={"source": "copy.txt", "document_id": "doc-first"},
            ),
            Document(
                page_content="Second duplicate to preserve.",
                metadata={"source": "copy.txt", "document_id": "doc-second"},
            ),
        ],
    )

    add_documents_to_store(
        store,
        [
            Document(
                page_content="Updated first duplicate.",
                metadata={"source": "copy.txt", "document_id": "doc-first"},
            )
        ],
    )

    stored = store.get(include=["documents", "metadatas"])
    rows = {
        metadata["document_id"]: text
        for text, metadata in zip(
            stored["documents"], stored["metadatas"], strict=True
        )
    }
    assert rows == {
        "doc-first": "Updated first duplicate.",
        "doc-second": "Second duplicate to preserve.",
    }


def test_clear_only_the_named_collection(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    for collection_name in ("clear_me", "keep_me"):
        store = create_vector_store(
            collection_name,
            embedding_function=fake_embeddings,
            persist_directory=tmp_path,
        )
        add_documents_to_store(
            store,
            [_document(f"Data for {collection_name}.", f"{collection_name}.txt")],
        )

    clear_collection(
        "clear_me",
        persist_directory=tmp_path,
    )

    assert list_collections(persist_directory=tmp_path) == ["keep_me"]
    kept = get_vector_store(
        "keep_me",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    assert get_collection_count(kept) == 1


def test_empty_document_list_is_a_safe_no_op(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "empty_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )

    assert add_documents_to_store(store, []) == []
    assert get_collection_count(store) == 0


def test_non_document_input_is_rejected(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "invalid_docs",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )

    with pytest.raises(TypeError, match="sequence of LangChain Document"):
        add_documents_to_store(store, None)  # type: ignore[arg-type]


def test_session_specific_collections_are_isolated(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    for name in ("session_alpha", "session_beta"):
        store = create_vector_store(
            name,
            embedding_function=fake_embeddings,
            persist_directory=tmp_path,
        )
        add_documents_to_store(
            store,
            [_document(f"Private synthetic data for {name}.", f"{name}.txt")],
        )

    alpha = get_vector_store(
        "session_alpha",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    beta = get_vector_store(
        "session_beta",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )

    assert get_collection_count(alpha) == 1
    assert get_collection_count(beta) == 1
    assert similarity_search(alpha, "session alpha", top_k=1)[0].metadata[
        "source"
    ] == "session_alpha.txt"


@pytest.mark.parametrize("name", ["", "   "])
def test_invalid_collection_name_is_rejected(name: str, tmp_path: Path) -> None:
    with pytest.raises(VectorStoreConfigurationError, match="must not be empty"):
        create_vector_store(
            name,
            embedding_function=DeterministicFakeEmbeddings(),
            persist_directory=tmp_path,
        )


def test_invalid_top_k_is_rejected(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "bad_top_k",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        similarity_search(store, "synthetic query", top_k=0)


def test_non_finite_chroma_metadata_is_rejected(
    tmp_path: Path,
    fake_embeddings: DeterministicFakeEmbeddings,
) -> None:
    store = create_vector_store(
        "bad_metadata",
        embedding_function=fake_embeddings,
        persist_directory=tmp_path,
    )
    document = Document(
        page_content="Synthetic metadata validation.",
        metadata={"source": "bad.txt", "score": float("nan")},
    )

    with pytest.raises(VectorStoreConfigurationError, match="not supported"):
        add_documents_to_store(store, [document])

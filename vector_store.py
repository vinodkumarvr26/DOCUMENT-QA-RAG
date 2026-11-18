"""Persistent ChromaDB vector storage and similarity retrieval."""

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import get_settings
from embeddings import (
    TaskSpecificEmbeddingFunction,
    create_embedding_model,
    embed_documents,
)


DEFAULT_COLLECTION_NAME = "document_chunks"
_IDENTITY_METADATA_KEYS = (
    "source",
    "page_index",
    "page_label",
    "chunk_index",
    "start_index",
    "document_id",
    "document_title",
    "title",
)


class VectorStoreError(Exception):
    """Base exception for vector store failures."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector-store settings or metadata are invalid."""


def _resolve_persistence_directory(
    persist_directory: str | Path | None = None,
) -> Path:
    path = (
        Path(persist_directory)
        if persist_directory is not None
        else get_settings().chroma_persist_directory
    )
    if path.exists() and not path.is_dir():
        raise VectorStoreConfigurationError(
            f"Chroma persistence path is not a directory: {path}"
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise VectorStoreConfigurationError(
            f"Could not create Chroma persistence directory: {path}"
        ) from None
    return path


def _validate_collection_name(collection_name: str) -> None:
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise VectorStoreConfigurationError("Collection name must not be empty.")


def _as_provider(embedding_function: Embeddings | None) -> Embeddings:
    if embedding_function is None:
        return create_embedding_model()
    if isinstance(embedding_function, TaskSpecificEmbeddingFunction):
        return embedding_function.provider
    return embedding_function


def create_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    embedding_function: Embeddings | None = None,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Create or open a persistent, isolated Chroma collection.

    Callers can supply a session-specific ``collection_name`` to keep separate
    users' uploaded documents in different collections.
    """
    _validate_collection_name(collection_name)
    directory = _resolve_persistence_directory(persist_directory)
    provider = _as_provider(embedding_function)
    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=TaskSpecificEmbeddingFunction(provider),
            persist_directory=str(directory),
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise VectorStoreConfigurationError(
            f"Could not open Chroma collection {collection_name!r}: "
            f"{type(exc).__name__}."
        ) from None


def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    embedding_function: Embeddings | None = None,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Reopen an existing persistent collection or create it if not present."""
    return create_vector_store(
        collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )


def _chunk_id(document: Document) -> str:
    identity = {
        key: document.metadata[key]
        for key in _IDENTITY_METADATA_KEYS
        if key in document.metadata
    }
    identity["page_content"] = document.page_content
    serialized = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_metadata(documents: Sequence[Document]) -> list[dict[str, Any] | None]:
    allowed_scalar = (str, int, float, bool)
    metadata_rows: list[dict[str, Any] | None] = []
    for document in documents:
        metadata = document.metadata
        if not metadata:
            metadata_rows.append(None)
            continue
        if not all(isinstance(key, str) for key in metadata):
            raise VectorStoreConfigurationError(
                "Chroma metadata keys must all be strings."
            )
        clean: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, list) and all(
                isinstance(item, allowed_scalar)
                and not (isinstance(item, float) and not math.isfinite(item))
                for item in value
            ):
                clean[key] = value
            elif isinstance(value, allowed_scalar) and not (
                isinstance(value, float) and not math.isfinite(value)
            ):
                clean[key] = value
            else:
                raise VectorStoreConfigurationError(
                    f"Metadata value for {key!r} is not supported by Chroma."
                )
        metadata_rows.append(clean)
    return metadata_rows


def add_documents_to_store(
    vector_store: Chroma,
    documents: Sequence[Document],
    *,
    embedding_model: Embeddings | None = None,
) -> list[str]:
    """Upsert chunks and remove stale chunks for each re-indexed source.

    A batch containing a ``source`` is treated as the complete current set of
    chunks for that source. Other sources in the same collection are untouched.
    The original page content and metadata are stored; task-prefixed text is
    sent only to the embedding model.
    """
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence of LangChain Document objects.")
    if any(not isinstance(document, Document) for document in documents):
        raise TypeError("every item in documents must be a LangChain Document.")
    if not documents:
        return []
    if any(not document.page_content.strip() for document in documents):
        raise VectorStoreConfigurationError(
            "Cannot add empty document chunks to the vector store."
        )

    metadata_rows = _validate_metadata(documents)
    ids = [_chunk_id(document) for document in documents]
    if len(ids) != len(set(ids)):
        unique_documents: dict[str, tuple[Document, dict[str, Any] | None]] = {}
        for document, document_id, metadata in zip(
            documents, ids, metadata_rows, strict=True
        ):
            unique_documents[document_id] = (document, metadata)
        ids = list(unique_documents)
        documents = [item[0] for item in unique_documents.values()]
        metadata_rows = [item[1] for item in unique_documents.values()]

    provider = embedding_model or _as_provider(
        getattr(vector_store, "_embedding_function", None)
    )
    vectors = embed_documents(documents, embedding_model=provider)
    if len(vectors) != len(documents):
        raise VectorStoreConfigurationError(
            "The embedding provider returned an unexpected number of vectors."
        )

    source_ids: dict[tuple[str, str], set[str]] = {}
    for document in documents:
        document_id = document.metadata.get("document_id")
        source = document.metadata.get("source")
        if isinstance(document_id, str) and document_id:
            source_ids.setdefault(("document_id", document_id), set())
        elif isinstance(source, str):
            source_ids.setdefault(("source", source), set())
    for (metadata_key, metadata_value) in source_ids:
        existing = vector_store.get(
            where={metadata_key: metadata_value},
            include=["metadatas"],
        )
        source_ids[(metadata_key, metadata_value)] = set(existing["ids"])

    collection = vector_store._collection
    metadata_indexes = [
        index for index, metadata in enumerate(metadata_rows) if metadata
    ]
    if metadata_indexes:
        collection.upsert(
            ids=[ids[index] for index in metadata_indexes],
            embeddings=[vectors[index] for index in metadata_indexes],
            documents=[
                documents[index].page_content for index in metadata_indexes
            ],
            metadatas=[metadata_rows[index] for index in metadata_indexes],
        )
    metadata_free_indexes = [
        index for index, metadata in enumerate(metadata_rows) if not metadata
    ]
    if metadata_free_indexes:
        collection.upsert(
            ids=[ids[index] for index in metadata_free_indexes],
            embeddings=[vectors[index] for index in metadata_free_indexes],
            documents=[
                documents[index].page_content for index in metadata_free_indexes
            ],
        )

    desired_ids_by_source: dict[tuple[str, str], set[str]] = {}
    for document, document_id in zip(documents, ids, strict=True):
        source_identity = document.metadata.get("document_id")
        source = document.metadata.get("source")
        if isinstance(source_identity, str) and source_identity:
            desired_ids_by_source.setdefault(
                ("document_id", source_identity), set()
            ).add(document_id)
        elif isinstance(source, str):
            desired_ids_by_source.setdefault(("source", source), set()).add(document_id)
    for source_identity, existing_ids in source_ids.items():
        stale_ids = existing_ids - desired_ids_by_source.get(source_identity, set())
        if stale_ids:
            vector_store.delete(ids=sorted(stale_ids))
    return ids


def similarity_search(
    vector_store: Chroma,
    query: str,
    *,
    top_k: int | None = None,
) -> list[Document]:
    """Return the top semantically similar chunks for a non-empty query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")
    resolved_top_k = get_settings().retriever_top_k if top_k is None else top_k
    if isinstance(resolved_top_k, bool) or not isinstance(resolved_top_k, int):
        raise ValueError("top_k must be a positive integer.")
    if resolved_top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    try:
        return vector_store.similarity_search(query, k=resolved_top_k)
    except Exception as exc:
        raise VectorStoreError(
            f"Chroma similarity search failed: {type(exc).__name__}."
        ) from None


def get_collection_count(vector_store: Chroma) -> int:
    """Return the number of records in a Chroma collection."""
    try:
        return vector_store._collection.count()
    except Exception as exc:
        raise VectorStoreError(
            f"Could not count Chroma collection records: {type(exc).__name__}."
        ) from None


def clear_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    persist_directory: str | Path | None = None,
) -> None:
    """Delete only the named Chroma collection in the selected persistence path."""
    _validate_collection_name(collection_name)
    directory = _resolve_persistence_directory(persist_directory)
    try:
        client = chromadb.PersistentClient(path=str(directory))
        existing_names = {
            collection.name
            if hasattr(collection, "name")
            else str(collection)
            for collection in client.list_collections()
        }
        if collection_name in existing_names:
            client.delete_collection(name=collection_name)
    except Exception as exc:
        raise VectorStoreError(
            f"Could not clear Chroma collection {collection_name!r}: "
            f"{type(exc).__name__}."
        ) from None


def list_collections(
    *,
    persist_directory: str | Path | None = None,
) -> list[str]:
    """List collection names persisted in the configured local Chroma directory."""
    directory = _resolve_persistence_directory(persist_directory)
    try:
        client = chromadb.PersistentClient(path=str(directory))
        return sorted(
            collection.name
            if hasattr(collection, "name")
            else str(collection)
            for collection in client.list_collections()
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Could not list Chroma collections: {type(exc).__name__}."
        ) from None

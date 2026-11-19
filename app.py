"""DocuQuery's single-page Streamlit interface and testable UI helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import uuid

import streamlit as st
from langchain_core.documents import Document

from config import get_settings
from document_loader import (
    DocumentLoadingError,
    DocumentNotFoundError,
    EmptyDocumentError,
    EncryptedPDFError,
    InvalidDocumentError,
    UnsupportedDocumentError,
    load_document,
)
from embeddings import (
    EmbeddingAPIUnavailableError,
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingQuotaError,
    MissingAPIKeyError as MissingEmbeddingAPIKeyError,
)
from rag_chain import (
    EmptyModelResponseError,
    EmptyQuestionError,
    MissingAPIKeyError,
    MissingVectorCollectionError,
    NOT_FOUND_ANSWER,
    RAGAuthenticationError,
    RAGError,
    RAGQuotaError,
    RAGServiceError,
    RAGServiceUnavailableError,
    answer_question,
)
from text_splitter import split_documents
from vector_store import (
    VectorStoreConfigurationError,
    VectorStoreError,
    add_documents_to_store,
    clear_collection,
    create_vector_store,
    get_collection_count,
)


APPLICATION_NAME = "DocuQuery — Document Q&A Assistant"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
_SESSION_PREFIX = "docuquery_"
_SESSION_STORAGE_ROOT = Path(tempfile.gettempdir()) / "docuquery_uploads"
_STATE_DEFAULTS: dict[str, Any] = {
    "chat_history": [],
    "indexed_chunk_count": 0,
    "indexed_files": [],
}
_INDEX_NOTICE_KEY = "indexing_notice"


class UploadValidationError(Exception):
    """Raised when an uploaded file is empty, unsafe, or unsupported."""


class IndexingError(Exception):
    """Raised when supported documents cannot be indexed."""


@dataclass(frozen=True)
class SavedUpload:
    """Safe on-disk upload path and citation-preserving source identity."""

    path: Path
    original_filename: str
    document_id: str


@dataclass(frozen=True)
class IndexResult:
    """Summary of a completed indexing attempt."""

    chunks_indexed: int
    collection_count: int
    files_indexed: list[dict[str, str]]
    errors: list[dict[str, str]]


def collection_name_for_session(session_id: str) -> str:
    """Return a Chroma-valid, deterministic collection name for a session ID."""
    try:
        normalized_id = uuid.UUID(str(session_id)).hex
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("A valid UUID session identifier is required.") from exc
    return f"{_SESSION_PREFIX}{normalized_id}"


def initialize_session_state(
    state: MutableMapping[str, Any],
    *,
    storage_root: str | Path | None = None,
    session_id_factory: Callable[[], uuid.UUID] | None = None,
) -> None:
    """Initialize stable per-session collection, storage, and chat state."""
    if "session_id" not in state:
        factory = session_id_factory or uuid.uuid4
        state["session_id"] = factory().hex
    if "collection_name" not in state:
        state["collection_name"] = collection_name_for_session(state["session_id"])

    root = Path(storage_root) if storage_root is not None else _SESSION_STORAGE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if "upload_directory" not in state:
        session_directory = root / str(state["session_id"])
        if session_directory.resolve().parent != root.resolve():
            raise ValueError("Session upload directory is outside its storage root.")
        session_directory.mkdir(parents=True, exist_ok=True)
        state["upload_directory"] = str(session_directory)

    for key, default in _STATE_DEFAULTS.items():
        if key not in state:
            state[key] = list(default) if isinstance(default, list) else default


def safe_filename(filename: str) -> str:
    """Reduce an uploaded name to a safe basename while preserving its filename."""
    if not isinstance(filename, str):
        raise UploadValidationError("The uploaded file has an invalid filename.")
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    basename = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", basename)
    if basename in {"", ".", ".."}:
        raise UploadValidationError("The uploaded file has an invalid filename.")
    return basename[:180]


def validate_uploaded_file(uploaded_file: Any) -> tuple[str, bytes]:
    """Validate an upload's safe filename, extension, and non-empty content."""
    try:
        filename = safe_filename(uploaded_file.name)
    except AttributeError as exc:
        raise UploadValidationError("The uploaded file has no filename.") from exc
    extension = Path(filename).suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(
            f"{filename}: only PDF and TXT documents are supported."
        )
    try:
        content = uploaded_file.getvalue()
    except Exception as exc:
        raise UploadValidationError(
            f"{filename}: the uploaded file could not be read."
        ) from exc
    if not isinstance(content, bytes) or not content:
        raise UploadValidationError(f"{filename}: the uploaded file is empty.")
    return filename, content


def save_uploaded_file(
    uploaded_file: Any,
    upload_directory: str | Path,
    *,
    duplicate_index: int = 0,
) -> SavedUpload:
    """Write upload bytes to a generated path and retain original source naming."""
    filename, content = validate_uploaded_file(uploaded_file)
    if isinstance(duplicate_index, bool) or duplicate_index < 0:
        raise ValueError("duplicate_index must be a non-negative integer.")

    directory = Path(upload_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    identity_material = f"{filename.casefold()}\0{duplicate_index}".encode("utf-8")
    document_id = hashlib.sha256(identity_material).hexdigest()
    content_digest = hashlib.sha256(content).hexdigest()
    extension = Path(filename).suffix.casefold()
    destination = (directory / f"{document_id}_{content_digest}{extension}").resolve()
    if destination.parent != directory:
        raise UploadValidationError("The uploaded file path is invalid.")
    destination.write_bytes(content)
    return SavedUpload(
        path=destination,
        original_filename=filename,
        document_id=document_id,
    )


def _upload_error_message(error: Exception, filename: str) -> str:
    if isinstance(error, UploadValidationError):
        return str(error)
    if isinstance(error, EncryptedPDFError):
        return f"{filename}: encrypted PDFs are not supported."
    if isinstance(error, EmptyDocumentError):
        return f"{filename}: no extractable text was found."
    if isinstance(error, UnsupportedDocumentError):
        return f"{filename}: only PDF and TXT documents are supported."
    if isinstance(error, DocumentNotFoundError):
        return f"{filename}: the uploaded file could not be accessed."
    if isinstance(error, InvalidDocumentError):
        return f"{filename}: the document is invalid or could not be read."
    if isinstance(error, DocumentLoadingError):
        return f"{filename}: the document could not be loaded."
    return f"{filename}: the document could not be processed."


def index_uploaded_files(
    uploaded_files: Sequence[Any],
    collection_name: str,
    upload_directory: str | Path,
    *,
    loader_fn: Callable[[str | Path], list[Document]] | None = None,
    splitter_fn: Callable[[Sequence[Document]], list[Document]] | None = None,
    store_factory: Callable[[str], Any] | None = None,
    add_fn: Callable[[Any, Sequence[Document]], list[str]] | None = None,
    count_fn: Callable[[Any], int] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> IndexResult:
    """Load, split, embed, and store submitted files as one explicit operation.

    Invalid files are reported individually. Valid files are batched into one
    add call so duplicate filenames in this upload do not evict each other's
    chunks.
    """
    loader = loader_fn or load_document
    splitter = splitter_fn or split_documents
    create_store = store_factory or create_vector_store
    add_documents = add_fn or add_documents_to_store
    count_documents = count_fn or get_collection_count

    errors: list[dict[str, str]] = []
    loaded_documents: list[Document] = []
    indexed_files: dict[str, dict[str, str]] = {}
    seen_names: Counter[str] = Counter()
    total_files = len(uploaded_files)

    for position, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            source_name = safe_filename(uploaded_file.name)
        except (AttributeError, UploadValidationError):
            source_name = "Uploaded file"
        duplicate_index = seen_names[source_name.casefold()]
        seen_names[source_name.casefold()] += 1
        if progress_callback:
            progress_callback(position, total_files, source_name)

        try:
            saved = save_uploaded_file(
                uploaded_file,
                upload_directory,
                duplicate_index=duplicate_index,
            )
            documents = loader(saved.path)
            if not documents:
                raise EmptyDocumentError("No document content was extracted.")
            for document in documents:
                document.metadata["source"] = saved.original_filename
                document.metadata["original_filename"] = saved.original_filename
                document.metadata["document_id"] = saved.document_id
                loaded_documents.append(document)
            indexed_files[saved.document_id] = {
                "source": saved.original_filename,
                "document_id": saved.document_id,
            }
        except Exception as exc:
            errors.append(
                {
                    "filename": source_name,
                    "message": _upload_error_message(exc, source_name),
                }
            )

    if not loaded_documents:
        return IndexResult(
            chunks_indexed=0,
            collection_count=0,
            files_indexed=[],
            errors=errors,
        )

    try:
        chunks = splitter(loaded_documents)
        if not chunks:
            if not errors:
                errors.append(
                    {
                        "filename": "Uploaded documents",
                        "message": "No non-empty text chunks were produced.",
                    }
                )
            return IndexResult(0, 0, [], errors)
        vector_store = create_store(collection_name)
        add_documents(vector_store, chunks)
        collection_count = count_documents(vector_store)
        if collection_count <= 0:
            raise IndexingError(
                "No document chunks were found in the session collection after "
                "indexing. Please try indexing the documents again."
            )
    except IndexingError:
        raise
    except (MissingEmbeddingAPIKeyError, EmbeddingAPIUnavailableError,
            EmbeddingAuthenticationError, EmbeddingQuotaError,
            VectorStoreConfigurationError, VectorStoreError) as exc:
        raise IndexingError(_safe_indexing_error(exc)) from None
    except Exception:
        raise IndexingError(
            "The documents could not be indexed. Check the API configuration "
            "and local vector-store availability."
        ) from None

    return IndexResult(
        chunks_indexed=len(chunks),
        collection_count=collection_count,
        files_indexed=list(indexed_files.values()),
        errors=errors,
    )


def refresh_indexed_chunk_count(
    state: MutableMapping[str, Any],
    *,
    store_factory: Callable[[str], Any] | None = None,
    count_fn: Callable[[Any], int] | None = None,
) -> int:
    """Reconcile known indexed state with the actual current session collection."""
    collection_name = state.get("collection_name")
    if not isinstance(collection_name, str) or not collection_name:
        state["indexed_chunk_count"] = 0
        return 0

    if not state.get("indexed_chunk_count") and not state.get("indexed_files"):
        return 0

    try:
        vector_store = (store_factory or create_vector_store)(collection_name)
        actual_count = (count_fn or get_collection_count)(vector_store)
    except Exception as exc:
        state["indexed_chunk_count"] = 0
        raise IndexingError(
            "Could not verify this session's indexed documents. Re-index your "
            "files before asking questions."
        ) from exc

    state["indexed_chunk_count"] = max(0, actual_count)
    if actual_count <= 0:
        state["indexed_files"] = []
        clear_chat_history(state)
    return state["indexed_chunk_count"]


def record_indexing_result(
    state: MutableMapping[str, Any],
    result: IndexResult,
) -> bool:
    """Update UI index state only when the collection contains stored chunks."""
    if result.chunks_indexed <= 0 or result.collection_count <= 0:
        return False
    state["indexed_chunk_count"] = result.collection_count
    state["indexed_files"] = merge_indexed_files(
        state.get("indexed_files", []),
        result.files_indexed,
    )
    return True


def chat_input_is_disabled(indexed_chunk_count: int) -> bool:
    """Enable chat only when the current session collection has stored chunks."""
    return indexed_chunk_count <= 0


def _safe_indexing_error(error: Exception) -> str:
    if isinstance(error, MissingEmbeddingAPIKeyError):
        return "Gemini API key is missing. Configure GEMINI_API_KEY before indexing."
    if isinstance(error, EmbeddingQuotaError):
        return "Gemini embedding quota or rate limit was reached. Try again later."
    if isinstance(error, EmbeddingAuthenticationError):
        return "Gemini rejected the configured credentials or project access."
    if isinstance(error, EmbeddingAPIUnavailableError):
        return "Gemini embeddings are unavailable. Check network access and try again."
    return "Could not initialize or update the session's document collection."


def citation_display_metadata(citation: MutableMapping[str, Any]) -> dict[str, Any]:
    """Select safe, human-readable source details for the interface."""
    source = citation.get("source")
    result: dict[str, Any] = {
        "source_id": str(citation.get("source_id", "")),
        "source": source if isinstance(source, str) else "Unknown source",
    }
    page = citation.get("page")
    if (
        isinstance(page, int)
        and not isinstance(page, bool)
        and page > 0
        and not result["source"].casefold().endswith(".txt")
    ):
        result["page"] = page
    page_label = citation.get("page_label")
    if isinstance(page_label, str) and page_label:
        result["page_label"] = page_label
    chunk_indexes = citation.get("chunk_indexes")
    if isinstance(chunk_indexes, list):
        result["chunk_indexes"] = [
            value
            for value in chunk_indexes
            if isinstance(value, int) and not isinstance(value, bool)
        ]
    elif isinstance(citation.get("chunk_index"), int):
        result["chunk_indexes"] = [citation["chunk_index"]]
    return result


def append_chat_turn(
    state: MutableMapping[str, Any],
    question: str,
    answer: str,
    sources: Sequence[MutableMapping[str, Any]],
    retrieved_count: int,
) -> None:
    """Append one completed Q&A turn without retaining model/provider objects."""
    state.setdefault("chat_history", []).append(
        {
            "question": question,
            "answer": answer,
            "sources": [dict(source) for source in sources],
            "retrieved_count": retrieved_count,
        }
    )


def citations_for_answer_display(
    answer: str,
    sources: Sequence[MutableMapping[str, Any]],
) -> list[MutableMapping[str, Any]]:
    """Do not present retrieved sources as evidence for the not-found fallback."""
    if answer.strip() == NOT_FOUND_ANSWER:
        return []
    return list(sources)


def merge_indexed_files(
    existing_files: Sequence[MutableMapping[str, str]],
    updated_files: Sequence[MutableMapping[str, str]],
) -> list[dict[str, str]]:
    """Replace only previously indexed uploads with matching document IDs."""
    updated_ids = {
        item["document_id"]
        for item in updated_files
        if isinstance(item.get("document_id"), str)
    }
    merged = [
        dict(item)
        for item in existing_files
        if item.get("document_id") not in updated_ids
    ]
    merged.extend(dict(item) for item in updated_files)
    return merged


def clear_chat_history(state: MutableMapping[str, Any]) -> None:
    """Clear only the current session's displayed conversation."""
    state["chat_history"] = []


def safe_error_message(error: Exception) -> str:
    """Map known failures to UI-safe text without showing raw exceptions."""
    if isinstance(error, UploadValidationError):
        return str(error)
    if isinstance(error, EncryptedPDFError):
        return "Encrypted PDFs are not supported."
    if isinstance(error, EmptyDocumentError):
        return "The document is empty or contains no extractable text."
    if isinstance(error, UnsupportedDocumentError):
        return "Only PDF and TXT documents are supported."
    if isinstance(error, InvalidDocumentError):
        return "The document is invalid or could not be read."
    if isinstance(error, EmptyQuestionError):
        return "Enter a question before sending."
    if isinstance(error, (MissingAPIKeyError, MissingEmbeddingAPIKeyError)):
        return "Gemini API key is not configured. Set GEMINI_API_KEY to use this feature."
    if isinstance(error, RAGQuotaError):
        return "Gemini quota or rate limit reached. Try again later."
    if isinstance(error, EmbeddingQuotaError):
        return "Gemini embedding quota or rate limit reached. Try again later."
    if isinstance(error, (RAGAuthenticationError, EmbeddingAuthenticationError)):
        return "Gemini authentication failed. Check the API key and project permissions."
    if isinstance(error, (RAGServiceUnavailableError, EmbeddingAPIUnavailableError)):
        return "Gemini is unavailable or the network request timed out. Try again later."
    if isinstance(error, MissingVectorCollectionError):
        return "This session's document collection is unavailable. Re-index your documents."
    if isinstance(error, EmptyModelResponseError):
        return "Gemini returned no answer text. Please try again."
    if isinstance(error, RAGServiceError):
        return "Gemini could not complete the request. Please try again."
    if isinstance(error, RAGError):
        return "The question could not be answered. Please check the input and try again."
    if isinstance(error, DocumentLoadingError):
        return "The uploaded document could not be loaded."
    if isinstance(error, EmbeddingError):
        return "Gemini embeddings could not be created. Check configuration and try again."
    if isinstance(error, (VectorStoreError, VectorStoreConfigurationError, IndexingError)):
        return (
            error.args[0]
            if isinstance(error, IndexingError)
            else "The document collection could not be accessed."
        )
    return "The request could not be completed. Please try again."


def reset_session_controls(
    state: MutableMapping[str, Any],
    *,
    storage_root: str | Path | None = None,
) -> None:
    """Clear the current Chroma collection and remove only its upload directory."""
    collection_name = state.get("collection_name")
    if isinstance(collection_name, str):
        clear_collection(collection_name)

    root = Path(storage_root) if storage_root is not None else _SESSION_STORAGE_ROOT
    upload_directory = state.get("upload_directory")
    if isinstance(upload_directory, str):
        target = Path(upload_directory).resolve()
        if target.parent == root.resolve() and target.name == state.get("session_id"):
            if target.exists():
                shutil.rmtree(target)

    for key in (
        "session_id",
        "collection_name",
        "upload_directory",
        "chat_history",
        "indexed_chunk_count",
        "indexed_files",
    ):
        state.pop(key, None)


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --docuquery-background: #F7F9FF;
            --docuquery-ink: #253047;
            --docuquery-heading: #18223A;
            --docuquery-muted: #53627A;
            --docuquery-blue: #5669D8;
            --docuquery-sidebar: #202128;
            --docuquery-sidebar-text: #F1F3F8;
            --docuquery-card: #FFFFFF;
            --docuquery-border: #D7DEEA;
            --docuquery-disabled: #E5E9F2;
            --docuquery-disabled-text: #53627A;
        }
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
            background: var(--docuquery-background) !important;
            color: var(--docuquery-ink) !important;
            font-family: "Nunito", "Segoe UI", sans-serif;
            font-size: 15px;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1040px;
            padding-top: 2rem;
        }
        .stMain h1,
        .stMain h2,
        .stMain h3,
        .stMain h4 {
            color: var(--docuquery-heading) !important;
            letter-spacing: -0.025em;
        }
        h1 { font-size: 2.15rem; }
        .stMain p,
        .stMain label,
        .stMain li,
        .stMain [data-testid="stMarkdownContainer"],
        .stMain [data-testid="stCaptionContainer"] {
            color: var(--docuquery-ink) !important;
        }
        .stMain [data-testid="stCaptionContainer"] {
            color: var(--docuquery-muted) !important;
        }
        p, label, li, input, textarea, button { font-size: 15px !important; }
        .stMain button,
        .stMain [role="button"] {
            color: var(--docuquery-ink);
        }
        .stMain button[kind="primary"],
        .stMain div.stButton > button[kind="primary"] {
            color: #FFFFFF !important;
            background: var(--docuquery-blue) !important;
            border-color: var(--docuquery-blue) !important;
        }
        .stMain button:disabled,
        .stMain button[disabled],
        .stMain div.stButton > button[kind="primary"]:disabled {
            color: var(--docuquery-disabled-text) !important;
            background: var(--docuquery-disabled) !important;
            border-color: var(--docuquery-border) !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {
            color: var(--docuquery-ink) !important;
            background: var(--docuquery-card) !important;
            border-color: #9AA8C0 !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] button {
            color: var(--docuquery-ink) !important;
        }
        [data-testid="stAlert"] {
            color: var(--docuquery-ink) !important;
            background: var(--docuquery-card) !important;
            border: 1px solid var(--docuquery-border) !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
            color: var(--docuquery-ink) !important;
        }
        [data-testid="stChatMessage"] {
            color: var(--docuquery-ink) !important;
            border: 1px solid var(--docuquery-border) !important;
            border-radius: 14px;
            background: var(--docuquery-card) !important;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            color: var(--docuquery-ink) !important;
        }
        [data-testid="stExpander"] {
            color: var(--docuquery-ink) !important;
            background: var(--docuquery-card) !important;
            border-color: var(--docuquery-border) !important;
        }
        [data-testid="stSidebar"] {
            color: var(--docuquery-sidebar-text) !important;
            background: var(--docuquery-sidebar) !important;
        }
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: var(--docuquery-sidebar-text) !important;
        }
        [data-testid="stSidebar"] button {
            color: var(--docuquery-sidebar-text) !important;
            background: #30313A !important;
            border-color: #53545F !important;
        }
        [data-testid="stSidebar"] button:disabled {
            color: #B5BAC6 !important;
            background: #30313A !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"] {
            color: var(--docuquery-sidebar-text) !important;
            background: var(--docuquery-sidebar) !important;
            border: 1px solid #444651 !important;
            border-radius: 14px !important;
        }
        [data-testid="stChatInput"] > div:first-child {
            background: var(--docuquery-sidebar) !important;
            border-radius: 14px !important;
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input {
            color: var(--docuquery-sidebar-text) !important;
            -webkit-text-fill-color: var(--docuquery-sidebar-text) !important;
            background: transparent !important;
            caret-color: var(--docuquery-sidebar-text) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stChatInput"] input::placeholder {
            color: #C4CAD6 !important;
            -webkit-text-fill-color: #C4CAD6 !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:disabled {
            color: #C4CAD6 !important;
            background: #30313A !important;
            border-color: #53545F !important;
            opacity: 1 !important;
        }
        [data-testid="stBottomBlockContainer"] {
            background: linear-gradient(
                to bottom,
                rgba(247, 249, 255, 0),
                #F7F9FF 32%
            ) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_citations(sources: Sequence[MutableMapping[str, Any]]) -> None:
    for citation in sources:
        details = citation_display_metadata(citation)
        heading = f'{details["source_id"]} · {details["source"]}'
        with st.expander(heading):
            st.write(f'Source: {details["source"]}')
            if "page" in details:
                label = details.get("page_label")
                page_text = f'Page {details["page"]}'
                if label:
                    page_text += f' (label {label})'
                st.write(page_text)
            if details.get("chunk_indexes"):
                st.write("Chunk(s): " + ", ".join(map(str, details["chunk_indexes"])))


def main() -> None:
    """Render the single-page document upload, indexing, and Q&A application."""
    st.set_page_config(
        page_title=APPLICATION_NAME,
        page_icon="D",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    initialize_session_state(st.session_state)
    _render_styles()
    try:
        indexed_chunk_count = refresh_indexed_chunk_count(st.session_state)
    except IndexingError as exc:
        indexed_chunk_count = 0
        st.error(safe_error_message(exc))
    notice = st.session_state.pop(_INDEX_NOTICE_KEY, None)
    if isinstance(notice, dict):
        success_notice = notice.get("success")
        if isinstance(success_notice, str) and success_notice:
            st.success(success_notice)
        errors = notice.get("errors", [])
        if isinstance(errors, list):
            for message in errors:
                if isinstance(message, str):
                    st.warning(message)

    with st.sidebar:
        st.markdown("### Your workspace")
        st.caption("Documents and chat are isolated to this browser session.")
        st.caption(f"Session: {st.session_state.session_id[:8]}")
        st.metric("Indexed chunks", indexed_chunk_count)
        if st.button("Clear chat history", use_container_width=True):
            clear_chat_history(st.session_state)
            st.rerun()
        if st.button(
            "Clear indexed documents",
            disabled=indexed_chunk_count == 0,
            use_container_width=True,
        ):
            try:
                clear_collection(st.session_state.collection_name)
                st.session_state.indexed_chunk_count = 0
                st.session_state.indexed_files = []
                clear_chat_history(st.session_state)
                indexed_chunk_count = 0
                st.session_state[_INDEX_NOTICE_KEY] = {
                    "success": "This session's indexed documents were cleared.",
                    "errors": [],
                }
                st.rerun()
            except Exception as exc:
                st.error(safe_error_message(exc))
        if st.button("Reset session", use_container_width=True):
            try:
                reset_session_controls(st.session_state)
                st.rerun()
            except Exception as exc:
                st.error(safe_error_message(exc))
        st.divider()
        settings = get_settings()
        st.caption(f"Retrieval: top {settings.retriever_top_k}")
        st.caption("PDF and TXT · no OCR")

    st.title(APPLICATION_NAME)
    st.write(
        "Upload documents, ask questions and receive answers grounded in their "
        "content, with source references."
    )

    st.subheader("1. Add your documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploads_{st.session_state.session_id}",
        help="Text-based PDFs and UTF-8 text files are supported. Scanned PDFs are not.",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
        for item in uploaded_files:
            filename = safe_filename(item.name)
            st.write(f"• {filename}")
    else:
        st.caption("Select one or more PDF or TXT files to begin.")

    if st.button(
        "Index documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        status = st.status("Preparing documents…", expanded=True)
        try:
            result = index_uploaded_files(
                uploaded_files,
                st.session_state.collection_name,
                st.session_state.upload_directory,
                progress_callback=lambda current, total, name: status.update(
                    label=f"Processing {current} of {total}: {name}",
                    state="running",
                ),
            )
            if result.chunks_indexed:
                if record_indexing_result(st.session_state, result):
                    indexed_chunk_count = result.collection_count
                    status.update(
                        label=f"Indexed {result.chunks_indexed} chunks successfully.",
                        state="complete",
                    )
                    st.success(
                        f"Indexed {result.chunks_indexed} chunks from "
                        f"{len(result.files_indexed)} file(s)."
                    )
                    st.session_state[_INDEX_NOTICE_KEY] = {
                        "success": (
                            f"Indexed {result.chunks_indexed} chunks from "
                            f"{len(result.files_indexed)} file(s)."
                        ),
                        "errors": [
                            error["message"] for error in result.errors
                        ],
                    }
                    st.rerun()
                else:
                    indexed_chunk_count = refresh_indexed_chunk_count(
                        st.session_state
                    )
                    status.update(label="No documents were indexed.", state="error")
            else:
                status.update(label="No documents were indexed.", state="error")
            for error in result.errors:
                st.warning(error["message"])
        except Exception as exc:
            status.update(label="Indexing did not complete.", state="error")
            st.error(safe_error_message(exc))

    if st.session_state.indexed_files:
        with st.expander(
            f"Indexed sources ({len(st.session_state.indexed_files)})",
            expanded=False,
        ):
            counts = Counter(item["source"] for item in st.session_state.indexed_files)
            seen: Counter[str] = Counter()
            for item in st.session_state.indexed_files:
                source = item["source"]
                seen[source] += 1
                copy_suffix = (
                    f" · copy {seen[source]}"
                    if counts[source] > 1
                    else ""
                )
                st.write(f"• {source}{copy_suffix}")

    st.divider()
    st.subheader("2. Ask your documents")
    if indexed_chunk_count == 0:
        st.info(
            "Upload one or more PDF or TXT documents, then select **Index "
            "documents** to enable questions."
        )

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            display_sources = citations_for_answer_display(
                turn["answer"],
                turn["sources"],
            )
            if display_sources:
                st.markdown("**Sources**")
                _render_citations(display_sources)

    question = st.chat_input(
        "Ask a question about your indexed documents…",
        disabled=chat_input_is_disabled(indexed_chunk_count),
    )
    if question:
        with st.spinner("Searching your documents…"):
            try:
                result = answer_question(
                    question,
                    collection_name=st.session_state.collection_name,
                    top_k=get_settings().retriever_top_k,
                )
                append_chat_turn(
                    st.session_state,
                    question,
                    result["answer"],
                    result["sources"],
                    result["retrieved_count"],
                )
            except Exception as exc:
                append_chat_turn(
                    st.session_state,
                    question,
                    safe_error_message(exc),
                    [],
                    0,
                )
        st.rerun()


if __name__ == "__main__":
    main()

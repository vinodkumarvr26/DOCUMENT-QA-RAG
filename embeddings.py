"""Gemini Embedding 2 setup and task-specific embedding helpers."""

from collections.abc import Callable, Sequence
import math
from numbers import Real
from typing import NoReturn, TypeVar

from google.genai import types
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import get_settings


class EmbeddingError(Exception):
    """Base exception for embedding configuration and service failures."""


class MissingAPIKeyError(EmbeddingError):
    """Raised when a Gemini API key is not configured."""


class UnsupportedEmbeddingModelError(EmbeddingError):
    """Raised when the configured model is not Gemini Embedding 2."""


class EmptyEmbeddingInputError(EmbeddingError):
    """Raised when document or query text is empty."""


class InvalidEmbeddingResponseError(EmbeddingError):
    """Raised when an embedding provider returns invalid vectors."""


class EmbeddingServiceError(EmbeddingError):
    """Raised when the embedding service cannot complete a request."""


class EmbeddingQuotaError(EmbeddingServiceError):
    """Raised when the embedding service reports a quota or rate limit."""


class EmbeddingAPIUnavailableError(EmbeddingServiceError):
    """Raised when the embedding service cannot be reached."""


class EmbeddingAuthenticationError(EmbeddingServiceError):
    """Raised when the embedding service rejects credentials or access."""


class GeminiEmbedding2Embeddings(GoogleGenerativeAIEmbeddings):
    """Use the Google integration client without its legacy task_type defaults."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed prepared documents without setting a legacy task type."""
        vectors: list[list[float]] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.models.embed_content(
                model=self.model,
                contents=[
                    {"parts": [{"text": text}]}
                    for text in batch
                ],
                config=types.EmbedContentConfig(
                    output_dimensionality=self.output_dimensionality
                ),
            )
            vectors.extend(
                [list(embedding.values) for embedding in response.embeddings or []]
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a prepared query without setting a legacy task type."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality
            ),
        )
        if not response.embeddings:
            return []
        return list(response.embeddings[0].values)


class TaskSpecificEmbeddingFunction(Embeddings):
    """Apply Gemini Embedding 2 document/query formats to plain text inputs."""

    def __init__(self, provider: Embeddings) -> None:
        self.provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed plain document texts with the no-title document prefix."""
        prepared = [
            prepare_document_text(text, title=None)
            for text in texts
        ]
        return _call_embed_documents(self.provider, prepared)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query using the question-answering query prefix."""
        return embed_query(text, embedding_model=self.provider)


def _configured_model_name() -> str:
    model = get_settings().gemini_embedding_model.strip()
    normalized = model.removeprefix("models/")
    if normalized != "gemini-embedding-2":
        raise UnsupportedEmbeddingModelError(
            "Unsupported Gemini embedding model. Configure "
            "'GEMINI_EMBEDDING_MODEL=gemini-embedding-2'."
        )
    return normalized


def create_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Initialize Gemini Embedding 2 with configured key and output dimensions.

    The Google API key is passed directly to the provider and is never included
    in messages or logs.
    """
    settings = get_settings()
    if not settings.gemini_api_key.strip():
        raise MissingAPIKeyError(
            "Gemini API key is missing. Set GEMINI_API_KEY in the project "
            "environment before requesting embeddings."
        )
    model = _configured_model_name()
    dimensions = settings.gemini_embedding_dimensions
    if dimensions <= 0:
        raise ValueError("GEMINI_EMBEDDING_DIMENSIONS must be greater than zero.")
    try:
        return GeminiEmbedding2Embeddings(
            model=model,
            api_key=settings.gemini_api_key,
            output_dimensionality=dimensions,
        )
    except Exception:
        raise EmbeddingServiceError(
            "Could not initialize the Gemini embedding client. Check the "
            "model configuration and installed Google integration."
        ) from None


def prepare_document_text(text: str, *, title: str | None = None) -> str:
    """Format document text for Gemini Embedding 2 without altering its source."""
    if not isinstance(text, str) or not text.strip():
        raise EmptyEmbeddingInputError("Document text must not be empty.")
    normalized_title = title.strip() if isinstance(title, str) else ""
    return f"title: {normalized_title or 'none'} | text: {text}"


def prepare_document_for_embedding(document: Document) -> str:
    """Format a LangChain document, using its title metadata when available."""
    if not isinstance(document, Document):
        raise TypeError("document must be a LangChain Document.")
    title = document.metadata.get("document_title") or document.metadata.get("title")
    if not isinstance(title, str):
        title = None
    return prepare_document_text(document.page_content, title=title)


def prepare_query_for_embedding(user_question: str) -> str:
    """Format a user question for Gemini Embedding 2 question-answer retrieval."""
    if not isinstance(user_question, str) or not user_question.strip():
        raise EmptyEmbeddingInputError("Query text must not be empty.")
    return f"task: question answering | query: {user_question}"


def _raise_service_error(error: Exception) -> NoReturn:
    """Map provider exceptions to safe messages without exposing raw details."""
    detail = str(error).casefold()
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "code", None)
    status_text = str(status_code).casefold()
    combined = f"{status_text} {detail}"

    if any(term in combined for term in ("429", "quota", "rate limit", "resource_exhausted")):
        raise EmbeddingQuotaError(
            "Gemini embedding quota or rate limit was reached. Check the "
            "Google Cloud quota and try again later."
        ) from None
    if any(term in combined for term in ("404", "not found", "unsupported model", "unknown model")):
        raise UnsupportedEmbeddingModelError(
            "Gemini does not recognize the configured embedding model. "
            "Check GEMINI_EMBEDDING_MODEL."
        ) from None
    if any(term in combined for term in ("401", "403", "unauthorized", "permission", "api key")):
        raise EmbeddingAuthenticationError(
            "Gemini rejected the configured credentials or project access. "
            "Check GEMINI_API_KEY and API permissions."
        ) from None
    if any(term in combined for term in ("503", "unavailable", "timeout", "connection", "dns")):
        raise EmbeddingAPIUnavailableError(
            "Gemini embedding service is unavailable. Check network access "
            "and try again."
        ) from None
    raise EmbeddingServiceError(
        "Gemini embedding request failed. Check service availability and "
        "model configuration."
    ) from None


def _validate_vectors(
    vectors: object,
    *,
    expected_count: int,
    expected_dimensions: int,
) -> list[list[float]]:
    """Check provider vector count, dimensionality, numeric type, and finiteness."""
    if not isinstance(vectors, (list, tuple)):
        raise InvalidEmbeddingResponseError(
            "Embedding service returned a response that is not a vector list."
        )
    if len(vectors) != expected_count:
        raise InvalidEmbeddingResponseError(
            f"Embedding service returned {len(vectors)} vectors; "
            f"expected {expected_count}."
        )
    validated: list[list[float]] = []
    for vector_index, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)):
            raise InvalidEmbeddingResponseError(
                f"Embedding {vector_index} is not a numeric vector."
            )
        if len(vector) != expected_dimensions:
            raise InvalidEmbeddingResponseError(
                f"Embedding {vector_index} has {len(vector)} dimensions; "
                f"expected {expected_dimensions}."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise InvalidEmbeddingResponseError(
                f"Embedding {vector_index} contains non-finite or non-numeric values."
            )
        validated.append([float(value) for value in vector])
    return validated


Result = TypeVar("Result")


def _call_provider(call: Callable[[], Result]) -> Result:
    """Call the provider once and translate failures to safe domain errors."""
    try:
        return call()
    except EmbeddingError:
        raise
    except Exception as exc:
        _raise_service_error(exc)


def _call_embed_documents(
    embedding_model: Embeddings,
    prepared_texts: list[str],
) -> list[list[float]]:
    if not prepared_texts:
        return []
    vectors = _call_provider(lambda: embedding_model.embed_documents(prepared_texts))
    dimensions = get_settings().gemini_embedding_dimensions
    return _validate_vectors(
        vectors,
        expected_count=len(prepared_texts),
        expected_dimensions=dimensions,
    )


def embed_documents(
    documents: Sequence[Document],
    *,
    embedding_model: Embeddings | None = None,
) -> list[list[float]]:
    """Embed document chunks while leaving original documents and metadata intact."""
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence of LangChain Document objects.")
    if any(not isinstance(document, Document) for document in documents):
        raise TypeError("every item in documents must be a LangChain Document.")
    if not documents:
        return []
    prepared = [prepare_document_for_embedding(document) for document in documents]
    model = embedding_model or create_embedding_model()
    return _call_embed_documents(model, prepared)


def embed_query(
    user_question: str,
    *,
    embedding_model: Embeddings | None = None,
) -> list[float]:
    """Embed a question using the configured question-answering query format."""
    prepared = prepare_query_for_embedding(user_question)
    model = embedding_model or create_embedding_model()
    vectors = _call_provider(lambda: [model.embed_query(prepared)])
    dimensions = get_settings().gemini_embedding_dimensions
    validated = _validate_vectors(
        vectors,
        expected_count=1,
        expected_dimensions=dimensions,
    )
    return validated[0]

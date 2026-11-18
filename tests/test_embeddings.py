"""Offline tests for Gemini Embedding 2 preparation and error handling."""

from dataclasses import replace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import embeddings
from config import get_settings
from embeddings import (
    EmbeddingAPIUnavailableError,
    EmbeddingAuthenticationError,
    EmbeddingQuotaError,
    EmbeddingServiceError,
    EmptyEmbeddingInputError,
    InvalidEmbeddingResponseError,
    MissingAPIKeyError,
    UnsupportedEmbeddingModelError,
    create_embedding_model,
    embed_documents,
    embed_query,
    prepare_document_for_embedding,
    prepare_query_for_embedding,
)


_UNSET = object()


class RecordingEmbeddings(Embeddings):
    """Small test embedding provider that records prepared text."""

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions
        self.document_inputs: list[str] = []
        self.query_inputs: list[str] = []
        self.document_response: object = _UNSET
        self.query_response: list[float] | None = None
        self.failure: Exception | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs = texts
        if self.failure:
            raise self.failure
        if self.document_response is not _UNSET:
            return self.document_response  # type: ignore[return-value]
        return [[0.25] * self.dimensions for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        if self.failure:
            raise self.failure
        if self.query_response is not None:
            return self.query_response
        return [0.5] * self.dimensions


def _settings(**changes: object):
    return replace(get_settings(), **changes)


def test_embedding_model_initialization_uses_configured_model_key_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeGoogleEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(embeddings, "get_settings", lambda: _settings(
        gemini_api_key="not-a-real-test-key",
        gemini_embedding_model="gemini-embedding-2",
        gemini_embedding_dimensions=768,
    ))
    monkeypatch.setattr(
        embeddings, "GeminiEmbedding2Embeddings", FakeGoogleEmbeddings
    )

    create_embedding_model()

    assert captured == {
        "model": "gemini-embedding-2",
        "api_key": "not-a-real-test-key",
        "output_dimensionality": 768,
    }


def test_gemini_embedding_2_calls_google_client_without_legacy_task_type() -> None:
    calls: list[dict[str, object]] = []

    class FakeModels:
        def embed_content(self, **kwargs: object):
            calls.append(kwargs)

            class Response:
                embeddings = [type("Embedding", (), {"values": [0.1, 0.2, 0.3]})()]

            return Response()

    class FakeClient:
        models = FakeModels()

    provider = embeddings.GeminiEmbedding2Embeddings.model_construct(
        model="gemini-embedding-2",
        output_dimensionality=3,
        client=FakeClient(),
    )

    assert provider.embed_documents(["title: none | text: synthetic"]) == [
        [0.1, 0.2, 0.3]
    ]
    assert provider.embed_query(
        "task: question answering | query: synthetic question"
    ) == [0.1, 0.2, 0.3]
    assert len(calls) == 2
    for call in calls:
        assert call["model"] == "gemini-embedding-2"
        assert call["config"].output_dimensionality == 3
        assert call["config"].task_type is None


def test_document_text_uses_title_metadata() -> None:
    document = Document(
        page_content="Synthetic document content.",
        metadata={"document_title": "Synthetic report"},
    )

    prepared = prepare_document_for_embedding(document)

    assert prepared == (
        "title: Synthetic report | text: Synthetic document content."
    )
    assert document.page_content == "Synthetic document content."
    assert document.metadata == {"document_title": "Synthetic report"}


def test_document_title_metadata_alias_is_supported() -> None:
    document = Document(
        page_content="Synthetic content.",
        metadata={"title": "Short title"},
    )

    assert prepare_document_for_embedding(document) == (
        "title: Short title | text: Synthetic content."
    )


def test_missing_document_title_uses_none() -> None:
    document = Document(page_content="Synthetic content.")

    assert prepare_document_for_embedding(document) == (
        "title: none | text: Synthetic content."
    )


def test_query_uses_question_answering_prefix() -> None:
    assert prepare_query_for_embedding("Where is the synthetic result?") == (
        "task: question answering | query: Where is the synthetic result?"
    )


def test_empty_document_and_query_are_rejected() -> None:
    with pytest.raises(EmptyEmbeddingInputError, match="Document text"):
        prepare_document_for_embedding(Document(page_content=" \n"))
    with pytest.raises(EmptyEmbeddingInputError, match="Query text"):
        prepare_query_for_embedding(" \t")


def test_missing_api_key_is_reported_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: _settings(gemini_api_key=""),
    )

    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        create_embedding_model()


def test_unsupported_model_is_reported_before_client_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: _settings(
            gemini_api_key="not-a-real-test-key",
            gemini_embedding_model="unsupported-model",
        ),
    )

    with pytest.raises(UnsupportedEmbeddingModelError, match="gemini-embedding-2"):
        create_embedding_model()


def test_document_embedding_uses_prepared_input_and_validates_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: _settings(gemini_embedding_dimensions=3),
    )
    provider = RecordingEmbeddings(dimensions=3)
    document = Document(
        page_content="Synthetic text.",
        metadata={"title": "Test title"},
    )

    vectors = embed_documents([document], embedding_model=provider)

    assert provider.document_inputs == ["title: Test title | text: Synthetic text."]
    assert vectors == [[0.25, 0.25, 0.25]]
    assert document.page_content == "Synthetic text."


def test_query_embedding_uses_prepared_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: _settings(gemini_embedding_dimensions=3),
    )
    provider = RecordingEmbeddings(dimensions=3)

    vector = embed_query("Synthetic query", embedding_model=provider)

    assert provider.query_inputs == [
        "task: question answering | query: Synthetic query"
    ]
    assert vector == [0.5, 0.5, 0.5]


@pytest.mark.parametrize(
    ("failure", "error_type", "message"),
    [
        (RuntimeError("429 quota exceeded"), EmbeddingQuotaError, "quota"),
        (
            RuntimeError("503 service unavailable"),
            EmbeddingAPIUnavailableError,
            "unavailable",
        ),
        (
            RuntimeError("401 invalid API key secret"),
            EmbeddingAuthenticationError,
            "credentials",
        ),
        (
            RuntimeError("unexpected provider failure"),
            EmbeddingServiceError,
            "request failed",
        ),
    ],
)
def test_service_errors_are_mapped_to_safe_messages(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    error_type: type[EmbeddingServiceError],
    message: str,
) -> None:
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: _settings(gemini_embedding_dimensions=3),
    )
    provider = RecordingEmbeddings(dimensions=3)
    provider.failure = failure

    with pytest.raises(error_type, match=message) as error:
        embed_query("Synthetic query", embedding_model=provider)

    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        None,
        [[0.1, 0.2]],
        [None],
        [[0.1, 0.2, float("nan")]],
        [[0.1, 0.2, float("inf")]],
        [[0.1, True, 0.3]],
        [],
    ],
)
def test_invalid_embedding_responses_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: _settings(gemini_embedding_dimensions=3),
    )
    provider = RecordingEmbeddings(dimensions=3)
    provider.document_response = response

    with pytest.raises(InvalidEmbeddingResponseError):
        embed_documents(
            [Document(page_content="Synthetic content.")],
            embedding_model=provider,
        )


def test_empty_document_list_returns_empty_without_api_call() -> None:
    assert embed_documents([]) == []


def test_non_document_inputs_are_rejected() -> None:
    with pytest.raises(TypeError, match="sequence of LangChain Document"):
        embed_documents(None)  # type: ignore[arg-type]

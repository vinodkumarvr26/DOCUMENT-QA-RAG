"""Offline tests for the manually executed Gemini chat smoke test."""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from rag_chain import (
    MissingAPIKeyError,
    RAGAuthenticationError,
    RAGQuotaError,
    RAGServiceError,
    RAGServiceUnavailableError,
)
from scripts import gemini_chat_smoke_test


def test_extract_response_text_accepts_plain_string() -> None:
    assert (
        gemini_chat_smoke_test.extract_response_text("Synthetic response.")
        == "Synthetic response."
    )


def test_extract_response_text_accepts_ai_message_string() -> None:
    assert (
        gemini_chat_smoke_test.extract_response_text(
            AIMessage(content="Synthetic response.")
        )
        == "Synthetic response."
    )


def test_extract_response_text_accepts_text_blocks_only() -> None:
    response = AIMessage(
        content=[
            {"type": "text", "text": "First synthetic sentence."},
            {"type": "image", "url": "private-metadata-not-for-printing"},
            {"type": "text", "text": "Second synthetic sentence."},
        ]
    )

    assert gemini_chat_smoke_test.extract_response_text(response) == (
        "First synthetic sentence.\nSecond synthetic sentence."
    )


def test_extract_response_text_returns_empty_for_no_text() -> None:
    response = AIMessage(
        content=[{"type": "image", "url": "private-metadata-not-for-printing"}]
    )

    assert gemini_chat_smoke_test.extract_response_text(response) == ""


def test_smoke_test_reports_safe_empty_response_diagnostic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeModel:
        def invoke(self, _prompt: str) -> Any:
            return AIMessage(content=[])

    result = gemini_chat_smoke_test.main(model_factory=FakeModel)
    captured = capsys.readouterr()

    assert result == 1
    assert "no textual content blocks" in captured.err
    assert "private-metadata" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (MissingAPIKeyError("secret API key value"), "GEMINI_API_KEY"),
        (RAGQuotaError("private quota response"), "quota or rate limit"),
        (RAGAuthenticationError("secret credential"), "authentication"),
        (RAGServiceUnavailableError("private network error"), "unavailable"),
        (RAGServiceError("private provider response"), "model configuration"),
        (RuntimeError("secret provider payload"), "unexpected chat request"),
    ],
)
def test_smoke_test_errors_are_safe(
    failure: Exception,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_factory() -> Any:
        raise failure

    result = gemini_chat_smoke_test.main(model_factory=fail_factory)
    captured = capsys.readouterr()

    assert result == 1
    assert message.casefold() in captured.err.casefold()
    assert "secret" not in captured.err.casefold()
    assert "private" not in captured.err.casefold()
    assert captured.out == ""


def test_smoke_test_prints_only_extracted_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeModel:
        def invoke(self, _prompt: str) -> Any:
            return AIMessage(
                content=[{"type": "text", "text": "Synthetic connection success."}],
                response_metadata={"private": "metadata"},
            )

    assert gemini_chat_smoke_test.main(model_factory=FakeModel) == 0
    captured = capsys.readouterr()

    assert captured.out == "Synthetic connection success.\n"
    assert "metadata" not in captured.out
    assert captured.err == ""


def test_smoke_test_redacts_configured_key_from_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_key = "synthetic-private-key"
    monkeypatch.setattr(
        gemini_chat_smoke_test,
        "get_settings",
        lambda: SimpleNamespace(gemini_api_key=configured_key),
    )

    class FakeModel:
        def invoke(self, _prompt: str) -> Any:
            return AIMessage(content=f"Unexpected echo: {configured_key}")

    assert gemini_chat_smoke_test.main(model_factory=FakeModel) == 0
    captured = capsys.readouterr()

    assert configured_key not in captured.out
    assert "[REDACTED]" in captured.out

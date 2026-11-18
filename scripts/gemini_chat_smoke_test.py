"""Optional live Gemini chat smoke test; never run by pytest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_chain import (  # noqa: E402
    MissingAPIKeyError,
    RAGAuthenticationError,
    RAGQuotaError,
    RAGServiceError,
    RAGServiceUnavailableError,
    create_chat_model,
)
from config import get_settings  # noqa: E402


def extract_response_text(response: Any) -> str:
    """Extract only textual content from supported LangChain/Gemini responses."""
    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_blocks: list[str] = []
    for block in content:
        if isinstance(block, str):
            text = block
        elif isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
        else:
            continue
        if isinstance(text, str) and text.strip():
            text_blocks.append(text.strip())
    return "\n".join(text_blocks)


def _redact_response_secrets(text: str) -> str:
    """Prevent configured or recognizable Google API keys from being printed."""
    api_key = get_settings().gemini_api_key
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", text)


def main(
    *,
    model_factory: Callable[[], Any] | None = None,
) -> int:
    """Run one manually requested live chat call and print only response text."""
    try:
        model = (model_factory or create_chat_model)()
        response = model.invoke(
            "Reply with exactly: DocuQuery Gemini chat connection successful."
        )
    except MissingAPIKeyError:
        print(
            "Gemini chat smoke test failed: GEMINI_API_KEY is not configured.",
            file=sys.stderr,
        )
        return 1
    except RAGQuotaError:
        print(
            "Gemini chat smoke test failed: API quota or rate limit reached.",
            file=sys.stderr,
        )
        return 1
    except RAGAuthenticationError:
        print(
            "Gemini chat smoke test failed: authentication or permissions error.",
            file=sys.stderr,
        )
        return 1
    except RAGServiceUnavailableError:
        print(
            "Gemini chat smoke test failed: service unavailable or network timeout.",
            file=sys.stderr,
        )
        return 1
    except RAGServiceError:
        print(
            "Gemini chat smoke test failed: model configuration or service error.",
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            "Gemini chat smoke test failed: unexpected chat request error.",
            file=sys.stderr,
        )
        return 1

    content = extract_response_text(response)
    if not content:
        print(
            "Gemini chat smoke test failed: response contained no text. "
            "The model returned no textual content blocks; verify the model "
            "configuration and response modality.",
            file=sys.stderr,
        )
        return 1
    print(_redact_response_secrets(content))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

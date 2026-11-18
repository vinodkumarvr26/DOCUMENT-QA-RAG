"""Application configuration and environment variables."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os


_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _read_positive_int(name: str, default: int) -> int:
    """Read a positive integer setting, raising a useful error for invalid values."""
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero; received {parsed}.")
    return parsed


def _read_non_negative_int(name: str, default: int) -> int:
    """Read a non-negative integer setting, raising a useful error if invalid."""
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {value!r}.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative; received {parsed}.")
    return parsed


@dataclass(frozen=True)
class Settings:
    """Validated application settings loaded from the project environment."""

    gemini_api_key: str
    gemini_chat_model: str
    gemini_embedding_model: str
    gemini_embedding_dimensions: int
    chroma_persist_directory: Path
    chunk_size: int
    chunk_overlap: int
    retriever_top_k: int


def get_settings() -> Settings:
    """Return settings from environment variables, using safe development defaults."""
    chunk_size = _read_positive_int("CHUNK_SIZE", 1000)
    chunk_overlap = _read_non_negative_int("CHUNK_OVERLAP", 150)
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    persist_directory = Path(
        os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    )
    if not persist_directory.is_absolute():
        persist_directory = _PROJECT_ROOT / persist_directory

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"
        ),
        gemini_embedding_dimensions=_read_positive_int(
            "GEMINI_EMBEDDING_DIMENSIONS", 768
        ),
        chroma_persist_directory=persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retriever_top_k=_read_positive_int("RETRIEVER_TOP_K", 4),
    )

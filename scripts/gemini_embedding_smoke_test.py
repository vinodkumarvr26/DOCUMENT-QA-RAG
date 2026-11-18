"""Optional live Gemini Embedding 2 smoke test; never run by pytest."""

from __future__ import annotations

import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from config import get_settings
from embeddings import EmbeddingError, create_embedding_model, embed_documents, embed_query


def main() -> int:
    settings = get_settings()
    try:
        model = create_embedding_model()
        document_vectors = embed_documents(
            [Document(page_content="A synthetic example about document search.")],
            embedding_model=model,
        )
        query_vector = embed_query(
            "How does synthetic document search work?",
            embedding_model=model,
        )
    except EmbeddingError as exc:
        print(f"Gemini embedding smoke test failed: {exc}", file=sys.stderr)
        return 1

    vectors = [*document_vectors, query_vector]
    if not document_vectors or not query_vector or any(
        not math.isfinite(value) for vector in vectors for value in vector
    ):
        print(
            "Gemini embedding smoke test failed: response vectors were empty "
            "or contained non-finite values.",
            file=sys.stderr,
        )
        return 1
    if any(len(vector) != settings.gemini_embedding_dimensions for vector in vectors):
        print(
            "Gemini embedding smoke test failed: vector dimensions did not "
            "match GEMINI_EMBEDDING_DIMENSIONS.",
            file=sys.stderr,
        )
        return 1

    print(f"Document vector dimensions: {len(document_vectors[0])}")
    print(f"Question vector dimensions: {len(query_vector)}")
    print("Gemini Embedding 2 smoke test succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

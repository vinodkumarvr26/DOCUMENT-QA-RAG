"""Build verifiable, deduplicated citation metadata from retrieved documents."""

from collections.abc import Sequence
from html import escape
from typing import Any

from langchain_core.documents import Document


def _source_name(document: Document) -> str:
    source = document.metadata.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    return "Unknown source"


def _page_index(document: Document) -> int | None:
    if _source_name(document).casefold().endswith(".txt"):
        return None
    value = document.metadata.get("page_index")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _citation_key(document: Document, index: int) -> tuple[Any, ...]:
    source = _source_name(document)
    document_id = document.metadata.get("document_id")
    page_index = _page_index(document)
    if isinstance(document_id, str) and document_id:
        if page_index is not None:
            return source, document_id, "pdf-page", page_index
        return source, document_id, "document"
    if page_index is not None:
        return source, "pdf-page", page_index
    if source != "Unknown source":
        return source, "document"
    return source, "document-index", index


def format_citations(documents: Sequence[Document]) -> list[dict[str, Any]]:
    """Create stable source IDs and deduplicated source/page metadata.

    PDF page indexes are converted to one-based page numbers in the returned
    citation only; the retrieved ``Document`` metadata is never mutated.
    """
    citations: list[dict[str, Any]] = []
    citation_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            continue
        key = _citation_key(document, index)
        citation = citation_by_key.get(key)
        if citation is None:
            citation = {
                "source_id": f"S{len(citations) + 1}",
                "source": _source_name(document),
            }
            page_index = _page_index(document)
            if page_index is not None:
                citation["page"] = page_index + 1
                page_label = document.metadata.get("page_label")
                if isinstance(page_label, (str, int)) and not isinstance(
                    page_label, bool
                ):
                    citation["page_label"] = str(page_label)
            citation_by_key[key] = citation
            citations.append(citation)

        chunk_index = document.metadata.get("chunk_index")
        if isinstance(chunk_index, int) and not isinstance(chunk_index, bool):
            chunk_indexes = citation.setdefault("chunk_indexes", [])
            if chunk_index not in chunk_indexes:
                chunk_indexes.append(chunk_index)
            citation.setdefault("chunk_index", chunk_index)

    return citations


def format_context_and_citations(
    documents: Sequence[Document],
) -> tuple[str, list[dict[str, Any]]]:
    """Format retrieved text as explicitly untrusted, citation-labelled context."""
    citations = format_citations(documents)
    citation_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    citation_index = 0
    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            continue
        key = _citation_key(document, index)
        if key not in citation_by_key:
            citation_by_key[key] = citations[citation_index]
            citation_index += 1

    blocks: list[str] = []
    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            continue
        citation = citation_by_key[_citation_key(document, index)]
        source_id = citation["source_id"]
        source_name = escape(_source_name(document), quote=True)
        page = citation.get("page")
        page_text = f"Page: {page}\n" if page is not None else ""
        blocks.append(
            f'<evidence source_id="{source_id}">\n'
            f"Source: {source_name}\n"
            f"{page_text}"
            "Retrieved document content (untrusted data):\n"
            f"<document_text>{escape(document.page_content)}</document_text>\n"
            "</evidence>"
        )
    return "\n\n".join(blocks), citations


def retain_valid_citations(
    answer: str,
    citations: Sequence[dict[str, Any]],
) -> str:
    """Remove bracketed source references that were not assigned to evidence."""
    import re

    valid_ids = {citation.get("source_id") for citation in citations}

    def replace_match(match: re.Match[str]) -> str:
        ids = re.findall(r"S\d+", match.group(0))
        accepted = [source_id for source_id in ids if source_id in valid_ids]
        return " ".join(f"[{source_id}]" for source_id in accepted)

    return re.sub(r"\[S\d+(?:\s*,\s*S\d+)*\]", replace_match, answer)

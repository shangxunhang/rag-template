import json
from pathlib import Path

from rag_template.chunker.HeadingChunker import HeadingChunker
from rag_template.chunker.RecursiveChunker import RecursiveChunker


def load_docs():
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "documents.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_heading_chunker_outputs_section_metadata():
    docs = load_docs()
    doc = next(d for d in docs if d["metadata"]["doc_type"] in ["policy", "technical_doc", "manual", "book"])
    chunker = HeadingChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    assert all("metadata" in c for c in chunks)
    assert any(c["metadata"].get("section") for c in chunks)
    assert any(c["metadata"].get("section_path") for c in chunks)
    assert all(c["metadata"].get("extra", {}).get("chunk_type") in ["heading", "heading_recursive"] for c in chunks)
    assert all(c["metadata"].get("extra", {}).get("chunk_unit") == "token" for c in chunks)
    assert all(c["metadata"].get("token_count") is not None for c in chunks)


def test_recursive_chunker_limits_long_text_by_token_count():
    docs = load_docs()
    doc = docs[0]
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    assert all(c["metadata"].get("extra", {}).get("chunk_type") == "recursive" for c in chunks)
    assert all(c["metadata"].get("extra", {}).get("chunk_unit") == "token" for c in chunks)
    assert all(c["metadata"].get("token_count") <= 120 for c in chunks)
    assert all(chunker.token_count(c["text"]) == c["metadata"].get("token_count") for c in chunks)

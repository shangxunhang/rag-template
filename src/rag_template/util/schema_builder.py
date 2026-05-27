# src/rag_template/util/schema_builder.py
from pathlib import Path
from typing import Dict, Any, Optional

import copy
from datetime import date

from rag_template.configs.SchemaConfig import *


def build_document(
    file_path: Path,
    text: str,
    doc_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    title: Optional[str] = None,
    source: Optional[str] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    **metadata_kwargs,
) -> Dict[str, Any]:
    """
    构造统一 Document Schema。
    """
    metadata = copy.deepcopy(DOCUMENT_METADATA_TEMPLATE)

    metadata["source"] = source or file_path.name
    metadata["source_path"] = str(file_path)
    metadata["doc_type"] = doc_type or file_path.suffix.replace(".", "")
    metadata["title"] = title
    metadata["created_at"] = created_at or str(date.today())
    metadata["updated_at"] = updated_at

    for key, value in metadata_kwargs.items():
        if key in metadata:
            metadata[key] = value
        else:
            metadata.setdefault("extra", {})
            metadata["extra"][key] = value

    return {
        "doc_id": doc_id or file_path.stem,
        "text": text,
        "metadata": metadata,
    }


def build_chunk(
    doc_id: str,
    text: str,
    idx: int,
    doc_metadata: Dict[str, Any],
    *,
    chunk_type: str = "fixed",
    page: Optional[int] = None,
    section: Optional[str] = None,
    section_path: Optional[str] = None,
    parent_chunk_id: Optional[str] = None,
    start_char: Optional[int] = None,
    end_char: Optional[int] = None,
    heading_level: Optional[int] = None,
    semantic_score: Optional[float] = None,
    token_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构造统一 Chunk Schema。

    兼容旧调用：build_chunk(doc_id, text, idx, doc_metadata)
    新增字段主要服务 HeadingChunker / RecursiveChunker / SemanticChunker。
    """
    metadata = copy.deepcopy(CHUNK_METADATA_TEMPLATE)

    # 只继承 Chunk metadata 模板中存在的字段，避免 document.extra 污染 chunk 顶层 metadata。
    for key in metadata.keys():
        if key in doc_metadata and key != "extra":
            metadata[key] = doc_metadata[key]

    metadata["chunk_index"] = idx
    metadata["page"] = page
    metadata["section"] = section
    metadata["section_path"] = section_path
    metadata["created_at"] = doc_metadata.get("created_at") or str(date.today())
    metadata["updated_at"] = doc_metadata.get("updated_at")
    metadata["char_count"] = len(text) if text is not None else 0
    metadata["token_count"] = token_count

    inherited_extra = copy.deepcopy(doc_metadata.get("extra", {})) if isinstance(doc_metadata.get("extra"), dict) else {}
    chunk_extra = {
        "chunk_type": chunk_type,
        "parent_chunk_id": parent_chunk_id,
        "start_char": start_char,
        "end_char": end_char,
        "heading_level": heading_level,
        "semantic_score": semantic_score,
    }
    if extra:
        chunk_extra.update(extra)

    metadata["extra"] = {
        "document_extra": inherited_extra,
        **chunk_extra,
    }

    return {
        "chunk_id": f"{doc_id}_chunk_{idx:04d}",
        "doc_id": doc_id,
        "text": text,
        "metadata": metadata,
    }


def build_retrieval_result(
    chunk: Dict[str, Any],
    rank: int,
    score: float,
    rerank_score: float = None,
) -> Dict[str, Any]:
    result = copy.deepcopy(RETRIEVAL_RESULT_TEMPLATE)
    result["rank"] = rank
    result["score"] = score
    result["rerank_score"] = rerank_score
    result["chunk_id"] = chunk["chunk_id"]
    result["doc_id"] = chunk["doc_id"]
    result["text"] = chunk["text"]
    result["metadata"] = chunk["metadata"]
    return result

# chunker/chunk_schema.py

from copy import deepcopy
from datetime import date
from typing import Dict, Optional

from rag_template.configs.SchemaConfig import CHUNK_METADATA_TEMPLATE


def build_chunk(
    doc: Dict,
    chunk_text: str,
    chunk_index: int,
    section: Optional[str] = None,
    section_path: Optional[str] = None,
    page: Optional[int] = None,
    parent_chunk_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> Dict:
    """
    统一构造 chunk schema。
    """

    doc_id = doc["doc_id"]
    doc_metadata = doc.get("metadata", {})

    metadata = deepcopy(CHUNK_METADATA_TEMPLATE)

    metadata.update({
        "source": doc_metadata.get("source"),
        "source_path": doc_metadata.get("source_path"),
        "doc_type": doc_metadata.get("doc_type"),
        "title": doc_metadata.get("title"),

        "chunk_index": chunk_index,
        "page": page,
        "section": section,
        "section_path": section_path,

        "created_at": str(date.today()),
        "updated_at": doc_metadata.get("updated_at"),

        "version": doc_metadata.get("version"),
        "status": doc_metadata.get("status", "active"),
        "is_latest": doc_metadata.get("is_latest", True),

        "department": doc_metadata.get("department"),
        "project_id": doc_metadata.get("project_id"),
        "project_name": doc_metadata.get("project_name"),
        "security_level": doc_metadata.get("security_level"),

        "char_count": len(chunk_text),
        "token_count": None,

        "extra": {
            "parent_chunk_id": parent_chunk_id,
            "chunk_type": chunk_type,
        }
    })

    return {
        "chunk_id": f"{doc_id}_chunk_{chunk_index:04d}",
        "doc_id": doc_id,
        "text": chunk_text,
        "metadata": metadata,
    }
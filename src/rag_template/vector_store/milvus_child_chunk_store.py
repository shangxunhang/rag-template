# -*- coding: utf-8 -*-
"""
rag_template/vector_store/milvus_child_chunk_store.py
====================================================

Milvus Lite child_chunk_v1 向量存储层。

职责：
1. 创建适配 child_chunk_v1 的 Milvus collection schema。
2. 构造 Milvus 物理入库 record。
3. 批量 insert。
4. 可选 search smoke test。

设计约定：
- Milvus 只索引 child_chunk_v1。
- chunk_id == child_chunk_id，作为 Milvus primary key。
- parent_chunk_id 必须入 Milvus metadata，后续检索命中 child 后用它回填 parent。
- parent_chunk_v1 暂时不入向量库，落 JSONL/HDFS/MySQL 均可。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pymilvus import DataType, MilvusClient


DEFAULT_INDEXED_GRANULARITY = "child"


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def safe_int(value: Any, default: int = -1) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _json_field_length(max_text_chars: int) -> int:
    # Milvus VARCHAR 最大 65535。给 JSON 字段保守一点，避免 schema 过大。
    return min(max(max_text_chars, 4096), 65535)


def build_milvus_schema_for_child_chunk_v1(dim: int, max_text_chars: int):
    """Build physical Milvus schema for child_chunk_v1 indexing."""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

    # Primary key / vector.
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=int(dim))

    # Parent-child linkage.
    schema.add_field(field_name="child_chunk_id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_chunk_id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="indexed_granularity", datatype=DataType.VARCHAR, max_length=64)

    # Child text and location.
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=min(max_text_chars, 65535))
    schema.add_field(field_name="text_length", datatype=DataType.INT64)
    schema.add_field(field_name="token_count", datatype=DataType.INT64)
    schema.add_field(field_name="source_unit_ids_json", datatype=DataType.VARCHAR, max_length=_json_field_length(max_text_chars))
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="section_level", datatype=DataType.INT64)
    schema.add_field(field_name="page_start", datatype=DataType.INT64)
    schema.add_field(field_name="page_end", datatype=DataType.INT64)

    # Child chunk positions.
    schema.add_field(field_name="child_chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="child_index_in_parent", datatype=DataType.INT64)
    schema.add_field(field_name="child_chunk_strategy", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="char_start_in_parent", datatype=DataType.INT64)
    schema.add_field(field_name="char_end_in_parent", datatype=DataType.INT64)

    # Versions / lineage.
    schema.add_field(field_name="cleaning_version", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="parent_chunk_version", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="child_chunk_version", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="chunk_created_at", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="extra_json", datatype=DataType.VARCHAR, max_length=_json_field_length(max_text_chars))
    schema.add_field(field_name="child_chunk_schema_version", datatype=DataType.VARCHAR, max_length=128)

    # Embedding metadata.
    schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="embedding_dim", datatype=DataType.INT64)
    schema.add_field(field_name="embedding_version", datatype=DataType.VARCHAR, max_length=256)
    return schema


def create_or_reset_child_chunk_collection(
    client: MilvusClient,
    collection_name: str,
    dim: int,
    metric_type: str,
    recreate: bool,
    max_text_chars: int,
) -> None:
    """Create child_chunk_v1 collection; optionally drop existing one."""
    if client.has_collection(collection_name):
        if recreate:
            client.drop_collection(collection_name)
            print(f"Dropped existing collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")
            return

    schema = build_milvus_schema_for_child_chunk_v1(dim=dim, max_text_chars=max_text_chars)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type=metric_type,
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection created: {collection_name}, dim={dim}")
    print(f"Vector index created: AUTOINDEX / {metric_type}")


def build_milvus_child_chunk_record(
    child_chunk: Dict[str, Any],
    vector: np.ndarray,
    embedding_model: str,
    embedding_dim: int,
    embedding_version: str,
    max_text_chars: int,
) -> Dict[str, Any]:
    """Convert child_chunk_v1 + vector into physical Milvus record."""
    child_chunk_id = safe_str(child_chunk.get("child_chunk_id") or child_chunk.get("chunk_id"))
    chunk_id = safe_str(child_chunk.get("chunk_id") or child_chunk_id)

    return {
        "chunk_id": chunk_id,
        "vector": vector.astype(np.float32).tolist(),

        "child_chunk_id": child_chunk_id,
        "parent_chunk_id": safe_str(child_chunk.get("parent_chunk_id")),
        "doc_id": safe_str(child_chunk.get("doc_id")),
        "source_type": safe_str(child_chunk.get("source_type"), "offline"),
        "indexed_granularity": DEFAULT_INDEXED_GRANULARITY,

        "text": truncate_text(safe_str(child_chunk.get("text")), max_text_chars),
        "text_length": safe_int(child_chunk.get("text_length"), len(safe_str(child_chunk.get("text")))),
        "token_count": safe_int(child_chunk.get("token_count"), len(safe_str(child_chunk.get("text")))),
        "source_unit_ids_json": json_dumps_compact(safe_list(child_chunk.get("source_unit_ids"))),
        "title": safe_str(child_chunk.get("title")),
        "section": safe_str(child_chunk.get("section")),
        "section_level": safe_int(child_chunk.get("section_level"), -1),
        "page_start": safe_int(child_chunk.get("page_start"), -1),
        "page_end": safe_int(child_chunk.get("page_end"), -1),

        "child_chunk_index": safe_int(child_chunk.get("child_chunk_index"), -1),
        "child_index_in_parent": safe_int(child_chunk.get("child_index_in_parent"), -1),
        "child_chunk_strategy": safe_str(child_chunk.get("child_chunk_strategy")),
        "char_start_in_parent": safe_int(child_chunk.get("char_start_in_parent"), -1),
        "char_end_in_parent": safe_int(child_chunk.get("char_end_in_parent"), -1),

        "cleaning_version": safe_str(child_chunk.get("cleaning_version")),
        "parent_chunk_version": safe_str(child_chunk.get("parent_chunk_version")),
        "child_chunk_version": safe_str(child_chunk.get("child_chunk_version")),
        "chunk_created_at": safe_str(child_chunk.get("created_at")),
        "extra_json": json_dumps_compact(safe_dict(child_chunk.get("extra"))),
        "child_chunk_schema_version": safe_str(child_chunk.get("schema_version"), "child_chunk_v1"),

        "embedding_model": embedding_model,
        "embedding_dim": int(embedding_dim),
        "embedding_version": embedding_version,
    }


def insert_child_chunk_records(
    client: MilvusClient,
    collection_name: str,
    records: Sequence[Dict[str, Any]],
    batch_size: int,
) -> int:
    total = 0
    for start in range(0, len(records), batch_size):
        batch = list(records[start:start + batch_size])
        result = client.insert(collection_name=collection_name, data=batch)
        total += len(batch)
        print(f"Inserted batch: {len(batch)}, total={total}, result={result}")
    return total


def search_child_chunk_smoke_test(
    client: MilvusClient,
    collection_name: str,
    query_vector: np.ndarray,
    top_k: int,
    metric_type: str,
) -> List[Dict[str, Any]]:
    """Run a minimal search and return Milvus hits with parent_chunk_id output."""
    try:
        client.load_collection(collection_name)
    except Exception as exc:
        print(f"WARN: load_collection skipped or failed: {exc}")

    result = client.search(
        collection_name=collection_name,
        data=[query_vector.astype(np.float32).tolist()],
        anns_field="vector",
        limit=top_k,
        search_params={"metric_type": metric_type},
        output_fields=[
            "chunk_id",
            "child_chunk_id",
            "parent_chunk_id",
            "doc_id",
            "source_type",
            "indexed_granularity",
            "text",
            "title",
            "section",
            "page_start",
            "page_end",
            "child_chunk_index",
            "child_index_in_parent",
            "child_chunk_version",
            "parent_chunk_version",
            "embedding_model",
            "embedding_version",
        ],
    )
    if not result:
        return []
    return result[0]

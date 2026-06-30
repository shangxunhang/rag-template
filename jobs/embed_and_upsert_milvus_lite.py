#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embed chunk_v1 records and upsert them into Milvus Lite.

Input schema: chunk_v1
Output sidecar schema: vector_index_record_v1

Typical Windows smoke test:
python jobs/embed_and_upsert_milvus_lite.py ^
  --input D:\\MyCode\\rag-template\\data\\processed\\chunk_unit_test\\part-00000 ^
  --db-file D:\\MyCode\\rag-template\\vector_store\\milvus_rag.db ^
  --collection-name rag_chunk_units ^
  --index-record-output D:\\MyCode\\rag-template\\data\\processed\\vector_index_record\\vector_index_record_v1.jsonl ^
  --recreate ^
  --hash-embedding ^
  --query 什么是高效学习 ^
  --top-k 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pymilvus import DataType, MilvusClient


CHUNK_SCHEMA_VERSION = "chunk_v1"
VECTOR_INDEX_RECORD_SCHEMA_VERSION = "vector_index_record_v1"
DEFAULT_COLLECTION_NAME = "rag_chunk_units"
DEFAULT_EMBEDDING_VERSION = "embedding_v1.0"
DEFAULT_VECTOR_DB = "milvus"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: str | Path) -> None:
    p = Path(path)
    parent = p.parent
    if parent and str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


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


def safe_nullable_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl_paths(input_path: str | Path) -> Iterable[Path]:
    p = Path(input_path)
    if p.is_file():
        yield p
        return
    if p.is_dir():
        # Spark output directory: part-00000, part-00001..., sometimes *.jsonl
        candidates = sorted([x for x in p.iterdir() if x.is_file() and (x.name.startswith("part-") or x.suffix.lower() in {".jsonl", ".json"})])
        for item in candidates:
            yield item
        return
    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_chunks(input_path: str | Path, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for path in iter_jsonl_paths(input_path):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON line: file={path}, line={line_no}, err={exc}") from exc
                if not isinstance(obj, dict):
                    continue
                chunks.append(normalize_chunk_v1(obj))
                if max_records is not None and len(chunks) >= max_records:
                    return chunks
    return chunks


def normalize_chunk_v1(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize input to the user's chunk_v1 contract.

    The job is tolerant of earlier chunk_unit_v1 outputs, but the normalized object
    uses the chunk_v1 field names.
    """
    text = safe_str(raw.get("text"))
    extra = safe_dict(raw.get("extra"))

    token_count = raw.get("token_count")
    if token_count is None:
        # Cheap fallback; formal token count can be filled by a tokenizer in the chunk job later.
        token_count = len(text)

    text_length = raw.get("text_length")
    if text_length is None:
        text_length = len(text)

    return {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "input_schema_version": safe_str(raw.get("schema_version"), CHUNK_SCHEMA_VERSION),
        "chunk_id": safe_str(raw.get("chunk_id")),
        "doc_id": safe_str(raw.get("doc_id")),
        "source_type": safe_str(raw.get("source_type"), "offline"),
        "text": text,
        "text_length": safe_int(text_length, len(text)),
        "token_count": safe_int(token_count, len(text)),
        "source_unit_ids": [safe_str(x) for x in safe_list(raw.get("source_unit_ids"))],
        "title": raw.get("title"),
        "section": raw.get("section"),
        "section_level": safe_nullable_int(raw.get("section_level")),
        "page_start": safe_nullable_int(raw.get("page_start")),
        "page_end": safe_nullable_int(raw.get("page_end")),
        "chunk_index": safe_int(raw.get("chunk_index"), -1),
        "chunk_strategy": safe_str(raw.get("chunk_strategy")),
        "cleaning_version": safe_str(raw.get("cleaning_version")),
        "chunk_version": safe_str(raw.get("chunk_version")),
        "created_at": safe_str(raw.get("created_at"), now_iso()),
        "extra": extra,
    }


def validate_chunks(chunks: Sequence[Dict[str, Any]]) -> None:
    if not chunks:
        raise ValueError("No chunk records loaded.")
    missing: List[str] = []
    for i, c in enumerate(chunks):
        if not c.get("chunk_id"):
            missing.append(f"idx={i}:chunk_id")
        if not c.get("doc_id"):
            missing.append(f"idx={i}:doc_id")
        if not c.get("text"):
            missing.append(f"idx={i}:text")
        if len(missing) >= 10:
            break
    if missing:
        raise ValueError("Invalid chunk_v1 records, missing required fields: " + ", ".join(missing))


def chunk_text_for_hash(text: str) -> Iterable[str]:
    text = "".join(text.split())
    if not text:
        return []
    if len(text) == 1:
        return [text]
    # Character bi-grams work tolerably for Chinese smoke testing.
    return (text[i : i + 2] for i in range(len(text) - 1))


def hash_embedding_one(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in chunk_text_for_hash(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="little", signed=False) % dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def hash_embed_texts(texts: Sequence[str], dim: int) -> np.ndarray:
    return np.vstack([hash_embedding_one(t, dim) for t in texts]).astype(np.float32)


def resolve_default_embedding_model() -> Optional[str]:
    """Try to reuse the project's RAGConfig if available."""
    try:
        # Make project root importable when this script is run from jobs/.
        this_file = Path(__file__).resolve()
        project_root = this_file.parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from rag_template.configs.RAGConfig import EMBEDDING_MODEL_NAME  # type: ignore

        if EMBEDDING_MODEL_NAME:
            return str(EMBEDDING_MODEL_NAME)
    except Exception:
        return None
    return None


def real_embed_texts(
    texts: Sequence[str],
    model_name: Optional[str],
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, str]:
    if not model_name:
        model_name = resolve_default_embedding_model()
    if not model_name:
        raise ValueError(
            "No embedding model resolved. Pass --embedding-model <local_model_path_or_name> "
            "or use --hash-embedding for smoke test."
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32), str(model_name)


def build_milvus_schema(dim: int, max_text_chars: int):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    # Physical retrieval fields: chunk_v1 + embedding metadata.
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=min(max_text_chars, 65535))
    schema.add_field(field_name="text_length", datatype=DataType.INT64)
    schema.add_field(field_name="token_count", datatype=DataType.INT64)
    schema.add_field(field_name="source_unit_ids_json", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="section_level", datatype=DataType.INT64)
    schema.add_field(field_name="page_start", datatype=DataType.INT64)
    schema.add_field(field_name="page_end", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_strategy", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="cleaning_version", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="chunk_version", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="chunk_created_at", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="extra_json", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="chunk_schema_version", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="input_schema_version", datatype=DataType.VARCHAR, max_length=128)

    schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="embedding_dim", datatype=DataType.INT64)
    schema.add_field(field_name="embedding_version", datatype=DataType.VARCHAR, max_length=256)
    return schema


def create_or_reset_collection(
    client: MilvusClient,
    collection_name: str,
    dim: int,
    metric_type: str,
    recreate: bool,
    max_text_chars: int,
) -> None:
    if client.has_collection(collection_name):
        if recreate:
            client.drop_collection(collection_name)
            print(f"Dropped existing collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")
            return

    schema = build_milvus_schema(dim=dim, max_text_chars=max_text_chars)
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


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_milvus_record(
    chunk: Dict[str, Any],
    vector: np.ndarray,
    embedding_model: str,
    embedding_dim: int,
    embedding_version: str,
    max_text_chars: int,
) -> Dict[str, Any]:
    return {
        "chunk_id": safe_str(chunk.get("chunk_id")),
        "vector": vector.astype(np.float32).tolist(),
        "doc_id": safe_str(chunk.get("doc_id")),
        "source_type": safe_str(chunk.get("source_type"), "offline"),
        "text": truncate_text(safe_str(chunk.get("text")), max_text_chars),
        "text_length": safe_int(chunk.get("text_length"), len(safe_str(chunk.get("text")))),
        "token_count": safe_int(chunk.get("token_count"), len(safe_str(chunk.get("text")))),
        "source_unit_ids_json": json_dumps_compact(safe_list(chunk.get("source_unit_ids"))),
        "title": safe_str(chunk.get("title")),
        "section": safe_str(chunk.get("section")),
        "section_level": safe_int(chunk.get("section_level"), -1),
        "page_start": safe_int(chunk.get("page_start"), -1),
        "page_end": safe_int(chunk.get("page_end"), -1),
        "chunk_index": safe_int(chunk.get("chunk_index"), -1),
        "chunk_strategy": safe_str(chunk.get("chunk_strategy")),
        "cleaning_version": safe_str(chunk.get("cleaning_version")),
        "chunk_version": safe_str(chunk.get("chunk_version")),
        "chunk_created_at": safe_str(chunk.get("created_at")),
        "extra_json": json_dumps_compact(safe_dict(chunk.get("extra"))),
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "input_schema_version": safe_str(chunk.get("input_schema_version"), CHUNK_SCHEMA_VERSION),
        "embedding_model": embedding_model,
        "embedding_dim": int(embedding_dim),
        "embedding_version": embedding_version,
    }


def insert_records(
    client: MilvusClient,
    collection_name: str,
    records: Sequence[Dict[str, Any]],
    batch_size: int,
) -> int:
    total = 0
    for start in range(0, len(records), batch_size):
        batch = list(records[start : start + batch_size])
        result = client.insert(collection_name=collection_name, data=batch)
        total += len(batch)
        print(f"Inserted batch: {len(batch)}, total={total}, result={result}")
    return total


def build_vector_index_record(
    chunk: Dict[str, Any],
    embedding_model: str,
    embedding_dim: int,
    index_name: str,
    vector_db: str,
    embedding_version: str,
    db_file: str,
    metric_type: str,
) -> Dict[str, Any]:
    # Exactly follows the user's vector_index_record_v1 top-level contract.
    # Operational details are placed into extra.
    return {
        "schema_version": VECTOR_INDEX_RECORD_SCHEMA_VERSION,
        "chunk_id": safe_str(chunk.get("chunk_id")),
        "doc_id": safe_str(chunk.get("doc_id")),
        "source_type": safe_str(chunk.get("source_type"), "offline"),
        "embedding_model": embedding_model,
        "embedding_dim": int(embedding_dim),
        "index_name": index_name,
        "vector_db": vector_db,
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "page_start": safe_nullable_int(chunk.get("page_start")),
        "page_end": safe_nullable_int(chunk.get("page_end")),
        "source_unit_ids": [safe_str(x) for x in safe_list(chunk.get("source_unit_ids"))],
        "cleaning_version": safe_str(chunk.get("cleaning_version")),
        "chunk_version": safe_str(chunk.get("chunk_version")),
        "embedding_version": embedding_version,
        "created_at": now_iso(),
        "extra": {
            "db_file": db_file,
            "metric_type": metric_type,
            "chunk_index": safe_int(chunk.get("chunk_index"), -1),
            "text_length": safe_int(chunk.get("text_length"), 0),
            "token_count": safe_int(chunk.get("token_count"), 0),
            "input_schema_version": safe_str(chunk.get("input_schema_version"), CHUNK_SCHEMA_VERSION),
            "index_status": "success",
        },
    }


def write_vector_index_records(records: Sequence[Dict[str, Any]], output_path: Optional[str]) -> None:
    if not output_path:
        return
    ensure_parent(output_path)
    with Path(output_path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Vector index records written: {output_path}, count={len(records)}")


def search_smoke_test(
    client: MilvusClient,
    collection_name: str,
    query: str,
    query_vector: np.ndarray,
    top_k: int,
    metric_type: str,
) -> None:
    print("\n========== Search Smoke Test ==========")
    if not query:
        print("Search skipped: empty query")
        return
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
            "doc_id",
            "source_type",
            "text",
            "title",
            "section",
            "page_start",
            "page_end",
            "chunk_index",
            "chunk_version",
            "embedding_model",
            "embedding_version",
        ],
    )

    if not result or not result[0]:
        print("Search result is empty.")
        return

    for rank, hit in enumerate(result[0], start=1):
        entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
        score = hit.get("distance", hit.get("score", None)) if isinstance(hit, dict) else None
        text = safe_str(entity.get("text"))
        print(f"[{rank}] score={score}")
        print(f"    chunk_id={entity.get('chunk_id')}")
        print(f"    doc_id={entity.get('doc_id')} page={entity.get('page_start')}~{entity.get('page_end')}")
        print(f"    chunk_version={entity.get('chunk_version')} embedding_version={entity.get('embedding_version')}")
        print(f"    text={text[:220]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunk_v1 and upsert into Milvus Lite; output vector_index_record_v1.")
    parser.add_argument("--input", required=True, help="Path to chunk_v1 JSONL file or Spark output directory.")
    parser.add_argument("--db-file", default="vector_store/milvus_rag.db", help="Milvus Lite db path.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Milvus collection name / index_name.")
    parser.add_argument("--index-record-output", default=None, help="Output JSONL path for vector_index_record_v1.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection if it exists.")

    parser.add_argument("--embedding-model", default=None, help="SentenceTransformer model name or local path. If omitted, tries RAGConfig.")
    parser.add_argument("--embedding-device", default="cuda", help="cuda or cpu for real embedding.")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-version", default=DEFAULT_EMBEDDING_VERSION)
    parser.add_argument("--hash-embedding", action="store_true", help="Use deterministic hash embeddings for smoke test.")
    parser.add_argument("--hash-dim", type=int, default=768)

    parser.add_argument("--metric-type", choices=["COSINE", "IP", "L2"], default="COSINE")
    parser.add_argument("--insert-batch-size", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-text-chars", type=int, default=8192, help="Text length stored in Milvus VARCHAR field.")
    parser.add_argument("--vector-db", default=DEFAULT_VECTOR_DB)

    parser.add_argument("--skip-short-chunk", action="store_true", help="Skip chunks with text_length < 20.")
    parser.add_argument("--skip-low-quality", action="store_true", help="Reserved flag for future quality_flags filtering.")
    parser.add_argument("--query", default="", help="Query for search smoke test.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-search-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("========== Load chunk_v1 ==========")
    chunks = load_chunks(args.input, max_records=args.max_records)
    if args.skip_short_chunk:
        chunks = [c for c in chunks if safe_int(c.get("text_length"), len(safe_str(c.get("text")))) >= 20]
    validate_chunks(chunks)
    input_versions = sorted({safe_str(c.get("input_schema_version"), CHUNK_SCHEMA_VERSION) for c in chunks})
    print(f"Loaded chunks: {len(chunks)}")
    print(f"Input schema versions: {input_versions}")
    print(f"First chunk_id: {chunks[0].get('chunk_id')}")

    print("\n========== Embedding ==========")
    texts = [safe_str(c.get("text")) for c in chunks]
    if args.hash_embedding:
        embeddings = hash_embed_texts(texts, dim=args.hash_dim)
        embedding_model = f"hash_embedding_dim_{args.hash_dim}_for_smoke_test"
        if args.embedding_version == DEFAULT_EMBEDDING_VERSION:
            args.embedding_version = "hash_embedding_v1.0"
    else:
        embeddings, embedding_model = real_embed_texts(
            texts,
            model_name=args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
    if len(embeddings.shape) != 2:
        raise ValueError(f"Embedding shape must be 2D, got {embeddings.shape}")
    dim = int(embeddings.shape[1])
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding model: {embedding_model}")
    print(f"Embedding dim: {dim}")
    print(f"Embedding version: {args.embedding_version}")

    print("\n========== Milvus Lite ==========")
    ensure_parent(args.db_file)
    print(f"DB file: {args.db_file}")
    print(f"Collection / index_name: {args.collection_name}")
    client = MilvusClient(args.db_file)

    create_or_reset_collection(
        client=client,
        collection_name=args.collection_name,
        dim=dim,
        metric_type=args.metric_type,
        recreate=args.recreate,
        max_text_chars=args.max_text_chars,
    )

    milvus_records = [
        build_milvus_record(
            chunk=c,
            vector=embeddings[i],
            embedding_model=embedding_model,
            embedding_dim=dim,
            embedding_version=args.embedding_version,
            max_text_chars=args.max_text_chars,
        )
        for i, c in enumerate(chunks)
    ]
    inserted = insert_records(
        client=client,
        collection_name=args.collection_name,
        records=milvus_records,
        batch_size=args.insert_batch_size,
    )

    # Milvus Lite on Windows can fail explicit flush with WinError 183 because os.rename
    # cannot overwrite an existing manifest.json. For local smoke test, skip explicit flush.
    if os.name == "nt":
        print("Skip explicit flush on Windows Milvus Lite.")
    else:
        client.flush(args.collection_name)
        print("Flush success.")

    vector_index_records = [
        build_vector_index_record(
            chunk=c,
            embedding_model=embedding_model,
            embedding_dim=dim,
            index_name=args.collection_name,
            vector_db=args.vector_db,
            embedding_version=args.embedding_version,
            db_file=args.db_file,
            metric_type=args.metric_type,
        )
        for c in chunks
    ]
    write_vector_index_records(vector_index_records, args.index_record_output)

    print("\n========== Insert Summary ==========")
    print(f"inputPath       = {args.input}")
    print(f"dbFile          = {args.db_file}")
    print(f"collectionName  = {args.collection_name}")
    print(f"indexRecordPath = {args.index_record_output}")
    print(f"inserted        = {inserted}")
    print(f"dim             = {dim}")

    if not args.no_search_test:
        if args.query:
            if args.hash_embedding:
                query_vec = hash_embedding_one(args.query, dim=dim)
            else:
                q_emb, _ = real_embed_texts([args.query], model_name=args.embedding_model or embedding_model, device=args.embedding_device, batch_size=1)
                query_vec = q_emb[0]
            search_smoke_test(
                client=client,
                collection_name=args.collection_name,
                query=args.query,
                query_vector=query_vec,
                top_k=args.top_k,
                metric_type=args.metric_type,
            )
        else:
            print("\n========== Search Smoke Test ==========")
            print("Search skipped: --query not provided")

    print("\nMilvus Lite upsert job finished")


if __name__ == "__main__":
    main()

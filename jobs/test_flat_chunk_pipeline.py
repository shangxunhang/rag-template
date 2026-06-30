#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flat Chunk Pipeline Test
========================

Purpose:
    Validate the refactored flat-chunk pipeline after modularization.

It tests:
    1. Read flat chunk JSONL / Spark part-* output
    2. Normalize to chunk_v1
    3. Validate required chunk fields
    4. Generate embeddings: hash by default, real model optionally
    5. Create Milvus Lite collection
    6. Insert chunk vectors
    7. Write vector_index_record_v1 JSONL
    8. Optional top-k search check

Typical quick test on Windows:

python jobs/test_flat_chunk_pipeline.py ^
  --input D:\\MyCode\\rag-template\\data\\processed\\chunk_unit_test\\part-00000 ^
  --db-file D:\\MyCode\\rag-template\\vector_store\\test_flat_chunk_milvus.db ^
  --collection-name test_flat_chunk_units ^
  --index-record-output D:\\MyCode\\rag-template\\data\\processed\\vector_index_record\\test_vector_index_record_v1.jsonl ^
  --clean-db ^
  --embedding-mode hash ^
  --query 什么是高效学习 ^
  --top-k 3

Real embedding test:

python jobs/test_flat_chunk_pipeline.py ^
  --input D:\\MyCode\\rag-template\\data\\processed\\chunk_unit_test\\part-00000 ^
  --db-file D:\\MyCode\\rag-template\\vector_store\\test_flat_chunk_milvus_real.db ^
  --collection-name test_flat_chunk_units_real ^
  --index-record-output D:\\MyCode\\rag-template\\data\\processed\\vector_index_record\\test_vector_index_record_v1_real.jsonl ^
  --clean-db ^
  --embedding-mode real ^
  --embedding-device cuda ^
  --query 什么是高效学习 ^
  --top-k 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from pymilvus import MilvusClient

# Make project root and src importable when running jobs/*.py directly.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rag_template.configs.SchemaConfig import DEFAULT_VECTOR_DB  # noqa: E402
from rag_template.util.jsonl_utils import ensure_parent, load_jsonl_dicts, write_jsonl  # noqa: E402
from rag_template.schema.Chunk_Schema import normalize_chunk_v1, validate_chunk_v1_records  # noqa: E402
from rag_template.embed.embedding_service import (  # noqa: E402
    encode_query_with_hash,
    encode_query_with_model,
    encode_texts_with_hash,
    encode_texts_with_model,
)
from rag_template.vector_store.milvus_chunk_store import (  # noqa: E402
    build_milvus_chunk_record,
    create_or_reset_chunk_collection,
    insert_records,
    safe_int,
    safe_str,
    search_smoke_test,
)
from rag_template.schema.VectorIndexRecord_Schema import build_vector_index_record_v1  # noqa: E402


VECTOR_INDEX_RECORD_V1_REQUIRED_FIELDS = [
    "schema_version",
    "chunk_id",
    "doc_id",
    "source_type",
    "embedding_model",
    "embedding_dim",
    "index_name",
    "vector_db",
    "title",
    "section",
    "page_start",
    "page_end",
    "source_unit_ids",
    "cleaning_version",
    "chunk_version",
    "embedding_version",
    "created_at",
    "extra",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test flat chunk_v1 -> embedding -> Milvus Lite -> vector_index_record_v1.")

    parser.add_argument(
        "--input",
        default="data/processed/chunk_unit_test/part-00000",
        help="Flat chunk_v1 JSONL file or Spark output directory.",
    )
    parser.add_argument(
        "--db-file",
        default="vector_store/test_flat_chunk_milvus.db",
        help="Milvus Lite db path used by this test.",
    )
    parser.add_argument(
        "--collection-name",
        default="test_flat_chunk_units",
        help="Milvus collection name used by this test.",
    )
    parser.add_argument(
        "--index-record-output",
        default="data/processed/vector_index_record/test_vector_index_record_v1.jsonl",
        help="Output JSONL path for vector_index_record_v1 test records.",
    )

    parser.add_argument("--clean-db", action="store_true", help="Delete --db-file before testing. Useful for Milvus Lite on Windows.")
    parser.add_argument("--recreate", action="store_true", default=True, help="Drop and recreate collection if it exists.")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records for a faster test. Default tests all records.")

    parser.add_argument("--embedding-mode", choices=["hash", "real"], default="hash", help="Use hash or real embedding.")
    parser.add_argument("--hash-dim", type=int, default=768)
    parser.add_argument("--embedding-model", default=None, help="SentenceTransformer model path/name. If omitted, tries RAGConfig.")
    parser.add_argument("--embedding-device", default="cuda", help="cuda or cpu for real embedding.")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-version", default=None, help="Override embedding version. Default comes from embedding service.")

    parser.add_argument("--metric-type", choices=["COSINE", "IP", "L2"], default="COSINE")
    parser.add_argument("--insert-batch-size", type=int, default=128)
    parser.add_argument("--max-text-chars", type=int, default=8192)
    parser.add_argument("--vector-db", default=DEFAULT_VECTOR_DB)

    parser.add_argument("--query", default="什么是高效学习", help="Query for top-k search test.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-search-test", action="store_true")
    parser.add_argument("--skip-milvus", action="store_true", help="Only test JSONL loading, schema normalization and embedding.")

    return parser.parse_args()


def remove_path_if_exists(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def load_normalized_chunks(input_path: str, max_records: int | None) -> List[Dict[str, Any]]:
    raw_records = load_jsonl_dicts(input_path, max_records=max_records)
    chunks = [normalize_chunk_v1(r) for r in raw_records]
    validate_chunk_v1_records(chunks)
    return chunks


def assert_flat_chunk_contract(chunks: Sequence[Dict[str, Any]]) -> None:
    if not chunks:
        raise AssertionError("No chunks loaded.")

    required = [
        "schema_version",
        "chunk_id",
        "doc_id",
        "source_type",
        "text",
        "text_length",
        "token_count",
        "source_unit_ids",
        "chunk_index",
        "chunk_strategy",
        "cleaning_version",
        "chunk_version",
        "created_at",
        "extra",
    ]

    problems: List[str] = []
    for idx, chunk in enumerate(chunks):
        for field in required:
            if field not in chunk:
                problems.append(f"idx={idx}: missing {field}")
        if chunk.get("schema_version") != "chunk_v1":
            problems.append(f"idx={idx}: schema_version={chunk.get('schema_version')}")
        if not chunk.get("chunk_id"):
            problems.append(f"idx={idx}: empty chunk_id")
        if not chunk.get("doc_id"):
            problems.append(f"idx={idx}: empty doc_id")
        if not chunk.get("text"):
            problems.append(f"idx={idx}: empty text")
        if len(problems) >= 20:
            break

    if problems:
        raise AssertionError("Flat chunk contract failed:\n" + "\n".join(problems))


def build_embeddings(args: argparse.Namespace, texts: List[str]):
    if args.embedding_mode == "hash":
        vectors, embedding_model, embedding_version = encode_texts_with_hash(texts, dim=args.hash_dim)
    else:
        version = args.embedding_version or "embedding_v1.0"
        vectors, embedding_model, embedding_version = encode_texts_with_model(
            texts=texts,
            model_name=args.embedding_model,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            embedding_version=version,
        )

    if args.embedding_version:
        embedding_version = args.embedding_version

    if len(vectors.shape) != 2:
        raise AssertionError(f"Embedding shape must be 2D, got {vectors.shape}")
    if len(vectors) != len(texts):
        raise AssertionError(f"Embedding count mismatch: vectors={len(vectors)}, texts={len(texts)}")
    if not np.isfinite(vectors).all():
        raise AssertionError("Embedding contains NaN or Inf.")

    return vectors.astype("float32"), embedding_model, embedding_version


def build_query_vector(args: argparse.Namespace, embedding_model: str, dim: int) -> np.ndarray:
    if args.embedding_mode == "hash":
        return encode_query_with_hash(args.query, dim=dim)
    return encode_query_with_model(
        query=args.query,
        model_name=args.embedding_model or embedding_model,
        device=args.embedding_device,
        batch_size=1,
    )


def assert_vector_index_record_contract(records: Sequence[Dict[str, Any]], expected_count: int) -> None:
    if len(records) != expected_count:
        raise AssertionError(f"vector_index_record count mismatch: {len(records)} != {expected_count}")

    problems: List[str] = []
    for idx, rec in enumerate(records):
        for field in VECTOR_INDEX_RECORD_V1_REQUIRED_FIELDS:
            if field not in rec:
                problems.append(f"idx={idx}: missing {field}")
        if rec.get("schema_version") != "vector_index_record_v1":
            problems.append(f"idx={idx}: schema_version={rec.get('schema_version')}")
        if not rec.get("chunk_id"):
            problems.append(f"idx={idx}: empty chunk_id")
        if not rec.get("embedding_model"):
            problems.append(f"idx={idx}: empty embedding_model")
        if safe_int(rec.get("embedding_dim"), 0) <= 0:
            problems.append(f"idx={idx}: invalid embedding_dim")
        if len(problems) >= 20:
            break

    if problems:
        raise AssertionError("vector_index_record_v1 contract failed:\n" + "\n".join(problems))


def print_search_hits(hits: Sequence[Dict[str, Any]]) -> None:
    print("\n========== Search Result ==========")
    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
        score = hit.get("distance", hit.get("score", None)) if isinstance(hit, dict) else None
        print(f"[{rank}] score={score}")
        print(f"    chunk_id={entity.get('chunk_id')}")
        print(f"    doc_id={entity.get('doc_id')} page={entity.get('page_start')}~{entity.get('page_end')}")
        print(f"    chunk_version={entity.get('chunk_version')} embedding_version={entity.get('embedding_version')}")
        print(f"    text={safe_str(entity.get('text'))[:180]}")


def main() -> None:
    args = parse_args()

    print("========== Flat Chunk Test: Load ==========")
    chunks = load_normalized_chunks(args.input, max_records=args.max_records)
    assert_flat_chunk_contract(chunks)
    input_versions = sorted({safe_str(c.get("input_schema_version"), "chunk_v1") for c in chunks})
    doc_count = len({safe_str(c.get("doc_id")) for c in chunks})
    print(f"inputPath       = {args.input}")
    print(f"loadedChunks    = {len(chunks)}")
    print(f"docCount        = {doc_count}")
    print(f"inputVersions   = {input_versions}")
    print(f"firstChunkId    = {chunks[0].get('chunk_id')}")
    print("PASS: chunk_v1 load + normalize + validate")

    print("\n========== Flat Chunk Test: Embedding ==========")
    texts = [safe_str(c.get("text")) for c in chunks]
    embeddings, embedding_model, embedding_version = build_embeddings(args, texts)
    dim = int(embeddings.shape[1])
    print(f"embeddingMode   = {args.embedding_mode}")
    print(f"embeddingModel  = {embedding_model}")
    print(f"embeddingVersion= {embedding_version}")
    print(f"embeddingShape  = {embeddings.shape}")
    print("PASS: embedding generated")

    if args.skip_milvus:
        print("\nPASS: flat chunk test finished without Milvus (--skip-milvus).")
        return

    print("\n========== Flat Chunk Test: Milvus Lite ==========")
    if args.clean_db:
        remove_path_if_exists(args.db_file)
        print(f"cleanDb         = {args.db_file}")
    ensure_parent(args.db_file)

    client = MilvusClient(args.db_file)
    create_or_reset_chunk_collection(
        client=client,
        collection_name=args.collection_name,
        dim=dim,
        metric_type=args.metric_type,
        recreate=True,
        max_text_chars=args.max_text_chars,
    )

    milvus_records = [
        build_milvus_chunk_record(
            chunk=chunk,
            vector=embeddings[i],
            embedding_model=embedding_model,
            embedding_dim=dim,
            embedding_version=embedding_version,
            max_text_chars=args.max_text_chars,
        )
        for i, chunk in enumerate(chunks)
    ]
    inserted = insert_records(
        client=client,
        collection_name=args.collection_name,
        records=milvus_records,
        batch_size=args.insert_batch_size,
    )
    if inserted != len(chunks):
        raise AssertionError(f"Inserted count mismatch: {inserted} != {len(chunks)}")

    if os.name == "nt":
        print("Skip explicit flush on Windows Milvus Lite.")
    else:
        client.flush(args.collection_name)
        print("Flush success.")
    print("PASS: Milvus collection created + records inserted")

    print("\n========== Flat Chunk Test: VectorIndexRecord ==========")
    vector_index_records = [
        build_vector_index_record_v1(
            chunk=chunk,
            embedding_model=embedding_model,
            embedding_dim=dim,
            index_name=args.collection_name,
            vector_db=args.vector_db,
            embedding_version=embedding_version,
            extra={
                "db_file": args.db_file,
                "metric_type": args.metric_type,
                "chunk_index": safe_int(chunk.get("chunk_index"), -1),
                "text_length": safe_int(chunk.get("text_length"), 0),
                "token_count": safe_int(chunk.get("token_count"), 0),
                "input_schema_version": safe_str(chunk.get("input_schema_version"), "chunk_v1"),
                "index_status": "success",
                "test_script": "jobs/test_flat_chunk_pipeline.py",
            },
        )
        for chunk in chunks
    ]
    assert_vector_index_record_contract(vector_index_records, expected_count=len(chunks))
    write_jsonl(vector_index_records, args.index_record_output)
    print(f"indexRecordPath = {args.index_record_output}")
    print(f"indexRecordCount= {len(vector_index_records)}")
    print("PASS: vector_index_record_v1 generated + validated")

    if not args.no_search_test:
        print("\n========== Flat Chunk Test: Search ==========")
        if not args.query:
            raise AssertionError("Search test enabled but --query is empty.")
        query_vector = build_query_vector(args, embedding_model=embedding_model, dim=dim)
        hits = search_smoke_test(
            client=client,
            collection_name=args.collection_name,
            query_vector=query_vector,
            top_k=args.top_k,
            metric_type=args.metric_type,
        )
        if not hits:
            raise AssertionError("Milvus search returned empty results.")
        print_search_hits(hits)
        print("PASS: Milvus top-k search returned results")

    print("\n========== TEST PASSED ==========")
    print("flat chunk_v1 -> embedding -> Milvus Lite -> vector_index_record_v1 is OK")


if __name__ == "__main__":
    main()

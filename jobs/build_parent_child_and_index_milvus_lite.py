#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jobs/build_parent_child_and_index_milvus_lite.py
================================================

今天目标版：cleaned_text_unit_v1 -> parent_chunk_v1 / child_chunk_v1 -> child embedding -> Milvus Lite。

输入：cleaned_text_unit_v1 JSONL 文件或目录。
输出：
1. parent_chunks.jsonl
2. child_chunks.jsonl
3. Milvus Lite child chunk collection
4. vector_index_record_v2.jsonl

设计约定：
- parent_chunk_v1 不入向量库，先落 JSONL，后续检索阶段通过 parent_chunk_id 回填。
- child_chunk_v1 做 embedding 并入 Milvus。
- Milvus metadata 必须包含 parent_chunk_id。

Windows smoke test 示例：
python jobs/build_parent_child_and_index_milvus_lite.py ^
  --input data\\processed\\cleaned_text_unit\\cleaned_text_unit_v1.jsonl ^
  --parent-output data\\processed\\parent_child_chunks\\parent_chunks.jsonl ^
  --child-output data\\processed\\parent_child_chunks\\child_chunks.jsonl ^
  --db-file vector_store\\milvus_parent_child.db ^
  --collection-name rag_child_chunks ^
  --index-record-output data\\processed\\vector_index_record\\vector_index_record_v2.jsonl ^
  --recreate ^
  --embedding-model D:\\models\\huggingface\\embedding\\m3e-base ^
  --embedding-device cuda ^
  --query "什么是 RAG 父子块" ^
  --top-k 5

无模型 smoke test：
python jobs/build_parent_child_and_index_milvus_lite.py ^
  --input data\\processed\\cleaned_text_unit\\cleaned_text_unit_v1.jsonl ^
  --parent-output data\\processed\\parent_child_chunks\\parent_chunks.jsonl ^
  --child-output data\\processed\\parent_child_chunks\\child_chunks.jsonl ^
  --db-file vector_store\\milvus_parent_child.db ^
  --collection-name rag_child_chunks ^
  --index-record-output data\\processed\\vector_index_record\\vector_index_record_v2.jsonl ^
  --recreate ^
  --hash-embedding ^
  --query "什么是 RAG 父子块"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pymilvus import MilvusClient


# Make project root and src importable when script is run directly.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rag_template.chunker.ChildParentChunker import ChildParentChunker
from rag_template.configs.SchemaConfig import DEFAULT_EMBEDDING_VERSION, DEFAULT_VECTOR_DB
from rag_template.schema.VectorIndexRecord_Schema import build_vector_index_record_v2
from rag_template.vector_store.milvus_child_chunk_store import (
    build_milvus_child_chunk_record,
    create_or_reset_child_chunk_collection,
    insert_child_chunk_records,
    search_child_chunk_smoke_test,
)


DEFAULT_COLLECTION_NAME = "rag_child_chunks"
DEFAULT_HASH_DIM = 768


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


def iter_jsonl_paths(input_path: str | Path) -> Iterable[Path]:
    p = Path(input_path)
    if p.is_file():
        yield p
        return
    if p.is_dir():
        candidates = sorted(
            x for x in p.iterdir()
            if x.is_file() and (x.name.startswith("part-") or x.suffix.lower() in {".jsonl", ".json"})
        )
        for item in candidates:
            yield item
        return
    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_jsonl_records(input_path: str | Path, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in iter_jsonl_paths(input_path):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL: file={path}, line={line_no}, err={exc}") from exc
                if not isinstance(obj, dict):
                    continue
                records.append(obj)
                if max_records is not None and len(records) >= max_records:
                    return records
    return records


def write_jsonl(records: Sequence[Dict[str, Any]], output_path: str | Path) -> None:
    ensure_parent(output_path)
    with Path(output_path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"JSONL written: {output_path}, count={len(records)}")


def validate_parent_child_records(parents: Sequence[Dict[str, Any]], children: Sequence[Dict[str, Any]]) -> None:
    if not parents:
        raise ValueError("No parent_chunk_v1 records generated.")
    if not children:
        raise ValueError("No child_chunk_v1 records generated.")

    parent_map = {safe_str(p.get("parent_chunk_id")): p for p in parents if p.get("parent_chunk_id")}
    child_map = {safe_str(c.get("child_chunk_id")): c for c in children if c.get("child_chunk_id")}

    if len(parent_map) != len(parents):
        raise ValueError("Duplicate or empty parent_chunk_id detected.")
    if len(child_map) != len(children):
        raise ValueError("Duplicate or empty child_chunk_id detected.")

    errors: List[str] = []

    for c in children:
        child_id = safe_str(c.get("child_chunk_id"))
        chunk_id = safe_str(c.get("chunk_id"))
        parent_id = safe_str(c.get("parent_chunk_id"))
        if not child_id:
            errors.append("child missing child_chunk_id")
        if chunk_id != child_id:
            errors.append(f"child chunk_id != child_chunk_id: {chunk_id} != {child_id}")
        if parent_id not in parent_map:
            errors.append(f"child parent not found: child={child_id}, parent={parent_id}")
        if not safe_str(c.get("doc_id")):
            errors.append(f"child missing doc_id: {child_id}")
        if not safe_str(c.get("text")).strip():
            errors.append(f"child empty text: {child_id}")
        if len(errors) >= 20:
            break

    for p in parents:
        parent_id = safe_str(p.get("parent_chunk_id"))
        child_ids = p.get("child_chunk_ids") or []
        if safe_int(p.get("child_count"), -1) != len(child_ids):
            errors.append(f"parent child_count mismatch: {parent_id}")
        for child_id in child_ids:
            child_id = safe_str(child_id)
            if child_id not in child_map:
                errors.append(f"parent references missing child: parent={parent_id}, child={child_id}")
            elif safe_str(child_map[child_id].get("parent_chunk_id")) != parent_id:
                errors.append(f"child reverse parent mismatch: parent={parent_id}, child={child_id}")
        if len(errors) >= 20:
            break

    if errors:
        raise ValueError("Parent-child validation failed:\n" + "\n".join(errors))


# =========================================================
# Embedding
# =========================================================


def chunk_text_for_hash(text: str) -> Iterable[str]:
    text = "".join(text.split())
    if not text:
        return []
    if len(text) == 1:
        return [text]
    return (text[i:i + 2] for i in range(len(text) - 1))


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
    try:
        from rag_template.configs.RAGConfig import EMBEDDING_MODEL_NAME  # type: ignore
        if EMBEDDING_MODEL_NAME:
            return str(EMBEDDING_MODEL_NAME)
    except Exception:
        return None
    return None


def resolve_default_embedding_device() -> str:
    try:
        from rag_template.configs.RAGConfig import EMBEDDING_DEVICE  # type: ignore
        if EMBEDDING_DEVICE:
            return str(EMBEDDING_DEVICE)
    except Exception:
        return "cuda"
    return "cuda"


def resolve_default_embedding_batch_size() -> int:
    try:
        from rag_template.configs.RAGConfig import EMBEDDING_BATCH_SIZE  # type: ignore
        return int(EMBEDDING_BATCH_SIZE)
    except Exception:
        return 32


def real_embed_texts(
    texts: Sequence[str],
    model_name: Optional[str],
    device: Optional[str],
    batch_size: Optional[int],
) -> Tuple[np.ndarray, str]:
    if not model_name:
        model_name = resolve_default_embedding_model()
    if not model_name:
        raise ValueError(
            "No embedding model resolved. Pass --embedding-model <local_model_path_or_name> "
            "or use --hash-embedding for smoke test."
        )
    if not device:
        device = resolve_default_embedding_device()
    if not batch_size:
        batch_size = resolve_default_embedding_batch_size()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_name), device=str(device))
    vectors = model.encode(
        list(texts),
        batch_size=int(batch_size),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32), str(model_name)


def embed_children(
    children: Sequence[Dict[str, Any]],
    *,
    hash_embedding: bool,
    hash_dim: int,
    embedding_model: Optional[str],
    embedding_device: Optional[str],
    embedding_batch_size: Optional[int],
    embedding_version: str,
) -> Tuple[np.ndarray, str, str]:
    texts = [safe_str(c.get("text")) for c in children]
    if hash_embedding:
        vectors = hash_embed_texts(texts, dim=hash_dim)
        model_name = f"hash_embedding_dim_{hash_dim}_for_smoke_test"
        version = "hash_embedding_v1.0" if embedding_version == DEFAULT_EMBEDDING_VERSION else embedding_version
        return vectors, model_name, version

    vectors, model_name = real_embed_texts(
        texts,
        model_name=embedding_model,
        device=embedding_device,
        batch_size=embedding_batch_size,
    )
    return vectors, model_name, embedding_version


# =========================================================
# CLI
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build parent/child chunks from cleaned_text_unit_v1, index child chunks into Milvus Lite, and write vector_index_record_v2."
    )
    parser.add_argument("--input", required=True, help="Path to cleaned_text_unit_v1 JSONL file or Spark output directory.")
    parser.add_argument("--parent-output", required=True, help="Output path for parent_chunk_v1 JSONL.")
    parser.add_argument("--child-output", required=True, help="Output path for child_chunk_v1 JSONL.")
    parser.add_argument("--index-record-output", required=True, help="Output path for vector_index_record_v2 JSONL.")

    parser.add_argument("--db-file", default="vector_store/milvus_parent_child.db", help="Milvus Lite db path.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Milvus collection name / index_name.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection if exists.")
    parser.add_argument("--metric-type", choices=["COSINE", "IP", "L2"], default="COSINE")
    parser.add_argument("--insert-batch-size", type=int, default=128)
    parser.add_argument("--max-text-chars", type=int, default=8192)
    parser.add_argument("--vector-db", default=DEFAULT_VECTOR_DB)

    parser.add_argument("--embedding-model", default=None, help="SentenceTransformer model name or local path. If omitted, tries RAGConfig.")
    parser.add_argument("--embedding-device", default=None, help="cuda or cpu. If omitted, tries RAGConfig.EMBEDDING_DEVICE.")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-version", default=DEFAULT_EMBEDDING_VERSION)
    parser.add_argument("--hash-embedding", action="store_true", help="Use deterministic hash embeddings for smoke test.")
    parser.add_argument("--hash-dim", type=int, default=DEFAULT_HASH_DIM)

    parser.add_argument("--max-records", type=int, default=None, help="Limit input records for smoke testing.")
    parser.add_argument("--parent-chunk-size", type=int, default=None)
    parser.add_argument("--parent-chunk-overlap", type=int, default=None)
    parser.add_argument("--child-chunk-size", type=int, default=None)
    parser.add_argument("--child-chunk-overlap", type=int, default=None)
    parser.add_argument("--unit", choices=["char", "token"], default=None)

    parser.add_argument("--query", default="", help="Query for search smoke test.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-search-test", action="store_true")
    return parser.parse_args()


def print_search_hits(hits: Sequence[Dict[str, Any]]) -> None:
    if not hits:
        print("Search result is empty.")
        return
    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
        score = hit.get("distance", hit.get("score", None)) if isinstance(hit, dict) else None
        text = safe_str(entity.get("text"))
        print(f"[{rank}] score={score}")
        print(f"    child_chunk_id={entity.get('child_chunk_id') or entity.get('chunk_id')}")
        print(f"    parent_chunk_id={entity.get('parent_chunk_id')}")
        print(f"    doc_id={entity.get('doc_id')} page={entity.get('page_start')}~{entity.get('page_end')}")
        print(f"    child_index_in_parent={entity.get('child_index_in_parent')}")
        print(f"    text={text[:220]}")


def main() -> None:
    args = parse_args()

    print("========== Load cleaned_text_unit_v1 ==========")
    cleaned_records = load_jsonl_records(args.input, max_records=args.max_records)
    if not cleaned_records:
        raise ValueError(f"No cleaned_text_unit_v1 records loaded: {args.input}")
    print(f"Loaded cleaned records: {len(cleaned_records)}")

    print("\n========== Build parent/child chunks ==========")
    chunker = ChildParentChunker(
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        unit=args.unit,
    )
    result = chunker.chunk_records(cleaned_records)
    parents = result.parents
    children = result.children
    validate_parent_child_records(parents, children)
    print(f"Generated parents: {len(parents)}")
    print(f"Generated children: {len(children)}")
    print(f"First parent_chunk_id: {parents[0].get('parent_chunk_id')}")
    print(f"First child_chunk_id : {children[0].get('child_chunk_id')}")
    print(f"First parent link    : {children[0].get('parent_chunk_id')}")

    write_jsonl(parents, args.parent_output)
    write_jsonl(children, args.child_output)

    print("\n========== Embedding child chunks ==========")
    embeddings, embedding_model, embedding_version = embed_children(
        children,
        hash_embedding=args.hash_embedding,
        hash_dim=args.hash_dim,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_version=args.embedding_version,
    )
    if len(embeddings.shape) != 2:
        raise ValueError(f"Embedding shape must be 2D, got {embeddings.shape}")
    dim = int(embeddings.shape[1])
    print(f"Embedding shape  : {embeddings.shape}")
    print(f"Embedding model  : {embedding_model}")
    print(f"Embedding dim    : {dim}")
    print(f"Embedding version: {embedding_version}")

    print("\n========== Milvus Lite child index ==========")
    ensure_parent(args.db_file)
    print(f"DB file             : {args.db_file}")
    print(f"Collection/indexName: {args.collection_name}")
    client = MilvusClient(args.db_file)

    create_or_reset_child_chunk_collection(
        client=client,
        collection_name=args.collection_name,
        dim=dim,
        metric_type=args.metric_type,
        recreate=args.recreate,
        max_text_chars=args.max_text_chars,
    )

    milvus_records = [
        build_milvus_child_chunk_record(
            child_chunk=child,
            vector=embeddings[i],
            embedding_model=embedding_model,
            embedding_dim=dim,
            embedding_version=embedding_version,
            max_text_chars=args.max_text_chars,
        )
        for i, child in enumerate(children)
    ]
    inserted = insert_child_chunk_records(
        client=client,
        collection_name=args.collection_name,
        records=milvus_records,
        batch_size=args.insert_batch_size,
    )

    if os.name == "nt":
        print("Skip explicit flush on Windows Milvus Lite.")
    else:
        client.flush(args.collection_name)
        print("Flush success.")

    print("\n========== Write vector_index_record_v2 ==========")
    vector_index_records = [
        build_vector_index_record_v2(
            child_chunk=child,
            embedding_model=embedding_model,
            embedding_dim=dim,
            index_name=args.collection_name,
            vector_db=args.vector_db,
            embedding_version=embedding_version,
            extra={
                "db_file": args.db_file,
                "metric_type": args.metric_type,
                "index_status": "success",
                "indexed_granularity": "child",
                "milvus_primary_key": child.get("chunk_id") or child.get("child_chunk_id"),
            },
        )
        for child in children
    ]
    write_jsonl(vector_index_records, args.index_record_output)

    print("\n========== Insert Summary ==========")
    print(f"input               = {args.input}")
    print(f"parentOutput        = {args.parent_output}")
    print(f"childOutput         = {args.child_output}")
    print(f"indexRecordOutput   = {args.index_record_output}")
    print(f"dbFile              = {args.db_file}")
    print(f"collectionName      = {args.collection_name}")
    print(f"parents             = {len(parents)}")
    print(f"children            = {len(children)}")
    print(f"inserted            = {inserted}")
    print(f"dim                 = {dim}")

    if not args.no_search_test:
        print("\n========== Search Smoke Test ==========")
        if args.query:
            if args.hash_embedding:
                query_vec = hash_embedding_one(args.query, dim=dim)
            else:
                q_emb, _ = real_embed_texts(
                    [args.query],
                    model_name=args.embedding_model or embedding_model,
                    device=args.embedding_device,
                    batch_size=1,
                )
                query_vec = q_emb[0]
            hits = search_child_chunk_smoke_test(
                client=client,
                collection_name=args.collection_name,
                query_vector=query_vec,
                top_k=args.top_k,
                metric_type=args.metric_type,
            )
            print_search_hits(hits)
        else:
            print("Search skipped: --query not provided")

    print("\nParent-child chunk indexing job finished")


if __name__ == "__main__":
    main()

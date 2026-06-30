# -*- coding: utf-8 -*-
"""
Scripts/test_hybrid_parent_child_retriever.py
============================================

P2 smoke test:
query -> dense child retrieval + BM25 child retrieval -> RRF fusion -> parent backfill -> retrieval_result_v2.

PyCharm recommended:
- Script path: D:\\MyCode\\rag-template\\Scripts\\test_hybrid_parent_child_retriever.py
- Working directory: D:\\MyCode\\rag-template
- Environment variables: PYTHONPATH=D:\\MyCode\\rag-template\\src
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rag_template.configs.RAGConfig import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    PARENT_CHILD_MILVUS_COLLECTION_NAME,
    PARENT_CHILD_MILVUS_DB_FILE,
    PARENT_CHILD_SEARCH_METRIC_TYPE,
    PARENT_CHUNKS_FILE,
    CHILD_CHUNKS_FILE,
)
from rag_template.store.parent_chunk_store import ParentChunkStore
from rag_template.retriever.bm25_child_retriever import BM25ChildRetriever
from rag_template.retriever.milvus_child_retriever import MilvusChildRetriever
from rag_template.retriever.hybrid_parent_child_retriever import HybridParentChildRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2 hybrid parent-child retriever smoke test")
    parser.add_argument("--query", type=str, default="整体性学习是什么")

    parser.add_argument("--parent-file", type=str, default=PARENT_CHUNKS_FILE)
    parser.add_argument("--child-file", type=str, default=CHILD_CHUNKS_FILE)

    parser.add_argument("--db-file", type=str, default=PARENT_CHILD_MILVUS_DB_FILE)
    parser.add_argument("--collection-name", type=str, default=PARENT_CHILD_MILVUS_COLLECTION_NAME)
    parser.add_argument("--metric-type", type=str, default=PARENT_CHILD_SEARCH_METRIC_TYPE)

    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--embedding-device", type=str, default=EMBEDDING_DEVICE)
    parser.add_argument("--hash-embedding", action="store_true")
    parser.add_argument("--hash-dim", type=int, default=768)

    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--keyword-top-k", type=int, default=10)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument("--filter-expr", type=str, default=None)
    parser.add_argument("--keyword-doc-id", type=str, default=None)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-keyword", action="store_true")
    parser.add_argument("--no-dedup-parent", action="store_true")

    return parser.parse_args()


def preview_text(text: str, n: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n]


def main() -> None:
    args = parse_args()

    print("========== Load ParentChunkStore ==========")
    print(f"parent_file = {args.parent_file}")
    parent_store = ParentChunkStore.from_jsonl(args.parent_file)
    print(f"parents     = {len(parent_store)}")

    print("\n========== Load BM25ChildRetriever ==========")
    print(f"child_file  = {args.child_file}")
    keyword_retriever = None if args.no_keyword else BM25ChildRetriever.from_jsonl(args.child_file)
    print(f"children    = {len(keyword_retriever) if keyword_retriever is not None else 0}")

    print("\n========== Init MilvusChildRetriever ==========")
    print(f"db_file       = {args.db_file}")
    print(f"collection    = {args.collection_name}")
    print(f"metric_type   = {args.metric_type}")
    print(f"embedding     = {args.embedding_model}")
    dense_retriever = None
    if not args.no_dense:
        dense_retriever = MilvusChildRetriever(
            db_file=args.db_file,
            collection_name=args.collection_name,
            metric_type=args.metric_type,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            hash_embedding=args.hash_embedding,
            hash_dim=args.hash_dim,
        )

    retriever = HybridParentChildRetriever(
        dense_retriever=dense_retriever,
        keyword_retriever=keyword_retriever,
        parent_store=parent_store,
        rrf_k=args.rrf_k,
        dedup_parent=not args.no_dedup_parent,
    )

    print("\n========== Hybrid Retrieve ==========")
    results = retriever.retrieve(
        query=args.query,
        final_top_k=args.final_top_k,
        dense_top_k=args.dense_top_k,
        keyword_top_k=args.keyword_top_k,
        filter_expr=args.filter_expr,
        keyword_doc_id=args.keyword_doc_id,
        use_dense=not args.no_dense,
        use_keyword=not args.no_keyword,
        dedup_parent=not args.no_dedup_parent,
    )

    for item in results:
        meta = item.get("metadata", {})
        print(
            f"[{item.get('rank')}] fusion={item.get('score')} "
            f"dense={meta.get('dense_score')} keyword={meta.get('keyword_score')} "
            f"sources={meta.get('retrieval_sources')}"
        )
        print(f"    child_chunk_id  ={item.get('child_chunk_id')}")
        print(f"    parent_chunk_id ={item.get('parent_chunk_id')}")
        print(f"    doc_id          ={item.get('doc_id')}")
        print(f"    title/section   ={item.get('title')} / {item.get('section')}")
        print(f"    page            ={item.get('page_start')}~{item.get('page_end')}")
        print(f"    parent_found    ={meta.get('parent_found')}")
        print(f"    matched_children={meta.get('matched_child_count')} {meta.get('matched_child_chunk_ids')}")
        print(f"    child_text      ={preview_text(item.get('child_text', ''))}")
        print(f"    parent_text     ={preview_text(item.get('parent_text', ''))}")

    print("\n========== Assertions ==========")
    assert results, "No hybrid retrieval results returned"
    for item in results:
        assert item.get("schema_version") == "retrieval_result_v2"
        assert item.get("child_chunk_id")
        assert item.get("parent_chunk_id")
        assert item.get("metadata", {}).get("parent_found") is True
        assert item.get("metadata", {}).get("retrieval_stage") == "p2_hybrid_rrf_parent_backfill"
        assert item.get("text") == item.get("parent_text")
        sources = item.get("metadata", {}).get("retrieval_sources", [])
        assert sources, "retrieval_sources should not be empty"
    print("P2 hybrid parent-child retriever test passed")

    print("\n========== JSON Preview ==========")
    print(json.dumps(results[0], ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()

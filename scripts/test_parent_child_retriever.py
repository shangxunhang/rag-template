#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts/test_parent_child_retriever.py
======================================

P1 smoke test:
query -> Milvus child search -> parent_chunk_id 回填 parent -> retrieval_result_v2。

PyCharm 运行配置：
- Script path: D:\\MyCode\\rag-template\\Scripts\\test_parent_child_retriever.py
- Working directory: D:\\MyCode\\rag-template
- Environment: PYTHONPATH=D:\\MyCode\\rag-template\\src
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rag_template.configs.RAGConfig import (  # noqa: E402
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    MILVUS_DIM,
    PARENT_CHILD_MILVUS_COLLECTION_NAME,
    PARENT_CHILD_MILVUS_DB_FILE,
    PARENT_CHUNKS_FILE,
    PARENT_CHILD_SEARCH_METRIC_TYPE,
)
from rag_template.retriever.milvus_child_retriever import MilvusChildRetriever  # noqa: E402
from rag_template.retriever.parent_child_retriever import ParentChildRetriever  # noqa: E402
from rag_template.store.parent_chunk_store import ParentChunkStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 parent-child dense retrieval smoke test.")
    parser.add_argument("--query", default="什么是RAG父子块", help="User query.")
    parser.add_argument("--parent-file", default=PARENT_CHUNKS_FILE, help="Path to parent_chunks.jsonl.")
    parser.add_argument("--db-file", default=PARENT_CHILD_MILVUS_DB_FILE, help="Milvus Lite db path.")
    parser.add_argument("--collection-name", default=PARENT_CHILD_MILVUS_COLLECTION_NAME)
    parser.add_argument("--metric-type", default=PARENT_CHILD_SEARCH_METRIC_TYPE, choices=["COSINE", "IP", "L2"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--child-top-k", type=int, default=None)
    parser.add_argument("--filter-expr", default=None)
    parser.add_argument("--dedup-parent", action="store_true")
    parser.add_argument("--context-granularity", choices=["parent", "child"], default="parent")

    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--embedding-device", default=EMBEDDING_DEVICE)
    parser.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--hash-embedding", action="store_true", help="Only for smoke test. If index was built by real embedding, ranking is meaningless.")
    parser.add_argument("--hash-dim", type=int, default=MILVUS_DIM)
    return parser.parse_args()


def print_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results.")
        return
    for item in results:
        print(f"[{item.get('rank')}] score={item.get('score')} rerank={item.get('rerank_score')}")
        print(f"    child_chunk_id ={item.get('child_chunk_id')}")
        print(f"    parent_chunk_id={item.get('parent_chunk_id')}")
        print(f"    doc_id         ={item.get('doc_id')}")
        print(f"    title/section  ={item.get('title')} / {item.get('section')}")
        print(f"    page           ={item.get('page_start')}~{item.get('page_end')}")
        print(f"    parent_found   ={item.get('metadata', {}).get('parent_found')}")
        child_text = (item.get("child_text") or "").replace("\n", " ")
        parent_text = (item.get("parent_text") or "").replace("\n", " ")
        print(f"    child_text     ={child_text[:180]}")
        print(f"    parent_text    ={parent_text[:240]}")


def main() -> None:
    args = parse_args()

    print("========== Load ParentChunkStore ==========")
    parent_store = ParentChunkStore.from_jsonl(args.parent_file)
    print(f"parent_file = {args.parent_file}")
    print(f"parents     = {len(parent_store)}")

    print("\n========== Init MilvusChildRetriever ==========")
    child_retriever = MilvusChildRetriever(
        db_file=args.db_file,
        collection_name=args.collection_name,
        metric_type=args.metric_type,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        hash_embedding=args.hash_embedding,
        hash_dim=args.hash_dim,
    )
    print(f"db_file       = {args.db_file}")
    print(f"collection    = {args.collection_name}")
    print(f"metric_type   = {args.metric_type}")
    print(f"embedding     = {'hash' if args.hash_embedding else args.embedding_model}")

    print("\n========== Retrieve ==========")
    retriever = ParentChildRetriever(
        child_retriever=child_retriever,
        parent_store=parent_store,
        context_granularity=args.context_granularity,
        dedup_parent=args.dedup_parent,
    )
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        child_top_k=args.child_top_k,
        filter_expr=args.filter_expr,
    )
    print_results(results)

    print("\n========== Assertions ==========")
    assert results, "P1 failed: empty retrieval results"
    for item in results:
        assert item.get("child_chunk_id"), "missing child_chunk_id"
        assert item.get("parent_chunk_id"), "missing parent_chunk_id"
        assert item.get("child_text"), "missing child_text"
        assert item.get("metadata", {}).get("parent_found") is True, "parent backfill failed"
        if args.context_granularity == "parent":
            assert item.get("parent_text"), "missing parent_text"
            assert item.get("text") == item.get("parent_text"), "text should equal parent_text in parent context mode"
    print("P1 parent-child retriever test passed")

    print("\n========== JSON Preview ==========")
    print(json.dumps(results[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()

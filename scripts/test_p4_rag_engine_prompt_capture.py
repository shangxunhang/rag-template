# -*- coding: utf-8 -*-
"""
Scripts/test_p4_rag_engine_prompt_capture.py
===========================================

P4-lite smoke test:
P3 hybrid retrieval + rerank + context packing
  -> ParentChildRAGEngine
  -> ParentChildPromptBuilder
  -> RagRunCapture JSONL

PyCharm:
- Script path: D:\\MyCode\\rag-template\\Scripts\\test_p4_rag_engine_prompt_capture.py
- Working directory: D:\\MyCode\\rag-template
- Environment variables: PYTHONPATH=D:\\MyCode\\rag-template\\src
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from rag_template.context.context_packer import ContextPacker
from rag_template.data_capture.rag_run_capture import RagRunCapture
from rag_template.prompt.parent_child_prompt_builder import ParentChildPromptBuilder
from rag_template.rag_engine.parent_child_rag_engine import ParentChildRAGEngine
from rag_template.reranker.parent_child_reranker import NoOpParentChildReranker, ParentChildReranker
from rag_template.retriever.bm25_child_retriever import BM25ChildRetriever
from rag_template.retriever.hybrid_parent_child_retriever import HybridParentChildRetriever
from rag_template.retriever.milvus_child_retriever import MilvusChildRetriever
from rag_template.store.parent_chunk_store import ParentChunkStore

try:
    from rag_template.configs.RAGConfig import (
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DEVICE,
        RERANKER_MODEL_NAME,
        RERANKER_DEVICE,
        RERANKER_BATCH_SIZE,
    )
except Exception:
    EMBEDDING_MODEL_NAME = ""
    EMBEDDING_DEVICE = "cuda"
    RERANKER_MODEL_NAME = r"D:\models\huggingface\reranker\bge-reranker-v2-m3"
    RERANKER_DEVICE = "cuda"
    RERANKER_BATCH_SIZE = 16

DEFAULT_RERANKER_MODEL_NAME = r"D:\models\huggingface\reranker\bge-reranker-v2-m3"
DEFAULT_RERANKER_DEVICE = "cuda"


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4-lite RAGEngine + PromptBuilder + DataCapture smoke test")

    parser.add_argument("--query", default="整体性学习是什么")

    parser.add_argument("--parent-file", default="data/processed/parent_child_chunks/parent_chunks.jsonl")
    parser.add_argument("--child-file", default="data/processed/parent_child_chunks/child_chunks.jsonl")
    parser.add_argument("--db-file", default="data/processed/vector_store/milvus_parent_child.db")
    parser.add_argument("--collection-name", default="rag_child_chunks")
    parser.add_argument("--metric-type", default="COSINE")

    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--embedding-device", default=EMBEDDING_DEVICE)
    parser.add_argument("--hash-embedding", action="store_true")

    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--keyword-top-k", type=int, default=10)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL_NAME or DEFAULT_RERANKER_MODEL_NAME)
    parser.add_argument("--reranker-device", default=RERANKER_DEVICE or DEFAULT_RERANKER_DEVICE)
    parser.add_argument("--reranker-batch-size", type=int, default=int(RERANKER_BATCH_SIZE or 16))
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-local-files-only", action="store_true", default=True)
    parser.add_argument("--rerank-top-k", type=int, default=5)

    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-context-items", type=int, default=3)

    parser.add_argument("--expected-doc-ids", default="doc_001_native_text")
    parser.add_argument("--expected-parent-chunk-ids", default="")
    parser.add_argument("--expected-child-chunk-ids", default="")
    parser.add_argument("--expected-keywords", default="整体性学习,学习")
    parser.add_argument("--eval-top-k", type=int, default=5)

    parser.add_argument("--capture-output", default="data/runs/rag_runs.jsonl")

    return parser.parse_args()


def print_results(title: str, results: List[dict], max_items: int = 5) -> None:
    print(f"\n========== {title} ==========")
    for item in results[:max_items]:
        meta = item.get("metadata") or {}
        print(
            f"[{item.get('rank')}] score={item.get('score')} rerank={item.get('rerank_score')} "
            f"sources={meta.get('retrieval_sources')}"
        )
        print(f"    child_chunk_id  ={item.get('child_chunk_id')}")
        print(f"    parent_chunk_id ={item.get('parent_chunk_id')}")
        print(f"    doc_id          ={item.get('doc_id')}")
        print(f"    title/section   ={item.get('title')} / {item.get('section')}")
        print(f"    page            ={item.get('page_start')}~{item.get('page_end')}")
        print(f"    parent_found    ={meta.get('parent_found')}")
        text = str(item.get("text") or "").replace("\n", " ")
        print(f"    text_preview    ={text[:220]}")


def main() -> None:
    args = parse_args()

    print("========== Load ParentChunkStore ==========")
    print(f"parent_file = {args.parent_file}")
    parent_store = ParentChunkStore.from_jsonl(args.parent_file)
    print(f"parents     = {len(parent_store)}")

    print("\n========== Load BM25ChildRetriever ==========")
    print(f"child_file  = {args.child_file}")
    keyword_retriever = BM25ChildRetriever.from_jsonl(args.child_file)
    print(f"children    = {len(keyword_retriever)}")

    print("\n========== Init MilvusChildRetriever ==========")
    print(f"db_file       = {args.db_file}")
    print(f"collection    = {args.collection_name}")
    print(f"metric_type   = {args.metric_type}")
    print(f"embedding     = {args.embedding_model}")
    dense_retriever = MilvusChildRetriever(
        db_file=args.db_file,
        collection_name=args.collection_name,
        metric_type=args.metric_type,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        hash_embedding=args.hash_embedding,
    )

    hybrid_retriever = HybridParentChildRetriever(
        dense_retriever=dense_retriever,
        keyword_retriever=keyword_retriever,
        parent_store=parent_store,
        rrf_k=args.rrf_k,
        dedup_parent=True,
    )

    print("\n========== Init Reranker ==========")
    if args.skip_rerank:
        print("reranker = noop")
        reranker = NoOpParentChildReranker()
    else:
        print(f"reranker_model = {args.reranker_model}")
        reranker = ParentChildReranker(
            model_name=args.reranker_model,
            device=args.reranker_device,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            local_files_only=args.reranker_local_files_only,
        )

    context_packer = ContextPacker(
        max_context_chars=args.max_context_chars,
        max_items=args.max_context_items,
        text_field="text",
        dedup_parent=True,
        include_metadata=True,
    )
    prompt_builder = ParentChildPromptBuilder()
    run_capture = RagRunCapture(args.capture_output)

    engine = ParentChildRAGEngine(
        retriever=hybrid_retriever,
        reranker=reranker,
        context_packer=context_packer,
        prompt_builder=prompt_builder,
        run_capture=run_capture,
    )

    print("\n========== P4 RAGEngine ==========")
    result = engine.run(
        query=args.query,
        dense_top_k=args.dense_top_k,
        keyword_top_k=args.keyword_top_k,
        candidate_top_k=args.candidate_top_k,
        rrf_k=args.rrf_k,
        rerank_top_k=args.rerank_top_k,
        eval_top_k=args.eval_top_k,
        expected_doc_ids=_split_csv(args.expected_doc_ids),
        expected_parent_chunk_ids=_split_csv(args.expected_parent_chunk_ids),
        expected_child_chunk_ids=_split_csv(args.expected_child_chunk_ids),
        expected_keywords=_split_csv(args.expected_keywords),
    )

    print_results("P4 Reranked Retrieval Results", result["retrieval_results"], max_items=args.rerank_top_k)

    print("\n========== Context Pack ==========")
    pack = result["context_pack"]
    print(f"selected_count    = {pack.get('selected_count')}")
    print(f"dropped_count     = {pack.get('dropped_count')}")
    print(f"used_chars        = {pack.get('used_chars')}")
    print(f"max_context_chars = {pack.get('max_context_chars')}")
    print("\n----- Packed Context Preview -----")
    print(result["packed_context"][:1500])

    print("\n========== Prompt Preview ==========")
    print(f"prompt_id      = {result.get('prompt_id')}")
    print(f"prompt_version = {result.get('prompt_version')}")
    print(result["prompt"][:2000])

    print("\n========== Retrieval Eval ==========")
    print(json.dumps(result["eval_result"], ensure_ascii=False, indent=2))

    print("\n========== DataCapture ==========")
    capture = result.get("capture_result") or {}
    print(f"capture_output = {capture.get('output_path')}")
    print(f"saved          = {capture.get('saved')}")
    print(f"run_id         = {capture.get('run_id')}")

    print("\n========== Assertions ==========")
    assert result["retrieval_results"], "retrieval_results should not be empty"
    assert result["packed_context"].strip(), "packed_context should not be empty"
    assert "【资料】" in result["prompt"], "prompt should include context marker"
    assert args.query in result["prompt"], "prompt should include query"
    assert capture.get("saved") is True, "DataCapture should save run record"
    assert Path(args.capture_output).exists(), "capture output file should exist"
    assert result["run_record"].get("answer") is None, "P4-lite should not call LLM yet"
    print("P4 RAGEngine + PromptBuilder + DataCapture test passed")

    print("\n========== JSON Preview ==========")
    preview = {
        "run_id": result["run_id"],
        "query": result["query"],
        "top_result": result["retrieval_results"][0],
        "context_pack_summary": {
            "selected_count": pack.get("selected_count"),
            "dropped_count": pack.get("dropped_count"),
            "used_chars": pack.get("used_chars"),
        },
        "prompt_id": result.get("prompt_id"),
        "prompt_version": result.get("prompt_version"),
        "eval_result": result.get("eval_result"),
        "capture_result": capture,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Scripts/test_full_parent_child_rag_capture_pipeline.py
=====================================================

完整端到端回归测试：
cleaned_text_unit_v1
  -> parent_chunk_v1 / child_chunk_v1
  -> child embedding
  -> Milvus Lite child index
  -> vector_index_record_v2
  -> P1 child dense retrieval + parent backfill
  -> P2 dense + BM25 + RRF hybrid retrieval
  -> P3 rerank + context packing + retrieval eval
  -> P4-lite RAGEngine + PromptBuilder + DataCapture
  -> data/processed/runs/rag_runs.jsonl

PyCharm:
- Script path: D:\\MyCode\\rag-template\\Scripts\\test_full_parent_child_rag_capture_pipeline.py
- Working directory: D:\\MyCode\\rag-template
- Environment variables: PYTHONPATH=D:\\MyCode\\rag-template\\src

注意：
- 本脚本不调用真实 LLM；P4-lite 只生成 prompt 和保存运行轨迹。
- answer/model_name 为 None 是正常现象。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pymilvus import MilvusClient

# Make project root and src importable when script is run directly.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jobs.build_parent_child_and_index_milvus_lite import (  # noqa: E402
    DEFAULT_HASH_DIM,
    embed_children,
    ensure_parent,
    load_jsonl_records,
    real_embed_texts,
    safe_str,
    validate_parent_child_records,
    write_jsonl,
)
from rag_template.chunker.ChildParentChunker import ChildParentChunker  # noqa: E402
from rag_template.configs.SchemaConfig import DEFAULT_EMBEDDING_VERSION, DEFAULT_VECTOR_DB  # noqa: E402
from rag_template.context.context_packer import ContextPacker  # noqa: E402
from rag_template.data_capture.rag_run_capture import RagRunCapture  # noqa: E402
from rag_template.eval.p3_retrieval_eval import evaluate_retrieval_results_v2  # noqa: E402
from rag_template.prompt.parent_child_prompt_builder import ParentChildPromptBuilder  # noqa: E402
from rag_template.rag_engine.parent_child_rag_engine import ParentChildRAGEngine  # noqa: E402
from rag_template.reranker.parent_child_reranker import NoOpParentChildReranker, ParentChildReranker  # noqa: E402
from rag_template.retriever.bm25_child_retriever import BM25ChildRetriever  # noqa: E402
from rag_template.retriever.hybrid_parent_child_retriever import HybridParentChildRetriever  # noqa: E402
from rag_template.retriever.milvus_child_retriever import MilvusChildRetriever  # noqa: E402
from rag_template.schema.VectorIndexRecord_Schema import build_vector_index_record_v2  # noqa: E402
from rag_template.store.parent_chunk_store import ParentChunkStore  # noqa: E402
from rag_template.vector_store.milvus_child_chunk_store import (  # noqa: E402
    build_milvus_child_chunk_record,
    create_or_reset_child_chunk_collection,
    insert_child_chunk_records,
)

try:
    from rag_template.configs.RAGConfig import (  # type: ignore  # noqa: E402
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DEVICE,
        EMBEDDING_BATCH_SIZE,
        RERANKER_MODEL_NAME,
        RERANKER_DEVICE,
        RERANKER_BATCH_SIZE,
    )
except Exception:
    EMBEDDING_MODEL_NAME = r"D:\models\huggingface\embedding\m3e-base"
    EMBEDDING_DEVICE = "cuda"
    EMBEDDING_BATCH_SIZE = 32
    RERANKER_MODEL_NAME = r"D:\models\huggingface\reranker\bge-reranker-v2-m3"
    RERANKER_DEVICE = "cuda"
    RERANKER_BATCH_SIZE = 16

DEFAULT_RERANKER_MODEL_NAME = r"D:\models\huggingface\reranker\bge-reranker-v2-m3"


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _remove_path(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def _count_jsonl(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    count = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _print_results(title: str, results: Sequence[Dict[str, Any]], max_items: int = 5) -> None:
    print(f"\n========== {title} ==========")
    for item in list(results)[:max_items]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full parent-child hybrid RAG capture pipeline test: P0 -> P1 -> P2 -> P3 -> P4-lite"
    )

    # Input / output paths.
    parser.add_argument("--input", default="data/raw/jsonl/cleaned_text_unit_all.jsonl")
    parser.add_argument("--parent-output", default="data/processed/parent_child_chunks/parent_chunks.jsonl")
    parser.add_argument("--child-output", default="data/processed/parent_child_chunks/child_chunks.jsonl")
    parser.add_argument("--index-record-output", default="data/processed/vector_index_record/vector_index_record_v2.jsonl")
    parser.add_argument("--db-file", default="data/processed/vector_store/milvus_parent_child.db")
    parser.add_argument("--capture-output", default="data/processed/runs/rag_runs.jsonl")

    # Milvus / index.
    parser.add_argument("--collection-name", default="rag_child_chunks")
    parser.add_argument("--metric-type", choices=["COSINE", "IP", "L2"], default="COSINE")
    parser.add_argument("--vector-db", default=DEFAULT_VECTOR_DB)
    parser.add_argument("--insert-batch-size", type=int, default=128)
    parser.add_argument("--max-text-chars", type=int, default=8192)
    parser.add_argument("--recreate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-output", action="store_true", help="Delete generated parent/child/index/db/capture outputs before running.")

    # Chunker.
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--parent-chunk-size", type=int, default=None)
    parser.add_argument("--parent-chunk-overlap", type=int, default=None)
    parser.add_argument("--child-chunk-size", type=int, default=None)
    parser.add_argument("--child-chunk-overlap", type=int, default=None)
    parser.add_argument("--unit", choices=["char", "token"], default=None)

    # Embedding.
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME or r"D:\models\huggingface\embedding\m3e-base")
    parser.add_argument("--embedding-device", default=EMBEDDING_DEVICE or "cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=int(EMBEDDING_BATCH_SIZE or 32))
    parser.add_argument("--embedding-version", default=DEFAULT_EMBEDDING_VERSION)
    parser.add_argument("--hash-embedding", action="store_true")
    parser.add_argument("--hash-dim", type=int, default=DEFAULT_HASH_DIM)

    # Query / retrieval.
    parser.add_argument("--query", default="整体性学习是什么")
    parser.add_argument("--p1-top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--keyword-top-k", type=int, default=10)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)

    # Rerank / context.
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL_NAME or DEFAULT_RERANKER_MODEL_NAME)
    parser.add_argument("--reranker-device", default=RERANKER_DEVICE or "cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=int(RERANKER_BATCH_SIZE or 16))
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-context-items", type=int, default=3)

    # Eval.
    parser.add_argument("--expected-doc-ids", default="doc_001_native_text")
    parser.add_argument("--expected-parent-chunk-ids", default="")
    parser.add_argument("--expected-child-chunk-ids", default="")
    parser.add_argument("--expected-keywords", default="整体性学习,学习")
    parser.add_argument("--eval-top-k", type=int, default=5)

    return parser.parse_args()


def run_p0_build_chunks_and_index(args: argparse.Namespace) -> Dict[str, Any]:
    print("========== P0 Load cleaned_text_unit_v1 ==========")
    print(f"input = {args.input}")
    cleaned_records = load_jsonl_records(args.input, max_records=args.max_records)
    if not cleaned_records:
        raise ValueError(f"No cleaned_text_unit_v1 records loaded: {args.input}")
    print(f"cleaned_records = {len(cleaned_records)}")

    print("\n========== P0 Build parent/child chunks ==========")
    chunker = ChildParentChunker(
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        unit=args.unit,
    )
    chunk_result = chunker.chunk_records(cleaned_records)
    parents = chunk_result.parents
    children = chunk_result.children
    validate_parent_child_records(parents, children)
    print(f"parents  = {len(parents)}")
    print(f"children = {len(children)}")
    print(f"first_parent = {parents[0].get('parent_chunk_id')}")
    print(f"first_child  = {children[0].get('child_chunk_id')}")

    write_jsonl(parents, args.parent_output)
    write_jsonl(children, args.child_output)

    print("\n========== P0 Embedding child chunks ==========")
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
    print(f"embedding_shape   = {embeddings.shape}")
    print(f"embedding_model   = {embedding_model}")
    print(f"embedding_dim     = {dim}")
    print(f"embedding_version = {embedding_version}")

    print("\n========== P0 Milvus Lite child index ==========")
    ensure_parent(args.db_file)
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
    if os.name != "nt":
        client.flush(args.collection_name)
    print(f"db_file      = {args.db_file}")
    print(f"collection   = {args.collection_name}")
    print(f"inserted     = {inserted}")

    print("\n========== P0 Write vector_index_record_v2 ==========")
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

    assert len(parents) > 0, "P0 parents should not be empty"
    assert len(children) > 0, "P0 children should not be empty"
    assert inserted == len(children), f"Milvus inserted mismatch: inserted={inserted}, children={len(children)}"
    assert len(vector_index_records) == len(children), "vector_index_record count should equal child count"

    return {
        "parents": parents,
        "children": children,
        "vector_index_records": vector_index_records,
        "embedding_model": embedding_model,
        "embedding_dim": dim,
        "embedding_version": embedding_version,
    }


def run_p1_p2_p3_p4(args: argparse.Namespace) -> Dict[str, Any]:
    print("\n========== P1/P2 Init Stores and Retrievers ==========")
    parent_store = ParentChunkStore.from_jsonl(args.parent_output)
    keyword_retriever = BM25ChildRetriever.from_jsonl(args.child_output)
    dense_retriever = MilvusChildRetriever(
        db_file=args.db_file,
        collection_name=args.collection_name,
        metric_type=args.metric_type,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=1,
        hash_embedding=args.hash_embedding,
        hash_dim=args.hash_dim,
    )
    print(f"parents  = {len(parent_store)}")
    print(f"children = {len(keyword_retriever)}")

    print("\n========== P1 Dense child retrieval + parent backfill smoke ==========")
    p1_dense_hits = dense_retriever.search(args.query, top_k=args.p1_top_k)
    print(f"dense_child_hits = {len(p1_dense_hits)}")
    for hit in p1_dense_hits[: args.p1_top_k]:
        print(
            f"[{hit.get('rank')}] score={hit.get('score')} "
            f"child={hit.get('child_chunk_id')} parent={hit.get('parent_chunk_id')} doc={hit.get('doc_id')}"
        )
    assert p1_dense_hits, "P1 dense child hits should not be empty"
    assert all(hit.get("parent_chunk_id") for hit in p1_dense_hits), "P1 dense hits should have parent_chunk_id"

    hybrid_retriever = HybridParentChildRetriever(
        dense_retriever=dense_retriever,
        keyword_retriever=keyword_retriever,
        parent_store=parent_store,
        rrf_k=args.rrf_k,
        dedup_parent=True,
    )

    print("\n========== P2 Hybrid retrieval ==========")
    p2_results = hybrid_retriever.retrieve(
        query=args.query,
        dense_top_k=args.dense_top_k,
        keyword_top_k=args.keyword_top_k,
        final_top_k=args.candidate_top_k,
    )
    _print_results("P2 Results", p2_results, max_items=args.candidate_top_k)
    assert p2_results, "P2 hybrid results should not be empty"
    assert all((x.get("metadata") or {}).get("parent_found") is True for x in p2_results), "P2 results should backfill parent"

    print("\n========== P3 Rerank ==========")
    if args.skip_rerank:
        print("reranker = NoOpParentChildReranker")
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

    p3_results = reranker.rerank(
        query=args.query,
        results=p2_results,
        top_k=args.rerank_top_k,
        text_field="parent_text",
    )
    _print_results("P3 Reranked Results", p3_results, max_items=args.rerank_top_k)
    assert p3_results, "P3 reranked results should not be empty"
    assert all(x.get("rerank_score") is not None for x in p3_results), "P3 results should have rerank_score"

    print("\n========== P3 Context Packing ==========")
    context_packer = ContextPacker(
        max_context_chars=args.max_context_chars,
        max_items=args.max_context_items,
        text_field="text",
        dedup_parent=True,
        include_metadata=True,
    )
    context_pack = context_packer.pack(p3_results)
    print(f"selected_count    = {len(context_pack.selected_results)}")
    print(f"dropped_count     = {len(context_pack.dropped_results)}")
    print(f"used_chars        = {context_pack.used_chars}")
    print(f"max_context_chars = {context_pack.max_context_chars}")
    print("\n----- Packed Context Preview -----")
    print(context_pack.context[:1500])
    assert context_pack.context.strip(), "packed context should not be empty"

    print("\n========== P3 Retrieval Eval ==========")
    eval_result = evaluate_retrieval_results_v2(
        p3_results,
        top_k=args.eval_top_k,
        expected_doc_ids=_split_csv(args.expected_doc_ids),
        expected_parent_chunk_ids=_split_csv(args.expected_parent_chunk_ids),
        expected_child_chunk_ids=_split_csv(args.expected_child_chunk_ids),
        expected_keywords=_split_csv(args.expected_keywords),
    )
    print(json.dumps(eval_result, ensure_ascii=False, indent=2))

    print("\n========== P4-lite RAGEngine + PromptBuilder + DataCapture ==========")
    prompt_builder = ParentChildPromptBuilder()
    run_capture = RagRunCapture(args.capture_output)
    engine = ParentChildRAGEngine(
        retriever=hybrid_retriever,
        reranker=reranker,
        context_packer=context_packer,
        prompt_builder=prompt_builder,
        run_capture=run_capture,
    )
    engine_result = engine.run(
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
        extra_metadata={
            "full_pipeline_script": "Scripts/test_full_parent_child_rag_capture_pipeline.py",
            "p0_parent_output": args.parent_output,
            "p0_child_output": args.child_output,
            "p0_index_record_output": args.index_record_output,
            "capture_output": args.capture_output,
        },
    )

    print("\n----- Prompt Preview -----")
    print(f"prompt_id      = {engine_result.get('prompt_id')}")
    print(f"prompt_version = {engine_result.get('prompt_version')}")
    print(engine_result["prompt"][:2000])

    print("\n----- DataCapture -----")
    capture_result = engine_result.get("capture_result") or {}
    print(f"capture_output = {capture_result.get('output_path')}")
    print(f"saved          = {capture_result.get('saved')}")
    print(f"run_id         = {capture_result.get('run_id')}")

    assert "【资料】" in engine_result["prompt"], "Prompt should include context marker"
    assert args.query in engine_result["prompt"], "Prompt should include query"
    assert capture_result.get("saved") is True, "DataCapture should save run record"
    assert Path(args.capture_output).exists(), "capture output file should exist"
    assert engine_result["run_record"].get("answer") is None, "P4-lite should not call LLM yet"

    return {
        "p1_dense_hits": p1_dense_hits,
        "p2_results": p2_results,
        "p3_results": p3_results,
        "context_pack": context_pack.to_dict(),
        "eval_result": eval_result,
        "engine_result": engine_result,
    }


def main() -> None:
    args = parse_args()

    print("========== Full Parent-Child RAG Capture Pipeline ==========")
    print(f"project_root = {_PROJECT_ROOT}")
    print(f"query        = {args.query}")

    if args.clean_output:
        print("\n========== Clean old outputs ==========")
        for path in [
            args.parent_output,
            args.child_output,
            args.index_record_output,
            args.db_file,
            args.capture_output,
        ]:
            print(f"remove if exists: {path}")
            _remove_path(path)

    # Ensure parent dirs exist.
    for path in [args.parent_output, args.child_output, args.index_record_output, args.db_file, args.capture_output]:
        ensure_parent(path)

    p0_summary = run_p0_build_chunks_and_index(args)
    downstream = run_p1_p2_p3_p4(args)

    print("\n========== Final Assertions ==========")
    parent_count = _count_jsonl(args.parent_output)
    child_count = _count_jsonl(args.child_output)
    index_record_count = _count_jsonl(args.index_record_output)
    capture_count = _count_jsonl(args.capture_output)
    print(f"parent_chunks       = {parent_count}")
    print(f"child_chunks        = {child_count}")
    print(f"vector_records      = {index_record_count}")
    print(f"capture_records     = {capture_count}")

    assert parent_count == len(p0_summary["parents"]), "parent_chunks.jsonl count mismatch"
    assert child_count == len(p0_summary["children"]), "child_chunks.jsonl count mismatch"
    assert index_record_count == len(p0_summary["children"]), "vector_index_record_v2 count should equal child count"
    assert capture_count >= 1, "rag_runs.jsonl should contain at least one run"

    final_summary = {
        "query": args.query,
        "parent_output": args.parent_output,
        "child_output": args.child_output,
        "index_record_output": args.index_record_output,
        "db_file": args.db_file,
        "capture_output": args.capture_output,
        "parent_count": parent_count,
        "child_count": child_count,
        "vector_record_count": index_record_count,
        "capture_record_count": capture_count,
        "top_result": downstream["engine_result"]["retrieval_results"][0],
        "eval_result": downstream["engine_result"]["eval_result"],
        "run_id": downstream["engine_result"]["run_id"],
    }
    print("\n========== JSON Preview ==========")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2)[:6000])

    print("\nFULL parent-child RAG capture pipeline test passed")


if __name__ == "__main__":
    main()

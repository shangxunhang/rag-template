# -*- coding: utf-8 -*-
"""
Scripts/test_p4_full_local_llm_answer.py
========================================

P4-full 回归测试：复用已经构建好的 parent/child chunk 与 Milvus Lite 索引，
跑通：

query
  -> dense + BM25 + RRF hybrid retrieval
  -> rerank
  -> context packing
  -> PromptBuilder
  -> LocalLLMGenerator
  -> answer
  -> DataCapture
  -> data/processed/runs/rag_runs.jsonl

PyCharm:
- Script path: D:\\MyCode\\rag-template\\Scripts\\test_p4_full_local_llm_answer.py
- Working directory: D:\\MyCode\\rag-template
- Environment variables: PYTHONPATH=D:\\MyCode\\rag-template\\src

注意：
- 本脚本不重新切 chunk、不重新 embedding、不重建 Milvus 索引。
- 它要求你已经跑通过 test_full_parent_child_rag_capture_pipeline.py。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rag_template.context.context_packer import ContextPacker  # noqa: E402
from rag_template.data_capture.rag_run_capture import RagRunCapture  # noqa: E402
from rag_template.llm.local_llm import LocalLLMGenerator  # noqa: E402
from rag_template.prompt.parent_child_prompt_builder import ParentChildPromptBuilder  # noqa: E402
from rag_template.rag_engine.parent_child_rag_engine import ParentChildRAGEngine  # noqa: E402
from rag_template.reranker.parent_child_reranker import NoOpParentChildReranker, ParentChildReranker  # noqa: E402
from rag_template.retriever.bm25_child_retriever import BM25ChildRetriever  # noqa: E402
from rag_template.retriever.hybrid_parent_child_retriever import HybridParentChildRetriever  # noqa: E402
from rag_template.retriever.milvus_child_retriever import MilvusChildRetriever  # noqa: E402
from rag_template.store.parent_chunk_store import ParentChunkStore  # noqa: E402

try:
    from rag_template.configs.RAGConfig import (  # type: ignore  # noqa: E402
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DEVICE,
        RERANKER_MODEL_NAME,
        RERANKER_DEVICE,
        RERANKER_BATCH_SIZE,
    )
except Exception:
    EMBEDDING_MODEL_NAME = r"D:\models\huggingface\embedding\m3e-base"
    EMBEDDING_DEVICE = "cuda"
    RERANKER_MODEL_NAME = r"D:\models\huggingface\reranker\bge-reranker-v2-m3"
    RERANKER_DEVICE = "cuda"
    RERANKER_BATCH_SIZE = 16

try:
    from rag_template.configs.LLMConfig import (  # type: ignore  # noqa: E402
        LLM_MODEL_NAME,
        LLM_DEVICE,
        LLM_MAX_NEW_TOKENS,
        LLM_TEMPERATURE,
        LLM_TOP_P,
        LLM_DO_SAMPLE,
    )
except Exception:
    LLM_MODEL_NAME = r"D:\models\huggingface\llm\Qwen2.5-1.5B-Instruct"
    LLM_DEVICE = "cuda"
    LLM_MAX_NEW_TOKENS = 256
    LLM_TEMPERATURE = 0.7
    LLM_TOP_P = 0.9
    LLM_DO_SAMPLE = False


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


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
        text = str(item.get("text") or "").replace("\n", " ")
        print(f"    text_preview    ={text[:220]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4-full local LLM answer generation test")

    parser.add_argument("--parent-file", default="data/processed/parent_child_chunks/parent_chunks.jsonl")
    parser.add_argument("--child-file", default="data/processed/parent_child_chunks/child_chunks.jsonl")
    parser.add_argument("--db-file", default="data/processed/vector_store/milvus_parent_child.db")
    parser.add_argument("--capture-output", default="data/processed/runs/rag_runs.jsonl")
    parser.add_argument("--collection-name", default="rag_child_chunks")
    parser.add_argument("--metric-type", choices=["COSINE", "IP", "L2"], default="COSINE")

    parser.add_argument("--query", default="整体性学习是什么")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME or r"D:\models\huggingface\embedding\m3e-base")
    parser.add_argument("--embedding-device", default=EMBEDDING_DEVICE or "cuda")
    parser.add_argument("--hash-embedding", action="store_true")
    parser.add_argument("--hash-dim", type=int, default=768)

    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--keyword-top-k", type=int, default=10)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL_NAME or r"D:\models\huggingface\reranker\bge-reranker-v2-m3")
    parser.add_argument("--reranker-device", default=RERANKER_DEVICE or "cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=int(RERANKER_BATCH_SIZE or 16))
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-context-items", type=int, default=3)

    parser.add_argument("--expected-doc-ids", default="doc_001_native_text")
    parser.add_argument("--expected-parent-chunk-ids", default="")
    parser.add_argument("--expected-child-chunk-ids", default="")
    parser.add_argument("--expected-keywords", default="整体性学习,学习")
    parser.add_argument("--eval-top-k", type=int, default=5)

    parser.add_argument("--llm-model", default=LLM_MODEL_NAME)
    parser.add_argument("--llm-device", default=LLM_DEVICE or "cuda")
    parser.add_argument("--max-new-tokens", type=int, default=int(LLM_MAX_NEW_TOKENS or 256))
    parser.add_argument("--temperature", type=float, default=float(LLM_TEMPERATURE or 0.7))
    parser.add_argument("--top-p", type=float, default=float(LLM_TOP_P or 0.9))
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=bool(LLM_DO_SAMPLE))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("========== P4-full Local LLM RAG Answer Test ==========")
    print(f"project_root = {_PROJECT_ROOT}")
    print(f"query        = {args.query}")
    print(f"llm_model    = {args.llm_model}")

    for path_name, path in [
        ("parent_file", args.parent_file),
        ("child_file", args.child_file),
        ("db_file", args.db_file),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path_name} not found: {path}. Please run test_full_parent_child_rag_capture_pipeline.py first.")

    print("\n========== Init Stores / Retrievers ==========")
    parent_store = ParentChunkStore.from_jsonl(args.parent_file)
    keyword_retriever = BM25ChildRetriever.from_jsonl(args.child_file)
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

    hybrid_retriever = HybridParentChildRetriever(
        dense_retriever=dense_retriever,
        keyword_retriever=keyword_retriever,
        parent_store=parent_store,
        rrf_k=args.rrf_k,
        dedup_parent=True,
    )

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

    context_packer = ContextPacker(
        max_context_chars=args.max_context_chars,
        max_items=args.max_context_items,
        text_field="text",
        dedup_parent=True,
        include_metadata=True,
    )
    prompt_builder = ParentChildPromptBuilder()
    run_capture = RagRunCapture(args.capture_output)

    print("\n========== Load Local LLM ==========")
    llm = LocalLLMGenerator(
        model_name=args.llm_model,
        device=args.llm_device,
    )

    engine = ParentChildRAGEngine(
        retriever=hybrid_retriever,
        reranker=reranker,
        context_packer=context_packer,
        prompt_builder=prompt_builder,
        run_capture=run_capture,
        llm_generator=llm,
        model_name=args.llm_model,
        model_provider="local",
        pipeline_name="parent_child_hybrid_rag_p4_full",
        pipeline_version="v1.0",
    )

    print("\n========== Run P4-full ==========")
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
    }
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
        generate_answer=True,
        generation_params=generation_params,
        extra_metadata={
            "script": "Scripts/test_p4_full_local_llm_answer.py",
            "parent_file": args.parent_file,
            "child_file": args.child_file,
            "db_file": args.db_file,
            "capture_output": args.capture_output,
        },
    )

    _print_results("Retrieval Results", result["retrieval_results"], max_items=args.rerank_top_k)

    print("\n----- Prompt Preview -----")
    print(result["prompt"][:2000])

    print("\n----- Answer -----")
    print(result["answer"])

    print("\n----- Eval Result -----")
    print(json.dumps(result["eval_result"], ensure_ascii=False, indent=2))

    print("\n----- DataCapture -----")
    capture_result = result.get("capture_result") or {}
    print(f"capture_output = {capture_result.get('output_path')}")
    print(f"saved          = {capture_result.get('saved')}")
    print(f"run_id         = {capture_result.get('run_id')}")

    capture_count = _count_jsonl(args.capture_output)
    print("\n========== Final Assertions ==========")
    print(f"capture_records = {capture_count}")

    assert result["answer"] and result["answer"].strip(), "P4-full should generate non-empty answer"
    assert result["run_record"].get("answer") == result["answer"], "run_record should save answer"
    assert result["run_record"].get("model_name"), "run_record should save model_name"
    assert result["run_record"].get("generation_params"), "run_record should save generation_params"
    assert capture_result.get("saved") is True, "DataCapture should save run record"
    assert Path(args.capture_output).exists(), "capture output file should exist"

    summary = {
        "query": args.query,
        "answer_preview": result["answer"][:500],
        "model_name": result["model_name"],
        "generation_params": result["generation_params"],
        "run_id": result["run_id"],
        "capture_output": args.capture_output,
        "capture_record_count": capture_count,
        "eval_result": result["eval_result"],
    }
    print("\n========== JSON Preview ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nP4-full local LLM answer test passed")


if __name__ == "__main__":
    main()

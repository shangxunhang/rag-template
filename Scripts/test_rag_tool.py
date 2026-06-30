# -*- coding: utf-8 -*-
r"""
scripts/test_rag_tool.py
========================

Smoke test for RAGTool.

PyCharm:
- Script path: D:\MyCode\rag-template\scripts\test_rag_tool.py
- Working directory: D:\MyCode\rag-template
- Environment variables: PYTHONPATH=D:\MyCode\rag-template\src

Command:
D:\mysoftware\anaconda\envs\rag\python.exe D:\MyCode\rag-template\scripts\test_rag_tool.py --query 整体性学习是什么
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rag_template.tools.rag_tool import RAGTool, RAGToolConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RAGTool")
    parser.add_argument("--query", default="整体性学习是什么")
    parser.add_argument("--parent-file", default="data/processed/parent_child_chunks/parent_chunks.jsonl")
    parser.add_argument("--child-file", default="data/processed/parent_child_chunks/child_chunks.jsonl")
    parser.add_argument("--db-file", default="data/processed/vector_store/milvus_parent_child.db")
    parser.add_argument("--capture-output", default="data/processed/runs/rag_tool_runs.jsonl")
    parser.add_argument("--collection-name", default="rag_child_chunks")
    parser.add_argument("--metric-type", choices=["COSINE", "IP", "L2"], default="COSINE")

    parser.add_argument("--embedding-model", default=r"D:\models\huggingface\embedding\m3e-base")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--hash-embedding", action="store_true")
    parser.add_argument("--hash-dim", type=int, default=768)

    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--reranker-model", default=r"D:\models\huggingface\reranker\bge-reranker-v2-m3")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=16)

    parser.add_argument("--llm-model", default=r"D:\models\huggingface\llm\Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm-device", default="cuda")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")

    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--keyword-top-k", type=int, default=10)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-context-items", type=int, default=3)

    parser.add_argument("--expected-doc-ids", default="doc_001_native_text")
    parser.add_argument("--expected-parent-chunk-ids", default="")
    parser.add_argument("--expected-child-chunk-ids", default="")
    parser.add_argument("--expected-keywords", default="整体性学习,学习")
    parser.add_argument("--return-prompt", action="store_true")
    parser.add_argument("--return-full-record", action="store_true")
    return parser.parse_args()


def _print_json(title: str, obj: Dict[str, Any]) -> None:
    print(f"\n========== {title} ==========")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()

    print("========== RAGTool Smoke Test ==========")
    print(f"project_root = {_PROJECT_ROOT}")
    print(f"query        = {args.query}")

    cfg = RAGToolConfig(
        parent_file=args.parent_file,
        child_file=args.child_file,
        db_file=args.db_file,
        capture_output=args.capture_output,
        collection_name=args.collection_name,
        metric_type=args.metric_type,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        hash_embedding=args.hash_embedding,
        hash_dim=args.hash_dim,
        skip_rerank=args.skip_rerank,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        enable_llm=not args.no_llm,
        llm_model=args.llm_model,
        llm_device=args.llm_device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        dense_top_k=args.dense_top_k,
        keyword_top_k=args.keyword_top_k,
        candidate_top_k=args.candidate_top_k,
        rerank_top_k=args.rerank_top_k,
        eval_top_k=args.eval_top_k,
        max_context_chars=args.max_context_chars,
        max_context_items=args.max_context_items,
    )

    tool = RAGTool(cfg, project_root=_PROJECT_ROOT)
    result = tool.run(
        {
            "query": args.query,
            "expected_doc_ids": args.expected_doc_ids,
            "expected_parent_chunk_ids": args.expected_parent_chunk_ids,
            "expected_child_chunk_ids": args.expected_child_chunk_ids,
            "expected_keywords": args.expected_keywords,
            "return_prompt": args.return_prompt,
            "return_full_record": args.return_full_record,
        }
    )

    if not result.get("success"):
        _print_json("Tool Failed", result)
        raise SystemExit(1)

    data = result.get("data") or {}
    print("\n========== Answer ==========")
    print(data.get("answer"))

    _print_json(
        "Compact Result",
        {
            "success": result.get("success"),
            "tool_name": result.get("tool_name"),
            "run_id": data.get("run_id"),
            "query": data.get("query"),
            "model_name": data.get("model_name"),
            "contexts_count": len(data.get("contexts") or []),
            "citations_count": len(data.get("citations") or []),
            "eval_result": data.get("eval_result"),
            "capture_result": data.get("capture_result"),
            "metadata": result.get("metadata"),
        },
    )

    print("\nRAGTool smoke test passed")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Scripts/test_ragas_style_eval_from_runs.py
==========================================

Run local RAGAS-style proxy evaluation from captured rag_runs.jsonl.

This script does NOT rerun retrieval or LLM generation. It evaluates existing
DataCapture records, so it is fast and stable.

PyCharm:
- Script path: D:\\MyCode\\rag-template\\Scripts\\test_ragas_style_eval_from_runs.py
- Working directory: D:\\MyCode\\rag-template
- Environment variables: PYTHONPATH=D:\\MyCode\\rag-template\\src
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

from rag_template.eval.rag_eval_case_schema import (  # noqa: E402
    index_cases_by_query,
    load_rag_eval_cases,
    match_case_for_run,
)
from rag_template.eval.ragas_style_eval import (  # noqa: E402
    aggregate_ragas_style_proxy,
    evaluate_ragas_style_proxy,
    load_jsonl_records,
    write_json,
    write_jsonl,
)


def _resolve_project_path(path: str | Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(_PROJECT_ROOT / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate captured RAG runs with RAGAS-style proxy metrics.")
    parser.add_argument("--runs-file", default="data/processed/runs/rag_runs.jsonl")
    parser.add_argument("--eval-cases-file", default="data/eval_set/rag_eval_cases.jsonl")
    parser.add_argument("--report-output", default="data/processed/eval_reports/ragas_style_eval_report.json")
    parser.add_argument("--details-output", default="data/processed/eval_reports/ragas_style_eval_details.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fail-on-missing-case", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.runs_file = _resolve_project_path(args.runs_file)
    args.eval_cases_file = _resolve_project_path(args.eval_cases_file)
    args.report_output = _resolve_project_path(args.report_output)
    args.details_output = _resolve_project_path(args.details_output)

    print("========== RAGAS-style Proxy Eval From Runs ==========")
    print(f"project_root    = {_PROJECT_ROOT}")
    print(f"runs_file       = {args.runs_file}")
    print(f"eval_cases_file = {args.eval_cases_file}")
    print(f"report_output   = {args.report_output}")
    print(f"details_output  = {args.details_output}")
    print(f"top_k           = {args.top_k}")

    runs = load_jsonl_records(args.runs_file)
    cases = load_rag_eval_cases(args.eval_cases_file)
    cases_by_query = index_cases_by_query(cases)

    print("\n========== Loaded ==========")
    print(f"runs  = {len(runs)}")
    print(f"cases = {len(cases)}")

    details: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for run in runs:
        case = match_case_for_run(run, cases_by_query)
        if case is None:
            skipped.append({
                "run_id": run.get("run_id"),
                "query": run.get("query"),
                "reason": "no eval case matched by query",
            })
            continue
        result = evaluate_ragas_style_proxy(
            run_record=run,
            eval_case=case,
            top_k=args.top_k,
        )
        details.append(result)

    if skipped and args.fail_on_missing_case:
        raise RuntimeError(f"Missing eval cases for {len(skipped)} run(s): {skipped[:3]}")

    report = aggregate_ragas_style_proxy(details)
    report.update({
        "runs_file": args.runs_file,
        "eval_cases_file": args.eval_cases_file,
        "details_output": args.details_output,
        "top_k": args.top_k,
        "skipped_count": len(skipped),
        "skipped": skipped,
    })

    write_json(args.report_output, report)
    write_jsonl(args.details_output, details)

    print("\n========== Summary ==========")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])

    assert len(details) > 0, "No RAG run was evaluated. Check query matching between runs and eval cases."
    assert report["avg_context_precision_proxy"] >= 0.0
    assert report["avg_context_recall_proxy"] >= 0.0
    assert report["avg_faithfulness_proxy"] >= 0.0
    assert report["avg_answer_relevancy_proxy"] >= 0.0

    print("\nRAGAS-style proxy eval test passed")


if __name__ == "__main__":
    main()

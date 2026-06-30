# -*- coding: utf-8 -*-
"""
rag_template/rag_engine/parent_child_rag_engine.py
=================================================

ParentChildRAGEngine for P4-lite / P4-full.

P4-lite:
HybridParentChildRetriever
  -> ParentChildReranker / NoOpParentChildReranker
  -> ContextPacker
  -> ParentChildPromptBuilder
  -> RagRunCapture

P4-full:
P4-lite
  -> LocalLLMGenerator / compatible llm_generator
  -> answer
  -> RagRunCapture(answer/model_name/generation_params)

The engine intentionally depends on abstract behavior only:
- retriever.retrieve(...)
- reranker.rerank(...)
- context_packer.pack(...)
- prompt_builder.build(...)
- llm_generator.generate(prompt, **generation_params)  [optional]
- run_capture.capture(record)                         [optional]
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rag_template.context.context_packer import ContextPacker
from rag_template.data_capture.rag_run_capture import RagRunCapture
from rag_template.eval.p3_retrieval_eval import evaluate_retrieval_results_v2
from rag_template.prompt.parent_child_prompt_builder import ParentChildPromptBuilder
from rag_template.reranker.parent_child_reranker import NoOpParentChildReranker, ParentChildReranker
from rag_template.retriever.hybrid_parent_child_retriever import HybridParentChildRetriever


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id(prefix: str = "rag_run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _infer_model_name(llm_generator: Any, fallback: Optional[str] = None) -> Optional[str]:
    if fallback:
        return fallback
    if llm_generator is None:
        return None
    for attr in ("model_name", "model_path", "model", "name"):
        value = getattr(llm_generator, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return llm_generator.__class__.__name__


class ParentChildRAGEngine:
    """Application-level engine for parent-child hybrid RAG."""

    def __init__(
        self,
        *,
        retriever: HybridParentChildRetriever,
        reranker: ParentChildReranker | NoOpParentChildReranker,
        context_packer: ContextPacker,
        prompt_builder: ParentChildPromptBuilder,
        run_capture: Optional[RagRunCapture] = None,
        llm_generator: Optional[Any] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        pipeline_name: str = "parent_child_hybrid_rag",
        pipeline_version: str = "v1.0",
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_packer = context_packer
        self.prompt_builder = prompt_builder
        self.run_capture = run_capture
        self.llm_generator = llm_generator
        self.model_name = model_name
        self.model_provider = model_provider
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version

    def run(
        self,
        query: str,
        *,
        dense_top_k: int = 10,
        keyword_top_k: int = 10,
        candidate_top_k: int = 10,
        rrf_k: Optional[int] = None,
        rerank_top_k: int = 5,
        eval_top_k: int = 5,
        expected_doc_ids: Optional[List[str]] = None,
        expected_parent_chunk_ids: Optional[List[str]] = None,
        expected_child_chunk_ids: Optional[List[str]] = None,
        expected_keywords: Optional[List[str]] = None,
        generate_answer: Optional[bool] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run RAG and optionally generate an answer with a local/compatible LLM.

        Args:
            query: User query.
            generate_answer: If None, generate answer when llm_generator is configured.
                If True and no llm_generator is configured, raise ValueError.
                If False, run as P4-lite.
            generation_params: kwargs passed to llm_generator.generate().
        """
        if not query or not str(query).strip():
            raise ValueError("query cannot be empty")

        if generate_answer is None:
            generate_answer = self.llm_generator is not None
        if generate_answer and self.llm_generator is None:
            raise ValueError("generate_answer=True requires llm_generator")

        run_id = _new_run_id()
        started_at = _utc_now_iso()

        p2_results = self.retriever.retrieve(
            query=query,
            dense_top_k=dense_top_k,
            keyword_top_k=keyword_top_k,
            final_top_k=candidate_top_k,
        )

        reranked_results = self.reranker.rerank(
            query=query,
            results=p2_results,
            top_k=rerank_top_k,
            text_field="parent_text",
        )

        context_pack = self.context_packer.pack(reranked_results)
        prompt_result = self.prompt_builder.build(
            query=query,
            packed_context=context_pack.context,
            citations=context_pack.citations,
        )

        eval_result = evaluate_retrieval_results_v2(
            reranked_results,
            top_k=eval_top_k,
            expected_doc_ids=expected_doc_ids or [],
            expected_parent_chunk_ids=expected_parent_chunk_ids or [],
            expected_child_chunk_ids=expected_child_chunk_ids or [],
            expected_keywords=expected_keywords or [],
        )

        answer: Optional[str] = None
        llm_latency_ms: Optional[int] = None
        final_generation_params: Optional[Dict[str, Any]] = None
        final_model_name = _infer_model_name(self.llm_generator, self.model_name) if generate_answer else None
        final_model_provider = self.model_provider if generate_answer else None

        if generate_answer:
            final_generation_params = dict(generation_params or {})
            t0 = time.perf_counter()
            answer = self.llm_generator.generate(prompt_result.prompt, **final_generation_params)
            llm_latency_ms = int((time.perf_counter() - t0) * 1000)
            if not isinstance(answer, str):
                answer = str(answer)
            answer = answer.strip()

        finished_at = _utc_now_iso()
        metadata: Dict[str, Any] = {
            "pipeline_stage": "p4_full_answer_capture" if generate_answer else "p4_lite_prompt_capture",
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "retriever": self.retriever.__class__.__name__,
            "reranker": self.reranker.__class__.__name__,
            "context_packer": self.context_packer.__class__.__name__,
            "prompt_builder": self.prompt_builder.__class__.__name__,
            "dense_top_k": dense_top_k,
            "keyword_top_k": keyword_top_k,
            "candidate_top_k": candidate_top_k,
            "rerank_top_k": rerank_top_k,
            "eval_top_k": eval_top_k,
            "rrf_k": rrf_k if rrf_k is not None else getattr(self.retriever, "rrf_k", None),
            "llm_enabled": bool(generate_answer),
            "llm_latency_ms": llm_latency_ms,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        run_record: Dict[str, Any] = {
            "schema_version": "rag_run_v1",
            "run_id": run_id,
            "created_at": started_at,
            "finished_at": finished_at,
            "query": query,
            "answer": answer,
            "model_name": final_model_name,
            "model_provider": final_model_provider,
            "generation_params": final_generation_params,
            "p2_results": p2_results,
            "retrieval_results": reranked_results,
            "context_pack": context_pack.to_dict(),
            "packed_context": context_pack.context,
            "citations": context_pack.citations,
            "prompt": prompt_result.prompt,
            "prompt_id": prompt_result.prompt_id,
            "prompt_version": prompt_result.prompt_version,
            "prompt_build": prompt_result.to_dict(),
            "eval_result": eval_result,
            "metadata": metadata,
        }

        capture_result = None
        if self.run_capture is not None:
            capture_result = self.run_capture.capture(run_record)

        return {
            "run_id": run_id,
            "query": query,
            "answer": answer,
            "model_name": final_model_name,
            "model_provider": final_model_provider,
            "generation_params": final_generation_params,
            "p2_results": p2_results,
            "retrieval_results": reranked_results,
            "context_pack": context_pack.to_dict(),
            "packed_context": context_pack.context,
            "citations": context_pack.citations,
            "prompt": prompt_result.prompt,
            "prompt_id": prompt_result.prompt_id,
            "prompt_version": prompt_result.prompt_version,
            "eval_result": eval_result,
            "run_record": run_record,
            "capture_result": capture_result,
        }

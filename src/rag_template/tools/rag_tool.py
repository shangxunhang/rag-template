# -*- coding: utf-8 -*-
"""
rag_template/tools/rag_tool.py
==============================

RAGTool wraps the completed Parent-Child RAG QA pipeline into a Tool interface.

Current capability:
query -> hybrid retrieval -> rerank -> context packing -> prompt -> local LLM answer -> capture

Agent-facing usage:
    tool = RAGTool.from_default_config()
    result = tool.run({"query": "整体性学习是什么"})

The Agent layer should only call RAGTool.run(...). It should not directly depend on:
Milvus / BM25 / RRF / Reranker / ContextPacker / PromptBuilder / LocalLLM / DataCapture.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_template.context.context_packer import ContextPacker
from rag_template.data_capture.rag_run_capture import RagRunCapture
from rag_template.llm.local_llm import LocalLLMGenerator
from rag_template.prompt.parent_child_prompt_builder import ParentChildPromptBuilder
from rag_template.rag_engine.parent_child_rag_engine import ParentChildRAGEngine
from rag_template.reranker.parent_child_reranker import NoOpParentChildReranker, ParentChildReranker
from rag_template.retriever.bm25_child_retriever import BM25ChildRetriever
from rag_template.retriever.hybrid_parent_child_retriever import HybridParentChildRetriever
from rag_template.retriever.milvus_child_retriever import MilvusChildRetriever
from rag_template.store.parent_chunk_store import ParentChunkStore
from rag_template.tools.base_tool import BaseTool

try:
    from rag_template.configs.RAGConfig import (
        CHILD_CHUNKS_FILE,
        EMBEDDING_BATCH_SIZE,
        EMBEDDING_DEVICE,
        EMBEDDING_MODEL_NAME,
        PARENT_CHILD_DEDUP_PARENT,
        PARENT_CHILD_DENSE_TOP_K,
        PARENT_CHILD_EVAL_TOP_K,
        PARENT_CHILD_HYBRID_FINAL_TOP_K,
        PARENT_CHILD_HYBRID_KEYWORD_TOP_K,
        PARENT_CHILD_MAX_CONTEXT_CHARS,
        PARENT_CHILD_MAX_CONTEXT_ITEMS,
        PARENT_CHILD_MILVUS_COLLECTION_NAME,
        PARENT_CHILD_MILVUS_DB_FILE,
        PARENT_CHILD_RERANK_LOCAL_FILES_ONLY,
        PARENT_CHILD_RERANK_MAX_LENGTH,
        PARENT_CHILD_RERANK_TOP_K,
        PARENT_CHILD_RRF_K,
        PARENT_CHILD_SEARCH_METRIC_TYPE,
        PARENT_CHUNKS_FILE,
        RERANKER_BATCH_SIZE,
        RERANKER_DEVICE,
        RERANKER_MODEL_NAME,
    )
except Exception:  # pragma: no cover - defensive defaults for standalone use
    PARENT_CHUNKS_FILE = "data/processed/parent_child_chunks/parent_chunks.jsonl"
    CHILD_CHUNKS_FILE = "data/processed/parent_child_chunks/child_chunks.jsonl"
    PARENT_CHILD_MILVUS_DB_FILE = "data/processed/vector_store/milvus_parent_child.db"
    PARENT_CHILD_MILVUS_COLLECTION_NAME = "rag_child_chunks"
    PARENT_CHILD_SEARCH_METRIC_TYPE = "COSINE"
    EMBEDDING_MODEL_NAME = r"D:\models\huggingface\embedding\m3e-base"
    EMBEDDING_DEVICE = "cuda"
    EMBEDDING_BATCH_SIZE = 32
    RERANKER_MODEL_NAME = r"D:\models\huggingface\reranker\bge-reranker-v2-m3"
    RERANKER_DEVICE = "cuda"
    RERANKER_BATCH_SIZE = 16
    PARENT_CHILD_DENSE_TOP_K = 10
    PARENT_CHILD_HYBRID_KEYWORD_TOP_K = 10
    PARENT_CHILD_HYBRID_FINAL_TOP_K = 5
    PARENT_CHILD_RRF_K = 60
    PARENT_CHILD_RERANK_TOP_K = 5
    PARENT_CHILD_RERANK_MAX_LENGTH = 512
    PARENT_CHILD_RERANK_LOCAL_FILES_ONLY = True
    PARENT_CHILD_MAX_CONTEXT_CHARS = 6000
    PARENT_CHILD_MAX_CONTEXT_ITEMS = 3
    PARENT_CHILD_EVAL_TOP_K = 5
    PARENT_CHILD_DEDUP_PARENT = True

try:
    from rag_template.configs.LLMConfig import (
        LLM_DEVICE,
        LLM_DO_SAMPLE,
        LLM_MAX_NEW_TOKENS,
        LLM_MODEL_NAME,
        LLM_TEMPERATURE,
        LLM_TOP_P,
    )
except Exception:  # pragma: no cover
    LLM_MODEL_NAME = r"D:\models\huggingface\llm\Qwen2.5-1.5B-Instruct"
    LLM_DEVICE = "cuda"
    LLM_MAX_NEW_TOKENS = 256
    LLM_TEMPERATURE = 0.7
    LLM_TOP_P = 0.9
    LLM_DO_SAMPLE = False


@dataclass
class RAGToolConfig:
    """Configuration for RAGTool.

    Keep this config local to the tool stage. When migrating to the enterprise
    framework, this dataclass can be replaced by backend/core/config.py.
    """

    parent_file: str = str(PARENT_CHUNKS_FILE)
    child_file: str = str(CHILD_CHUNKS_FILE)
    db_file: str = str(PARENT_CHILD_MILVUS_DB_FILE)
    capture_output: str = "data/processed/runs/rag_runs.jsonl"

    collection_name: str = str(PARENT_CHILD_MILVUS_COLLECTION_NAME)
    metric_type: str = str(PARENT_CHILD_SEARCH_METRIC_TYPE)

    embedding_model: str = str(EMBEDDING_MODEL_NAME)
    embedding_device: str = str(EMBEDDING_DEVICE)
    embedding_batch_size: int = int(EMBEDDING_BATCH_SIZE)
    hash_embedding: bool = False
    hash_dim: int = 768

    dense_top_k: int = int(PARENT_CHILD_DENSE_TOP_K)
    keyword_top_k: int = int(PARENT_CHILD_HYBRID_KEYWORD_TOP_K)
    candidate_top_k: int = int(PARENT_CHILD_HYBRID_FINAL_TOP_K)
    rrf_k: int = int(PARENT_CHILD_RRF_K)
    dedup_parent: bool = bool(PARENT_CHILD_DEDUP_PARENT)

    skip_rerank: bool = False
    reranker_model: str = str(RERANKER_MODEL_NAME)
    reranker_device: str = str(RERANKER_DEVICE)
    reranker_batch_size: int = int(RERANKER_BATCH_SIZE)
    reranker_max_length: int = int(PARENT_CHILD_RERANK_MAX_LENGTH)
    reranker_local_files_only: bool = bool(PARENT_CHILD_RERANK_LOCAL_FILES_ONLY)
    rerank_top_k: int = int(PARENT_CHILD_RERANK_TOP_K)

    max_context_chars: int = int(PARENT_CHILD_MAX_CONTEXT_CHARS)
    max_context_items: int = int(PARENT_CHILD_MAX_CONTEXT_ITEMS)
    eval_top_k: int = int(PARENT_CHILD_EVAL_TOP_K)

    enable_llm: bool = True
    llm_model: str = str(LLM_MODEL_NAME)
    llm_device: str = str(LLM_DEVICE)
    max_new_tokens: int = int(LLM_MAX_NEW_TOKENS)
    temperature: float = float(LLM_TEMPERATURE)
    top_p: float = float(LLM_TOP_P)
    do_sample: bool = bool(LLM_DO_SAMPLE)

    model_provider: str = "local"
    pipeline_name: str = "parent_child_hybrid_rag_tool"
    pipeline_version: str = "v1.0"


def _split_csv(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _resolve_path(path: str | Path, project_root: Optional[str | Path] = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    root = Path(project_root) if project_root else Path.cwd()
    return str((root / p).resolve())


def _assert_exists(path: str | Path, name: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _compact_contexts(retrieval_results: List[Dict[str, Any]], max_text_chars: int = 500) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for item in retrieval_results:
        text = str(item.get("text") or item.get("parent_text") or item.get("child_text") or "")
        contexts.append(
            {
                "rank": item.get("rank"),
                "doc_id": item.get("doc_id"),
                "parent_chunk_id": item.get("parent_chunk_id"),
                "child_chunk_id": item.get("child_chunk_id"),
                "title": item.get("title"),
                "section": item.get("section"),
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
                "text_preview": text[:max_text_chars],
                "metadata": item.get("metadata") or {},
            }
        )
    return contexts


class RAGTool(BaseTool):
    """Agent-facing wrapper for ParentChildRAGEngine."""

    name = "rag_tool"
    description = "Parent-Child hybrid RAG QA tool."

    def __init__(self, config: Optional[RAGToolConfig] = None, *, project_root: Optional[str | Path] = None):
        self.config = config or RAGToolConfig()
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.engine: Optional[ParentChildRAGEngine] = None
        self._initialized = False

    @classmethod
    def from_default_config(cls, *, project_root: Optional[str | Path] = None) -> "RAGTool":
        return cls(RAGToolConfig(), project_root=project_root)

    def _resolved_config(self) -> RAGToolConfig:
        cfg = RAGToolConfig(**asdict(self.config))
        cfg.parent_file = _resolve_path(cfg.parent_file, self.project_root)
        cfg.child_file = _resolve_path(cfg.child_file, self.project_root)
        cfg.db_file = _resolve_path(cfg.db_file, self.project_root)
        cfg.capture_output = _resolve_path(cfg.capture_output, self.project_root)
        return cfg

    def initialize(self) -> None:
        """Load stores, retrievers, reranker, prompt builder, capture, and optional LLM."""
        if self._initialized and self.engine is not None:
            return

        cfg = self._resolved_config()
        _assert_exists(cfg.parent_file, "parent_file")
        _assert_exists(cfg.child_file, "child_file")
        _assert_exists(cfg.db_file, "db_file")

        parent_store = ParentChunkStore.from_jsonl(cfg.parent_file)
        keyword_retriever = BM25ChildRetriever.from_jsonl(cfg.child_file)
        dense_retriever = MilvusChildRetriever(
            db_file=cfg.db_file,
            collection_name=cfg.collection_name,
            metric_type=cfg.metric_type,
            embedding_model=cfg.embedding_model,
            embedding_device=cfg.embedding_device,
            embedding_batch_size=1,
            hash_embedding=cfg.hash_embedding,
            hash_dim=cfg.hash_dim,
        )
        hybrid_retriever = HybridParentChildRetriever(
            dense_retriever=dense_retriever,
            keyword_retriever=keyword_retriever,
            parent_store=parent_store,
            rrf_k=cfg.rrf_k,
            dedup_parent=cfg.dedup_parent,
        )

        if cfg.skip_rerank:
            reranker = NoOpParentChildReranker()
        else:
            reranker = ParentChildReranker(
                model_name=cfg.reranker_model,
                device=cfg.reranker_device,
                batch_size=cfg.reranker_batch_size,
                max_length=cfg.reranker_max_length,
                local_files_only=cfg.reranker_local_files_only,
            )

        context_packer = ContextPacker(
            max_context_chars=cfg.max_context_chars,
            max_items=cfg.max_context_items,
            text_field="text",
            dedup_parent=True,
            include_metadata=True,
        )
        prompt_builder = ParentChildPromptBuilder()
        run_capture = RagRunCapture(cfg.capture_output)

        llm_generator = None
        model_name = None
        model_provider = None
        if cfg.enable_llm:
            llm_generator = LocalLLMGenerator(
                model_name=cfg.llm_model,
                device=cfg.llm_device,
            )
            model_name = cfg.llm_model
            model_provider = cfg.model_provider

        self.engine = ParentChildRAGEngine(
            retriever=hybrid_retriever,
            reranker=reranker,
            context_packer=context_packer,
            prompt_builder=prompt_builder,
            run_capture=run_capture,
            llm_generator=llm_generator,
            model_name=model_name,
            model_provider=model_provider,
            pipeline_name=cfg.pipeline_name,
            pipeline_version=cfg.pipeline_version,
        )
        self.config = cfg
        self._initialized = True

    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run RAGTool.

        Required input:
            {"query": "..."}

        Optional input overrides:
            dense_top_k, keyword_top_k, candidate_top_k, rerank_top_k, eval_top_k,
            expected_doc_ids, expected_parent_chunk_ids, expected_child_chunk_ids,
            expected_keywords, generate_answer, generation_params, return_prompt,
            return_full_record.
        """
        try:
            if not isinstance(tool_input, dict):
                raise TypeError("tool_input must be a dict")
            query = str(tool_input.get("query") or "").strip()
            if not query:
                raise ValueError("tool_input['query'] cannot be empty")

            self.initialize()
            assert self.engine is not None
            cfg = self.config

            generation_params = dict(tool_input.get("generation_params") or {})
            if not generation_params:
                generation_params = {
                    "max_new_tokens": int(tool_input.get("max_new_tokens", cfg.max_new_tokens)),
                    "temperature": float(tool_input.get("temperature", cfg.temperature)),
                    "top_p": float(tool_input.get("top_p", cfg.top_p)),
                    "do_sample": bool(tool_input.get("do_sample", cfg.do_sample)),
                }

            generate_answer = tool_input.get("generate_answer", cfg.enable_llm)
            result = self.engine.run(
                query=query,
                dense_top_k=int(tool_input.get("dense_top_k", cfg.dense_top_k)),
                keyword_top_k=int(tool_input.get("keyword_top_k", cfg.keyword_top_k)),
                candidate_top_k=int(tool_input.get("candidate_top_k", cfg.candidate_top_k)),
                rrf_k=int(tool_input.get("rrf_k", cfg.rrf_k)),
                rerank_top_k=int(tool_input.get("rerank_top_k", cfg.rerank_top_k)),
                eval_top_k=int(tool_input.get("eval_top_k", cfg.eval_top_k)),
                expected_doc_ids=_split_csv(tool_input.get("expected_doc_ids")),
                expected_parent_chunk_ids=_split_csv(tool_input.get("expected_parent_chunk_ids")),
                expected_child_chunk_ids=_split_csv(tool_input.get("expected_child_chunk_ids")),
                expected_keywords=_split_csv(tool_input.get("expected_keywords")),
                generate_answer=bool(generate_answer),
                generation_params=generation_params if bool(generate_answer) else None,
                extra_metadata={
                    "tool_name": self.name,
                    "tool_stage": "rag_tool_v1",
                },
            )

            data: Dict[str, Any] = {
                "run_id": result.get("run_id"),
                "query": result.get("query"),
                "answer": result.get("answer"),
                "contexts": _compact_contexts(result.get("retrieval_results") or []),
                "citations": result.get("citations") or [],
                "eval_result": result.get("eval_result") or {},
                "capture_result": result.get("capture_result"),
                "model_name": result.get("model_name"),
                "model_provider": result.get("model_provider"),
            }
            if bool(tool_input.get("return_prompt", False)):
                data["prompt"] = result.get("prompt")
            if bool(tool_input.get("return_full_record", False)):
                data["run_record"] = result.get("run_record")
                data["retrieval_results"] = result.get("retrieval_results") or []
                data["context_pack"] = result.get("context_pack") or {}

            return self._ok(
                data=data,
                metadata={
                    "pipeline_name": cfg.pipeline_name,
                    "pipeline_version": cfg.pipeline_version,
                    "capture_output": cfg.capture_output,
                },
            )
        except Exception as exc:
            return self._fail(
                error=f"{exc.__class__.__name__}: {exc}",
                metadata={"tool_stage": "rag_tool_v1"},
            )

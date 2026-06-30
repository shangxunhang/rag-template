# -*- coding: utf-8 -*-
"""
rag_template/context/context_packer.py
======================================

P3 context packer:
- Input: reranked retrieval_result_v2 list.
- Output: packed context string + selected results + citation metadata.

职责边界：
- 只做上下文预算控制和格式化，不调用 LLM。
- 第一版用字符预算，避免强依赖 tokenizer；后续可替换成 token budget。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


@dataclass
class ContextPack:
    """Packed context result."""

    context: str
    selected_results: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    dropped_results: List[Dict[str, Any]] = field(default_factory=list)
    max_context_chars: int = 0
    used_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "selected_results": self.selected_results,
            "citations": self.citations,
            "dropped_results": self.dropped_results,
            "max_context_chars": self.max_context_chars,
            "used_chars": self.used_chars,
            "selected_count": len(self.selected_results),
            "dropped_count": len(self.dropped_results),
        }


class ContextPacker:
    """Pack retrieval results into prompt-ready context."""

    def __init__(
        self,
        *,
        max_context_chars: int = 6000,
        max_items: int = 5,
        text_field: str = "text",
        dedup_parent: bool = True,
        include_metadata: bool = True,
    ):
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be > 0")
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self.max_context_chars = int(max_context_chars)
        self.max_items = int(max_items)
        self.text_field = text_field
        self.dedup_parent = bool(dedup_parent)
        self.include_metadata = bool(include_metadata)

    @staticmethod
    def build_citation(result: Dict[str, Any], context_rank: int) -> Dict[str, Any]:
        meta = result.get("metadata") or {}
        return {
            "context_rank": context_rank,
            "rank": result.get("rank"),
            "doc_id": result.get("doc_id"),
            "chunk_id": result.get("chunk_id"),
            "child_chunk_id": result.get("child_chunk_id"),
            "parent_chunk_id": result.get("parent_chunk_id"),
            "title": result.get("title"),
            "section": result.get("section"),
            "page_start": result.get("page_start"),
            "page_end": result.get("page_end"),
            "score": result.get("score"),
            "rerank_score": result.get("rerank_score"),
            "retrieval_sources": meta.get("retrieval_sources", []),
            "matched_child_chunk_ids": meta.get("matched_child_chunk_ids", []),
        }

    def _get_text(self, result: Dict[str, Any]) -> str:
        if self.text_field and result.get(self.text_field):
            return _safe_str(result.get(self.text_field))
        return _safe_str(result.get("text") or result.get("parent_text") or result.get("child_text"))

    def _format_item(self, result: Dict[str, Any], context_rank: int, text: str) -> str:
        title = _safe_str(result.get("title"))
        section = _safe_str(result.get("section"))
        doc_id = _safe_str(result.get("doc_id"))
        parent_id = _safe_str(result.get("parent_chunk_id"))
        child_id = _safe_str(result.get("child_chunk_id"))
        page_start = result.get("page_start")
        page_end = result.get("page_end")
        score = _safe_float(result.get("score"))
        rerank_score = result.get("rerank_score")
        meta = result.get("metadata") or {}
        sources = meta.get("retrieval_sources", [])

        if page_start is not None and page_end is not None:
            page = f"{page_start}~{page_end}"
        elif page_start is not None:
            page = str(page_start)
        else:
            page = ""

        if self.include_metadata:
            header = (
                f"[资料 {context_rank}]\n"
                f"doc_id: {doc_id}\n"
                f"parent_chunk_id: {parent_id}\n"
                f"child_chunk_id: {child_id}\n"
                f"title: {title}\n"
                f"section: {section}\n"
                f"page: {page}\n"
                f"score: {score}\n"
                f"rerank_score: {rerank_score}\n"
                f"retrieval_sources: {sources}\n"
                f"text:\n"
            )
        else:
            header = f"[资料 {context_rank}]\n"
        return f"{header}{text}".strip()

    def pack(self, results: Iterable[Dict[str, Any]]) -> ContextPack:
        selected: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        parts: List[str] = []
        seen_parents: Set[str] = set()
        used_chars = 0

        for result in results:
            if len(selected) >= self.max_items:
                dropped.append(result)
                continue
            parent_id = _safe_str(result.get("parent_chunk_id"))
            if self.dedup_parent and parent_id and parent_id in seen_parents:
                dropped.append(result)
                continue

            text = self._get_text(result).strip()
            if not text:
                dropped.append(result)
                continue

            context_rank = len(selected) + 1
            item = self._format_item(result, context_rank=context_rank, text=text)
            extra_len = len(item) + (2 if parts else 0)
            remaining = self.max_context_chars - used_chars

            if remaining <= 0:
                dropped.append(result)
                continue

            if extra_len > remaining:
                # Allow a truncated first item, but avoid adding very tiny fragments later.
                if not parts and remaining > 300:
                    item = item[:remaining].rstrip() + "\n...[TRUNCATED]"
                    extra_len = len(item)
                else:
                    dropped.append(result)
                    continue

            parts.append(item)
            selected.append(result)
            citations.append(self.build_citation(result, context_rank=context_rank))
            if parent_id:
                seen_parents.add(parent_id)
            used_chars += extra_len

        context = "\n\n".join(parts)
        return ContextPack(
            context=context,
            selected_results=selected,
            citations=citations,
            dropped_results=dropped,
            max_context_chars=self.max_context_chars,
            used_chars=len(context),
        )

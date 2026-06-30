# -*- coding: utf-8 -*-
"""
rag_template/tools/base_tool.py
===============================

Minimal Tool interface for the next Agent stage.

Design goal:
- Agent only depends on BaseTool.run(tool_input).
- Agent should not know RAG internal modules such as Milvus, BM25, RRF,
  reranker, context packing, prompt builder, or local LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Standard tool result used by Agent / Workflow layers."""

    success: bool
    tool_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool interface."

    @abstractmethod
    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with a dict input and return a serializable dict."""
        raise NotImplementedError

    def _ok(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return ToolResult(
            success=True,
            tool_name=self.name,
            data=data,
            error=None,
            metadata=metadata or {},
        ).to_dict()

    def _fail(self, error: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return ToolResult(
            success=False,
            tool_name=self.name,
            data={},
            error=error,
            metadata=metadata or {},
        ).to_dict()

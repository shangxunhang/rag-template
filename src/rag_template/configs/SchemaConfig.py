"""
configs/SchemaConfig.py
=======================

Mini-RAG 的数据结构配置文件。

职责：
1. 统一定义 Document Schema
2. 统一定义 Chunk Schema
3. 统一定义 Chunk Meta Schema
4. 统一定义 Retrieval Result Schema
5. 提供标准结构构造函数，避免 reader / chunker / retriever 中硬编码字段

注意：
- 这里只定义“数据结构”和“默认字段”
- 不负责读取文件
- 不负责清洗文本
- 不负责切 chunk
- 不负责向量化
"""

from datetime import datetime
from copy import deepcopy
from typing import Dict, Any, Optional


# =========================
# 1. 通用默认值
# =========================

DEFAULT_NONE = None
DEFAULT_DOC_TYPE = "unknown"
DEFAULT_TITLE = None
DEFAULT_PAGE = None
DEFAULT_SECTION = None
DEFAULT_VERSION = None
DEFAULT_STATUS = "active"
DEFAULT_LANGUAGE = "zh"
DEFAULT_SECURITY_LEVEL = "internal"


def current_time_str() -> str:
    """
    返回当前时间字符串。

    Returns:
        当前时间，格式：YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# 2. Document Metadata 默认结构
# =========================

DOCUMENT_METADATA_TEMPLATE: Dict[str, Any] = {
    # 来源信息
    "source": None,              # 原始文件名 / 来源表 / 来源系统
    "source_path": None,         # 文件完整路径，可选
    "doc_type": DEFAULT_DOC_TYPE, # txt / json / jsonl / pdf / docx / csv / mysql

    # 文档描述信息
    "title": DEFAULT_TITLE,
    "author": None,
    "language": DEFAULT_LANGUAGE,

    # 时间信息
    "created_at": None,
    "updated_at": None,
    "indexed_at": None,

    # 版本治理
    "version": DEFAULT_VERSION,
    "status": DEFAULT_STATUS,     # active / archived / deleted
    "is_latest": True,

    # 企业场景扩展
    "department": None,
    "project_id": None,
    "project_name": None,
    "security_level": DEFAULT_SECURITY_LEVEL,

    # 预留字段
    "extra": {},
}


# =========================
# 3. Document 默认结构
# =========================

DOCUMENT_TEMPLATE: Dict[str, Any] = {
    "doc_id": None,
    "text": "",
    "metadata": DOCUMENT_METADATA_TEMPLATE,
}


# =========================
# 4. Chunk Metadata 默认结构
# =========================

CHUNK_METADATA_TEMPLATE: Dict[str, Any] = {
    # 来源信息，通常从 document.metadata 继承
    "source": None,
    "source_path": None,
    "doc_type": DEFAULT_DOC_TYPE,
    "title": DEFAULT_TITLE,

    # chunk 位置信息
    "chunk_index": None,
    "page": DEFAULT_PAGE,
    "section": DEFAULT_SECTION,
    "section_path": None,

    # 时间信息
    "created_at": None,
    "updated_at": None,
    "indexed_at": None,

    # 版本治理
    "version": DEFAULT_VERSION,
    "status": DEFAULT_STATUS,
    "is_latest": True,

    # 企业场景扩展
    "department": None,
    "project_id": None,
    "project_name": None,
    "security_level": DEFAULT_SECURITY_LEVEL,

    # chunk 统计信息
    "char_count": None,
    "token_count": None,

    # 预留字段
    "extra": {},
}


# =========================
# 5. Chunk 默认结构
# =========================

CHUNK_TEMPLATE: Dict[str, Any] = {
    "chunk_id": None,
    "doc_id": None,
    "text": "",
    "metadata": CHUNK_METADATA_TEMPLATE,
}


# =========================
# 6. Retrieval Result 默认结构
# =========================

RETRIEVAL_RESULT_TEMPLATE: Dict[str, Any] = {
    "rank": None,
    "score": None,
    "rerank_score": None,

    "chunk_id": None,
    "doc_id": None,
    "text": "",

    "metadata": CHUNK_METADATA_TEMPLATE,
}


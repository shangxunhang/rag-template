"""
scripts/build_milvus_lite_index.py
==================================

构建 Milvus Lite 向量索引。

完整流程：
1. 读取 BaseConfig.CHUNKS_FILE
2. 使用 TextEmbedder 生成 chunk embedding
3. 写入 BaseConfig.MILVUS_LITE_DB_FILE
4. 创建 / 重建 RAGConfig.MILVUS_COLLECTION_NAME

注意：
- 路径配置全部来自 BaseConfig。
- Milvus 参数和 embedding 参数来自 RAGConfig。
- 运行前请先确保 rag-template 已执行 pip install -e .
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rag_template.configs.BaseConfig import (
    CHUNKS_FILE,
    MILVUS_LITE_DB_FILE,
)
from rag_template.configs.RAGConfig import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    MILVUS_COLLECTION_NAME,
    MILVUS_DIM,
)
from rag_template.embed.embedder import TextEmbedder
from rag_template.vector_store.milvus_lite_store import MilvusLiteStore


def load_chunks(chunks_file: str | Path) -> List[Dict[str, Any]]:
    chunks_file = Path(chunks_file)

    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks 文件不存在: {chunks_file}")

    with chunks_file.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json 顶层结构必须是 list。")

    return chunks


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    return str(
        chunk.get("text")
        or chunk.get("chunk_text")
        or chunk.get("content")
        or ""
    )


def build_milvus_records(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> List[Dict[str, Any]]:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks 数量和 embeddings 数量不一致: {len(chunks)} != {len(embeddings)}"
        )

    records: List[Dict[str, Any]] = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = chunk.get("metadata", {}) or {}

        doc_id = chunk.get("doc_id") or metadata.get("doc_id") or ""
        chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id") or ""
        text = get_chunk_text(chunk)

        source = chunk.get("source") or metadata.get("source") or ""

        record = {
            "id": index,
            "vector": embedding,
            "doc_id": str(doc_id),
            "chunk_id": str(chunk_id),
            "text": text,
            "source": str(source),
            "doc_type": str(chunk.get("doc_type") or metadata.get("doc_type") or ""),
            "title": str(chunk.get("title") or metadata.get("title") or ""),
            "chunk_index": int(
                chunk.get("chunk_index")
                if chunk.get("chunk_index") is not None
                else metadata.get("chunk_index", index)
            ),
            "security_level": str(
                chunk.get("security_level")
                or metadata.get("security_level")
                or "internal"
            ),
            "project_id": str(
                chunk.get("project_id")
                or metadata.get("project_id")
                or ""
            ),
            "status": str(
                chunk.get("status")
                or metadata.get("status")
                or "active"
            ),
            "is_latest": bool(
                chunk.get("is_latest")
                if chunk.get("is_latest") is not None
                else metadata.get("is_latest", True)
            ),
        }

        records.append(record)

    return records


def main() -> None:
    print("=" * 80)
    print("[Build Milvus Lite Index] 开始")
    print("=" * 80)
    print(f"[Path] CHUNKS_FILE            = {CHUNKS_FILE}")
    print(f"[Path] MILVUS_LITE_DB_FILE    = {MILVUS_LITE_DB_FILE}")
    print(f"[Config] MILVUS_COLLECTION_NAME = {MILVUS_COLLECTION_NAME}")
    print(f"[Config] MILVUS_DIM             = {MILVUS_DIM}")

    chunks = load_chunks(CHUNKS_FILE)
    print(f"[Chunks] chunks 数量 = {len(chunks)}")

    if not chunks:
        raise ValueError("chunks.json 为空，无法构建 Milvus 索引")

    texts = [get_chunk_text(chunk) for chunk in chunks]

    print("=" * 80)
    print("[Embedding] 初始化 TextEmbedder")
    print(f"[Embedding] EMBEDDING_MODEL_NAME = {EMBEDDING_MODEL_NAME}")
    print(f"[Embedding] EMBEDDING_DEVICE     = {EMBEDDING_DEVICE}")
    print(f"[Embedding] EMBEDDING_BATCH_SIZE = {EMBEDDING_BATCH_SIZE}")

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print("=" * 80)
    print("[Embedding] 生成 embeddings")

    embeddings = embedder.encode_texts(texts)

    embeddings = [
        embedding.tolist() if hasattr(embedding, "tolist") else embedding
        for embedding in embeddings
    ]

    if embeddings:
        actual_dim = len(embeddings[0])
        print(f"[Embedding] actual_dim = {actual_dim}")

        if actual_dim != MILVUS_DIM:
            raise ValueError(
                f"MILVUS_DIM 配置错误: config={MILVUS_DIM}, actual={actual_dim}"
            )

    print("=" * 80)
    print("[Milvus] 初始化 MilvusLiteStore")

    store = MilvusLiteStore(
        db_file=MILVUS_LITE_DB_FILE,
        collection_name=MILVUS_COLLECTION_NAME,
        dim=MILVUS_DIM,
    )

    print("=" * 80)
    print("[Milvus] 创建 / 重建 collection")
    store.create_collection(recreate=True)

    print("=" * 80)
    print("[Milvus] 构造 records")
    records = build_milvus_records(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"[Milvus] records 数量 = {len(records)}")

    print("=" * 80)
    print("[Milvus] 写入 Milvus Lite")
    result = store.insert(records)

    print("=" * 80)
    print("[Build Milvus Lite Index] 完成")
    print(f"[Milvus] insert result    = {result}")
    print(f"[Milvus] collection count = {store.count()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

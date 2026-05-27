"""
scripts/build_index.py
======================

Mini-RAG 的 FAISS 索引构建脚本。

完整流程：
1. 读取 raw 目录下的 txt / json / jsonl 文档
2. 转换为统一 Document Schema
3. 清洗 documents
4. 切分 chunks
5. 保存 documents.json / chunks.json
6. 使用 embedding 模型生成 chunk embeddings
7. 构建 FAISS index
8. 保存 faiss.index / chunk_meta.json

注意：
- 路径配置全部来自 BaseConfig。
- RAG 参数配置来自 RAGConfig。
- 运行前请先确保 rag-template 已执行 pip install -e .
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_template.configs.BaseConfig import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DOCUMENTS_FILE,
    CHUNKS_FILE,
    VECTOR_STORE_DIR,
    FAISS_INDEX_FILE,
    CHUNK_META_FILE,
)
from rag_template.configs.RAGConfig import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
)
from rag_template.reader.reader_factory import load_documents
from rag_template.cleaner.clean import clean_documents
from rag_template.chunker.chunk import chunk_documents
from rag_template.embed.embedder import TextEmbedder
from rag_template.Index.indexer import (
    build_faiss_index,
    save_faiss_index,
    save_chunk_meta,
)


def save_json(data: Any, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_doc_type_stats(documents: list[dict]) -> None:
    stats: dict[str, int] = {}

    for doc in documents:
        metadata = doc.get("metadata", {}) or {}
        doc_type = metadata.get("doc_type", "unknown")
        stats[doc_type] = stats.get(doc_type, 0) + 1

    print("[Reader] doc_type 统计:")
    for doc_type, count in sorted(stats.items()):
        print(f"  - {doc_type}: {count}")


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[Build FAISS Index] 开始")
    print("=" * 80)
    print(f"[Path] RAW_DATA_DIR       = {RAW_DATA_DIR}")
    print(f"[Path] PROCESSED_DATA_DIR = {PROCESSED_DATA_DIR}")
    print(f"[Path] DOCUMENTS_FILE     = {DOCUMENTS_FILE}")
    print(f"[Path] CHUNKS_FILE        = {CHUNKS_FILE}")
    print(f"[Path] VECTOR_STORE_DIR   = {VECTOR_STORE_DIR}")
    print(f"[Path] FAISS_INDEX_FILE   = {FAISS_INDEX_FILE}")
    print(f"[Path] CHUNK_META_FILE    = {CHUNK_META_FILE}")

    documents = load_documents(RAW_DATA_DIR)

    print(f"[Reader] 读取 document 数量: {len(documents)}")
    print_doc_type_stats(documents)

    if not documents:
        raise ValueError(f"未读取到任何支持的文档，请检查目录: {RAW_DATA_DIR}")

    cleaned_documents = clean_documents(documents)
    print("[Cleaner] 文档清洗完成")

    save_json(cleaned_documents, DOCUMENTS_FILE)
    print(f"[Writer] documents.json 已保存: {DOCUMENTS_FILE}")

    chunks = chunk_documents(
        documents=cleaned_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(f"[Chunker] chunk_size: {CHUNK_SIZE}")
    print(f"[Chunker] chunk_overlap: {CHUNK_OVERLAP}")
    print(f"[Chunker] chunk 数量: {len(chunks)}")

    if not chunks:
        raise ValueError("chunk 数量为 0，无法继续构建索引")

    save_json(chunks, CHUNKS_FILE)
    print(f"[Writer] chunks.json 已保存: {CHUNKS_FILE}")

    chunk_texts = [chunk["text"] for chunk in chunks]

    print("=" * 80)
    print("[Embedding] 初始化 TextEmbedder")
    print(f"[Embedding] model_name = {EMBEDDING_MODEL_NAME}")
    print(f"[Embedding] device     = {EMBEDDING_DEVICE}")
    print(f"[Embedding] batch_size = {EMBEDDING_BATCH_SIZE}")

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    embeddings = embedder.encode_texts(chunk_texts)

    print(f"[Embedding] embeddings shape = {embeddings.shape}")
    print(f"[Embedding] embeddings dtype  = {embeddings.dtype}")

    index = build_faiss_index(embeddings)

    print("=" * 80)
    print("[FAISS] index 构建完成")
    print(f"[FAISS] index.ntotal = {index.ntotal}")
    print(f"[FAISS] index.d      = {index.d}")

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index 向量数量和 chunks 数量不一致: "
            f"index.ntotal={index.ntotal}, len(chunks)={len(chunks)}"
        )

    save_faiss_index(index, FAISS_INDEX_FILE)
    print(f"[FAISS] faiss.index 已保存: {FAISS_INDEX_FILE}")

    save_chunk_meta(chunks, CHUNK_META_FILE)
    print(f"[Writer] chunk_meta.json 已保存: {CHUNK_META_FILE}")

    print("=" * 80)
    print("[Build FAISS Index] 完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

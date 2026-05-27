"""
ragtests/test_build_index_flow.py
==============================

测试 Mini-RAG 的索引构建全流程。

测试范围：
1. 读取 data/processed/chunks.json
2. 使用 Hash_embedder 生成 embeddings
3. 使用 indexer 构建 FAISS index
4. 保存 faiss.index
5. 保存 chunk_meta.json
6. 简单验证 FAISS index 和 chunk_meta 是否匹配

注意：
这里测试的是：
chunks.json -> embedding -> FAISS index -> 保存索引

暂时不测试：
reader / cleaner / chunker / retriever / generator
"""

import sys
import json
from pathlib import Path

from rag_template.Index.indexer import load_chunks, build_faiss_index, save_faiss_index, save_chunk_meta
from rag_template.embed.Hash_embedder import HashTextEmbedder

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))



from rag_template.configs.RAGConfig import *

# 这里按你的文件名 Hash_embedder.py 导入
# 如果你的类名不同，改这里



def test_build_index_flow():
    """
    测试 chunks.json 到 FAISS index 的完整构建流程。
    """

    # =========================
    # 1. 读取 chunks.json
    # =========================

    chunks = load_chunks(CHUNKS_FILE)

    print(f"[Load] chunks 数量: {len(chunks)}")

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert "chunk_id" in chunks[0]

    # =========================
    # 2. 提取 chunk 文本
    # =========================

    chunk_texts = [chunk["text"] for chunk in chunks]

    print(f"[Extract] chunk_texts 数量: {len(chunk_texts)}")

    assert len(chunk_texts) == len(chunks)

    # =========================
    # 3. 使用 HashEmbedder 生成 embeddings
    # =========================

    embedder = HashTextEmbedder(embedding_dim=384)

    embeddings = embedder.encode_texts(chunk_texts)

    print(f"[Embedding] embeddings shape: {embeddings.shape}")

    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384
    assert embeddings.dtype == "float32"

    # =========================
    # 4. 构建 FAISS index
    # =========================

    index = build_faiss_index(embeddings)

    print(f"[FAISS] index.ntotal: {index.ntotal}")
    print(f"[FAISS] index.d: {index.d}")

    assert index.ntotal == len(chunks)
    assert index.d == embeddings.shape[1]

    # =========================
    # 5. 保存 faiss.index
    # =========================

    save_faiss_index(index, FAISS_INDEX_FILE)

    print(f"[Save] FAISS index 已保存: {FAISS_INDEX_FILE}")

    assert FAISS_INDEX_FILE.exists()

    # =========================
    # 6. 保存 chunk_meta.json
    # =========================

    save_chunk_meta(chunks, CHUNK_META_FILE)

    print(f"[Save] chunk_meta 已保存: {CHUNK_META_FILE}")

    assert CHUNK_META_FILE.exists()

    # =========================
    # 7. 验证 chunk_meta.json 内容
    # =========================

    with open(CHUNK_META_FILE, "r", encoding="utf-8") as f:
        chunk_meta = json.load(f)

    print(f"[Check] chunk_meta 数量: {len(chunk_meta)}")

    assert isinstance(chunk_meta, list)
    assert len(chunk_meta) == len(chunks)
    assert chunk_meta[0]["chunk_id"] == chunks[0]["chunk_id"]
    assert chunk_meta[0]["text"] == chunks[0]["text"]

    print("\n[Test Passed] build index 全流程测试通过。")


if __name__ == "__main__":
    test_build_index_flow()
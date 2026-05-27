"""
ragtests/test_full_flow.py
=======================

Mini-RAG 全流程测试（不调用 LLM）

流程：
1. 读取文档
2. 清洗文本
3. 切分 chunks
4. 使用真实 embedding 编码
5. 构建 FAISS index
6. 保存 faiss.index 和 chunk_meta.json
7. 加载 FAISS retriever
8. 检索 query
9. 构造 prompt
"""

import sys
import json
from pathlib import Path


# =========================
# 1. 加入项目根目录
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


# =========================
# 2. 导入配置
# =========================


from rag_template.configs.BaseConfig import  *
from rag_template.configs.LLMConfig import  *
from rag_template.configs.RAGConfig import *

# =========================
# 3. 导入业务模块
# =========================

from rag_template.reader.txt_reader import load_txt_documents
from rag_template.cleaner.clean import clean_documents
from rag_template.chunker.chunk import chunk_documents
from rag_template.embed.embedder import TextEmbedder
from rag_template.Index.indexer import (
    build_faiss_index,
    save_faiss_index,
    save_chunk_meta,
)
from rag_template.retriever.retriever import FaissRetriever
from rag_template.prompt.prompt_builder import build_rag_prompt


def test_full_rag_flow():
    """
    测试完整 Mini-RAG 流程：
    TXT -> clean -> chunk -> embedding -> FAISS -> retrieve -> prompt
    """
    print("=" * 80)
    print("[Test] Mini-RAG 全流程测试开始")
    print("=" * 80)

    # =========================
    # 1. 读取文档
    # =========================

    documents = load_txt_documents(RAW_DATA_DIR)
    print(f"[Load] 文档数量: {len(documents)}")

    if len(documents) == 0:
        raise ValueError(f"未读取到任何 txt 文档，请检查目录: {RAW_DATA_DIR}")

    # =========================
    # 2. 清洗文本
    # =========================

    cleaned_documents = clean_documents(documents)
    print(f"[Clean] 清洗完成，文档数量: {len(cleaned_documents)}")

    # =========================
    # 3. 切分 chunks
    # =========================

    chunks = chunk_documents(
        documents=cleaned_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"[Chunk] chunk 数量: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("chunk 数量为 0，无法继续测试")

    # =========================
    # 4. 保存 chunks.json
    # =========================

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[Save] chunks.json 已保存: {CHUNKS_FILE}")

    # =========================
    # 5. 初始化真实 embedding
    # =========================

    print("[Embedding] 正在加载 embedding 模型...")
    print(f"[Embedding] model_name: {EMBEDDING_MODEL_NAME}")
    print(f"[Embedding] device: {EMBEDDING_DEVICE}")
    print(f"[Embedding] batch_size: {EMBEDDING_BATCH_SIZE}")

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print("[Embedding] 模型加载完成")

    # =========================
    # 6. 生成 embeddings
    # =========================

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode_texts(chunk_texts)

    print(f"[Embedding] embeddings shape: {embeddings.shape}")
    print(f"[Embedding] embeddings dtype: {embeddings.dtype}")

    # =========================
    # 7. 构建 FAISS index
    # =========================

    index = build_faiss_index(embeddings)

    print(f"[FAISS] index.ntotal: {index.ntotal}")
    print(f"[FAISS] index.d: {index.d}")

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS index 向量数量和 chunks 数量不一致: "
            f"index.ntotal={index.ntotal}, len(chunks)={len(chunks)}"
        )

    # =========================
    # 8. 保存 FAISS index 和 metadata
    # =========================

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    save_faiss_index(index, FAISS_INDEX_FILE)
    save_chunk_meta(chunks, CHUNK_META_FILE)

    print(f"[Save] faiss.index 已保存: {FAISS_INDEX_FILE}")
    print(f"[Save] chunk_meta.json 已保存: {CHUNK_META_FILE}")

    # =========================
    # 9. 测试检索
    # =========================

    retriever = FaissRetriever(
        index_path=FAISS_INDEX_FILE,
        chunk_meta_path=CHUNK_META_FILE,
        embedder=embedder,
    )

    query = "FAISS 是什么？"

    retrieved = retriever.search(
        query=query,
        top_k=TOP_K,
    )

    print(f"\n[Query] {query}")
    print(f"[Retrieved Chunks] {len(retrieved)} 条")

    for item in retrieved:
        print("-" * 80)
        print(f"rank: {item['rank']}")
        print(f"score: {item['score']:.4f}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"doc_id: {item['doc_id']}")
        print(f"source: {item['source']}")
        print(f"text: {item['text'][:200]} ...")

    if len(retrieved) == 0:
        raise ValueError("检索结果为空")

    # =========================
    # 10. 测试 prompt 构建
    # =========================

    prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=retrieved,
    )

    print("=" * 80)
    print("[Prompt]")
    print(prompt)
    print("=" * 80)

    # =========================
    # 11. 基础断言
    # =========================

    assert isinstance(prompt, str)
    assert query in prompt
    assert "chunk_id" in prompt
    assert "资料" in prompt

    print("[Test] Mini-RAG 全流程测试完成")


if __name__ == "__main__":
    test_full_rag_flow()
"""
ragtests/test_retriever.py
=======================

测试 src/retriever.py。

测试范围：
1. 加载 vector_store/faiss.index
2. 加载 vector_store/chunk_meta.json
3. 使用 Hash_embedder 编码 query
4. FAISS 检索 top-k chunks
5. 检查返回结果结构是否正确

注意：
这里不测试 LLM。
这里只测试：
query -> retrieval -> top-k chunks
"""

import sys
from pathlib import Path

from rag_template.embed.Hash_embedder import HashTextEmbedder

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))



from rag_template.configs.RAGConfig import *


from rag_template.retriever.retriever import FaissRetriever


def test_retriever_search():
    """
    测试 retriever 是否能根据 query 返回 top-k chunk。
    """

    # =========================
    # 1. 初始化 Hash embedder
    # =========================

    embedder = HashTextEmbedder(embedding_dim=384)

    # =========================
    # 2. 初始化 retriever
    # =========================

    retriever = FaissRetriever(
        index_path=FAISS_INDEX_FILE,
        chunk_meta_path=CHUNK_META_FILE,
        embedder=embedder,
    )

    # =========================
    # 3. 输入 query
    # =========================

    query = "FAISS 是什么？"

    # =========================
    # 4. 执行检索
    # =========================

    results = retriever.search(
        query=query,
        top_k=TOP_K,
    )

    # =========================
    # 5. 打印检索结果
    # =========================

    print("\n[Query]")
    print(query)

    print("\n[Retrieved Chunks]")
    for item in results:
        print("-" * 80)
        print(f"rank: {item['rank']}")
        print(f"score: {item['score']}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"doc_id: {item['doc_id']}")
        print(f"source: {item['source']}")
        print(f"chunk_index: {item['chunk_index']}")
        print(f"text: {item['text']}")

    # =========================
    # 6. 基础断言
    # =========================

    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= TOP_K

    first = results[0]

    assert "rank" in first
    assert "score" in first
    assert "chunk_id" in first
    assert "doc_id" in first
    assert "source" in first
    assert "chunk_index" in first
    assert "text" in first

    assert isinstance(first["rank"], int)
    assert isinstance(first["score"], float)
    assert isinstance(first["text"], str)


if __name__ == "__main__":
    test_retriever_search()

    print("\n[Test Passed] retriever 检索测试通过。")
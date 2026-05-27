"""
scripts/ask.py
==============

Mini-RAG 查询入口脚本。

当前版本只完成到 prompt，不调用 LLM。

流程：
1. 接收用户问题 query
2. 加载  embedder
3. 加载 FAISS retriever
4. 检索 top-k chunks
5. 构造 RAG prompt
6. 打印检索结果和最终 prompt

运行方式：
    python scripts/ask.py

或者：
    python scripts/ask.py "FAISS 是什么？"
"""

from pathlib import Path

from rag_template.configs.RAGConfig import *
from rag_template.embed.embedder import TextEmbedder
from rag_template.prompt.prompt_builder import *
from rag_template.reranker.reranker import TextReranker
from rag_template.retriever.retriever import *

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.append(str(ROOT_DIR))


def print_retrieved_chunks(results):
    """
    打印检索结果。
    """
    print("\n[Retrieved Chunks]")

    for item in results:
        print("-" * 80)
        print(f"rank: {item['rank']}")
        print(f"score: {item['score']:.4f}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"source: {item['source']}")
        print(f"text:\n{item['text']}")


def main():
    """
    Mini-RAG 查询主流程。
    """
    # =========================
    # 1. 获取 query
    # =========================

    if len(sys.argv) >= 2:
        query = " ".join(sys.argv[1:])
    else:
        query = input("请输入问题：").strip()

    if not query:
        raise ValueError("query 不能为空")

    # =========================
    # 2. 初始化 embedder
    # =========================

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    # =========================
    # 3. 初始化 retriever
    # =========================

    retriever = FaissRetriever(
        index_path=FAISS_INDEX_FILE,
        chunk_meta_path=CHUNK_META_FILE,
        embedder=embedder,
    )

    # =========================
    # 4. 检索 chunks
    # =========================

    # 1. FAISS 粗召回
    retrieved_chunks = retriever.search(
        query=query,
        top_k=RETRIEVAL_TOP_K if USE_RERANKER else FINAL_TOP_K,
    )

    # 2. 可选 rerank
    if USE_RERANKER:
        reranker = TextReranker(
            model_name=RERANKER_MODEL_NAME,
            device=RERANKER_DEVICE,
            batch_size=RERANKER_BATCH_SIZE,
        )

        retrieved_chunks = reranker.rerank(
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=FINAL_TOP_K,
        )
    else:
        retrieved_chunks = retrieved_chunks[:FINAL_TOP_K]

    # =========================
    # 5. 构造 prompt
    # =========================

    prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    # 6. 打印结果
    print("\n[Question]")
    print(query)

    print_retrieved_chunks(retrieved_chunks)

    print("\n[Final Prompt]")
    print(prompt)

if __name__ == "__main__":
    main()

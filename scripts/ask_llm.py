"""
scripts/ask_llm.py
==================

Mini-RAG + 本地 LLM 完整问答入口。

流程：
1. 接收用户 query
2. 加载 embedding 模型
3. 加载 FAISS retriever
4. 检索 top-k chunks
5. 构造 RAG prompt
6. 调用本地 LLM 生成 answer
7. 打印检索结果、prompt、answer

运行方式：
    python scripts/ask_llm.py

或者：
    python scripts/ask_llm.py "FAISS 是什么？"
"""

import sys
from pathlib import Path

from rag_template.reranker.reranker import TextReranker

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from rag_template.configs.BaseConfig import *
from rag_template.configs.LLMConfig import *
from rag_template.configs.RAGConfig import *

from rag_template.embed.embedder import TextEmbedder
from rag_template.retriever.retriever import FaissRetriever
from rag_template.prompt.prompt_builder import build_rag_prompt
from rag_template.llm.local_llm import LocalLLMGenerator


def print_retrieved_chunks(results):
    """
    打印检索结果。
    """
    print("\n[Retrieved Chunks]")

    for item in results:
        metadata = item.get("metadata", {})

        print("-" * 80)
        print(f"rank: {item.get('rank')}")
        print(f"score: {item.get('score')}")
        print(f"rerank_score: {item.get('rerank_score')}")
        print(f"chunk_id: {item.get('chunk_id')}")
        print(f"doc_id: {item.get('doc_id')}")
        print(f"source: {metadata.get('source')}")
        print(f"doc_type: {metadata.get('doc_type')}")
        print(f"title: {metadata.get('title')}")
        print(f"department: {metadata.get('department')}")
        print(f"project_id: {metadata.get('project_id')}")
        print(f"version: {metadata.get('version')}")
        print(f"status: {metadata.get('status')}")
        print(f"security_level: {metadata.get('security_level')}")
        print(f"page: {metadata.get('page')}")
        print(f"section: {metadata.get('section')}")
        print(f"text:\n{item.get('text')}")


def main():
    """
    RAG + 本地 LLM 主流程。
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

    # =========================
    # 6. 初始化本地 LLM
    # =========================

    llm = LocalLLMGenerator(
        model_name=LLM_MODEL_NAME,
        device=LLM_DEVICE,
    )

    # =========================
    # 7. 生成答案
    # =========================

    answer = llm.generate(
        prompt=prompt,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        do_sample=LLM_DO_SAMPLE,
    )

    # =========================
    # 8. 打印结果
    # =========================

    print("\n[Question]")
    print(query)

    print_retrieved_chunks(retrieved_chunks)

    print("\n[Final Prompt]")
    print(prompt)

    print("\n[Answer]")
    print(answer)


if __name__ == "__main__":
    main()

"""
scripts/test_rag_batch.py
=========================

批量测试 Mini-RAG + 本地大模型闭环。

功能：
1. 从列表读取测试问题
2. 使用 embedding + FAISS retriever 检索 top-k chunks
3. 构造 RAG prompt
4. 使用本地 LLM 生成答案
5. 打印并可保存结果到 JSON
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from rag_template.configs.RAGConfig import *
from rag_template.configs.LLMConfig import  *

from rag_template.embed.embedder import TextEmbedder
from rag_template.retriever.retriever import FaissRetriever
from rag_template.prompt.prompt_builder import build_rag_prompt
from rag_template.llm.local_llm import LocalLLMGenerator


# =========================
# 测试问题列表
# =========================

TEST_QUERIES = [
    "FAISS 是什么？",
    "什么是向量索引？",
    "RAG 是什么？",
    "Embedding 是什么？",
    "Mini-RAG 怎么工作的？",
]


# =========================
# 初始化组件
# =========================

embedder = TextEmbedder(
    model_name=EMBEDDING_MODEL_NAME,
    device=EMBEDDING_DEVICE,
    batch_size=EMBEDDING_BATCH_SIZE,
)

retriever = FaissRetriever(
    index_path=FAISS_INDEX_FILE,
    chunk_meta_path=CHUNK_META_FILE,
    embedder=embedder,
)

llm = LocalLLMGenerator(
    model_name=LLM_MODEL_NAME,
    device=LLM_DEVICE,
)


def run_rag(query: str) -> dict:
    """
    执行单条问题的 Mini-RAG 闭环，返回结果字典。
    """
    retrieved_chunks = retriever.search(query=query, top_k=TOP_K)
    prompt = build_rag_prompt(query=query, retrieved_chunks=retrieved_chunks)
    answer = llm.generate(
        prompt=prompt,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        do_sample=LLM_DO_SAMPLE,
    )

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "answer": answer,
    }


def main():
    results = []

    for q in TEST_QUERIES:
        print("=" * 80)
        print(f"[Query] {q}")
        result = run_rag(q)
        results.append(result)

        print("\n[Retrieved Chunks]")
        for c in result["retrieved_chunks"]:
            print("-" * 80)
            print(f"rank: {c['rank']}")
            print(f"score: {c['score']:.4f}")
            print(f"chunk_id: {c['chunk_id']}")
            print(f"source: {c['source']}")
            print(f"text:\n{c['text']}")

        print("\n[Final Prompt]")
        print(result["prompt"])
        print("\n[Answer]")
        print(result["answer"])

    # 保存 JSON
    save_path = ROOT_DIR / "ragtests" / "rag_batch_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n[Done] 所有测试完成，结果已保存到:", save_path)


if __name__ == "__main__":
    main()
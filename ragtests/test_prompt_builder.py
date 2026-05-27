"""
ragtests/test_prompt_builder.py
============================

测试 prompt_builder.py。

测试范围：
1. 使用 retriever 返回 top-k chunks
2. 将 retrieved_chunks 拼成 prompt
3. 打印最终 prompt

注意：
这里不调用 LLM。
"""

import sys
from pathlib import Path

from rag_template.embed.Hash_embedder import HashTextEmbedder

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))



from rag_template.configs.RAGConfig import *


from rag_template.retriever.retriever import FaissRetriever
from rag_template.prompt.prompt_builder import build_rag_prompt


def test_build_rag_prompt():
    """
    测试 query -> retrieved_chunks -> prompt。
    """

    embedder = HashTextEmbedder(embedding_dim=384)

    retriever = FaissRetriever(
        index_path=FAISS_INDEX_FILE,
        chunk_meta_path=CHUNK_META_FILE,
        embedder=embedder,
    )

    query = "FAISS 是什么？"

    retrieved_chunks = retriever.search(
        query=query,
        top_k=TOP_K,
    )

    prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    print("\n[Prompt]")
    print(prompt)

    assert isinstance(prompt, str)
    assert "FAISS 是什么？" in prompt
    assert "资料" in prompt
    assert "chunk_id" in prompt
    assert "doc_003_chunk_0000" in prompt


if __name__ == "__main__":
    test_build_rag_prompt()

    print("\n[Test Passed] prompt_builder 测试通过。")
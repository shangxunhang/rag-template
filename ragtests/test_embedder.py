"""
ragtests/test_embedder.py
======================

测试 embedding 模块。
"""

import sys
from pathlib import Path

from rag_template.configs.RAGConfig import *

from rag_template.embed.embedder import TextEmbedder

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))




def test_embedder():
    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    texts = [
        "RAG 是检索增强生成。",
        "FAISS 是向量检索库。",
        "Embedding 是文本向量化过程。",
    ]

    embeddings = embedder.encode_texts(texts)

    print("embeddings shape:", embeddings.shape)

    query_embedding = embedder.encode_query("FAISS 是什么？")

    print("query embedding shape:", query_embedding.shape)


if __name__ == "__main__":
    test_embedder()
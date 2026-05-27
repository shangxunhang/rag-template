from rag_template.configs.RAGConfig import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    MILVUS_LITE_DB_FILE,
    MILVUS_COLLECTION_NAME,
    MILVUS_DIM,
)

from rag_template.embed.embedder import TextEmbedder
from rag_template.retriever.milvus_retriever import MilvusRetriever


def main():
    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    retriever = MilvusRetriever(
        db_file=MILVUS_LITE_DB_FILE,
        collection_name=MILVUS_COLLECTION_NAME,
        dim=MILVUS_DIM,
        embedder=embedder,
    )

    results = retriever.retrieve(
        query="FAISS 是什么？",
        top_k=3,
    )

    for result in results:
        print("=" * 80)
        print("rank:", result["rank"])
        print("score:", result["score"])
        print("doc_id:", result["doc_id"])
        print("chunk_id:", result["chunk_id"])
        print("source:", result["source"])
        print("text:", result["text"][:200])


if __name__ == "__main__":
    main()
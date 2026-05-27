"""
ragtests/eval_full.py
=====================

运行基于 FAISS 的完整 RAG Pipeline 评估。

注意：
- 路径配置全部来自 BaseConfig。
- RAG 参数来自 RAGConfig。
- LLM 参数来自 LLMConfig。
"""

from __future__ import annotations

from rag_template.configs.BaseConfig import (
    FAISS_INDEX_FILE,
    CHUNK_META_FILE,
    PROCESSED_DATA_DIR,
)
from rag_template.configs.RAGConfig import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    USE_RERANKER,
    RETRIEVAL_TOP_K,
    FINAL_TOP_K,
    RERANKER_MODEL_NAME,
    RERANKER_DEVICE,
    RERANKER_BATCH_SIZE,
)
from rag_template.configs.LLMConfig import (
    LLM_MODEL_NAME,
    LLM_DEVICE,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_DO_SAMPLE,
)
from rag_template.embed.embedder import TextEmbedder
from rag_template.retriever.retriever import FaissRetriever
from rag_template.reranker.reranker import TextReranker
from rag_template.prompt.prompt_builder import build_rag_prompt
from rag_template.llm.local_llm import LocalLLMGenerator
from rag_template.eval.eval_runner import run_and_save_eval
from rag_template.schema.eval_schema import RetrievalResult


class RealRAGGenerator:
    def __init__(
        self,
        llm: LocalLLMGenerator,
        reranker: TextReranker | None = None,
        final_top_k: int = 3,
    ) -> None:
        self.llm = llm
        self.reranker = reranker
        self.final_top_k = final_top_k

    def generate(
        self,
        query: str,
        retrieved_results: list[RetrievalResult],
    ) -> dict:
        retrieved_chunks = self._retrieval_results_to_dicts(retrieved_results)

        if self.reranker is not None:
            retrieved_chunks = self.reranker.rerank(
                query=query,
                retrieved_chunks=retrieved_chunks,
                top_k=self.final_top_k,
            )
        else:
            retrieved_chunks = retrieved_chunks[: self.final_top_k]

        prompt = build_rag_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        answer = self.llm.generate(
            prompt=prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            do_sample=LLM_DO_SAMPLE,
        )

        return {
            "answer": answer,
            "cited_doc_ids": self._extract_cited_doc_ids(retrieved_chunks),
        }

    @staticmethod
    def _retrieval_results_to_dicts(
        retrieved_results: list[RetrievalResult],
    ) -> list[dict]:
        return [
            {
                "rank": result.rank,
                "score": result.score,
                "doc_id": result.doc_id,
                "chunk_id": result.chunk_id,
                "source": result.source,
                "text": result.text,
                "metadata": result.metadata or {},
            }
            for result in retrieved_results
        ]

    @staticmethod
    def _extract_cited_doc_ids(retrieved_chunks: list[dict]) -> list[str]:
        cited_doc_ids: list[str] = []

        for chunk in retrieved_chunks:
            doc_id = chunk.get("doc_id")
            if doc_id and doc_id not in cited_doc_ids:
                cited_doc_ids.append(doc_id)

        return cited_doc_ids


def main() -> None:
    print("=" * 80)
    print("[eval_full][FAISS] 初始化 TextEmbedder")
    print(f"[Config] EMBEDDING_MODEL_NAME = {EMBEDDING_MODEL_NAME}")
    print(f"[Config] EMBEDDING_DEVICE     = {EMBEDDING_DEVICE}")
    print("=" * 80)

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print("=" * 80)
    print("[eval_full][FAISS] 初始化 FaissRetriever")
    print(f"[Path] FAISS_INDEX_FILE = {FAISS_INDEX_FILE}")
    print(f"[Path] CHUNK_META_FILE  = {CHUNK_META_FILE}")
    print("=" * 80)

    retriever = FaissRetriever(
        index_path=FAISS_INDEX_FILE,
        chunk_meta_path=CHUNK_META_FILE,
        embedder=embedder,
    )

    reranker = None

    if USE_RERANKER:
        print("=" * 80)
        print("[eval_full][FAISS] 初始化 TextReranker")
        print(f"[Config] RERANKER_MODEL_NAME = {RERANKER_MODEL_NAME}")
        print(f"[Config] RERANKER_DEVICE     = {RERANKER_DEVICE}")
        print("=" * 80)

        reranker = TextReranker(
            model_name=RERANKER_MODEL_NAME,
            device=RERANKER_DEVICE,
            batch_size=RERANKER_BATCH_SIZE,
        )

    print("=" * 80)
    print("[eval_full][FAISS] 初始化 LocalLLMGenerator")
    print(f"[Config] LLM_MODEL_NAME = {LLM_MODEL_NAME}")
    print(f"[Config] LLM_DEVICE     = {LLM_DEVICE}")
    print("=" * 80)

    llm = LocalLLMGenerator(
        model_name=LLM_MODEL_NAME,
        device=LLM_DEVICE,
    )

    generator = RealRAGGenerator(
        llm=llm,
        reranker=reranker,
        final_top_k=FINAL_TOP_K,
    )

    eval_dataset_path = PROCESSED_DATA_DIR / "eval_dataset.json"
    eval_report_path = PROCESSED_DATA_DIR / "eval_full_report.json"

    eval_top_k = RETRIEVAL_TOP_K if USE_RERANKER else FINAL_TOP_K

    print("=" * 80)
    print("[eval_full][FAISS] 开始运行完整 RAG 评估")
    print(f"[Path] eval_dataset_path = {eval_dataset_path}")
    print(f"[Path] eval_report_path  = {eval_report_path}")
    print(f"[Config] eval_top_k      = {eval_top_k}")
    print("=" * 80)

    report = run_and_save_eval(
        retriever=retriever,
        generator=generator,
        eval_dataset_path=eval_dataset_path,
        output_path=eval_report_path,
        top_k=eval_top_k,
    )

    print("=" * 80)
    print("[eval_full][FAISS] 完整评估完成")
    print(f"[Report] num_samples = {report.num_samples}")
    print(f"[Report] top_k = {report.top_k}")
    print(f"[Report] avg_hit_at_k = {report.avg_hit_at_k:.4f}")
    print(f"[Report] avg_recall_at_k = {report.avg_recall_at_k:.4f}")
    print(f"[Report] avg_mrr = {report.avg_mrr:.4f}")
    print(f"[Report] avg_context_keyword_hit = {report.avg_context_keyword_hit:.4f}")

    if report.avg_answer_keyword_hit is not None:
        print(f"[Report] avg_answer_keyword_hit = {report.avg_answer_keyword_hit:.4f}")

    if report.avg_answer_context_overlap is not None:
        print(f"[Report] avg_answer_context_overlap = {report.avg_answer_context_overlap:.4f}")

    if report.avg_citation_hit is not None:
        print(f"[Report] avg_citation_hit = {report.avg_citation_hit:.4f}")

    print(f"[Report] saved to: {eval_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

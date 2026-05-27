"""
scripts/run_eval.py
===================

本脚本用于运行 RAG 离线评估。

它负责：
1. 读取 RAGConfig 中的路径和模型配置
2. 初始化 TextEmbedder
3. 初始化 FaissRetriever
4. 调用 eval_runner.run_and_save_eval()
5. 将评估结果保存为 eval_report.json

注意：
eval_runner.py 是通用评估调度模块；
run_eval.py 是当前项目的本地运行入口。
"""

from rag_template.configs.RAGConfig import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE,
    CHUNK_META_FILE,
    PROCESSED_DATA_DIR,
    TOP_K,
)

from rag_template.embed.embedder import TextEmbedder
from rag_template.retriever.retriever import FaissRetriever
from rag_template.eval.eval_runner import run_and_save_eval


def main() -> None:
    """
    运行 RAG 检索评估。

    第一版只传入 retriever，不传 generator。
    因此当前只会计算：
    1. hit@k
    2. recall@k
    3. MRR
    4. context_keyword_hit

    后续接入 generator 后，可以继续计算：
    1. answer_keyword_hit
    2. answer_context_overlap
    3. citation_hit
    """

    print("=" * 80)
    print("[run_eval] 初始化 TextEmbedder")
    print(f"[run_eval] EMBEDDING_MODEL_NAME = {EMBEDDING_MODEL_NAME}")
    print(f"[run_eval] EMBEDDING_DEVICE = {EMBEDDING_DEVICE}")
    print("=" * 80)

    embedder = TextEmbedder(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print("=" * 80)
    print("[run_eval] 初始化 FaissRetriever")
    print(f"[run_eval] FAISS_INDEX_FILE = {FAISS_INDEX_FILE}")
    print(f"[run_eval] CHUNK_META_FILE = {CHUNK_META_FILE}")
    print("=" * 80)

    retriever = FaissRetriever(
        index_path=FAISS_INDEX_FILE,
        chunk_meta_path=CHUNK_META_FILE,
        embedder=embedder,
    )

    eval_dataset_path = PROCESSED_DATA_DIR / "eval_dataset.json"
    eval_report_path = PROCESSED_DATA_DIR / "eval_report.json"

    print("=" * 80)
    print("[run_eval] 开始运行评估")
    print(f"[run_eval] eval_dataset_path = {eval_dataset_path}")
    print(f"[run_eval] eval_report_path = {eval_report_path}")
    print(f"[run_eval] top_k = {TOP_K}")
    print("=" * 80)

    report = run_and_save_eval(
        retriever=retriever,
        eval_dataset_path=eval_dataset_path,
        output_path=eval_report_path,
        top_k=TOP_K,
        generator=None,
    )

    print("=" * 80)
    print("[run_eval] 评估完成")
    print(f"[run_eval] num_samples = {report.num_samples}")
    print(f"[run_eval] top_k = {report.top_k}")
    print(f"[run_eval] avg_hit_at_k = {report.avg_hit_at_k:.4f}")
    print(f"[run_eval] avg_recall_at_k = {report.avg_recall_at_k:.4f}")
    print(f"[run_eval] avg_mrr = {report.avg_mrr:.4f}")
    print(
        f"[run_eval] avg_context_keyword_hit = "
        f"{report.avg_context_keyword_hit:.4f}"
    )
    print(f"[run_eval] report saved to: {eval_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
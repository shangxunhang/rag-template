"""
configs/LLMConfig.py
====================

RAG 项目内部 LLM 配置。

注意：
- Agent-RAG 场景通常不建议让 RAGEngine.answer() 再调用内部 LLM。
- Agent 项目更推荐调用 RAGEngine.retrieve_context()，然后由 Agent 的 Qwen Finalizer 生成最终回答。
- 这个配置主要保留给纯 RAG demo / CLI 使用。
"""

from rag_template.configs.BaseConfig import MODEL_ROOT_DIR


# =========================
# 本地 LLM 模型
# =========================

LLM_MODEL_DIR = (
    MODEL_ROOT_DIR
    / "llm"
    / "Qwen2.5-1.5B-Instruct"
)

LLM_MODEL_NAME = str(LLM_MODEL_DIR)

LLM_DEVICE = "cuda"
LLM_MAX_NEW_TOKENS = 256
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_DO_SAMPLE = False


if __name__ == "__main__":
    print(f"LLM_MODEL_NAME = {LLM_MODEL_NAME}")
    print(f"LLM_DEVICE     = {LLM_DEVICE}")

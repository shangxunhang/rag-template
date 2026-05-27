from rag_template.configs.BaseConfig import (
    MODEL_ROOT_DIR,
    VECTOR_STORE_DIR,
    FAISS_INDEX_FILE,
    CHUNK_META_FILE,
    MILVUS_LITE_DB_FILE,
)

# =========================
# 1. Chunk 参数
# =========================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHUNK_STRATEGY = "fixed"
CHUNK_UNIT = "token"
CHUNK_TOKENIZER_LOCAL_FILES_ONLY = True

# =========================
# 2. Embedding 模型
# =========================

EMBEDDING_MODEL_DIR = (
    MODEL_ROOT_DIR
    / "embedding"
    / "paraphrase-multilingual-MiniLM-L12-v2"
)

BGE_M3_MODEL_DIR = MODEL_ROOT_DIR / "embedding" / "bge-m3"
M3E_BASE_MODEL_DIR = MODEL_ROOT_DIR / "embedding" / "m3e-base"

EMBEDDING_MODEL_NAME = str(M3E_BASE_MODEL_DIR)

CHUNK_TOKENIZER_MODEL_NAME = EMBEDDING_MODEL_NAME

EMBEDDING_DEVICE = "cuda"
EMBEDDING_BATCH_SIZE = 32

# =========================
# 3. 检索参数
# =========================

TOP_K = 3

# =========================
# 4. Prompt 模板
# =========================

PROMPT_TEMPLATE = "strict_qa"

# =========================
# 5. Reranker 配置
# =========================

USE_RERANKER = True

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 3

RERANKER_MODEL_DIR = (
    MODEL_ROOT_DIR
    / "reranker"
    / "bge-reranker-v2-m3"
)

RERANKER_MODEL_NAME = str(RERANKER_MODEL_DIR)
RERANKER_DEVICE = "cuda"
RERANKER_BATCH_SIZE = 16

# =========================
# 6. Milvus Lite 配置
# =========================

MILVUS_COLLECTION_NAME = "rag_chunks"

MILVUS_DIM = 768

MILVUS_VECTOR_FIELD = "vector"
MILVUS_ID_FIELD = "id"

MILVUS_TOP_K = 10
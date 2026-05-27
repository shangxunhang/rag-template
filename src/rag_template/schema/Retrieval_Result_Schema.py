

import copy
from datetime import date

from rag_template.configs.SchemaConfig import *


def build_retrieval_result(chunk: Dict[str, Any], rank: int, score: float, rerank_score: float = None) -> Dict[
    str, Any]:
    result = copy.deepcopy(RETRIEVAL_RESULT_TEMPLATE)
    result["rank"] = rank
    result["score"] = score
    result["rerank_score"] = rerank_score
    result["chunk_id"] = chunk["chunk_id"]
    result["doc_id"] = chunk["doc_id"]
    result["text"] = chunk["text"]
    result["metadata"] = chunk["metadata"]
    return result

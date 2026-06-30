# src/rag_template/chunker/fixed_chunker.py
from typing import Dict, List

from rag_template.chunker.base_chunker import BaseChunker
from rag_template.schema.Chunk_Schema import build_chunk
from rag_template.util.text_utils import split_text_by_fixed_size


class FixedSizeChunker(BaseChunker):
    """
    固定长度切分。
    主要用于测试、兜底和没有结构的短文本。
    """

    def chunk_document(self, document: Dict) -> List[Dict]:
        doc_id = document["doc_id"]
        text = document.get("text", "")
        doc_metadata = document.get("metadata", {})

        chunk_texts = split_text_by_fixed_size(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks: List[Dict] = []
        cursor = 0
        for idx, chunk_text in enumerate(chunk_texts):
            start_char = text.find(chunk_text, cursor)
            if start_char < 0:
                start_char = None
                end_char = None
            else:
                end_char = start_char + len(chunk_text)
                cursor = max(start_char + 1, end_char - self.chunk_overlap)

            chunks.append(
                build_chunk(
                    doc_id=doc_id,
                    text=chunk_text,
                    idx=idx,
                    doc_metadata=doc_metadata,
                    chunk_type="fixed",
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return chunks

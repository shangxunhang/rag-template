"""
src/chunker.py
==============

文本切分模块。

职责：
1. 把单篇 document 切成多个 chunk
2. 给每个 chunk 添加 doc_id、chunk_id、source、text 等信息

第一版使用固定长度切分。
后续可以扩展：
- 按段落切分
- 按标题切分
- 递归字符切分
"""

# chunker/fixed_size_chunker.py

from typing import List, Dict

from rag_template.chunker.base_chunker import BaseChunker
from rag_template.schema.Chunk_Schema import build_chunk

from rag_template.util.text_utils import split_text_by_fixed_size


class FixedSizeChunker(BaseChunker):
    """
    固定长度切分。
    适合短文本、测试数据、玩具 Demo。
    """

    def chunk_document(self, document: Dict) -> List[Dict]:
        text = document["text"]

        chunk_texts = split_text_by_fixed_size(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = []

        for idx, chunk_text in enumerate(chunk_texts):
            chunk = build_chunk(
                doc=document,
                chunk_text=chunk_text,
                chunk_index=idx,
                chunk_type="fixed_size",
            )
            chunks.append(chunk)

        return chunks
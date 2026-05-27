from typing import List


def split_text_by_fixed_size(text: str, chunk_size: int, chunk_overlap: int, ) -> List[str]:
    """
    按固定长度切分文本。

    Args:
        text: 输入文本
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 的重叠字符数

    Returns:
        chunk 文本列表
    """
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks

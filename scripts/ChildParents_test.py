from rag_template.chunker.ChildParentChunker import ChildParentChunker

records = [
    {
        "unit_id": "doc1_unit_0001",
        "doc_id": "doc1",
        "source_type": "offline",
        "title": "测试文档",
        "section": "系统架构",
        "page_start": 1,
        "page_end": 1,
        "unit_type": "paragraph",
        "unit_order": 1,
        "text": "这是第一段，介绍 Hadoop、Kafka、Milvus 和 RAG 父子块策略。" * 30,
        "language": "zh",
        "quality_score": 0.9,
        "quality_flags": [],
        "cleaning_version": "cleaning_v1.0",
    },
    {
        "unit_id": "doc1_unit_0002",
        "doc_id": "doc1",
        "source_type": "offline",
        "title": "测试文档",
        "section": "系统架构",
        "page_start": 2,
        "page_end": 2,
        "unit_type": "paragraph",
        "unit_order": 2,
        "text": "这是第二段，介绍 child chunk 检索和 parent chunk 回填。" * 30,
        "language": "zh",
        "quality_score": 0.9,
        "quality_flags": [],
        "cleaning_version": "cleaning_v1.0",
    },
]

chunker = ChildParentChunker(
    parent_chunk_size=1500,
    parent_chunk_overlap=150,
    child_chunk_size=500,
    child_chunk_overlap=50,
    unit="char",  # 第一轮建议先用 char 验证结构，稳定后再换 token
)

result = chunker.chunk_records_for_doc(records)

print(len(result.parents))
print(len(result.children))

print(result.parents[0]["parent_chunk_id"])
print(result.parents[0]["child_chunk_ids"])
print(result.children[0]["child_chunk_id"])
print(result.children[0]["parent_chunk_id"])

assert len(result.parents) > 0
assert len(result.children) > 0

parent_map = {
    p["parent_chunk_id"]: p
    for p in result.parents
}

child_map = {
    c["child_chunk_id"]: c
    for c in result.children
}

for child in result.children:
    assert child["chunk_id"] == child["child_chunk_id"]
    assert child["parent_chunk_id"] in parent_map
    assert child["doc_id"] == parent_map[child["parent_chunk_id"]]["doc_id"]
    assert child["text"].strip()

for parent in result.parents:
    assert parent["text"].strip()
    assert parent["child_count"] == len(parent["child_chunk_ids"])

    for child_id in parent["child_chunk_ids"]:
        assert child_id in child_map
        assert child_map[child_id]["parent_chunk_id"] == parent["parent_chunk_id"]

print("parent-child chunker test passed")
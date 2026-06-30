# -*- coding: utf-8 -*-
"""
PySpark Chunk Job: cleaned_text_unit_v1 -> chunk_unit_v1

输入：HDFS 上的 cleaned_text_unit_v1 JSONL
输出：HDFS 上的 chunk_unit_v1 JSONL

典型运行：

spark-submit \
  --master yarn \
  --conf spark.pyspark.python=/opt/module/miniconda3/envs/agent_rag_py/bin/python \
  --conf spark.pyspark.driver.python=/opt/module/miniconda3/envs/agent_rag_py/bin/python \
  --conf spark.executorEnv.PYSPARK_PYTHON=/opt/module/miniconda3/envs/agent_rag_py/bin/python \
  /opt/agent_rag/jobs/chunk_cleaned_text_units.py \
  --input hdfs://node1-biz:8020/agent_rag/cleaned/text_unit/ \
  --output hdfs://node1-biz:8020/agent_rag/chunk/chunk_unit/ \
  --batch-id pdf_clean_batch_20260627 \
  --chunk-size 900 \
  --chunk-overlap 120 \
  --min-chunk-size 80 \
  --overwrite
"""

from __future__ import print_function

import argparse
import json
import re
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim

import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

SCHEMA_VERSION = "chunk_unit_v1"
CHUNK_VERSION = "chunk_v1.0"
CHUNK_STRATEGY = "rule_based_by_doc_unit_order_v1"


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def row_to_unit(row):
    d = row.asDict(recursive=True)
    text = normalize_text(d.get("text"))
    unit_id = d.get("unit_id")

    extra = d.get("extra") or {}
    source_parse_unit_id = extra.get("source_parse_unit_id")

    return {
        "unit_id": unit_id or source_parse_unit_id,
        "doc_id": d.get("doc_id"),
        "source_type": d.get("source_type"),
        "source_uri": d.get("source_uri"),
        "source_name": d.get("source_name"),
        "source_format": d.get("source_format"),
        "batch_id": d.get("batch_id"),
        "title": d.get("title"),
        "section": d.get("section"),
        "page_start": safe_int(d.get("page_start")),
        "page_end": safe_int(d.get("page_end")),
        "unit_type": d.get("unit_type") or "other",
        "unit_order": safe_int(d.get("unit_order"), 0),
        "text": text,
        "text_length": len(text),
        "language": d.get("language") or "unknown",
        "quality_score": safe_float(d.get("quality_score")),
        "quality_flags": as_list(d.get("quality_flags")),
        "cleaning_version": d.get("cleaning_version"),
        "extra": extra,
    }


def should_skip_unit(unit):
    text = unit.get("text") or ""
    unit_type = unit.get("unit_type") or "other"
    quality_flags = set(unit.get("quality_flags") or [])

    if not text.strip():
        return True

    # 图片块不参与文本 chunk。
    if unit_type == "image":
        return True

    # 第一版不直接删除所有 short_text，只过滤非常短且类型为 other 的噪声。
    if unit_type == "other" and "short_text" in quality_flags and len(text) < 10:
        return True

    return False


def split_long_text(text, chunk_size, overlap):
    """对超长单元做兜底切分。第一版使用字符窗口，避免外部 tokenizer 依赖。"""
    text = normalize_text(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start += step
    return pieces


def expand_long_units(units, chunk_size, overlap):
    expanded = []
    for unit in units:
        text = unit.get("text") or ""
        if len(text) <= chunk_size:
            expanded.append(unit)
            continue

        pieces = split_long_text(text, chunk_size, overlap)
        for idx, piece in enumerate(pieces, start=1):
            u = dict(unit)
            u["text"] = piece
            u["text_length"] = len(piece)
            u["unit_id"] = "%s_part_%04d" % (unit.get("unit_id"), idx)
            flags = list(u.get("quality_flags") or [])
            if "split_from_long_unit" not in flags:
                flags.append("split_from_long_unit")
            u["quality_flags"] = flags
            expanded.append(u)
    return expanded


def unique_keep_order(values):
    seen = set()
    out = []
    for v in values:
        if v is None:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def pick_language(units):
    counts = {}
    for u in units:
        lang = u.get("language") or "unknown"
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]


def compute_chunk_type(units):
    unit_types = set([u.get("unit_type") or "other" for u in units])
    if unit_types == {"table"}:
        return "table"
    if "table" in unit_types:
        return "mixed"
    return "text"


def avg_quality_score(units):
    scores = [u.get("quality_score") for u in units if u.get("quality_score") is not None]
    if not scores:
        return None
    return round(sum(scores) / float(len(scores)), 4)


def collect_quality_flags(units, text_length, min_chunk_size):
    flags = []
    for u in units:
        flags.extend(u.get("quality_flags") or [])
    flags = unique_keep_order(flags)
    if text_length < min_chunk_size and "short_chunk" not in flags:
        flags.append("short_chunk")
    return flags


def overlap_tail(units, overlap):
    if overlap <= 0:
        return []
    tail = []
    total = 0
    for u in reversed(units):
        tail.insert(0, u)
        total += len(u.get("text") or "")
        if total >= overlap:
            break
    return tail


def build_chunk(doc_id, chunk_index, units, min_chunk_size, batch_id_arg):
    texts = [u.get("text") or "" for u in units if u.get("text")]
    text = "\n".join(texts).strip()
    text_length = len(text)

    page_starts = [u.get("page_start") for u in units if u.get("page_start") is not None]
    page_ends = [u.get("page_end") for u in units if u.get("page_end") is not None]
    source_unit_ids = unique_keep_order([u.get("unit_id") for u in units])

    first = units[0] if units else {}
    titles = [u.get("title") for u in units if u.get("title")]
    sections = [u.get("section") for u in units if u.get("section")]

    return {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": "%s_chunk_%06d" % (doc_id, chunk_index),
        "doc_id": doc_id,
        "source_type": first.get("source_type"),
        "source_uri": first.get("source_uri"),
        "source_name": first.get("source_name"),
        "source_format": first.get("source_format"),
        "batch_id": first.get("batch_id") or batch_id_arg,
        "chunk_index": chunk_index,
        "chunk_type": compute_chunk_type(units),
        "title": titles[-1] if titles else None,
        "section": sections[-1] if sections else None,
        "page_start": min(page_starts) if page_starts else None,
        "page_end": max(page_ends) if page_ends else None,
        "text": text,
        "text_length": text_length,
        "language": pick_language(units),
        "source_unit_ids": source_unit_ids,
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_version": CHUNK_VERSION,
        "quality_score": avg_quality_score(units),
        "quality_flags": collect_quality_flags(units, text_length, min_chunk_size),
        "created_at": now_iso(),
        "extra": {
            "source_unit_count": len(source_unit_ids),
            "source_unit_types": unique_keep_order([u.get("unit_type") for u in units]),
            "cleaning_versions": unique_keep_order([u.get("cleaning_version") for u in units]),
        },
    }


def chunk_units_for_doc(doc_id, units, chunk_size, overlap, min_chunk_size, batch_id_arg):
    units = [u for u in units if not should_skip_unit(u)]
    units = expand_long_units(units, chunk_size, overlap)

    chunks = []
    current = []
    current_len = 0
    chunk_index = 1

    for unit in units:
        unit_len = len(unit.get("text") or "")
        sep_len = 1 if current else 0

        if current and current_len + sep_len + unit_len > chunk_size:
            chunk = build_chunk(doc_id, chunk_index, current, min_chunk_size, batch_id_arg)
            if chunk["text_length"] > 0:
                chunks.append(chunk)
                chunk_index += 1

            current = overlap_tail(current, overlap)
            current_len = sum([len(u.get("text") or "") for u in current]) + max(len(current) - 1, 0)

        current.append(unit)
        current_len += sep_len + unit_len

    if current:
        chunk = build_chunk(doc_id, chunk_index, current, min_chunk_size, batch_id_arg)
        if chunk["text_length"] > 0:
            chunks.append(chunk)

    return chunks


def chunk_partition(rows_iter, chunk_size, overlap, min_chunk_size, batch_id_arg):
    current_doc_id = None
    current_units = []

    for row in rows_iter:
        unit = row_to_unit(row)
        doc_id = unit.get("doc_id")
        if not doc_id:
            continue

        if current_doc_id is None:
            current_doc_id = doc_id

        if doc_id != current_doc_id:
            for chunk in chunk_units_for_doc(current_doc_id, current_units, chunk_size, overlap, min_chunk_size, batch_id_arg):
                yield json.dumps(chunk, ensure_ascii=False)
            current_doc_id = doc_id
            current_units = []

        current_units.append(unit)

    if current_doc_id is not None and current_units:
        for chunk in chunk_units_for_doc(current_doc_id, current_units, chunk_size, overlap, min_chunk_size, batch_id_arg):
            yield json.dumps(chunk, ensure_ascii=False)


def delete_output_if_exists(spark, output_path: str) -> None:
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()

    path = jvm.org.apache.hadoop.fs.Path(output_path)
    fs = path.getFileSystem(conf)

    if fs.exists(path):
        fs.delete(path, True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build chunk_unit_v1 from cleaned_text_unit_v1 on HDFS")
    parser.add_argument("--input", required=True, help="Input HDFS path of cleaned_text_unit_v1")
    parser.add_argument("--output", required=True, help="Output HDFS path for chunk_unit_v1")
    parser.add_argument("--batch-id", required=True, help="Batch id")
    parser.add_argument("--chunk-size", type=int, default=900, help="Target chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Overlap in characters, implemented by tail units")
    parser.add_argument("--min-chunk-size", type=int, default=80, help="Minimum chunk size flag threshold")
    parser.add_argument("--output-partitions", type=int, default=1, help="Output part file count")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output path if exists")
    return parser.parse_args()


def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("chunk_cleaned_text_units")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.blockManager.port", "0")
        .config("spark.driver.port", "0")
        .config("spark.python.worker.reuse", "false")
        .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
        .getOrCreate()
    )

    if args.overwrite:
        delete_output_if_exists(spark, args.output)

    df = spark.read.json(args.input)

    cleaned = (
        df
        .where(col("schema_version") == "cleaned_text_unit_v1")
        .where(col("doc_id").isNotNull())
        .where(col("text").isNotNull())
        .where(length(trim(col("text"))) > 0)
    )

    # 按 doc_id 重分区并在分区内排序，保证同一文档内按 unit_order 顺序切分。
    ordered = cleaned.repartition(col("doc_id")).sortWithinPartitions(col("doc_id"), col("unit_order"))

    chunk_rdd = ordered.rdd.mapPartitions(
        lambda it: chunk_partition(
            it,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            min_chunk_size=args.min_chunk_size,
            batch_id_arg=args.batch_id,
        )
    )

    if args.output_partitions and args.output_partitions > 0:
        chunk_rdd = chunk_rdd.coalesce(args.output_partitions)

    chunk_rdd.saveAsTextFile(args.output)

    print("Chunk job finished")
    print("inputPath      = %s" % args.input)
    print("outputPath     = %s" % args.output)
    print("batchId        = %s" % args.batch_id)
    print("chunkSize      = %s" % args.chunk_size)
    print("chunkOverlap   = %s" % args.chunk_overlap)
    print("minChunkSize   = %s" % args.min_chunk_size)

    spark.stop()


if __name__ == "__main__":
    main()

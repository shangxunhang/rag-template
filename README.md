# RAG QA v1.0 说明文档

## 1. 项目阶段定位

当前项目已完成 **Parent-Child RAG QA v1.0** 主链路。

本阶段目标不是做完整企业级 RAG 平台，也不是做微服务，而是先完成一个可运行、可评估、可沉淀数据的单机版 RAG QA 核心链路，为后续封装 RAGTool、迁移企业框架、接入 Agent 和构造后训练数据做准备。

当前版本可以定义为：

```text
RAG QA v1.0 completed
```

---

## 2. 当前已完成能力

当前 RAG 主链路包括：

```text
cleaned_text_unit_all.jsonl
-> parent/child chunk
-> child embedding
-> Milvus Lite vector index
-> vector_index_record_v2
-> dense retrieval
-> BM25 keyword retrieval
-> RRF fusion
-> parent backfill
-> bge-reranker-v2-m3 rerank
-> context packing
-> strict QA PromptBuilder
-> local LLM answer generation
-> DataCapture
-> rag_runs.jsonl
```

当前已经具备以下能力：

```text
1. cleaned_text_unit 输入
2. Parent-Child Chunk 切分
3. child chunk 向量化
4. Milvus Lite 入库
5. Dense 向量检索
6. BM25 关键词检索
7. RRF 多路召回融合
8. parent chunk 回填
9. Reranker 精排
10. Context Packing
11. PromptBuilder 构造严格问答 Prompt
12. 本地 Qwen LLM 生成答案
13. DataCapture 捕获完整运行记录
14. retrieval lightweight eval
15. RAGAS-style proxy eval
```

---

## 3. 当前使用模型

当前本地模型配置如下：

```text
Embedding:
D:\models\huggingface\embedding\m3e-base

Reranker:
D:\models\huggingface\reranker\bge-reranker-v2-m3

Local LLM:
D:\models\huggingface\llm\Qwen2.5-1.5B-Instruct
```

如果后续要切换到 3B 模型，可将 LLM 路径改为：

```text
D:\models\huggingface\llm\Qwen2.5-3B-Instruct
```

---

## 4. 核心输入文件

主链路输入文件：

```text
D:\MyCode\rag-template\data\raw\jsonl\cleaned_text_unit_all.jsonl
```

该文件来自前置数据清洗流程，是 RAG 知识构建链路的标准输入。

---

## 5. 核心输出文件

Parent-Child Chunk 输出：

```text
D:\MyCode\rag-template\data\processed\parent_child_chunks\parent_chunks.jsonl
D:\MyCode\rag-template\data\processed\parent_child_chunks\child_chunks.jsonl
```

向量索引记录：

```text
D:\MyCode\rag-template\data\processed\vector_index_record\vector_index_record_v2.jsonl
```

Milvus Lite 数据库：

```text
D:\MyCode\rag-template\data\processed\vector_store\milvus_parent_child.db
```

RAG 运行捕获数据：

```text
D:\MyCode\rag-template\data\processed\runs\rag_runs.jsonl
```

RAGAS-style proxy 评估报告：

```text
D:\MyCode\rag-template\data\processed\eval_reports\ragas_style_eval_report.json
D:\MyCode\rag-template\data\processed\eval_reports\ragas_style_eval_details.jsonl
```

---

## 6. 当前主链路脚本

完整 RAG QA 主链路脚本：

```text
D:\MyCode\rag-template\Scripts\test_full_parent_child_rag_qa_pipeline.py
```

该脚本从 cleaned text unit 开始，完整执行：

```text
切分
-> embedding
-> 入 Milvus
-> 检索
-> rerank
-> context packing
-> prompt
-> local LLM answer
-> DataCapture
```

---

## 7. 完整运行命令

在 Windows 环境下执行：

```bat
D:\mysoftware\anaconda\envs\rag\python.exe D:\MyCode\rag-template\Scripts\test_full_parent_child_rag_qa_pipeline.py --input D:\MyCode\rag-template\data\raw\jsonl\cleaned_text_unit_all.jsonl --parent-output D:\MyCode\rag-template\data\processed\parent_child_chunks\parent_chunks.jsonl --child-output D:\MyCode\rag-template\data\processed\parent_child_chunks\child_chunks.jsonl --index-record-output D:\MyCode\rag-template\data\processed\vector_index_record\vector_index_record_v2.jsonl --db-file D:\MyCode\rag-template\data\processed\vector_store\milvus_parent_child.db --capture-output D:\MyCode\rag-template\data\processed\runs\rag_runs.jsonl --collection-name rag_child_chunks --metric-type COSINE --embedding-model D:\models\huggingface\embedding\m3e-base --embedding-device cuda --embedding-batch-size 32 --dense-top-k 10 --keyword-top-k 10 --candidate-top-k 10 --rrf-k 60 --reranker-model D:\models\huggingface\reranker\bge-reranker-v2-m3 --reranker-device cuda --reranker-batch-size 16 --rerank-top-k 5 --max-context-chars 6000 --max-context-items 3 --llm-model D:\models\huggingface\llm\Qwen2.5-1.5B-Instruct --llm-device cuda --max-new-tokens 256 --temperature 0.7 --top-p 0.9 --no-do-sample --query 整体性学习是什么 --expected-doc-ids doc_001_native_text --expected-keywords 整体性学习,学习 --eval-top-k 5 --clean-output
```

通过标准：

```text
FULL parent-child RAG QA pipeline test passed
退出代码 0
```

---

## 8. RAGAS-style Proxy Eval

当前项目已新增 RAGAS-style proxy eval。

评估脚本：

```text
D:\MyCode\rag-template\Scripts\test_ragas_style_eval_from_runs.py
```

评估样本文件：

```text
D:\MyCode\rag-template\data\eval_set\rag_eval_cases.jsonl
```

运行命令：

```bat
D:\mysoftware\anaconda\envs\rag\python.exe D:\MyCode\rag-template\Scripts\test_ragas_style_eval_from_runs.py
```

当前评估指标包括：

```text
context_precision_proxy
context_recall_proxy
faithfulness_proxy
answer_relevancy_proxy
citation_hit_proxy
```

需要注意：

```text
当前实现的是 RAGAS-style proxy eval，不是官方 ragas package，也不是完整 LLM-as-a-judge 评估。
```

当前 proxy eval 的定位是：

```text
1. 本地低成本回归测试
2. 检查 RAG 运行记录是否完整
3. 检查检索、上下文、答案、引用是否基本匹配
4. 为后续 DeepSeek Judge / 官方 RAGAS / ARES 接入预留评估数据结构
```

后续正式评估需要继续扩展：

```text
1. 扩充 eval_cases.jsonl 到 20~50 条
2. 增加 expected_parent_chunk_ids
3. 增加困难样本、无答案样本、干扰样本、跨段样本
4. 接入 DeepSeek Judge
5. 可选接入官方 ragas package
```

---

## 9. 当前 DataCapture 内容

当前 `rag_runs.jsonl` 会保存完整 RAG 运行记录，包括：

```text
schema_version
run_id
created_at
finished_at
query
answer
model_name
model_provider
generation_params
retrieval_results
context_pack
packed_context
citations
prompt
prompt_id
prompt_version
eval_result
metadata
```

该文件是后续 DatasetBuilder 的第一类原始数据资产。

后续可从该文件构造：

```text
1. RAG eval 数据
2. SFT 问答数据
3. RAG 轨迹数据
4. 偏好数据候选
5. Agent 调用样本
```

---

## 10. 当前评估状态

当前已经完成：

```text
1. Full pipeline regression test
2. Lightweight retrieval eval
3. RAGAS-style proxy eval framework
```

当前尚未完成：

```text
1. 官方 RAGAS 评估
2. DeepSeek Judge 评估
3. ARES 风格 judge 评估
4. 大规模批量 eval set
5. 人工评分闭环
6. 业务可用性评估
```

因此当前版本的准确描述是：

```text
本项目已完成 Parent-Child RAG QA 主链路和本地 RAGAS-style proxy 评估框架。
当前评估适合工程回归测试，不代表完整业务质量验收。
```

---

## 11. 当前不做什么

当前阶段不做以下内容：

```text
1. 不做完整企业级 RAG 平台
2. 不做 FastAPI 服务化
3. 不做前端
4. 不做微服务
5. 不做 Kubernetes
6. 不做 Redis / Celery
7. 不做完整权限系统
8. 不做官方 RAGAS 深度集成
9. 不做 ARES judge 训练
10. 不做完整业务人工评估平台
```

这些能力后续在企业框架和 Agent 阶段逐步补充。

---

## 12. 旧代码说明

当前主链路以以下脚本为准：

```text
Scripts/test_full_parent_child_rag_qa_pipeline.py
```

早期 flat RAG 实现、旧 retriever、旧 eval runner 等文件暂时保留，但不作为当前主入口。

当前核心主链路是：

```text
Parent-Child RAG QA Pipeline
```

不是早期 flat chunk RAG。

---

## 13. PyCharm 运行配置

推荐 PyCharm 配置：

```text
Script path:
D:\MyCode\rag-template\Scripts\test_full_parent_child_rag_qa_pipeline.py

Working directory:
D:\MyCode\rag-template

Environment variables:
PYTHONPATH=D:\MyCode\rag-template\src
```

注意：

```text
Working directory 必须是项目根目录 D:\MyCode\rag-template
```

不要设置成：

```text
D:\MyCode\rag-template\Scripts
D:\MyCode\rag-template\src
```

否则相对路径可能会错误写入：

```text
D:\MyCode\rag-template\Scripts\data
```

---

## 14. Git 封版建议

当前版本建议提交为：

```text
rag-qa-v1.0
```

推荐提交命令：

```bat
cd /d D:\MyCode\rag-template
git status
git add .
git commit -m "feat(rag): complete parent-child RAG QA pipeline with eval capture"
git tag rag-qa-v1.0
git push
git push origin rag-qa-v1.0
```

如果远程分支是 `master`：

```bat
git push origin master
git push origin rag-qa-v1.0
```

如果远程分支是 `main`：

```bat
git push origin main
git push origin rag-qa-v1.0
```

---

## 15. 推荐 .gitignore

建议不要提交运行产物、模型文件、Milvus Lite 数据库和缓存文件。

推荐 `.gitignore`：

```gitignore
# Python
__pycache__/
*.pyc
.venv/
.env

# IDE
.idea/
.vscode/

# Logs
logs/
*.log

# Model files
models/
.cache/
*.bin
*.safetensors

# Milvus Lite / local db
*.db
*.db-shm
*.db-wal

# Runtime outputs
data/processed/
data/runs/
data/vector_store/
data/vector_index/
data/processed/runs/
data/processed/eval_reports/

# Keep eval set
!data/eval_set/
!data/eval_set/*.jsonl
```

如果需要保留空目录，可以使用 `.gitkeep`。

---

## 16. 后续路线

当前阶段结束后，不继续在 RAG 主链路上横向扩展算法。

后续路线：

```text
阶段 1：Git 固化 RAG QA v1.0
当前完成后提交并打 tag。

阶段 2：封装 RAGTool
在当前 rag-template 中先封装 Tool 接口，让调用方式变成：
rag_tool.run({"query": "..."})

阶段 3：迁移企业框架
将 rag-template 迁移到 enterprise-rag-agent-system 的 backend/rag 和 backend/tools/rag_tool.py。

阶段 4：进入 Agent
实现 SupervisorAgent、ToolRegistry、RAGTool、LLMTool 和 AgentTraceCapture。

阶段 5：Agent Eval
评估 tool 调用正确率、RAG 调用成功率、trace 完整性和最终答案质量。

阶段 6：DatasetBuilder
从 rag_runs.jsonl 和 agent_traces.jsonl 构造 SFT、DPO、RAG eval、tool-call 和 trajectory 数据。

阶段 7：DeepSeek Teacher
使用 DeepSeek 生成、改写、打分和构造偏好数据。

阶段 8：后训练
进行 SFT、DPO、GRPO 等后训练实验。
```

---

## 17. 当前结论

当前项目已经完成 RAG QA 主链路闭环：

```text
输入数据
-> 知识切分
-> 向量入库
-> 混合检索
-> 精排
-> 上下文构造
-> 本地 LLM 生成
-> 运行数据捕获
-> 轻量评估
```

当前版本可以作为后续 Agent-RAG 企业框架迁移的 RAG 基础能力。

一句话总结：

```text
RAG QA v1.0 已完成，下一步是封装 RAGTool，然后迁移到企业框架，进入 Agent 阶段。
```

# rag-template

一个从零实现的轻量级 RAG（Retrieval-Augmented Generation）项目模板，用于学习和验证企业级知识问答系统的核心流程。

本项目不依赖 LangChain 等高层封装框架，而是手动拆解 RAG 的关键模块，重点理解文档处理、切分、向量化、索引构建、检索、重排序、Prompt 构建和大模型生成之间的完整链路。

## 1. 项目目标

本项目的目标是构建一个可扩展的 RAG 基础框架，完成从原始文档到问答生成的最小闭环：

```text
原始文档
↓
文档读取
↓
文本清洗
↓
文本切分
↓
Embedding 向量化
↓
向量索引构建
↓
Query 检索
↓
Rerank 重排序
↓
Prompt 构建
↓
LLM 生成答案
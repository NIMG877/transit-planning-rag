# 基于 RAG 的轨道交通规划与政策智能问答助手

## 项目简介
本项目面向轨道交通规划与政策场景，提供两类问答能力：

- Vector RAG：基于向量检索的问答。
- PathRAG：在向量检索基础上，加入知识图谱路径证据进行增强推理。

项目已完成模块化重构，主流程统一收敛到 `app` 包，支持：

- 规划/政策文本抽取与清洗
- 文本分块与向量库构建
- 知识图谱构建（含批次与批内进度条）
- Vector RAG / PathRAG 问答

## 目录结构

```text
scripts/
├─ app/
│  ├─ config/
│  │  ├─ settings.py
│  │  └─ pdf_config.yaml
│  ├─ llm/
│  │  └─ qwen_manager.py
│  ├─ ingestion/
│  │  ├─ pdf_reader.py
│  │  └─ chunking.py
│  ├─ retrieval/
│  │  └─ vector_store.py
│  ├─ graph/
│  │  └─ pathrag_engine.py
│  ├─ rag/
│  │  └─ pipeline.py
│  ├─ evaluation/
│  │  └─ metrics.py
│  └─ entrypoints/
│     ├─ init_pipeline.py
│     └─ rag_entry.py
└─ faiss_db/
```

## 运行环境

建议环境：

- Python 3.10+
- Windows / Linux / macOS

常用依赖（按代码导入整理）：

- langchain-core
- langchain-classic
- transformers
- sentence-transformers
- faiss-cpu
- numpy
- pdfplumber
- pyyaml
- tqdm
- torch
- pydantic

如果你使用 `venv`：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install langchain-core langchain-classic transformers sentence-transformers faiss-cpu numpy pdfplumber pyyaml tqdm torch pydantic
```

## 配置说明

### 1) 模型与参数
在 `app/config/settings.py` 中配置：

- `HF_TOKEN`
- `QWEN_MODEL_NAME`
- `EMBEDDING_MODEL_NAME`
- PathRAG 相关超参数

### 2) 文档清单与清洗规则
在 `app/config/pdf_config.yaml` 中配置：

- `pdf_path`：PDF 路径
- `pages`：页码范围
- `rules`：清洗规则
- `year/publisher/doc_type`：元信息

注意：请在 `scripts` 目录下执行命令，以保证相对路径行为与当前配置一致。

## 快速开始

### 1) 初始化资源（向量库 / 知识图谱）

在 `scripts` 目录执行：

```powershell
python -m app.entrypoints.init_pipeline --target all
```

可选参数：

- `--target vector`：仅构建向量库
- `--target graph`：仅构建知识图谱
- `--target all`：全部构建

说明：构建知识图谱时会显示两层进度条（batch + doc）。

### 2) 进行问答

Vector RAG：

```powershell
python -m app.entrypoints.rag_entry "十四五期间上海综合交通发展的总体思路是什么？" --mode vector --top-k 3
```

PathRAG：

```powershell
python -m app.entrypoints.rag_entry "十四五期间上海综合交通发展的总体思路是什么？" --mode pathrag --top-k 3
```

## 评估说明

评估逻辑位于 `app/evaluation/metrics.py`，包含：

- 检索准确率
- 回答语义相似度
- 数据类问题精确率
- 分难度通过率

如需将评估暴露为命令行入口，可新增 `app/entrypoints/evaluate_entry.py` 并调用 `evaluate_dual_modes`。

## 常见问题

### 1) 初始化或问答很慢
- 首次运行会下载模型并构建索引，耗时较长。
- PathRAG 启用 LLM 三元组抽取时，图谱构建更慢但证据质量通常更好。

### 2) 路径找不到
- 确保在 `scripts` 目录执行命令。
- 检查 `app/config/pdf_config.yaml` 中的 `pdf_path` 是否正确。

### 3) 显存/内存不足
- 降低 `top-k`。
- 选择更小模型。
- 减少文档数量或分批构建。

## 维护建议

- 优先在 `app` 包内新增功能，避免回退到扁平脚本结构。
- 变更配置时，先验证初始化流程再验证问答流程。
- 如需复现实验，建议固定模型版本与评估数据版本。

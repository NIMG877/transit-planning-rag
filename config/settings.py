from functools import lru_cache
from pathlib import Path

import yaml

HF_TOKEN = ""
# RAG 各环节模型配置
RAG_REWRITE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RAG_CLASSIFIER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RAG_RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RAG_ANSWER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh"

MAX_NEW_TOKENS = 512
RAG_REWRITE_MAX_NEW_TOKENS = 48
RAG_CLASSIFIER_MAX_NEW_TOKENS = 32
RAG_ENABLE_QUERY_REWRITE = True
RAG_ENABLE_RERANK = False
RAG_RERANK_CANDIDATE_TOP_K = 12

PATHRAG_MAX_HOPS = 3
PATHRAG_TOP_PATHS = 5
PATHRAG_MAX_PATHS_PER_SEED = 20
PATHRAG_ENABLE_LLM_TRIPLET = True
PATHRAG_TRIPLET_TEXT_MAX_CHARS = 900
PATHRAG_HYBRID_PATH_VECTOR_ALPHA = 0.35
PATHRAG_HYBRID_DOC_PATH_BETA = 0.30

CONFIG_PATH = {
    "pdf_config": Path(__file__).with_name("pdf_config.yaml"),
    "prompts": Path(__file__).with_name("prompts.yaml"),
}


def PDF_CONFIG() -> list[dict]:
    return _load_config(CONFIG_PATH["pdf_config"])


def PROMPT_CONFIG() -> dict:
    config = _load_config(CONFIG_PATH["prompts"])
    return config if isinstance(config, dict) else {}


@lru_cache(maxsize=8)
def _load_config(config_path: str | Path):
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

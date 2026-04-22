"""统一的 Hugging Face 本地优先模型加载入口。"""

from functools import lru_cache
from typing import Any

from huggingface_hub import snapshot_download


@lru_cache(maxsize=64)
def resolve_model_path(model_name: str, token: str | None = None) -> str:
    """本地优先解析模型路径：本地存在则直接使用，否则自动联网下载。"""
    try:
        return snapshot_download(
            repo_id=model_name,
            token=token,
            local_files_only=True,
        )
    except Exception:
        try:
            return snapshot_download(
                repo_id=model_name,
                token=token,
                local_files_only=False,
            )
        except Exception as download_exc:
            raise RuntimeError(
                f"模型 {model_name} 本地缓存不存在，且联网下载失败。"
            ) from download_exc


@lru_cache(maxsize=64)
def ensure_local_model_available(model_name: str, token: str | None = None) -> str:
    """兼容旧调用名：返回本地优先解析后的可用模型路径。"""
    try:
        return resolve_model_path(model_name=model_name, token=token)
    except Exception as exc:
        raise RuntimeError(
            f"模型 {model_name} 加载失败：本地缓存缺失且无法完成联网下载。"
        ) from exc


def load_auto_tokenizer(model_name: str, token: str | None = None, **kwargs: Any) -> Any:
    """本地优先加载 transformers tokenizer。"""
    local_model_path = resolve_model_path(model_name=model_name, token=token)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        local_model_path,
        token=token,
        local_files_only=True,
        **kwargs,
    )


def load_auto_causal_lm(model_name: str, token: str | None = None, **kwargs: Any) -> Any:
    """本地优先加载 transformers causal language model。"""
    local_model_path = resolve_model_path(model_name=model_name, token=token)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        local_model_path,
        token=token,
        local_files_only=True,
        **kwargs,
    )


def load_sentence_transformer(model_name: str, token: str | None = None, **kwargs: Any) -> Any:
    """本地优先加载 sentence-transformers 模型。"""
    local_model_path = resolve_model_path(model_name=model_name, token=token)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        local_model_path,
        token=token,
        local_files_only=True,
        **kwargs,
    )


def load_cross_encoder(model_name: str, token: str | None = None, **kwargs: Any) -> Any:
    """本地优先加载 sentence-transformers CrossEncoder。"""
    local_model_path = resolve_model_path(model_name=model_name, token=token)
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        local_model_path,
        token=token,
        local_files_only=True,
        **kwargs,
    )

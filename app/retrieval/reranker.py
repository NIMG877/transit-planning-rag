from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from app.config.settings import HF_TOKEN, RAG_RERANK_MODEL_NAME


@lru_cache(maxsize=1)
def load_reranker(model_name: str = RAG_RERANK_MODEL_NAME) -> Any:
    return CrossEncoder(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        device="cuda",
    )


def rerank_documents(
    query: str,
    documents: list[tuple[str, float, dict[str, Any]]],
    top_k: int,
    reranker: Any,
) -> list[tuple[str, float, dict[str, Any], float]]:
    if not documents:
        return []

    pairs = [[query, content] for content, _distance, _metadata in documents]
    scores = reranker.predict(pairs)

    scored = []
    for (content, distance, metadata), score in zip(documents, scores):
        scored.append((content, distance, metadata, float(score)))

    scored.sort(key=lambda item: item[3], reverse=True)
    return scored[: max(top_k, 1)]

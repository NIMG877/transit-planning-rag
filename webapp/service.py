import json
from time import perf_counter
from typing import Any, Iterator

from langchain_core.documents import Document

from rag.pipeline import (
    ask,
    ask_with_multi_recall,
    ask_with_pathrag,
    prepare_multi_recall_answer,
    prepare_pathrag_answer,
    prepare_vector_answer,
)
from webapp.schemas import ChatRequest, ChatResponse, EvidenceDoc, PathEvidence


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_text_answer(raw_result: dict[str, Any]) -> str:
    answer = raw_result.get("result", "")
    return str(answer).strip()


def _normalize_paths(raw_result: dict[str, Any]) -> list[PathEvidence]:
    paths = raw_result.get("paths") or []
    normalized: list[PathEvidence] = []
    for path_item in paths:
        if not isinstance(path_item, dict):
            continue
        normalized.append(
            PathEvidence(
                path=str(path_item.get("path", "")),
                score=_safe_float(path_item.get("score")),
                hybrid_score=_safe_float(path_item.get("hybrid_score")),
                sources=[str(src) for src in (path_item.get("sources") or [])],
            )
        )
    return normalized


def _normalize_docs_from_vector(raw_result: dict[str, Any]) -> list[EvidenceDoc]:
    source_documents = raw_result.get("source_documents") or []
    normalized: list[EvidenceDoc] = []

    for doc in source_documents:
        if not isinstance(doc, Document):
            continue

        metadata = dict(doc.metadata or {})
        normalized.append(
            EvidenceDoc(
                content=doc.page_content,
                metadata=metadata,
                distance=_safe_float(metadata.get("distance")),
                rerank_score=_safe_float(metadata.get("rerank_score")),
            )
        )

    return normalized


def _normalize_docs_from_path_or_hybrid(raw_result: dict[str, Any]) -> list[EvidenceDoc]:
    vector_docs = raw_result.get("vector_context_docs") or []
    normalized: list[EvidenceDoc] = []

    for item in vector_docs:
        if not isinstance(item, dict):
            continue
        metadata = dict(item.get("metadata") or {})
        normalized.append(
            EvidenceDoc(
                content=str(item.get("content", "")),
                metadata=metadata,
                distance=_safe_float(item.get("distance")),
                score=_safe_float(item.get("score")),
                hybrid_score=_safe_float(item.get("hybrid_score")),
            )
        )

    return normalized


def _build_chat_response(
    request: ChatRequest,
    clean_question: str,
    raw_result: dict[str, Any],
    elapsed_ms: int,
) -> ChatResponse:
    if request.mode == "vector":
        evidence_docs = _normalize_docs_from_vector(raw_result)
    else:
        evidence_docs = _normalize_docs_from_path_or_hybrid(raw_result)

    return ChatResponse(
        mode=request.mode,
        question=clean_question,
        answer=_extract_text_answer(raw_result),
        rewritten_query=str(raw_result.get("rewritten_query") or "") or None,
        question_category=str(raw_result.get("question_category") or "") or None,
        query_entities=[str(item) for item in (raw_result.get("query_entities") or [])],
        paths=_normalize_paths(raw_result),
        evidence_docs=evidence_docs,
        elapsed_ms=elapsed_ms,
    )


def _prepare_chat(request: ChatRequest, clean_question: str) -> dict[str, Any]:
    if request.mode == "vector":
        return prepare_vector_answer(question=clean_question, top_k=request.top_k)
    if request.mode == "pathrag":
        return prepare_pathrag_answer(question=clean_question, vector_top_k=request.top_k)
    return prepare_multi_recall_answer(question=clean_question, vector_top_k=request.top_k)


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def run_chat(request: ChatRequest) -> ChatResponse:
    started = perf_counter()
    clean_question = request.question.strip()

    if request.mode == "vector":
        raw_result = ask(question=clean_question, top_k=request.top_k)
    elif request.mode == "pathrag":
        raw_result = ask_with_pathrag(question=clean_question, vector_top_k=request.top_k)
    else:
        raw_result = ask_with_multi_recall(question=clean_question, vector_top_k=request.top_k)
    elapsed_ms = int((perf_counter() - started) * 1000)
    return _build_chat_response(request, clean_question, raw_result, elapsed_ms)


def run_chat_stream(request: ChatRequest) -> Iterator[str]:
    started = perf_counter()
    clean_question = request.question.strip()
    prepared = _prepare_chat(request, clean_question)

    llm = prepared.pop("llm")
    prompt = prepared.pop("prompt")
    answer_parts: list[str] = []

    yield _sse_event({"type": "start"})

    for chunk in llm.stream(prompt):
        answer_parts.append(chunk)
        yield _sse_event({"type": "token", "delta": chunk})

    raw_result = {
        **prepared,
        "result": "".join(answer_parts).strip(),
    }
    elapsed_ms = int((perf_counter() - started) * 1000)
    payload = _build_chat_response(request, clean_question, raw_result, elapsed_ms).model_dump()
    yield _sse_event({"type": "done", "payload": payload})

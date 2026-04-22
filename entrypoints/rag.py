"""RAG 统一入口"""

import argparse
import json
import sys
from time import perf_counter

from rag.pipeline import (
    ask,
    ask_with_multi_recall,
    ask_with_pathrag,
    prepare_multi_recall_answer,
    prepare_pathrag_answer,
    prepare_vector_answer,
)


def _run_stream(question: str, mode: str, top_k: int) -> dict:
    if mode == "vector":
        prepared = prepare_vector_answer(question=question, top_k=top_k)
    elif mode == "pathrag":
        prepared = prepare_pathrag_answer(question=question, vector_top_k=top_k)
    else:
        prepared = prepare_multi_recall_answer(question=question, vector_top_k=top_k)

    llm = prepared.pop("llm")
    prompt = prepared.pop("prompt")

    sys.stdout.write("回答:\n")
    sys.stdout.flush()

    parts: list[str] = []
    pending_chunks: list[str] = []
    last_flush = perf_counter()
    for chunk in llm.stream(prompt):
        parts.append(chunk)
        pending_chunks.append(chunk)
        now = perf_counter()
        should_flush = (
            sum(len(part) for part in pending_chunks) >= 24
            or "\n" in chunk
            or now - last_flush >= 0.05
        )
        if should_flush:
            sys.stdout.write("".join(pending_chunks))
            sys.stdout.flush()
            pending_chunks.clear()
            last_flush = now

    if pending_chunks:
        sys.stdout.write("".join(pending_chunks))
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    prepared["result"] = "".join(parts).strip()
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 问答入口")
    parser.add_argument("question", type=str, help="用户问题")
    parser.add_argument("--mode", choices=["vector", "pathrag", "hybrid"], default="vector")
    parser.add_argument("--top-k", type=int, default=3, help="召回文档数量")
    parser.add_argument("--no-stream", action="store_true", help="关闭最终回答的流式输出")
    args = parser.parse_args()

    if args.no_stream:
        if args.mode == "vector":
            result = ask(question=args.question, top_k=args.top_k)
        elif args.mode == "pathrag":
            result = ask_with_pathrag(question=args.question, vector_top_k=args.top_k)
        else:
            result = ask_with_multi_recall(question=args.question, vector_top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        result = _run_stream(question=args.question, mode=args.mode, top_k=args.top_k)
        metadata = {key: value for key, value in result.items() if key != "result"}
        print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

"""RAG 统一入口"""

import argparse
import json

from rag.pipeline import ask, ask_with_multi_recall, ask_with_pathrag


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 问答入口")
    parser.add_argument("question", type=str, help="用户问题")
    parser.add_argument("--mode", choices=["vector", "pathrag", "hybrid"], default="vector")
    parser.add_argument("--top-k", type=int, default=3, help="召回文档数量")
    args = parser.parse_args()

    if args.mode == "vector":
        result = ask(question=args.question, top_k=args.top_k)
    elif args.mode == "pathrag":
        result = ask_with_pathrag(question=args.question, vector_top_k=args.top_k)
    else:
        result = ask_with_multi_recall(question=args.question, vector_top_k=args.top_k)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

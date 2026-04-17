"""初始化入口：构建向量数据库或知识图谱。"""

import argparse

from graph.pathrag_engine import rebuild_path_graph
from retrieval.vector_store import build_index, get_collection, get_embedding_model


def init_vector_db() -> None:
    collection = get_collection()
    embedding_model = get_embedding_model()
    build_index(collection=collection, embedding_model=embedding_model)


def init_knowledge_graph() -> None:
    rebuild_path_graph()


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 RAG 资源")
    parser.add_argument(
        "--target",
        choices=["vector", "graph", "all"],
        default="all",
        help="初始化目标：向量库、知识图谱或全部",
    )
    args = parser.parse_args()

    if args.target in {"vector", "all"}:
        print("开始构建向量数据库...")
        init_vector_db()

    if args.target in {"graph", "all"}:
        print("开始构建知识图谱...")
        init_knowledge_graph()

    print("初始化完成")


if __name__ == "__main__":
    main()

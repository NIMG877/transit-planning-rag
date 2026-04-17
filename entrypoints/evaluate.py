"""评估入口"""

import argparse

from evaluation.metrics import evaluate_dual_modes


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 RAG 模型性能")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="../datas/QA_pairs.csv",
        help="评估问答对CSV文件路径，默认: ../datas/QA_pairs.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="检索文档数量，默认 3",
    )
    args = parser.parse_args()

    evaluate_dual_modes(csv_path=args.csv_path, top_k=args.top_k)


if __name__ == "__main__":
    main()

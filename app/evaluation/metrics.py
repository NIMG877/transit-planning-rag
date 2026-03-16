import csv
import gc
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

import torch
from sentence_transformers import SentenceTransformer, util

from app.config.settings import EMBEDDING_MODEL_NAME, HF_TOKEN
from app.rag.pipeline import ask_with_pathrag, build_retrieval_qa_chain

SUPPORTED_MODES = {"vector", "pathrag"}


def compute_semantic_similarity(ans1: str, ans2: str, model=None) -> float:
    if model:
        with torch.no_grad():
            emb1 = model.encode(ans1, convert_to_tensor=True)
            emb2 = model.encode(ans2, convert_to_tensor=True)
            return util.cos_sim(emb1, emb2).item()
    return SequenceMatcher(None, ans1, ans2).ratio()


def extract_numbers(text: str) -> set:
    return set(re.findall(r"\d+\.?\d*|\d+%", text))


def _cache_path_for_mode(csv_path: str, mode: str) -> str:
    return os.path.join(os.path.dirname(csv_path), f"QA_cache_{mode}.csv")


def _read_csv_rows(csv_path: str) -> list[dict]:
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_cache(cache_path: str) -> dict[str, dict]:
    cache = {}
    if not os.path.exists(cache_path):
        return cache

    with open(cache_path, mode="r", encoding="utf-8-sig") as f:
        reader_cache = csv.DictReader(f)
        for row in reader_cache:
            cache[row["序号"]] = {
                "sys_ans": row["生成答案"],
                "retrieved_sources": row["召回来源"].split("|") if row["召回来源"] else [],
                "mode": row.get("模式", ""),
                "path_count": int(row.get("路径数", 0) or 0),
            }
    return cache


def _invoke_mode_answer(mode: str, question: str, qa_chain=None, top_k: int = 3) -> tuple[str, list[str], int]:
    if mode == "vector":
        result = qa_chain.invoke({"query": question})
        sys_ans = result["result"]
        retrieved_sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]
        return sys_ans, retrieved_sources, 0

    result = ask_with_pathrag(
        question=question,
        vector_top_k=top_k,
        max_hops=3,
        top_paths=5,
        enable_llm_triplet=True,
    )
    sys_ans = result["result"]

    source_set = set()
    for doc_info in result.get("vector_context_docs", []):
        source = (doc_info.get("metadata") or {}).get("source", "")
        if source:
            source_set.add(source)
    for path in result.get("paths", []):
        for src in path.get("sources", []):
            if src:
                source_set.add(src)

    return sys_ans, sorted(source_set), len(result.get("paths", []))


def generate_and_cache_answers(csv_path: str, mode: str, top_k: int = 3) -> str:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {SUPPORTED_MODES}, got: {mode}")

    qa_chain = build_retrieval_qa_chain(top_k=top_k) if mode == "vector" else None
    cache_path = _cache_path_for_mode(csv_path, mode)
    data = _read_csv_rows(csv_path)
    total_questions = len(data)
    cache = _load_cache(cache_path)

    write_mode = "a" if os.path.exists(cache_path) else "w"
    with open(cache_path, mode=write_mode, encoding="utf-8-sig", newline="") as f_cache:
        fieldnames = ["序号", "问题", "生成答案", "召回来源", "模式", "路径数"]
        writer = csv.DictWriter(f_cache, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()

        for row in data:
            idx = row["序号"]
            question = row["问题"]

            if idx in cache:
                print(f"[{mode}] [{idx}/{total_questions}] Reading from cache: {question}")
                continue

            print(f"[{mode}] [{idx}/{total_questions}] Generating answer: {question}")
            sys_ans, retrieved_sources, path_count = _invoke_mode_answer(
                mode=mode,
                question=question,
                qa_chain=qa_chain,
                top_k=top_k,
            )

            writer.writerow(
                {
                    "序号": idx,
                    "问题": question,
                    "生成答案": sys_ans,
                    "召回来源": "|".join(retrieved_sources),
                    "模式": mode,
                    "路径数": path_count,
                }
            )
            f_cache.flush()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return cache_path


def evaluate_cached_answers(csv_path: str, cache_path: str, mode: str) -> dict:
    print("\nLoading evaluation models...")
    try:
        semantic_model = SentenceTransformer(EMBEDDING_MODEL_NAME, token=HF_TOKEN)
    except Exception:
        semantic_model = None
        print("SentenceTransformer not found, falling back to SequenceMatcher.")

    data = _read_csv_rows(csv_path)
    cache = _load_cache(cache_path)

    total_questions = len(data)
    retrieval_success_count = 0
    total_semantic_similarity = 0.0
    data_questions_count = 0
    data_precision_success_count = 0
    difficulty_stats = defaultdict(lambda: {"total": 0, "pass": 0})

    print(f"\nStarting evaluation of metrics for mode={mode}...")
    for row in data:
        idx = row["序号"]
        q_type = row["问题类别"]
        ref_ans = row["参考答案"]
        source_doc_name = row["来源文档"]
        difficulty = row["难度"]

        if idx not in cache:
            print(f"[Warning] Question {idx} missing in cache. Skipping evaluation for this question.")
            total_questions -= 1
            continue

        sys_ans = cache[idx]["sys_ans"]
        retrieved_sources = cache[idx]["retrieved_sources"]

        is_retrieved = any(source_doc_name in src for src in retrieved_sources)
        if is_retrieved:
            retrieval_success_count += 1

        similarity = compute_semantic_similarity(sys_ans, ref_ans, model=semantic_model)
        total_semantic_similarity += similarity

        is_data_pass = False
        if q_type == "数据指标":
            data_questions_count += 1
            ref_nums = extract_numbers(ref_ans)
            sys_nums = extract_numbers(sys_ans)
            if (ref_nums and ref_nums.issubset(sys_nums)) or (not ref_nums):
                data_precision_success_count += 1
                is_data_pass = True

        difficulty_stats[difficulty]["total"] += 1
        if (q_type == "数据指标" and is_data_pass) or (q_type != "数据指标" and similarity > 0.6):
            difficulty_stats[difficulty]["pass"] += 1

    print(f"\n=== 评估结果 [{mode}] ===")
    retrieval_acc = retrieval_success_count / total_questions if total_questions else 0
    print(f"检索准确率 (Retrieval Accuracy): {retrieval_acc:.2%} ({retrieval_success_count}/{total_questions})")

    ans_acc = total_semantic_similarity / total_questions if total_questions else 0
    print(f"回答准确率 (Answer Accuracy - Avg Similarity): {ans_acc:.2%}")

    data_prec = data_precision_success_count / data_questions_count if data_questions_count else 0
    print(f"数据精确率 (Data Precision): {data_prec:.2%} ({data_precision_success_count}/{data_questions_count})")

    print("\n难度通过率 (Difficulty Pass Rate):")
    for diff, stats in difficulty_stats.items():
        total = stats["total"]
        passed = stats["pass"]
        rate = passed / total if total else 0
        print(f"  - {diff}: {rate:.2%} ({passed}/{total})")

    difficulty_rates = {}
    for diff, stats in difficulty_stats.items():
        total = stats["total"]
        passed = stats["pass"]
        difficulty_rates[diff] = (passed / total) if total else 0.0

    return {
        "mode": mode,
        "retrieval_acc": retrieval_acc,
        "answer_acc": ans_acc,
        "data_precision": data_prec,
        "difficulty_rates": difficulty_rates,
        "counts": {
            "total": total_questions,
            "retrieval_hit": retrieval_success_count,
            "data_total": data_questions_count,
            "data_hit": data_precision_success_count,
        },
    }


def compare_metrics(vector_metrics: dict, pathrag_metrics: dict) -> None:
    print("\n=== VectorRAG vs PathRAG 对比 ===")
    print(
        f"检索准确率: Vector={vector_metrics['retrieval_acc']:.2%} | "
        f"PathRAG={pathrag_metrics['retrieval_acc']:.2%} | "
        f"Delta={pathrag_metrics['retrieval_acc'] - vector_metrics['retrieval_acc']:+.2%}"
    )
    print(
        f"回答准确率: Vector={vector_metrics['answer_acc']:.2%} | "
        f"PathRAG={pathrag_metrics['answer_acc']:.2%} | "
        f"Delta={pathrag_metrics['answer_acc'] - vector_metrics['answer_acc']:+.2%}"
    )
    print(
        f"数据精确率: Vector={vector_metrics['data_precision']:.2%} | "
        f"PathRAG={pathrag_metrics['data_precision']:.2%} | "
        f"Delta={pathrag_metrics['data_precision'] - vector_metrics['data_precision']:+.2%}"
    )

    all_difficulties = sorted(set(vector_metrics["difficulty_rates"].keys()) | set(pathrag_metrics["difficulty_rates"].keys()))
    print("\n分难度通过率对比:")
    for diff in all_difficulties:
        vec_rate = vector_metrics["difficulty_rates"].get(diff, 0.0)
        path_rate = pathrag_metrics["difficulty_rates"].get(diff, 0.0)
        print(f"  - {diff}: Vector={vec_rate:.2%} | PathRAG={path_rate:.2%} | Delta={path_rate - vec_rate:+.2%}")

def evaluate_model(csv_path: str, mode: str, top_k: int = 3) -> None:
    cache_path = generate_and_cache_answers(csv_path=csv_path, mode=mode, top_k=top_k)
    metrics = evaluate_cached_answers(csv_path=csv_path, cache_path=cache_path, mode=mode)
    return metrics

def evaluate_dual_modes(csv_path: str, top_k: int = 3) -> None:
    vector_cache = generate_and_cache_answers(csv_path=csv_path, mode="vector", top_k=top_k)
    pathrag_cache = generate_and_cache_answers(csv_path=csv_path, mode="pathrag", top_k=top_k)

    vector_metrics = evaluate_cached_answers(csv_path=csv_path, cache_path=vector_cache, mode="vector")
    pathrag_metrics = evaluate_cached_answers(csv_path=csv_path, cache_path=pathrag_cache, mode="pathrag")
    compare_metrics(vector_metrics, pathrag_metrics)

if __name__ == "__main__":
    csv_path = "../datas/QA_pairs.csv"
    evaluate_model(csv_path=csv_path, mode="vector", top_k=3)
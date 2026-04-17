from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config.settings import (
    RAG_ANSWER_MODEL_NAME,
    RAG_CLASSIFIER_MAX_NEW_TOKENS,
    RAG_CLASSIFIER_MODEL_NAME,
    RAG_ENABLE_QUERY_REWRITE,
    RAG_ENABLE_RERANK,
    RAG_RERANK_CANDIDATE_TOP_K,
    RAG_REWRITE_MAX_NEW_TOKENS,
    RAG_REWRITE_MODEL_NAME,
    PROMPT_CONFIG,
    PATHRAG_ENABLE_LLM_TRIPLET,
    PATHRAG_HYBRID_DOC_PATH_BETA,
    PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
)
from llm.qwen_manager import load_qwen_llm
from retrieval.reranker import load_reranker, rerank_documents
from retrieval.vector_store import get_collection, get_embedding_model, retrieve

DEFAULT_TOP_K = 3

QUESTION_CATEGORIES = [
    "政策目标与愿景查询",
    "现状与成就查询",
    "规划行动与实施举措查询",
    "概念解释与术语定义查询",
    "政策逻辑与因果关系查询",
    "问题与挑战识别",
    "跨文件综合对比",
]
DEFAULT_CATEGORY = "概念解释与术语定义查询"


def _get_prompt_config() -> dict[str, Any]:
    return PROMPT_CONFIG()


def _get_rewrite_prompt_template() -> str:
    config = _get_prompt_config()
    template = config.get("rewrite", "")
    return str(template).strip()


def _get_classifier_prompt_template() -> str:
    config = _get_prompt_config()
    template = config.get("question_classifier", "")
    return str(template).strip()


def _get_category_instruction(category: str) -> str:
    config = _get_prompt_config()
    answer_style_prompts = config.get("answer_style_prompts", {}) or {}
    default_style = str(config.get("default_answer_style_prompt", "请保持回答结构清晰，优先提炼与问题最相关的政策信息。")).strip()
    selected = answer_style_prompts.get(category)
    return str(selected).strip() if selected else default_style


def _get_rag_answer_prompt_template(category: str) -> str:
    config = _get_prompt_config()
    base_template = str(config.get("rag_answer_base", "")).strip()
    category_instruction = _get_category_instruction(category)
    if "{category_instruction}" in base_template:
        return base_template.replace("{category_instruction}", category_instruction)
    return f"{base_template}\n\n分类回答风格要求：\n{category_instruction}"


def _get_multi_recall_answer_prompt_template() -> str:
    config = _get_prompt_config()
    base_template = str(config.get("multi_recall_answer_base", "")).strip()
    if base_template:
        return base_template
    return (
        "你是一名交通政策问答助手。请综合向量召回证据与实体关系路径证据回答问题。\n\n"
        "规则：\n"
        "1. 回答必须基于给定证据，不得编造。\n"
        "2. 先给出结论，再用路径逻辑与文本事实支撑。\n"
        "3. 若证据不足，回答：根据现有资料无法确定。\n\n"
        "[分类回答风格要求]\n"
        "{category_instruction}\n\n"
        "[问题]\n"
        "{question}\n\n"
        "[关系路径证据]\n"
        "{path_context}\n\n"
        "[向量文本证据]\n"
        "{vector_context}\n\n"
        "[回答]"
    )


def _normalize_category(raw_category: str) -> str:
    cleaned = (raw_category or "").strip().replace("：", "").replace(":", "")
    if cleaned in QUESTION_CATEGORIES:
        return cleaned

    for category in QUESTION_CATEGORIES:
        if category in cleaned:
            return category

    simple_alias = {
        "目标愿景": "政策目标与愿景查询",
        "现状成就": "现状与成就查询",
        "规划行动": "规划行动与实施举措查询",
        "概念解释": "概念解释与术语定义查询",
        "术语定义": "概念解释与术语定义查询",
        "政策逻辑": "政策逻辑与因果关系查询",
        "问题挑战": "问题与挑战识别",
        "跨文件对比": "跨文件综合对比",
    }
    for alias, category in simple_alias.items():
        if alias in cleaned:
            return category
    return DEFAULT_CATEGORY


class VectorDBRetriever(BaseRetriever):
    collection: Any
    embedding_model: Any
    top_k: int = DEFAULT_TOP_K
    candidate_top_k: int = RAG_RERANK_CANDIDATE_TOP_K
    use_rerank: bool = RAG_ENABLE_RERANK
    reranker: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        retrieval_k = max(self.top_k, self.candidate_top_k if self.use_rerank else self.top_k)
        retrieved_docs = retrieve(
            query=query,
            collection=self.collection,
            embedding_model=self.embedding_model,
            top_k=retrieval_k,
        )

        documents = []
        if self.use_rerank and self.reranker and retrieved_docs:
            reranked_docs = rerank_documents(
                query=query,
                documents=retrieved_docs,
                top_k=self.top_k,
                reranker=self.reranker,
            )
            for content, distance, metadata, rerank_score in reranked_docs:
                document_metadata = dict(metadata or {})
                document_metadata["distance"] = float(distance)
                document_metadata["rerank_score"] = float(rerank_score)
                documents.append(Document(page_content=content, metadata=document_metadata))
        else:
            for content, distance, metadata in retrieved_docs[: self.top_k]:
                document_metadata = dict(metadata or {})
                document_metadata["distance"] = float(distance)
                documents.append(Document(page_content=content, metadata=document_metadata))
        return documents


def rewrite_question(question: str) -> str:
    if not RAG_ENABLE_QUERY_REWRITE:
        return question

    rewrite_llm = load_qwen_llm(
        model_name=RAG_REWRITE_MODEL_NAME,
        max_new_tokens=RAG_REWRITE_MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
    )
    rewrite_prompt_template = _get_rewrite_prompt_template()
    if not rewrite_prompt_template:
        return question
    prompt = PromptTemplate(input_variables=["question"], template=rewrite_prompt_template)

    try:
        rewritten = rewrite_llm.invoke(prompt.format(question=question)).strip()
    except Exception:
        return question

    cleaned = rewritten.splitlines()[0].strip().strip("\"'“”") if rewritten else ""
    if not cleaned:
        return question
    return cleaned


def classify_question(question: str) -> str:
    classifier_prompt_template = _get_classifier_prompt_template()
    if not classifier_prompt_template:
        return DEFAULT_CATEGORY

    classifier_llm = load_qwen_llm(
        model_name=RAG_CLASSIFIER_MODEL_NAME,
        max_new_tokens=RAG_CLASSIFIER_MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
    )
    prompt = PromptTemplate(input_variables=["question"], template=classifier_prompt_template)

    try:
        raw_category = classifier_llm.invoke(prompt.format(question=question)).strip()
    except Exception:
        return DEFAULT_CATEGORY

    first_line = raw_category.splitlines()[0].strip() if raw_category else ""
    return _normalize_category(first_line)


def build_retrieval_qa_chain(top_k: int = DEFAULT_TOP_K, answer_prompt_template: str | None = None) -> RetrievalQA:
    collection, embedding_model = get_collection(), get_embedding_model()
    reranker = load_reranker() if RAG_ENABLE_RERANK else None
    retriever = VectorDBRetriever(
        collection=collection,
        embedding_model=embedding_model,
        top_k=top_k,
        candidate_top_k=max(top_k, RAG_RERANK_CANDIDATE_TOP_K),
        use_rerank=RAG_ENABLE_RERANK,
        reranker=reranker,
    )
    llm = load_qwen_llm(model_name=RAG_ANSWER_MODEL_NAME)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=answer_prompt_template or _get_rag_answer_prompt_template(DEFAULT_CATEGORY),
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    # rewritten_question = rewrite_question(question)
    rewritten_question = question
    question_category = classify_question(rewritten_question)
    answer_prompt_template = _get_rag_answer_prompt_template(question_category)
    qa_chain = build_retrieval_qa_chain(top_k=top_k, answer_prompt_template=answer_prompt_template)
    result = qa_chain.invoke({"query": rewritten_question})
    result["original_query"] = question
    result["rewritten_query"] = rewritten_question
    result["question_category"] = question_category
    return result


def ask_with_pathrag(
    question: str,
    vector_top_k: int = DEFAULT_TOP_K,
    max_hops: int = 3,
    top_paths: int = 5,
    enable_llm_triplet: bool = PATHRAG_ENABLE_LLM_TRIPLET,
    hybrid_path_vector_alpha: float = PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
    hybrid_doc_path_beta: float = PATHRAG_HYBRID_DOC_PATH_BETA,
) -> dict[str, Any]:
    from graph.pathrag_engine import ask_pathrag

    rewritten_question = rewrite_question(question)
    question_category = classify_question(rewritten_question)
    category_instruction = _get_category_instruction(question_category)

    result = ask_pathrag(
        question=rewritten_question,
        vector_top_k=vector_top_k,
        max_hops=max_hops,
        top_paths=top_paths,
        enable_llm_triplet=enable_llm_triplet,
        hybrid_path_vector_alpha=hybrid_path_vector_alpha,
        hybrid_doc_path_beta=hybrid_doc_path_beta,
        category_instruction=category_instruction,
    )
    result["original_query"] = question
    result["rewritten_query"] = rewritten_question
    result["question_category"] = question_category
    return result


def ask_with_multi_recall(
    question: str,
    vector_top_k: int = DEFAULT_TOP_K,
    max_hops: int = 3,
    top_paths: int = 5,
    enable_llm_triplet: bool = PATHRAG_ENABLE_LLM_TRIPLET,
    hybrid_path_vector_alpha: float = PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
    hybrid_doc_path_beta: float = PATHRAG_HYBRID_DOC_PATH_BETA,
) -> dict[str, Any]:
    from graph.pathrag_engine import build_pathrag_pipeline

    rewritten_question = rewrite_question(question)
    question_category = classify_question(rewritten_question)
    category_instruction = _get_category_instruction(question_category)

    collection, embedding_model = get_collection(), get_embedding_model()
    retrieval_k = max(vector_top_k, top_paths, RAG_RERANK_CANDIDATE_TOP_K if RAG_ENABLE_RERANK else vector_top_k)
    retrieved_docs = retrieve(
        query=rewritten_question,
        collection=collection,
        embedding_model=embedding_model,
        top_k=retrieval_k,
    )

    pathrag_pipeline = build_pathrag_pipeline(
        max_hops=max_hops,
        top_paths=top_paths,
        enable_llm_triplet=enable_llm_triplet,
        hybrid_path_vector_alpha=hybrid_path_vector_alpha,
        hybrid_doc_path_beta=hybrid_doc_path_beta,
    )
    query_entities = pathrag_pipeline.link_query_entities(rewritten_question, top_k_docs=max(3, vector_top_k))
    paths = pathrag_pipeline.retrieve_paths(rewritten_question, query_entities)
    paths = pathrag_pipeline._hybrid_rescore_paths(rewritten_question, paths, retrieved_docs)
    path_context = pathrag_pipeline.build_path_context(paths[:top_paths])

    selected_docs: list[tuple[str, float, dict[str, Any], float]]
    if RAG_ENABLE_RERANK and retrieved_docs:
        reranker = load_reranker()
        reranked = rerank_documents(
            query=rewritten_question,
            documents=retrieved_docs,
            top_k=vector_top_k,
            reranker=reranker,
        )
        selected_docs = [
            (content, distance, metadata, float(rerank_score))
            for content, distance, metadata, rerank_score in reranked
        ]
    else:
        selected_docs = pathrag_pipeline._hybrid_rerank_docs(rewritten_question, retrieved_docs, paths)[:vector_top_k]

    vector_context = "\n\n".join([doc for doc, _distance, _meta, _score in selected_docs])

    llm = load_qwen_llm(model_name=RAG_ANSWER_MODEL_NAME)
    prompt = PromptTemplate(
        input_variables=["question", "path_context", "vector_context", "category_instruction"],
        template=_get_multi_recall_answer_prompt_template(),
    )
    answer_text = llm.invoke(
        prompt.format(
            question=rewritten_question,
            path_context=path_context or "无可用路径证据",
            vector_context=vector_context or "无可用向量证据",
            category_instruction=category_instruction,
        )
    )

    return {
        "result": answer_text,
        "original_query": question,
        "rewritten_query": rewritten_question,
        "question_category": question_category,
        "query_entities": query_entities,
        "paths": [
            {
                "path": pathrag_pipeline.serialize_path(path),
                "score": path.score,
                "hybrid_score": path.hybrid_score,
                "sources": sorted({edge.doc_source for edge in path.edges if edge.doc_source}),
            }
            for path in paths[:top_paths]
        ],
        "path_context": path_context,
        "vector_context_docs": [
            {
                "content": doc,
                "distance": distance,
                "metadata": metadata,
                "score": score,
            }
            for doc, distance, metadata, score in selected_docs
        ],
    }

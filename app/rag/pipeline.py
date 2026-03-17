from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.config.settings import (
    RAG_ANSWER_MODEL_NAME,
    RAG_ENABLE_QUERY_REWRITE,
    RAG_ENABLE_RERANK,
    RAG_RERANK_CANDIDATE_TOP_K,
    RAG_REWRITE_MAX_NEW_TOKENS,
    RAG_REWRITE_MODEL_NAME,
    PATHRAG_ENABLE_LLM_TRIPLET,
    PATHRAG_HYBRID_DOC_PATH_BETA,
    PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
)
from app.llm.qwen_manager import load_qwen_llm
from app.retrieval.reranker import load_reranker, rerank_documents
from app.retrieval.vector_store import get_collection, get_embedding_model, retrieve

DEFAULT_TOP_K = 3

RAG_PROMPT_TEMPLATE = """
你是一名交通政策问答助手，请严格依据提供的【上下文】回答问题。

回答原则：
1. 回答必须基于【上下文】中的信息，不得使用任何外部知识或常识进行补充。
2. 优先直接使用【上下文】中的原始表述回答问题，尽量保持原句或关键表达。
3. 如果原文无法直接回答，但其含义可以支持回答，可以在保持原意的前提下进行简要概括。
4. 严禁编造、猜测或补充上下文中不存在的政策内容。
5. 如果【上下文】中的信息不足以支持问题的回答，必须输出：
   “根据现有资料无法确定”。

回答要求：
- 回答应简洁、准确。
- 仅回答与问题直接相关的政策内容。
- 不要输出推理过程。
- 不要添加无关解释。

【上下文】
{context}

【问题】
{question}

【回答】
"""

REWRITE_PROMPT_TEMPLATE = """
你是交通规划与政策相关检索查询改写助手。请将用户问题改写为更利于政策文档检索的查询语句。

要求：
1. 保留原问题核心语义，不改变意图。
2. 保留关键实体（如城市、时间、规划名称、政策术语）。
3. 输出一句简洁中文查询语句，不要解释。

用户问题：
{question}

改写查询：
"""


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
    prompt = PromptTemplate(input_variables=["question"], template=REWRITE_PROMPT_TEMPLATE)

    try:
        rewritten = rewrite_llm.invoke(prompt.format(question=question)).strip()
    except Exception:
        return question

    cleaned = rewritten.splitlines()[0].strip().strip("\"'“”") if rewritten else ""
    if not cleaned:
        return question
    return cleaned


def build_retrieval_qa_chain(top_k: int = DEFAULT_TOP_K) -> RetrievalQA:
    print("正在加载向量数据库集合、向量嵌入模型...", flush=True)
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
    print("正在加载大语言模型...", flush=True)
    llm = load_qwen_llm(model_name=RAG_ANSWER_MODEL_NAME)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    print("正在构建检索问答链...", flush=True)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    rewritten_question = rewrite_question(question)
    qa_chain = build_retrieval_qa_chain(top_k=top_k)
    result = qa_chain.invoke({"query": rewritten_question})
    result["original_query"] = question
    result["rewritten_query"] = rewritten_question
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
    from app.graph.pathrag_engine import ask_pathrag

    rewritten_question = rewrite_question(question)

    result = ask_pathrag(
        question=rewritten_question,
        vector_top_k=vector_top_k,
        max_hops=max_hops,
        top_paths=top_paths,
        enable_llm_triplet=enable_llm_triplet,
        hybrid_path_vector_alpha=hybrid_path_vector_alpha,
        hybrid_doc_path_beta=hybrid_doc_path_beta,
    )
    result["original_query"] = question
    result["rewritten_query"] = rewritten_question
    return result

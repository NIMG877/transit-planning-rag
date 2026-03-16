from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.config.settings import (
    PATHRAG_ENABLE_LLM_TRIPLET,
    PATHRAG_HYBRID_DOC_PATH_BETA,
    PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
)
from app.llm.qwen_manager import load_qwen_llm
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


class VectorDBRetriever(BaseRetriever):
    collection: Any
    embedding_model: Any
    top_k: int = DEFAULT_TOP_K

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        retrieved_docs = retrieve(
            query=query,
            collection=self.collection,
            embedding_model=self.embedding_model,
            top_k=self.top_k,
        )

        documents = []
        for content, distance, metadata in retrieved_docs:
            document_metadata = dict(metadata or {})
            document_metadata["distance"] = float(distance)
            documents.append(Document(page_content=content, metadata=document_metadata))
        return documents


def build_retrieval_qa_chain(top_k: int = DEFAULT_TOP_K) -> RetrievalQA:
    print("正在加载向量数据库集合、向量嵌入模型...", flush=True)
    collection, embedding_model = get_collection(), get_embedding_model()
    retriever = VectorDBRetriever(
        collection=collection,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    print("正在加载大语言模型...", flush=True)
    llm = load_qwen_llm()
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
    qa_chain = build_retrieval_qa_chain(top_k=top_k)
    return qa_chain.invoke({"query": question})


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

    return ask_pathrag(
        question=question,
        vector_top_k=vector_top_k,
        max_hops=max_hops,
        top_paths=top_paths,
        enable_llm_triplet=enable_llm_triplet,
        hybrid_path_vector_alpha=hybrid_path_vector_alpha,
        hybrid_doc_path_beta=hybrid_doc_path_beta,
    )

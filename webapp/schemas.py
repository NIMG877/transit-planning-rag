from typing import Any, Literal

from pydantic import BaseModel, Field

ChatMode = Literal["vector", "pathrag", "hybrid"]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    mode: ChatMode = Field(default="vector", description="问答模式")
    top_k: int = Field(default=3, ge=1, le=20, description="召回数量")


class EvidenceDoc(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float | None = None
    score: float | None = None
    rerank_score: float | None = None
    hybrid_score: float | None = None


class PathEvidence(BaseModel):
    path: str
    score: float | None = None
    hybrid_score: float | None = None
    sources: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    mode: ChatMode
    question: str
    answer: str
    rewritten_query: str | None = None
    question_category: str | None = None
    query_entities: list[str] = Field(default_factory=list)
    paths: list[PathEvidence] = Field(default_factory=list)
    evidence_docs: list[EvidenceDoc] = Field(default_factory=list)
    elapsed_ms: int

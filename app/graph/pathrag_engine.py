import json
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from tqdm import tqdm

from app.config.settings import (
    MAX_NEW_TOKENS,
    PATHRAG_ENABLE_LLM_TRIPLET,
    PATHRAG_HYBRID_DOC_PATH_BETA,
    PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
    PATHRAG_MAX_HOPS,
    PATHRAG_MAX_PATHS_PER_SEED,
    PATHRAG_TOP_PATHS,
    PATHRAG_TRIPLET_TEXT_MAX_CHARS,
)
from app.llm.qwen_manager import load_qwen_llm
from app.retrieval.vector_store import get_collection, get_embedding_model, retrieve

GRAPH_CACHE_PATH = "./chroma_db/path_graph.json"
DEFAULT_MAX_HOPS = PATHRAG_MAX_HOPS
DEFAULT_TOP_PATHS = PATHRAG_TOP_PATHS
DEFAULT_PATHS_PER_SEED = PATHRAG_MAX_PATHS_PER_SEED


@dataclass
class GraphEdge:
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    doc_source: str = ""


@dataclass
class PathCandidate:
    nodes: list[str]
    edges: list[GraphEdge]
    score: float = 0.0
    hybrid_score: float = 0.0


class RuleBasedTripletExtractor:
    """轻量级规则抽取器，便于快速构建可用图谱。"""

    def __init__(self) -> None:
        self._sentence_splitter = re.compile(r"[。！？；\n]")
        self._patterns: list[tuple[re.Pattern[str], str, float]] = [
            (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})是([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})"), "是", 0.75),
            (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})属于([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})"), "属于", 0.78),
            (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})位于([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})"), "位于", 0.8),
            (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})包括([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})"), "包括", 0.72),
            (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})提出([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})"), "提出", 0.7),
            (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})推进([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})"), "推进", 0.7),
            (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})实现([\u4e00-\u9fa5A-Za-z0-9·（）()《》\-]{2,40})"), "实现", 0.68),
        ]

    @staticmethod
    def _clean_entity(entity: str) -> str:
        entity = re.sub(r"\s+", "", entity)
        entity = entity.strip("，。；：、（）()[]【】")
        return entity

    @staticmethod
    def _is_valid_entity(entity: str) -> bool:
        if len(entity) < 2 or len(entity) > 40:
            return False
        if re.fullmatch(r"\d+", entity):
            return False
        return True

    def extract_triplets(self, text: str, doc_source: str = "") -> list[GraphEdge]:
        triplets: list[GraphEdge] = []
        for sentence in self._sentence_splitter.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            for pattern, relation, confidence in self._patterns:
                for match in pattern.finditer(sentence):
                    src = self._clean_entity(match.group(1))
                    tgt = self._clean_entity(match.group(2))
                    if src == tgt:
                        continue
                    if self._is_valid_entity(src) and self._is_valid_entity(tgt):
                        triplets.append(
                            GraphEdge(
                                source=src,
                                relation=relation,
                                target=tgt,
                                confidence=confidence,
                                doc_source=doc_source,
                            )
                        )
        return triplets


class LLMTripletExtractor:
    """使用 LLM 从文本中抽取结构化三元组，并返回 JSON。"""

    def __init__(self, llm: Any, text_max_chars: int = PATHRAG_TRIPLET_TEXT_MAX_CHARS) -> None:
        self.llm = llm
        self.text_max_chars = text_max_chars

    @staticmethod
    def _safe_float(value: Any, default: float = 0.7) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _extract_json_block(text: str) -> str:
        fenced = re.search(r"```json\s*(\[.*?\])\s*```", text, flags=re.S)
        if fenced:
            return fenced.group(1)
        raw = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.S)
        if raw:
            return raw.group(1)
        return "[]"

    def extract_triplets(self, text: str, doc_source: str = "") -> list[GraphEdge]:
        text = text.strip()
        if not text:
            return []

        snippet = text[: self.text_max_chars]
        prompt = (
            "你是信息抽取助手。请从给定文本中抽取实体关系三元组。\n"
            "输出要求：\n"
            "1. 只输出 JSON 数组，不要其他说明。\n"
            "2. 每个元素格式："
            "{\"source\":\"实体A\",\"relation\":\"关系\",\"target\":\"实体B\",\"confidence\":0-1}\n"
            "3. 仅抽取文本明确表达的关系；若无可抽取内容，输出 []。\n\n"
            f"文本：\n{snippet}"
        )

        try:
            raw = self.llm.invoke(prompt)
        except Exception:
            return []

        json_block = self._extract_json_block(raw)
        try:
            data = json.loads(json_block)
        except json.JSONDecodeError:
            return []

        edges: list[GraphEdge] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            src = RuleBasedTripletExtractor._clean_entity(str(item.get("source", "")))
            rel = str(item.get("relation", "")).strip()
            tgt = RuleBasedTripletExtractor._clean_entity(str(item.get("target", "")))
            confidence = self._safe_float(item.get("confidence", 0.7), default=0.7)
            if not rel:
                continue
            if src == tgt:
                continue
            if RuleBasedTripletExtractor._is_valid_entity(src) and RuleBasedTripletExtractor._is_valid_entity(tgt):
                edges.append(
                    GraphEdge(
                        source=src,
                        relation=rel,
                        target=tgt,
                        confidence=confidence,
                        doc_source=doc_source,
                    )
                )
        return edges


class HybridTripletExtractor:
    """融合规则抽取与 LLM 抽取。"""

    def __init__(self, rule_extractor: RuleBasedTripletExtractor, llm_extractor: LLMTripletExtractor | None = None) -> None:
        self.rule_extractor = rule_extractor
        self.llm_extractor = llm_extractor

    def extract_triplets(self, text: str, doc_source: str = "") -> list[GraphEdge]:
        rule_edges = self.rule_extractor.extract_triplets(text=text, doc_source=doc_source)
        llm_edges = self.llm_extractor.extract_triplets(text=text, doc_source=doc_source) if self.llm_extractor else []

        merged: dict[tuple[str, str, str], GraphEdge] = {}
        rule_keys = {(e.source, e.relation, e.target) for e in rule_edges}
        llm_keys = {(e.source, e.relation, e.target) for e in llm_edges}

        for edge in rule_edges + llm_edges:
            key = (edge.source, edge.relation, edge.target)
            if key not in merged or edge.confidence > merged[key].confidence:
                merged[key] = edge

        for key in rule_keys & llm_keys:
            merged[key].confidence = min(1.0, merged[key].confidence + 0.12)

        return sorted(merged.values(), key=lambda edge: edge.confidence, reverse=True)

    def extract_entities(self, text: str) -> list[str]:
        entities = set()
        for edge in self.extract_triplets(text):
            entities.add(edge.source)
            entities.add(edge.target)
        return sorted(entities)


class KnowledgeGraph:
    def __init__(self) -> None:
        self.adj_out: dict[str, list[GraphEdge]] = defaultdict(list)
        self.adj_in: dict[str, list[GraphEdge]] = defaultdict(list)
        self.node_sources: dict[str, set[str]] = defaultdict(set)
        self.edge_count = 0

    def add_edge(self, edge: GraphEdge) -> None:
        self.adj_out[edge.source].append(edge)
        self.adj_in[edge.target].append(edge)
        if edge.doc_source:
            self.node_sources[edge.source].add(edge.doc_source)
            self.node_sources[edge.target].add(edge.doc_source)
        self.edge_count += 1

    def nodes(self) -> list[str]:
        keys = set(self.adj_out.keys()) | set(self.adj_in.keys())
        return sorted(keys)

    def out_degree(self, node: str) -> int:
        return len(self.adj_out.get(node, []))

    def in_degree(self, node: str) -> int:
        return len(self.adj_in.get(node, []))

    def degree(self, node: str) -> int:
        return self.out_degree(node) + self.in_degree(node)

    def to_json(self) -> dict[str, Any]:
        edges = []
        for edge_list in self.adj_out.values():
            for edge in edge_list:
                edges.append(
                    {
                        "source": edge.source,
                        "relation": edge.relation,
                        "target": edge.target,
                        "confidence": edge.confidence,
                        "doc_source": edge.doc_source,
                    }
                )
        return {
            "edges": edges,
            "node_sources": {node: sorted(sources) for node, sources in self.node_sources.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        graph = cls()
        for edge in data.get("edges", []):
            graph.add_edge(
                GraphEdge(
                    source=edge["source"],
                    relation=edge["relation"],
                    target=edge["target"],
                    confidence=float(edge.get("confidence", 1.0)),
                    doc_source=edge.get("doc_source", ""),
                )
            )
        for node, sources in data.get("node_sources", {}).items():
            graph.node_sources[node].update(sources)
        return graph


class PathRAG:
    def __init__(
        self,
        collection: Any,
        embedding_model: Any,
        llm: Any,
        max_hops: int = DEFAULT_MAX_HOPS,
        top_paths: int = DEFAULT_TOP_PATHS,
        max_paths_per_seed: int = DEFAULT_PATHS_PER_SEED,
        enable_llm_triplet: bool = PATHRAG_ENABLE_LLM_TRIPLET,
        hybrid_path_vector_alpha: float = PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
        hybrid_doc_path_beta: float = PATHRAG_HYBRID_DOC_PATH_BETA,
    ) -> None:
        self.collection = collection
        self.embedding_model = embedding_model
        self.llm = llm
        self.max_hops = max_hops
        self.top_paths = top_paths
        self.max_paths_per_seed = max_paths_per_seed
        self.hybrid_path_vector_alpha = max(0.0, min(1.0, hybrid_path_vector_alpha))
        self.hybrid_doc_path_beta = max(0.0, min(1.0, hybrid_doc_path_beta))

        self.rule_extractor = RuleBasedTripletExtractor()
        llm_extractor = LLMTripletExtractor(self.llm) if enable_llm_triplet else None
        self.extractor = HybridTripletExtractor(self.rule_extractor, llm_extractor)
        self.graph = KnowledgeGraph()

    def load_or_build_graph(self, rebuild: bool = False) -> None:
        if not rebuild and os.path.exists(GRAPH_CACHE_PATH):
            with open(GRAPH_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = KnowledgeGraph.from_json(data)
            return

        self.graph = self._build_graph_from_collection()
        os.makedirs(os.path.dirname(GRAPH_CACHE_PATH), exist_ok=True)
        with open(GRAPH_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.graph.to_json(), f, ensure_ascii=False, indent=2)

    def _build_graph_from_collection(self, batch_size: int = 256) -> KnowledgeGraph:
        graph = KnowledgeGraph()
        total = self.collection.count()
        batch_offsets = range(0, total, batch_size)
        total_batches = (total + batch_size - 1) // batch_size if total > 0 else 0
        for offset in tqdm(batch_offsets, total=total_batches, desc="构建知识图谱", unit="batch"):
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            docs = batch.get("documents", [])
            metadatas = batch.get("metadatas", [])
            for doc, metadata in tqdm(
                zip(docs, metadatas),
                total=min(len(docs), len(metadatas)),
                desc="解析文档",
                unit="doc",
                leave=False,
            ):
                source = (metadata or {}).get("source", "")
                for edge in self.extractor.extract_triplets(doc, doc_source=source):
                    graph.add_edge(edge)
        return graph

    def link_query_entities(self, question: str, top_k_docs: int = 3) -> list[str]:
        if not self.graph.nodes():
            return []

        matched = [node for node in self.graph.nodes() if node in question]
        if matched:
            return sorted(matched, key=lambda n: len(n), reverse=True)[:6]

        doc_candidates = retrieve(
            query=question,
            collection=self.collection,
            embedding_model=self.embedding_model,
            top_k=top_k_docs,
        )

        entities = set()
        for doc, _distance, _meta in doc_candidates:
            for entity in self.extractor.extract_entities(doc):
                if entity in self.graph.adj_out or entity in self.graph.adj_in:
                    entities.add(entity)

        if entities:
            return sorted(entities, key=lambda n: self.graph.degree(n), reverse=True)[:6]

        return sorted(self.graph.nodes(), key=lambda n: self.graph.degree(n), reverse=True)[:3]

    def _find_paths_from_seed(self, seed: str, targets: set[str]) -> list[PathCandidate]:
        candidates: list[PathCandidate] = []
        queue = deque([(seed, [seed], [])])

        while queue and len(candidates) < self.max_paths_per_seed:
            current, path_nodes, path_edges = queue.popleft()
            if len(path_edges) >= self.max_hops:
                continue

            for edge in self.graph.adj_out.get(current, []):
                if edge.target in path_nodes:
                    continue
                next_nodes = path_nodes + [edge.target]
                next_edges = path_edges + [edge]

                if not targets or edge.target in targets:
                    candidates.append(PathCandidate(nodes=next_nodes, edges=next_edges))

                queue.append((edge.target, next_nodes, next_edges))

        return candidates

    def retrieve_paths(self, question: str, query_entities: list[str]) -> list[PathCandidate]:
        if not query_entities:
            return []

        targets = set(query_entities)
        all_candidates: list[PathCandidate] = []
        for entity in query_entities:
            all_candidates.extend(self._find_paths_from_seed(entity, targets - {entity}))

        if not all_candidates:
            for entity in query_entities:
                all_candidates.extend(self._find_paths_from_seed(entity, set()))

        for candidate in all_candidates:
            candidate.score = self.score_path(question, candidate)

        all_candidates.sort(key=lambda item: item.score, reverse=True)
        deduped: list[PathCandidate] = []
        seen = set()
        for item in all_candidates:
            key = tuple((edge.source, edge.relation, edge.target) for edge in item.edges)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= self.top_paths:
                break
        return deduped

    def score_path(self, question: str, candidate: PathCandidate) -> float:
        path_text = self.serialize_path(candidate)

        question_emb = self.embedding_model.encode([question])[0]
        path_emb = self.embedding_model.encode([path_text])[0]
        dot = sum(a * b for a, b in zip(question_emb, path_emb))
        q_norm = math.sqrt(sum(a * a for a in question_emb)) + 1e-9
        p_norm = math.sqrt(sum(a * a for a in path_emb)) + 1e-9
        semantic_score = dot / (q_norm * p_norm)

        avg_confidence = sum(edge.confidence for edge in candidate.edges) / max(len(candidate.edges), 1)
        avg_degree = sum(self.graph.degree(node) for node in candidate.nodes) / max(len(candidate.nodes), 1)
        length_penalty = 1.0 / max(len(candidate.edges), 1)
        importance_score = math.log1p(avg_degree) / 4.0

        return 0.45 * semantic_score + 0.25 * avg_confidence + 0.2 * importance_score + 0.1 * length_penalty

    def _score_doc_text_similarity(self, question: str, doc_text: str) -> float:
        question_emb = self.embedding_model.encode([question])[0]
        doc_emb = self.embedding_model.encode([doc_text])[0]
        dot = sum(a * b for a, b in zip(question_emb, doc_emb))
        q_norm = math.sqrt(sum(a * a for a in question_emb)) + 1e-9
        d_norm = math.sqrt(sum(a * a for a in doc_emb)) + 1e-9
        return dot / (q_norm * d_norm)

    def _hybrid_rescore_paths(
        self,
        question: str,
        paths: list[PathCandidate],
        retrieved_docs: list[tuple[str, float, dict[str, Any]]],
    ) -> list[PathCandidate]:
        if not paths:
            return []
        doc_sources = [str((meta or {}).get("source", "")) for _, _, meta in retrieved_docs]

        for candidate in paths:
            path_sources = {edge.doc_source for edge in candidate.edges if edge.doc_source}
            overlap = 0.0
            if path_sources and doc_sources:
                overlap_hits = sum(1 for src in doc_sources if src in path_sources)
                overlap = overlap_hits / len(doc_sources)
            candidate.hybrid_score = (1.0 - self.hybrid_path_vector_alpha) * candidate.score + self.hybrid_path_vector_alpha * overlap

        return sorted(paths, key=lambda item: item.hybrid_score, reverse=True)

    def _hybrid_rerank_docs(
        self,
        question: str,
        retrieved_docs: list[tuple[str, float, dict[str, Any]]],
        paths: list[PathCandidate],
    ) -> list[tuple[str, float, dict[str, Any], float]]:
        top_path_sources = set()
        for path in paths[: self.top_paths]:
            for edge in path.edges:
                if edge.doc_source:
                    top_path_sources.add(edge.doc_source)

        reranked = []
        for doc, distance, metadata in retrieved_docs:
            source = str((metadata or {}).get("source", ""))
            semantic = self._score_doc_text_similarity(question, doc)
            path_support = 1.0 if source and source in top_path_sources else 0.0
            hybrid_score = (1.0 - self.hybrid_doc_path_beta) * semantic + self.hybrid_doc_path_beta * path_support
            reranked.append((doc, distance, metadata, hybrid_score))

        reranked.sort(key=lambda item: item[3], reverse=True)
        return reranked

    @staticmethod
    def serialize_path(candidate: PathCandidate) -> str:
        if not candidate.edges:
            return ""
        segments = [candidate.nodes[0]]
        for edge in candidate.edges:
            segments.append(f"-[{edge.relation}]->{edge.target}")
        return "".join(segments)

    def build_path_context(self, paths: list[PathCandidate]) -> str:
        if not paths:
            return ""
        lines = []
        for idx, path in enumerate(paths, start=1):
            path_text = self.serialize_path(path)
            source_docs = sorted({edge.doc_source for edge in path.edges if edge.doc_source})
            source_text = "；".join(source_docs) if source_docs else "未知来源"
            lines.append(f"路径{idx}: {path_text} (score={path.score:.4f}, source={source_text})")
        return "\n".join(lines)

    def answer(self, question: str, vector_top_k: int = 3) -> dict[str, Any]:
        retrieved_docs = retrieve(
            query=question,
            collection=self.collection,
            embedding_model=self.embedding_model,
            top_k=max(vector_top_k, self.top_paths),
        )

        query_entities = self.link_query_entities(question)
        paths = self.retrieve_paths(question, query_entities)
        paths = self._hybrid_rescore_paths(question, paths, retrieved_docs)
        path_context = self.build_path_context(paths)

        reranked_docs = self._hybrid_rerank_docs(question, retrieved_docs, paths)
        selected_docs = reranked_docs[:vector_top_k]
        vector_context = "\n\n".join([doc for doc, _distance, _meta, _hyb in selected_docs])

        prompt = (
            "你是一名轨道交通政策问答助手。请根据给定问题、路径证据和文本证据回答。\n"
            "规则：\n"
            "1. 优先使用路径证据组织逻辑，再用文本证据补充事实。\n"
            "2. 不得编造，无法确定时回答：根据现有资料无法确定。\n"
            "3. 回答简洁且只输出最终答案。\n\n"
            f"[问题]\n{question}\n\n"
            f"[路径证据]\n{path_context or '无可用路径'}\n\n"
            f"[文本证据]\n{vector_context}\n\n"
            "[回答]"
        )

        answer_text = self.llm.invoke(prompt)
        return {
            "result": answer_text,
            "query_entities": query_entities,
            "paths": [
                {
                    "path": self.serialize_path(path),
                    "score": path.score,
                    "hybrid_score": path.hybrid_score,
                    "sources": sorted({edge.doc_source for edge in path.edges if edge.doc_source}),
                }
                for path in paths
            ],
            "path_context": path_context,
            "vector_context_docs": [
                {
                    "content": doc,
                    "distance": distance,
                    "metadata": metadata,
                    "hybrid_score": hybrid_score,
                }
                for doc, distance, metadata, hybrid_score in selected_docs
            ],
        }


@lru_cache(maxsize=1)
def build_pathrag_pipeline(
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    max_paths_per_seed: int = DEFAULT_PATHS_PER_SEED,
    enable_llm_triplet: bool = PATHRAG_ENABLE_LLM_TRIPLET,
    hybrid_path_vector_alpha: float = PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
    hybrid_doc_path_beta: float = PATHRAG_HYBRID_DOC_PATH_BETA,
) -> PathRAG:
    pipeline = create_pathrag_pipeline(
        max_hops=max_hops,
        top_paths=top_paths,
        max_paths_per_seed=max_paths_per_seed,
        enable_llm_triplet=enable_llm_triplet,
        hybrid_path_vector_alpha=hybrid_path_vector_alpha,
        hybrid_doc_path_beta=hybrid_doc_path_beta,
    )
    pipeline.load_or_build_graph(rebuild=False)
    return pipeline


def create_pathrag_pipeline(
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    max_paths_per_seed: int = DEFAULT_PATHS_PER_SEED,
    enable_llm_triplet: bool = PATHRAG_ENABLE_LLM_TRIPLET,
    hybrid_path_vector_alpha: float = PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
    hybrid_doc_path_beta: float = PATHRAG_HYBRID_DOC_PATH_BETA,
) -> PathRAG:
    collection = get_collection()
    embedding_model = get_embedding_model()
    llm = load_qwen_llm(max_new_tokens=MAX_NEW_TOKENS)

    pipeline = PathRAG(
        collection=collection,
        embedding_model=embedding_model,
        llm=llm,
        max_hops=max_hops,
        top_paths=top_paths,
        max_paths_per_seed=max_paths_per_seed,
        enable_llm_triplet=enable_llm_triplet,
        hybrid_path_vector_alpha=hybrid_path_vector_alpha,
        hybrid_doc_path_beta=hybrid_doc_path_beta,
    )
    return pipeline


def ask_pathrag(
    question: str,
    vector_top_k: int = 3,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    enable_llm_triplet: bool = PATHRAG_ENABLE_LLM_TRIPLET,
    hybrid_path_vector_alpha: float = PATHRAG_HYBRID_PATH_VECTOR_ALPHA,
    hybrid_doc_path_beta: float = PATHRAG_HYBRID_DOC_PATH_BETA,
) -> dict[str, Any]:
    pipeline = build_pathrag_pipeline(
        max_hops=max_hops,
        top_paths=top_paths,
        enable_llm_triplet=enable_llm_triplet,
        hybrid_path_vector_alpha=hybrid_path_vector_alpha,
        hybrid_doc_path_beta=hybrid_doc_path_beta,
    )
    return pipeline.answer(question=question, vector_top_k=vector_top_k)


def rebuild_path_graph(
    max_hops: int = DEFAULT_MAX_HOPS,
    top_paths: int = DEFAULT_TOP_PATHS,
    enable_llm_triplet: bool = PATHRAG_ENABLE_LLM_TRIPLET,
) -> None:
    pipeline = create_pathrag_pipeline(max_hops=max_hops, top_paths=top_paths, enable_llm_triplet=enable_llm_triplet)
    pipeline.load_or_build_graph(rebuild=True)

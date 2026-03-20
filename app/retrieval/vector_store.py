from functools import lru_cache
import json
import os
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import EMBEDDING_MODEL_NAME, HF_TOKEN, PDF_CONFIG
from app.ingestion.pdf_reader import extract_and_clean_all_pdf

FAISS_DB_PATH = "./faiss_db"
INDEX_FILE_NAME = "traffic_docs.faiss"
META_FILE_NAME = "traffic_docs_meta.json"


@dataclass
class _StoredDoc:
    doc_id: str
    document: str
    metadata: dict[str, Any]


class FaissCollection:
    """A tiny Chroma-compatible wrapper backed by FAISS."""

    def __init__(self, db_path: str = FAISS_DB_PATH) -> None:
        self.db_path = db_path
        self.index_path = os.path.join(db_path, INDEX_FILE_NAME)
        self.meta_path = os.path.join(db_path, META_FILE_NAME)

        self._entries: list[_StoredDoc] = []
        self._ids: set[str] = set()
        self._index: faiss.IndexFlatIP | None = None
        self._dimension: int | None = None

        self._load()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(dimension)
            self._dimension = dimension
            return
        if self._dimension != dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dimension}, got {dimension}")

    def _active_index(self) -> Any:
        if self._index is None:
            raise RuntimeError("FAISS index has not been initialized")
        # faiss Python bindings are dynamically typed; use runtime object for stable calls.
        return self._index

    def _load(self) -> None:
        os.makedirs(self.db_path, exist_ok=True)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._dimension = data.get("dimension")
            self._entries = [
                _StoredDoc(
                    doc_id=item["id"],
                    document=item["document"],
                    metadata=item.get("metadata") or {},
                )
                for item in data.get("entries", [])
            ]
            self._ids = {entry.doc_id for entry in self._entries}

        if os.path.exists(self.index_path):
            self._index = faiss.read_index(self.index_path)
            if self._dimension is None:
                index = self._active_index()
                self._dimension = int(index.d)

        if self._index is not None and self._index.ntotal != len(self._entries):
            raise RuntimeError("FAISS index and metadata count mismatch; please rebuild index.")

    def _persist(self) -> None:
        if self._index is None:
            return

        faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self._dimension,
                    "entries": [
                        {
                            "id": entry.doc_id,
                            "document": entry.document,
                            "metadata": entry.metadata,
                        }
                        for entry in self._entries
                    ],
                },
                f,
                ensure_ascii=False,
            )

    def count(self) -> int:
        return len(self._entries)

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not (len(documents) == len(embeddings) == len(ids)):
            raise ValueError("documents, embeddings and ids must have the same length")

        if metadatas is None:
            metadatas = [{} for _ in documents]

        if len(metadatas) != len(documents):
            raise ValueError("metadatas length must match documents length")

        pending_vectors = []
        pending_entries = []

        for doc, emb, doc_id, metadata in zip(documents, embeddings, ids, metadatas):
            if doc_id in self._ids:
                continue
            vector = np.asarray(emb, dtype=np.float32)
            pending_vectors.append(vector)
            pending_entries.append(
                _StoredDoc(
                    doc_id=doc_id,
                    document=doc,
                    metadata=metadata or {},
                )
            )

        if not pending_vectors:
            return

        vectors = np.vstack(pending_vectors).astype(np.float32)
        self._ensure_index(vectors.shape[1])
        normalized_vectors = self._normalize(vectors)
        index = self._active_index()
        index.add(normalized_vectors)

        for entry in pending_entries:
            self._entries.append(entry)
            self._ids.add(entry.doc_id)

        self._persist()

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        include = include or []

        if self._index is None or self.count() == 0:
            return {
                "documents": [[] for _ in query_embeddings],
                "distances": [[] for _ in query_embeddings],
                "metadatas": [[] for _ in query_embeddings],
            }

        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self._dimension:
            raise ValueError(f"Query embedding dimension mismatch: expected {self._dimension}, got {queries.shape[1]}")

        normalized_queries = self._normalize(queries)
        top_k = min(max(n_results, 1), self.count())
        index = self._active_index()
        similarities, indices = index.search(normalized_queries, top_k)

        all_docs: list[list[str]] = []
        all_distances: list[list[float]] = []
        all_metadatas: list[list[dict[str, Any]]] = []

        for sims, ids in zip(similarities, indices):
            docs = []
            distances = []
            metas = []
            for sim, idx in zip(sims.tolist(), ids.tolist()):
                if idx < 0 or idx >= len(self._entries):
                    continue
                entry = self._entries[idx]
                docs.append(entry.document)
                distances.append(float(1.0 - sim))
                metas.append(dict(entry.metadata))

            all_docs.append(docs if "documents" in include else [])
            all_distances.append(distances if "distances" in include else [])
            all_metadatas.append(metas if "metadatas" in include else [])

        return {
            "documents": all_docs,
            "distances": all_distances,
            "metadatas": all_metadatas,
        }

    def get(self, limit: int, offset: int = 0, include: list[str] | None = None) -> dict[str, list[Any]]:
        include = include or []
        subset = self._entries[offset : offset + limit]
        return {
            "documents": [entry.document for entry in subset] if "documents" in include else [],
            "metadatas": [dict(entry.metadata) for entry in subset] if "metadatas" in include else [],
        }


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, token=HF_TOKEN,device="cuda")


@lru_cache(maxsize=1)
def get_collection():
    return FaissCollection(db_path=FAISS_DB_PATH)


def retrieve(query, collection, embedding_model, top_k=5):
    query_embeddings = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    return list(zip(docs, distances, metadatas))


def build_index(collection, embedding_model, batch_size=100):
    pdf_config = PDF_CONFIG()
    all_chunks, chunk_sources = extract_and_clean_all_pdf(pdf_config)

    for start in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[start : start + batch_size]
        batch_ids = [f"chunk_{index}" for index in range(start, start + len(batch_chunks))]
        embeddings = embedding_model.encode(batch_chunks).tolist()
        collection.add(
            documents=batch_chunks,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=[{"source": chunk_sources[index]} for index in range(start, start + len(batch_chunks))],
        )

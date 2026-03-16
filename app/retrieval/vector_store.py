from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.config.settings import EMBEDDING_MODEL_NAME, HF_TOKEN, PDF_CONFIG
from app.ingestion.pdf_reader import extract_and_clean_all_pdf

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "traffic_docs"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, token=HF_TOKEN)


@lru_cache(maxsize=1)
def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


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

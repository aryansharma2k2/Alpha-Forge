"""
Vector store client (e.g. Qdrant, Pinecone, pgvector).
"""


def upsert(vectors: list, metadata: list[dict]) -> None:
    raise NotImplementedError


def search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    raise NotImplementedError

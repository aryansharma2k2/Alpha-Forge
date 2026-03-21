"""
High-level retriever: embed query → search vector store → return chunks.
"""


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    raise NotImplementedError

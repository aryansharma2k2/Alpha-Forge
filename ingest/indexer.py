"""
Qdrant indexer — embeds chunks and upserts them into a vector collection.

Embedding model : OpenAI text-embedding-3-small  (dim = 1536)
Vector store    : Qdrant

Each Qdrant point stores:
    id          – deterministic UUID derived from (ticker, filing_type, filed_date, chunk_index)
    vector      – 1536-d float embedding
    payload     – ticker, company, source_type, date, period, chunk_id, url,
                  chunk_index, text

Public surface:
    index_chunks(chunks, *, collection)         embed + upsert a list of ChunkResult
    refresh_collection(ticker, *, collection)   delete + re-index all docs for a company
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Sequence

import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest.chunker import ChunkResult

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM   = 1536
_EMBED_BATCH     = 100          # max texts per OpenAI embeddings call
_UPSERT_BATCH    = 200          # max points per Qdrant upsert call
_DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financial_docs")

# ── Clients (lazy singletons) ─────────────────────────────────────────────────

_qdrant_client: QdrantClient | None = None
_openai_client: openai.OpenAI | None = None


def _qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


def _openai() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# ── Collection bootstrap ──────────────────────────────────────────────────────


def _ensure_collection(name: str) -> None:
    """Create the Qdrant collection if it does not already exist."""
    client = _qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return

    logger.info("Creating Qdrant collection '%s' (dim=%d)", name, _EMBEDDING_DIM)
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=_EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # Payload index on ticker for fast filtered deletes / searches
    for field_name in ("ticker", "source_type", "date"):
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chunk_id(ticker: str, filing_type: str, filed_date: str, chunk_index: int) -> str:
    """
    Deterministic UUID-v5 so re-indexing the same filing overwrites existing
    points rather than creating duplicates.
    """
    key = f"{ticker.upper()}/{filing_type}/{filed_date}/{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (≤ _EMBED_BATCH) via the OpenAI API."""
    resp = _openai().embeddings.create(model=_EMBEDDING_MODEL, input=texts)
    # Response items are ordered to match input
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def _embed_all(texts: list[str]) -> list[list[float]]:
    """Embed an arbitrary-length list by splitting into _EMBED_BATCH batches."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        logger.debug("Embedding batch %d–%d / %d", i, i + len(batch), len(texts))
        vectors.extend(_embed_batch(batch))
    return vectors


def _build_points(
    chunks: Sequence[ChunkResult],
    vectors: list[list[float]],
) -> list[qmodels.PointStruct]:
    points = []
    for chunk, vector in zip(chunks, vectors):
        cid = _chunk_id(chunk.ticker, chunk.filing_type, chunk.filed_date, chunk.chunk_index)
        points.append(
            qmodels.PointStruct(
                id=cid,
                vector=vector,
                payload={
                    "ticker":       chunk.ticker,
                    "company":      chunk.company,
                    "source_type":  chunk.filing_type,   # "10-K" | "10-Q"
                    "date":         chunk.filed_date,
                    "period":       chunk.period,
                    "chunk_id":     cid,
                    "url":          chunk.source_url,
                    "chunk_index":  chunk.chunk_index,
                    "text":         chunk.text,
                },
            )
        )
    return points


# ── Public API ────────────────────────────────────────────────────────────────


def index_chunks(
    chunks: Sequence[ChunkResult],
    *,
    collection: str = _DEFAULT_COLLECTION,
) -> int:
    """
    Embed *chunks* and upsert them into Qdrant.

    Args:
        chunks:     Sequence of ChunkResult objects (from ``ingest.chunker``).
        collection: Qdrant collection name (defaults to $QDRANT_COLLECTION).

    Returns:
        Number of points upserted.

    Raises:
        openai.OpenAIError:  Embedding API failure.
        qdrant_client.*:     Any Qdrant transport / validation error.
    """
    if not chunks:
        logger.warning("index_chunks called with empty chunk list — nothing to do")
        return 0

    _ensure_collection(collection)

    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunk(s) with %s …", len(texts), _EMBEDDING_MODEL)
    vectors = _embed_all(texts)

    points = _build_points(chunks, vectors)

    client = _qdrant()
    upserted = 0
    for i in range(0, len(points), _UPSERT_BATCH):
        batch = points[i : i + _UPSERT_BATCH]
        client.upsert(collection_name=collection, points=batch, wait=True)
        upserted += len(batch)
        logger.debug("Upserted %d / %d points", upserted, len(points))

    logger.info("Indexed %d chunk(s) into collection '%s'", upserted, collection)
    return upserted


def refresh_collection(
    ticker: str,
    *,
    collection: str = _DEFAULT_COLLECTION,
    n_each: int = 1,
) -> int:
    """
    Delete all existing points for *ticker* and re-index from EDGAR.

    Imports the ingest pipeline lazily to avoid circular imports at module
    load time.  The full flow is:

        EDGAR → fetch_filings → parser.clean → chunker.chunk → index_chunks

    Args:
        ticker:     Stock ticker, e.g. "AAPL".
        collection: Qdrant collection name.
        n_each:     Number of 10-K and 10-Q filings to fetch per type.

    Returns:
        Total number of points upserted after refresh.
    """
    # Lazy imports — keeps module load fast and avoids circular deps
    from ingest.sec_fetcher import fetch_filings
    from ingest.parser import clean
    from ingest.chunker import chunk

    _ensure_collection(collection)

    # 1. Delete all existing points for this ticker
    logger.info("Deleting existing points for ticker '%s' from '%s'", ticker, collection)
    _qdrant().delete(
        collection_name=collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="ticker",
                        match=qmodels.MatchValue(value=ticker.upper()),
                    )
                ]
            )
        ),
        wait=True,
    )

    # 2. Fetch fresh filings from EDGAR
    logger.info("Fetching %d filing(s) per type for '%s' from EDGAR …", n_each, ticker)
    filings = fetch_filings(ticker, n_each=n_each)

    # 3. Clean → chunk → index each filing
    total = 0
    for filing in filings:
        logger.info(
            "Processing %s filed %s (%s) …",
            filing.filing_type, filing.filed_date, filing.company,
        )
        cleaned = clean(filing.text)
        chunks = chunk(
            cleaned,
            metadata={
                "ticker":       filing.ticker,
                "company":      filing.company,
                "filing_type":  filing.filing_type,
                "filed_date":   filing.filed_date,
                "period":       filing.period,
                "url":          filing.url,
            },
        )
        total += index_chunks(chunks, collection=collection)

    logger.info(
        "refresh_collection('%s'): %d total point(s) indexed into '%s'",
        ticker, total, collection,
    )
    return total

"""
Token-aware sliding-window chunker.

Splits cleaned filing text into overlapping windows of *chunk_tokens* tokens
with *overlap_tokens* of context carried forward. Uses tiktoken so token
counts match what the embedding model and LLM will see.

Each chunk is returned as a ChunkResult dataclass that carries both the
text slice and the full filing metadata needed downstream (vector store
upserts, citation rendering, eval harness).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import tiktoken

logger = logging.getLogger(__name__)

# Encoding used by OpenAI text-embedding-3-* and Claude-compatible tokenisers.
# Switch to "p50k_base" if you use an older OpenAI model.
_ENCODING_NAME = "cl100k_base"

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ChunkResult:
    # Slice content
    text: str
    chunk_index: int
    token_count: int

    # Filing provenance (passed through from FilingResult or equivalent dict)
    ticker: str
    company: str
    filing_type: str    # "10-K" | "10-Q"
    filed_date: str     # "YYYY-MM-DD"
    period: str         # reporting period end date
    source_url: str     # canonical EDGAR document URL

    # Optional extra metadata callers may attach
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Flat dict suitable for vector store metadata payloads."""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "ticker": self.ticker,
            "company": self.company,
            "filing_type": self.filing_type,
            "filed_date": self.filed_date,
            "period": self.period,
            "source_url": self.source_url,
            **self.extra,
        }


# ── Chunker ───────────────────────────────────────────────────────────────────


def chunk(
    text: str,
    *,
    metadata: dict,
    chunk_tokens: int = 512,
    overlap_tokens: int = 128,
    encoding_name: str = _ENCODING_NAME,
) -> list[ChunkResult]:
    """
    Chunk *text* into overlapping token windows and attach *metadata*.

    Args:
        text:           Cleaned filing text (output of ``ingest.parser.clean``).
        metadata:       Dict with keys: ticker, company, filing_type,
                        filed_date, period, source_url (url).
                        Any additional keys are stored in ChunkResult.extra.
        chunk_tokens:   Target window size in tokens (default 512).
        overlap_tokens: Tokens of overlap between consecutive chunks (default 128).
        encoding_name:  tiktoken encoding name (default "cl100k_base").

    Returns:
        Ordered list of ChunkResult objects, one per window.

    Raises:
        ValueError: If overlap_tokens >= chunk_tokens.
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < chunk_tokens ({chunk_tokens})"
        )

    enc = tiktoken.get_encoding(encoding_name)

    # Encode the full document once
    token_ids: list[int] = enc.encode(text, disallowed_special=())
    total_tokens = len(token_ids)

    if total_tokens == 0:
        logger.warning("chunker received empty text — returning no chunks")
        return []

    logger.debug(
        "Chunking %d tokens → windows of %d with %d overlap",
        total_tokens, chunk_tokens, overlap_tokens,
    )

    # Extract known metadata keys; remainder goes to extra
    known_keys = {"ticker", "company", "filing_type", "filed_date", "period", "source_url", "url"}
    source_url = metadata.get("source_url") or metadata.get("url", "")
    extra = {k: v for k, v in metadata.items() if k not in known_keys}

    base_meta = dict(
        ticker=metadata.get("ticker", ""),
        company=metadata.get("company", ""),
        filing_type=metadata.get("filing_type", ""),
        filed_date=metadata.get("filed_date", ""),
        period=metadata.get("period", ""),
        source_url=source_url,
    )

    step = chunk_tokens - overlap_tokens
    chunks: list[ChunkResult] = []
    start = 0
    index = 0

    while start < total_tokens:
        end = min(start + chunk_tokens, total_tokens)
        window_ids = token_ids[start:end]
        window_text = enc.decode(window_ids)

        chunks.append(
            ChunkResult(
                text=window_text,
                chunk_index=index,
                token_count=len(window_ids),
                extra=extra,
                **base_meta,
            )
        )

        if end == total_tokens:
            break

        start += step
        index += 1

    logger.info(
        "Produced %d chunk(s) from %d tokens (window=%d, overlap=%d)",
        len(chunks), total_tokens, chunk_tokens, overlap_tokens,
    )
    return chunks

"""
SEC EDGAR fetcher — retrieves the latest 10-K and 10-Q filings for a ticker.

Flow:
  1. Search EDGAR EFTS for filings matching the ticker symbol.
  2. For each hit, derive the document URL from the accession number + CIK.
  3. Fetch the raw document text.
  4. Return structured FilingResult objects.

All HTTP calls use a 30-second timeout and up to 4 attempts with
exponential backoff (1 s → 2 s → 4 s → 8 s).
"""

import logging
import re
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_EFTS_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# SEC requires a descriptive User-Agent: "<app> <contact-email>"
_HEADERS = {"User-Agent": "financial-rag research@example.com"}

FILING_TYPES = ("10-K", "10-Q")

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class FilingResult:
    ticker: str
    company: str
    filing_type: str        # "10-K" or "10-Q"
    filed_date: str         # "YYYY-MM-DD"
    period: str             # reporting period end date
    url: str                # canonical EDGAR document URL
    text: str               # raw extracted text


# ── HTTP helper ───────────────────────────────────────────────────────────────


def _get(
    url: str,
    *,
    params: dict | None = None,
    retries: int = 4,
    timeout: float = 30.0,
) -> httpx.Response:
    """
    HTTP GET with exponential backoff.

    Raises the final exception after *retries* failed attempts.
    """
    delay = 1.0
    last_exc: Exception = RuntimeError("no attempts made")

    for attempt in range(retries):
        try:
            resp = httpx.get(
                url,
                params=params,
                headers=_HEADERS,
                timeout=timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            logger.warning(
                "Request to %s failed (attempt %d/%d): %s — retrying in %.0fs",
                url, attempt + 1, retries, exc, delay,
            )
            time.sleep(delay)
            delay *= 2

    raise last_exc


# ── EFTS search ───────────────────────────────────────────────────────────────


def _search_efts(ticker: str, n_each: int) -> list[dict]:
    """
    Query the EDGAR full-text search index for recent 10-K / 10-Q filings.

    Returns a list of raw hit dicts from the EFTS response, capped at
    *n_each* per filing type.
    """
    params = {
        "q": f'"{ticker}"',
        "forms": ",".join(FILING_TYPES),
        "hits.hits._source": (
            "period_of_report,entity_name,file_date,form_type,"
            "entity_id,file_num,accession_no"
        ),
    }

    data = _get(_EFTS_SEARCH, params=params).json()
    hits = data.get("hits", {}).get("hits", [])

    selected: list[dict] = []
    counts = {ft: 0 for ft in FILING_TYPES}

    for hit in hits:
        src = hit.get("_source", {})
        form = src.get("form_type", "")
        if form not in FILING_TYPES:
            continue
        if counts[form] >= n_each:
            continue
        counts[form] += 1
        selected.append(hit)
        if all(v >= n_each for v in counts.values()):
            break

    return selected


# ── URL construction ──────────────────────────────────────────────────────────


def _doc_url(cik: str, accession_no: str, primary_doc: str) -> str:
    """
    Build the canonical EDGAR archive URL for a primary document.

    e.g. https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm
    """
    acc_nodash = accession_no.replace("-", "")
    return f"{_ARCHIVES_BASE}/{int(cik)}/{acc_nodash}/{primary_doc}"


def _index_url(cik: str, accession_no: str) -> str:
    """Return the filing index page URL (used to discover primary_doc)."""
    acc_nodash = accession_no.replace("-", "")
    return f"{_ARCHIVES_BASE}/{int(cik)}/{acc_nodash}/{accession_no}-index.json"


def _resolve_primary_doc(cik: str, accession_no: str) -> str:
    """
    Fetch the filing index JSON and return the name of the primary document.

    Falls back to the accession-derived HTM name if the index is unavailable.
    """
    try:
        index = _get(_index_url(cik, accession_no)).json()
        for item in index.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if name.lower().endswith((".htm", ".html")) and "index" not in name.lower():
                return name
    except Exception as exc:
        logger.warning("Could not fetch index for %s: %s", accession_no, exc)

    # Fallback: EDGAR sometimes uses <accession-no>.htm as the primary doc
    return accession_no + ".htm"


# ── Text extraction ───────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t]{2,}")


def _extract_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", html)
    text = _SPACE_RE.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _fetch_text(url: str) -> str:
    """Download a filing document and return its plain text."""
    content = _get(url).text
    if "<html" in content[:2000].lower():
        return _extract_text(content)
    return content


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_filings(ticker: str, n_each: int = 1) -> list[FilingResult]:
    """
    Fetch the *n_each* most recent 10-K and 10-Q filings for *ticker*.

    Args:
        ticker:  Stock ticker symbol, e.g. "AAPL".
        n_each:  Number of filings to retrieve per filing type (default 1).

    Returns:
        List of FilingResult objects ordered newest-first within each type.

    Raises:
        ValueError:            No filings found for the ticker.
        httpx.HTTPStatusError: Non-2xx response after all retries exhausted.
    """
    logger.info("Searching EDGAR for %s filings: %s", ticker, FILING_TYPES)
    hits = _search_efts(ticker, n_each)

    if not hits:
        raise ValueError(f"No 10-K / 10-Q filings found for ticker '{ticker}' on EDGAR.")

    results: list[FilingResult] = []

    for hit in hits:
        src = hit["_source"]

        company      = src.get("entity_name", "Unknown")
        filing_type  = src.get("form_type", "")
        filed_date   = src.get("file_date", "")
        period       = src.get("period_of_report", "")
        cik          = str(src.get("entity_id", "")).lstrip("0") or "0"
        accession_no = src.get("accession_no") or hit.get("_id", "")

        primary_doc = _resolve_primary_doc(cik, accession_no)
        url = _doc_url(cik, accession_no, primary_doc)

        logger.info("Fetching %s filed %s for %s …", filing_type, filed_date, company)
        text = _fetch_text(url)

        results.append(
            FilingResult(
                ticker=ticker.upper(),
                company=company,
                filing_type=filing_type,
                filed_date=filed_date,
                period=period,
                url=url,
                text=text,
            )
        )

    return results

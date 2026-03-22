"""
SEC filing text cleaner.

Takes raw text from EDGAR (HTML or plain-text SGML) and returns clean,
readable prose ready for chunking. Removes:
  - EDGAR SGML envelope headers / footers
  - HTML / XML tags and entities
  - XBRL inline tags
  - Repeated separator lines
  - Boilerplate cover-page and exhibit-index patterns
  - Excessive whitespace and blank lines
"""

import html
import re

# ── Compiled patterns (build once) ───────────────────────────────────────────

# EDGAR SGML envelope: everything between <SEC-HEADER>…</SEC-HEADER> and
# the privacy-enhanced-message banner at top/bottom.
_SGML_HEADER = re.compile(
    r"-----BEGIN PRIVACY-ENHANCED MESSAGE-----.*?-----END PRIVACY-ENHANCED MESSAGE-----",
    re.DOTALL,
)
_SEC_HEADER_BLOCK = re.compile(
    r"<SEC-HEADER>.*?</SEC-HEADER>", re.DOTALL | re.IGNORECASE
)
_SGML_TAGS = re.compile(
    r"<(?:DOCUMENT|TYPE|SEQUENCE|FILENAME|DESCRIPTION|TEXT|/DOCUMENT|/TEXT)>[^\n]*",
    re.IGNORECASE,
)

# Inline XBRL / iXBRL tags (keeps the text node)
_XBRL_TAG = re.compile(
    r"</?(?:ix|xbrli?|link|label|ref|schema|context|unit|measure|"
    r"startDate|endDate|instant|identifier|segment|entity|period|"
    r"scenario|decimals|scale|format|name|contextRef|unitRef|"
    r"nonNumeric|nonFraction|fraction|numerator|denominator|"
    r"xbrl[^>]*)?:[^>]*>",
    re.IGNORECASE,
)

# Generic HTML / XML tags
_HTML_TAG = re.compile(r"<[^>]{0,400}>", re.DOTALL)

# HTML entities (&amp; &nbsp; &#160; etc.)
_HTML_ENTITY = re.compile(r"&(?:#\d+|#x[\da-fA-F]+|[a-zA-Z]+);")

# Separator lines: rows of dashes, equals, underscores, stars (≥4 chars)
_SEPARATOR = re.compile(r"^[\-=_*#]{4,}\s*$", re.MULTILINE)

# Page numbers: "- 12 -" / "Page 12" / "F-12" on their own line
_PAGE_NUMBER = re.compile(
    r"^\s*(?:-\s*\d+\s*-|Page\s+\d+(?:\s+of\s+\d+)?|F-\d+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Common boilerplate phrases (full lines)
_BOILERPLATE_LINE = re.compile(
    r"^\s*(?:"
    r"this page intentionally left blank"
    r"|table of contents"
    r"|index to financial statements"
    r"|see accompanying notes"
    r"|see notes to (?:consolidated )?financial statements"
    r"|the accompanying notes (?:are|form) an integral part"
    r"|(?:filed (?:pursuant to|as part of)|incorporated herein by reference)"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# EDGAR filing header lines that leak into body (e.g. "FORM 10-K\nCIK: 0000320193")
_EDGAR_META_LINE = re.compile(
    r"^\s*(?:CIK|ACCESSION NUMBER|CONFORMED SUBMISSION TYPE|"
    r"FILED AS OF DATE|DATE AS OF CHANGE|EFFECTIVENESS DATE|"
    r"CENTRAL INDEX KEY|STANDARD INDUSTRIAL CLASSIFICATION|"
    r"IRS NUMBER|STATE OF INCORPORATION|FISCAL YEAR END|"
    r"FILER|COMPANY DATA|FILING VALUES|BUSINESS ADDRESS|"
    r"MAIL ADDRESS|FORMER COMPANY|FORMER NAME|DATE OF NAME CHANGE)"
    r"\s*:.*$",
    re.MULTILINE | re.IGNORECASE,
)

# Collapse runs of 3+ blank lines to single blank line
_MULTI_BLANK = re.compile(r"\n{3,}")

# Collapse inline whitespace (tabs, multiple spaces)
_INLINE_SPACE = re.compile(r"[ \t]{2,}")


# ── Public API ────────────────────────────────────────────────────────────────


def clean(raw: str) -> str:
    """
    Clean raw SEC filing text and return normalised prose.

    Args:
        raw: Raw string from EDGAR (HTML, SGML, or pre-stripped plain text).

    Returns:
        Clean plain text with boilerplate and markup removed.
    """
    text = raw

    # 1. Strip privacy-enhanced message wrapper
    text = _SGML_HEADER.sub("", text)

    # 2. Drop SGML <SEC-HEADER> block
    text = _SEC_HEADER_BLOCK.sub("", text)

    # 3. Drop SGML document envelope tags
    text = _SGML_TAGS.sub("", text)

    # 4. Drop inline XBRL tags (before generic HTML so the pattern is tighter)
    text = _XBRL_TAG.sub(" ", text)

    # 5. Drop all remaining HTML / XML tags
    text = _HTML_TAG.sub(" ", text)

    # 6. Decode HTML entities (&amp; becomes &, &nbsp; becomes space, etc.)
    text = html.unescape(text)

    # 7. Remove boilerplate lines
    text = _BOILERPLATE_LINE.sub("", text)
    text = _EDGAR_META_LINE.sub("", text)
    text = _PAGE_NUMBER.sub("", text)
    text = _SEPARATOR.sub("", text)

    # 8. Normalise whitespace
    text = _INLINE_SPACE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # 9. Collapse excessive blank lines
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()

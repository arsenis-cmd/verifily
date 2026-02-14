"""Unified PII detection for Verifily.

Consolidates duplicate PII patterns from report.py, ingest.py, and classify.py.
Supports regex detection (always available) and optional NER via spaCy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Canonical PII patterns ────────────────────────────────────────

PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}

REGEX_CONFIDENCE: Dict[str, float] = {
    "email": 0.95,
    "phone": 0.60,
    "ssn": 0.85,
    "ip_address": 0.50,
    "credit_card": 0.80,
}

# NER entity → PII type mapping
_NER_ENTITY_MAP: Dict[str, str] = {
    "PERSON": "person_name",
    "ORG": "organization",
    "GPE": "location",
}

_NER_CONFIDENCE: Dict[str, float] = {
    "person_name": 0.85,
    "organization": 0.70,
    "location": 0.60,
}


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class PIIMatch:
    """A single PII detection."""

    pii_type: str
    confidence: float
    method: str  # "regex" or "ner"


@dataclass
class PIIScanResult:
    """Result of scanning text for PII."""

    matches: List[PIIMatch] = field(default_factory=list)
    pii_types_found: set = field(default_factory=set)

    @property
    def has_pii(self) -> bool:
        return len(self.matches) > 0


# ── spaCy loader ──────────────────────────────────────────────────


def _load_spacy():
    """Try to load spaCy with en_core_web_sm. Returns None if unavailable."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return None
    except ImportError:
        return None


# ── Scanning functions ────────────────────────────────────────────


def scan_text_regex(text: str) -> PIIScanResult:
    """Scan text for PII using regex patterns only."""
    result = PIIScanResult()
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            result.matches.append(PIIMatch(
                pii_type=pii_type,
                confidence=REGEX_CONFIDENCE.get(pii_type, 0.5),
                method="regex",
            ))
            result.pii_types_found.add(pii_type)
    return result


def scan_text_ner(text: str, nlp=None) -> PIIScanResult:
    """Scan text for PII using regex + spaCy NER (if available).

    Falls back to regex-only if spaCy is not installed or model is missing.
    """
    result = scan_text_regex(text)

    if nlp is None:
        nlp = _load_spacy()
    if nlp is None:
        return result

    doc = nlp(text[:10_000])  # Cap text length for NER performance
    for ent in doc.ents:
        pii_type = _NER_ENTITY_MAP.get(ent.label_)
        if pii_type and pii_type not in result.pii_types_found:
            result.matches.append(PIIMatch(
                pii_type=pii_type,
                confidence=_NER_CONFIDENCE.get(pii_type, 0.5),
                method="ner",
            ))
            result.pii_types_found.add(pii_type)

    return result


def scan_row(
    row: Dict[str, Any],
    *,
    use_ner: bool = False,
    nlp=None,
) -> PIIScanResult:
    """Scan all text fields in a row for PII."""
    text = " ".join(
        str(v) for k, v in row.items()
        if k != "tags" and isinstance(v, (str, int, float))
    )
    # Also scan list fields (e.g. messages)
    for v in row.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for vv in item.values():
                        if isinstance(vv, str):
                            text += " " + vv
                elif isinstance(item, str):
                    text += " " + item

    if use_ner:
        return scan_text_ner(text, nlp=nlp)
    return scan_text_regex(text)


def scan_dataset(
    rows: List[Dict[str, Any]],
    *,
    use_ner: bool = False,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """Scan a dataset for PII. Returns format compatible with existing output.

    Returns::

        {
            "pii_scan": {type: {"count": int, "rows": [int]}},
            "pii_total_hits": int,
            "pii_clean": bool,
            "pii_confidence": {type: float},   # new field
        }
    """
    nlp = None
    if use_ner:
        nlp = _load_spacy()

    # Accumulate per-type results
    type_rows: Dict[str, List[int]] = {}
    type_confidence: Dict[str, float] = {}

    for i, row in enumerate(rows):
        result = scan_row(row, use_ner=use_ner, nlp=nlp)
        for match in result.matches:
            if match.confidence < min_confidence:
                continue
            if match.pii_type not in type_rows:
                type_rows[match.pii_type] = []
                type_confidence[match.pii_type] = match.confidence
            type_rows[match.pii_type].append(i)

    pii_scan: Dict[str, Dict[str, Any]] = {}
    for pii_type in sorted(type_rows.keys()):
        pii_scan[pii_type] = {
            "count": len(type_rows[pii_type]),
            "rows": type_rows[pii_type][:10],
        }

    total_hits = sum(d["count"] for d in pii_scan.values())

    return {
        "pii_scan": pii_scan,
        "pii_total_hits": total_hits,
        "pii_clean": total_hits == 0,
        "pii_confidence": type_confidence,
    }

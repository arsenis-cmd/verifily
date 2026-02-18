"""Centralized schema registry for Verifily.

Single source of truth for schema definitions, replacing fragmented definitions
across ingest.py, contract_check.py, and classify.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass(frozen=True)
class SchemaDefinition:
    """Immutable schema descriptor."""

    name: str
    # At least one group of fields must be present for a row to be valid.
    required_groups: tuple  # tuple of tuples of str
    optional: tuple = ()
    # {alt_field: canonical_field} mappings for contract validation
    alternates: tuple = ()  # tuple of dicts
    # Sets of field names that identify this schema during auto-detect
    detection_signatures: tuple = ()  # tuple of frozensets
    # Canonical output field names (after canonicalization)
    canonical_fields: tuple = ()
    # Fields to compute text statistics on (report)
    text_fields: tuple = ()
    # Fields to extract text from for contamination checking
    content_fields: tuple = ()


# ── Registry ──────────────────────────────────────────────────────

SCHEMA_REGISTRY: Dict[str, SchemaDefinition] = {}


def _register(schema: SchemaDefinition) -> None:
    SCHEMA_REGISTRY[schema.name] = schema


_register(SchemaDefinition(
    name="sft",
    required_groups=(("input", "output"), ("question", "answer")),
    optional=("tags", "id", "context"),
    alternates=(
        {"question": "instruction", "answer": "output", "context": "input"},
        {"input": "instruction", "output": "output"},
        {"prompt": "input", "completion": "output"},
    ),
    detection_signatures=(
        frozenset({"input", "output"}),
        frozenset({"instruction", "output"}),
        frozenset({"instruction", "response"}),
        frozenset({"prompt", "completion"}),
        frozenset({"question", "answer"}),
    ),
    canonical_fields=("input", "output"),
    text_fields=("input", "output"),
    content_fields=("input", "output", "instruction", "question", "answer", "context"),
))

_register(SchemaDefinition(
    name="qa",
    required_groups=(("question", "context", "answer"),),
    optional=("tags", "id"),
    alternates=(
        {"passage": "context"},
    ),
    detection_signatures=(
        frozenset({"question", "context", "answer"}),
        frozenset({"passage", "question", "answer"}),
    ),
    canonical_fields=("input", "output"),
    text_fields=("input", "output"),
    content_fields=("question", "context", "answer", "passage"),
))

_register(SchemaDefinition(
    name="classification",
    required_groups=(("text", "label"),),
    optional=("tags", "id"),
    alternates=(
        {"body": "text", "category": "label"},
        {"input": "text"},
        {"sentence": "text"},
    ),
    detection_signatures=(
        frozenset({"text", "label"}),
        frozenset({"body", "category"}),
        frozenset({"input", "label"}),
        frozenset({"sentence", "label"}),
        frozenset({"text", "class"}),
    ),
    canonical_fields=("text", "label"),
    text_fields=("text",),
    content_fields=("text", "body", "sentence", "input"),
))

_register(SchemaDefinition(
    name="chat",
    required_groups=(("messages",),),
    optional=("tags", "id", "conversations"),
    alternates=(
        {"conversations": "messages"},
    ),
    detection_signatures=(
        frozenset({"messages"}),
        frozenset({"conversations"}),
    ),
    canonical_fields=("input", "output"),
    text_fields=("input", "output"),
    content_fields=("messages", "conversations"),
))

_register(SchemaDefinition(
    name="summarization",
    required_groups=(("document", "summary"),),
    optional=("tags", "id"),
    alternates=(
        {"article": "document", "highlights": "summary"},
        {"text": "document", "summary": "summary"},
        {"source": "document", "target": "summary"},
    ),
    detection_signatures=(
        frozenset({"document", "summary"}),
        frozenset({"article", "highlights"}),
        frozenset({"article", "summary"}),
    ),
    canonical_fields=("input", "output"),
    text_fields=("input", "output"),
    content_fields=("document", "summary", "article", "highlights", "source", "target"),
))

_register(SchemaDefinition(
    name="translation",
    required_groups=(("source", "target"),),
    optional=("tags", "id", "source_lang", "target_lang"),
    alternates=(
        {"src", "tgt"},
    ),
    detection_signatures=(
        frozenset({"source", "target", "source_lang"}),
        frozenset({"source", "target", "target_lang"}),
        frozenset({"src", "tgt"}),
    ),
    canonical_fields=("input", "output"),
    text_fields=("input", "output"),
    content_fields=("source", "target", "src", "tgt"),
))

_register(SchemaDefinition(
    name="rm_pairwise",
    required_groups=(("prompt", "chosen", "rejected"),),
    optional=("tags", "id"),
    alternates=(),
    detection_signatures=(
        frozenset({"prompt", "chosen", "rejected"}),
    ),
    canonical_fields=("prompt", "chosen", "rejected"),
    text_fields=("prompt", "chosen", "rejected"),
    content_fields=("prompt", "chosen", "rejected"),
))

_register(SchemaDefinition(
    name="nl2sql",
    required_groups=(("question", "sql", "schema"), ("question", "sql", "schema_ref")),
    optional=("tags", "id", "db_id", "metadata"),
    alternates=({"natural_language": "question", "query": "sql"},),
    detection_signatures=(
        frozenset({"question", "sql", "schema"}),
        frozenset({"question", "sql", "schema_ref"}),
        frozenset({"question", "sql", "db_id"}),
    ),
    canonical_fields=("question", "sql", "schema", "schema_ref", "db_id"),
    text_fields=("question", "sql"),
    content_fields=("question", "sql", "natural_language", "query"),
))


# ── Detection priority ────────────────────────────────────────────
# More specific schemas checked first to avoid misclassification.
_DETECTION_ORDER = [
    "rm_pairwise",
    "nl2sql",
    "qa",
    "chat",
    "summarization",
    "translation",
    "sft",
    "classification",
]


# ── Public API ────────────────────────────────────────────────────

def get_schema(name: str) -> SchemaDefinition:
    """Look up a schema by name. Raises KeyError if unknown."""
    return SCHEMA_REGISTRY[name]


def schema_names() -> List[str]:
    """Return all registered schema names."""
    return list(SCHEMA_REGISTRY.keys())


def detect_schema_from_fields(fields: set) -> str:
    """Auto-detect schema from column/field names.

    Returns the schema name or raises ValueError if no match.
    Priority: rm_pairwise > qa > chat > summarization > translation > sft > classification.
    """
    lower_fields = {f.lower() for f in fields}
    for name in _DETECTION_ORDER:
        schema = SCHEMA_REGISTRY[name]
        for sig in schema.detection_signatures:
            if sig.issubset(lower_fields):
                return name
    raise ValueError(
        f"Cannot auto-detect schema from columns: {sorted(fields)}.\n"
        "Known schemas: " + ", ".join(_DETECTION_ORDER) + "\n"
        "Hint: use --map to rename your columns or --schema to specify explicitly."
    )


def get_content_fields(schema_name: str) -> List[str]:
    """Return the content fields used for contamination text extraction."""
    return list(SCHEMA_REGISTRY[schema_name].content_fields)


def get_text_fields(schema_name: str) -> List[str]:
    """Return the text fields used for report statistics."""
    return list(SCHEMA_REGISTRY[schema_name].text_fields)

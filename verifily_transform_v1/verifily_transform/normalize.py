"""Verifily Transform normalization: map raw rows to a canonical schema."""

import logging
from typing import Any, Dict, List, Optional

from verifily_transform.errors import NormalizeError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target schemas (compatible with Verifily Train v1)
# ---------------------------------------------------------------------------
# SFT: {"instruction": str, "output": str, "input": str (optional), "tags": dict}
# CLS: {"text": str, "label": str, "tags": dict}

# Common raw field names that map to our target fields
_INSTRUCTION_FIELDS = ["instruction", "question", "prompt", "query", "input_text", "request"]
_OUTPUT_FIELDS = ["output", "answer", "response", "completion", "target", "reply"]
_INPUT_FIELDS = ["input", "context", "passage", "document", "source_text"]
_TEXT_FIELDS = ["text", "content", "body", "message", "sentence", "document"]
_LABEL_FIELDS = ["label", "class", "category", "sentiment", "tag", "type"]


def normalize(
    rows: List[Dict[str, Any]],
    task: str,
    instruction_field: Optional[str] = None,
    output_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Normalize raw rows to the canonical schema.

    For SFT: ensures each row has 'instruction' and 'output'.
    For classification: ensures each row has 'text' and 'label'.

    Args:
        rows: raw ingested rows.
        task: "sft" or "classification".
        instruction_field: explicit field name override for instruction.
        output_field: explicit field name override for output.

    Returns:
        List of normalized rows.
    """
    if not rows:
        return []

    if task == "sft":
        return _normalize_sft(rows, instruction_field, output_field)
    elif task == "classification":
        return _normalize_cls(rows)
    else:
        raise NormalizeError(f"Unknown task: {task}")


def _normalize_sft(
    rows: List[Dict[str, Any]],
    instruction_field: Optional[str] = None,
    output_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Normalize to SFT schema: instruction, output, input (optional), tags."""
    # Auto-detect field mapping from the first row
    sample = rows[0]
    instr_key = instruction_field or _find_field(sample, _INSTRUCTION_FIELDS)
    out_key = output_field or _find_field(sample, _OUTPUT_FIELDS)
    inp_key = _find_field(sample, _INPUT_FIELDS)

    if not instr_key:
        raise NormalizeError(
            f"Cannot detect instruction field. Available: {list(sample.keys())}. "
            f"Set labeling.instruction_field in config."
        )
    if not out_key:
        raise NormalizeError(
            f"Cannot detect output field. Available: {list(sample.keys())}. "
            f"Set labeling.output_field in config."
        )

    logger.info("SFT mapping: instruction='%s', output='%s', input='%s'", instr_key, out_key, inp_key)

    normalized = []
    skipped = 0
    for row in rows:
        instruction = str(row.get(instr_key, "")).strip()
        output = str(row.get(out_key, "")).strip()

        if not instruction or not output:
            skipped += 1
            continue

        entry = {"instruction": instruction, "output": output}

        if inp_key and row.get(inp_key):
            entry["input"] = str(row[inp_key]).strip()

        # Carry forward any existing tags
        if "tags" in row and isinstance(row["tags"], dict):
            entry["tags"] = row["tags"]
        else:
            entry["tags"] = {}

        normalized.append(entry)

    if skipped:
        logger.warning("Skipped %d rows with empty instruction or output", skipped)
    logger.info("Normalized %d SFT rows", len(normalized))
    return normalized


def _normalize_cls(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize to classification schema: text, label, tags."""
    sample = rows[0]
    text_key = _find_field(sample, _TEXT_FIELDS)
    label_key = _find_field(sample, _LABEL_FIELDS)

    if not text_key:
        raise NormalizeError(
            f"Cannot detect text field. Available: {list(sample.keys())}. "
        )
    if not label_key:
        raise NormalizeError(
            f"Cannot detect label field. Available: {list(sample.keys())}. "
        )

    logger.info("CLS mapping: text='%s', label='%s'", text_key, label_key)

    normalized = []
    skipped = 0
    for row in rows:
        text = str(row.get(text_key, "")).strip()
        label = str(row.get(label_key, "")).strip()

        if not text or not label:
            skipped += 1
            continue

        entry = {"text": text, "label": label}
        if "tags" in row and isinstance(row["tags"], dict):
            entry["tags"] = row["tags"]
        else:
            entry["tags"] = {}
        normalized.append(entry)

    if skipped:
        logger.warning("Skipped %d rows with empty text or label", skipped)
    logger.info("Normalized %d classification rows", len(normalized))
    return normalized


def _find_field(sample: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Find the first matching field name in a sample row."""
    lower_keys = {k.lower(): k for k in sample.keys()}
    for candidate in candidates:
        if candidate in sample:
            return candidate
        if candidate.lower() in lower_keys:
            return lower_keys[candidate.lower()]
    return None

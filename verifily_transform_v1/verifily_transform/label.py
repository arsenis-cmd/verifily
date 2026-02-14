"""Verifily Transform labeling: heuristic rules and LLM-based weak supervision."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from verifily_transform.config import LabelingConfig
from verifily_transform.errors import LabelError

logger = logging.getLogger(__name__)


def label(
    rows: List[Dict[str, Any]],
    config: LabelingConfig,
) -> List[Dict[str, Any]]:
    """Apply labeling to rows that may be missing labels.

    For SFT: generates instruction/output pairs from raw text if needed.
    For classification: assigns labels via heuristic or LLM.

    The strategy can be:
      - 'heuristic': rule-based labeling
      - 'llm': LLM-based labeling via API
      - 'heuristic+llm': heuristic first, LLM for uncertain cases
    """
    strategy = config.strategy.lower()
    task = config.task

    # Check which rows already have labels
    if task == "sft":
        unlabeled = [r for r in rows if not r.get("instruction") or not r.get("output")]
        labeled = [r for r in rows if r.get("instruction") and r.get("output")]
    else:
        unlabeled = [r for r in rows if not r.get("label")]
        labeled = [r for r in rows if r.get("label")]

    logger.info("Labeling: %d already labeled, %d need labels", len(labeled), len(unlabeled))

    if not unlabeled:
        return rows

    if "heuristic" in strategy:
        newly_labeled, still_unlabeled = _heuristic_label(unlabeled, task, config)
        labeled.extend(newly_labeled)
        unlabeled = still_unlabeled

    if "llm" in strategy and unlabeled:
        newly_labeled = _llm_label(unlabeled, task, config)
        labeled.extend(newly_labeled)
        unlabeled = []

    if unlabeled:
        logger.warning("%d rows could not be labeled and will be dropped", len(unlabeled))

    # Apply label_map remapping if configured
    if config.label_map and task == "classification":
        for row in labeled:
            raw_label = row.get("label", "")
            if raw_label in config.label_map:
                row["label"] = config.label_map[raw_label]

    logger.info("Labeling complete: %d rows", len(labeled))
    return labeled


def _heuristic_label(
    rows: List[Dict[str, Any]],
    task: str,
    config: LabelingConfig,
) -> tuple:
    """Apply heuristic rules. Returns (labeled, still_unlabeled)."""
    labeled = []
    unlabeled = []

    for row in rows:
        if task == "sft":
            # Heuristic: if row has text with Q&A pattern, split it
            text = row.get("text", "")
            if not text:
                unlabeled.append(row)
                continue

            # Try to detect Q: ... A: ... pattern
            result = _extract_qa_from_text(text)
            if result:
                row["instruction"] = result["question"]
                row["output"] = result["answer"]
                if "input" in result:
                    row["input"] = result["input"]
                labeled.append(row)
            else:
                unlabeled.append(row)

        elif task == "classification":
            # Heuristic: simple keyword-based sentiment if no label
            text = row.get("text", "")
            if not text:
                unlabeled.append(row)
                continue

            label = _heuristic_classify(text)
            if label:
                row["label"] = label
                labeled.append(row)
            else:
                unlabeled.append(row)

    logger.info("Heuristic labeling: %d labeled, %d remain", len(labeled), len(unlabeled))
    return labeled, unlabeled


def _extract_qa_from_text(text: str) -> Optional[Dict[str, str]]:
    """Try to extract Q&A from text patterns."""
    import re

    # Pattern: Q: ... A: ...
    qa_match = re.search(
        r"(?:Q|Question|Ask)[:\s]+(.+?)(?:\n|$)\s*(?:A|Answer|Response)[:\s]+(.+)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if qa_match:
        return {"question": qa_match.group(1).strip(), "answer": qa_match.group(2).strip()}

    # Pattern: lines with ? followed by answer
    lines = text.strip().split("\n")
    for i, line in enumerate(lines):
        if "?" in line and i + 1 < len(lines):
            return {"question": line.strip(), "answer": lines[i + 1].strip()}

    return None


def _heuristic_classify(text: str) -> Optional[str]:
    """Simple keyword-based classification as a fallback."""
    text_lower = text.lower()

    positive_words = {"great", "good", "excellent", "amazing", "love", "wonderful", "fantastic"}
    negative_words = {"bad", "terrible", "awful", "horrible", "hate", "worst", "poor"}

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return None  # can't determine


def _llm_label(
    rows: List[Dict[str, Any]],
    task: str,
    config: LabelingConfig,
) -> List[Dict[str, Any]]:
    """Use an LLM API to label rows."""
    api_key = os.environ.get(config.llm_api_key_env or config.api_key_env if hasattr(config, 'api_key_env') else "OPENAI_API_KEY")
    if not api_key:
        logger.warning("No API key found in %s, skipping LLM labeling", config.llm_api_key_env)
        return []

    model = config.llm_model or "gpt-4o-mini"
    labeled = []

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai package not installed, skipping LLM labeling")
        return []

    for i, row in enumerate(rows):
        text = row.get("text", str(row))

        if task == "sft":
            prompt = (
                "Given the following text, create a question-answer pair. "
                "Return JSON with keys 'instruction' and 'output'.\n\n"
                f"Text: {text[:1000]}"
            )
        else:
            prompt = (
                "Classify the following text. Return JSON with key 'label'.\n\n"
                f"Text: {text[:1000]}"
            )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)

            if task == "sft" and "instruction" in result and "output" in result:
                row["instruction"] = result["instruction"]
                row["output"] = result["output"]
                labeled.append(row)
            elif task == "classification" and "label" in result:
                row["label"] = result["label"]
                labeled.append(row)

        except Exception as e:
            logger.debug("LLM labeling failed for row %d: %s", i, e)
            continue

    logger.info("LLM labeling: %d/%d rows labeled", len(labeled), len(rows))
    return labeled

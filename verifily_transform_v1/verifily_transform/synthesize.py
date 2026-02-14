"""Verifily Transform synthetic data generation via LLM APIs."""

import json
import logging
import os
import random
from typing import Any, Dict, List

from verifily_transform.config import SyntheticConfig
from verifily_transform.errors import SynthesisError

logger = logging.getLogger(__name__)


def synthesize(
    seed_rows: List[Dict[str, Any]],
    config: SyntheticConfig,
    task: str,
) -> List[Dict[str, Any]]:
    """Generate synthetic data by prompting an LLM to create variations.

    Args:
        seed_rows: existing labeled rows to use as seeds.
        config: synthetic generation config.
        task: "sft" or "classification".

    Returns:
        List of newly generated synthetic rows.
    """
    if not config.enabled:
        return []

    target_count = len(seed_rows) * config.expansion_factor
    logger.info(
        "Synthesizing %d rows (expansion=%dx from %d seeds) via %s",
        target_count, config.expansion_factor, len(seed_rows), config.model,
    )

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        logger.warning("No API key in %s, skipping synthesis", config.api_key_env)
        return []

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai not installed, skipping synthesis")
        return []

    synthetic = []
    attempts = 0
    max_attempts = target_count * 3  # allow some failures

    while len(synthetic) < target_count and attempts < max_attempts:
        # Pick a batch of seeds
        batch_seeds = random.choices(seed_rows, k=min(config.batch_size, len(seed_rows)))
        batch_results = _generate_batch(client, batch_seeds, config, task)
        synthetic.extend(batch_results)
        attempts += config.batch_size

        if len(synthetic) % 100 == 0 and synthetic:
            logger.info("Generated %d/%d synthetic rows", len(synthetic), target_count)

    synthetic = synthetic[:target_count]

    # Tag all synthetic rows
    for row in synthetic:
        row.setdefault("tags", {})
        row["tags"]["_origin"] = "synthetic"
        row["tags"]["_model"] = config.model

    logger.info("Synthesis complete: %d rows generated", len(synthetic))
    return synthetic


def _generate_batch(
    client,
    seeds: List[Dict[str, Any]],
    config: SyntheticConfig,
    task: str,
) -> List[Dict[str, Any]]:
    """Generate synthetic examples from a batch of seeds."""
    results = []

    for seed in seeds:
        prompt = _build_prompt(seed, task)
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": _system_prompt(task)},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            content = response.choices[0].message.content.strip()
            parsed = _parse_response(content, task)
            if parsed:
                results.append(parsed)
        except Exception as e:
            logger.debug("Synthesis failed for seed: %s", e)

    return results


def _system_prompt(task: str) -> str:
    if task == "sft":
        return (
            "You are a data generation assistant. Given a seed example, "
            "generate a NEW, different but thematically similar question-answer pair. "
            "Return valid JSON with keys 'instruction' and 'output'. "
            "Do not copy the seed. Create novel content."
        )
    else:
        return (
            "You are a data generation assistant. Given a seed example, "
            "generate a NEW, different text with the same label/class. "
            "Return valid JSON with keys 'text' and 'label'. "
            "Do not copy the seed. Create novel content."
        )


def _build_prompt(seed: Dict[str, Any], task: str) -> str:
    if task == "sft":
        instruction = seed.get("instruction", "")
        output = seed.get("output", "")
        inp = seed.get("input", "")
        parts = [f"Seed instruction: {instruction}"]
        if inp:
            parts.append(f"Seed context: {inp[:300]}")
        parts.append(f"Seed answer: {output}")
        parts.append(
            "\nGenerate a completely new question-answer pair on a similar topic. "
            "Return JSON: {\"instruction\": \"...\", \"output\": \"...\"}"
        )
        return "\n".join(parts)
    else:
        text = seed.get("text", "")
        label = seed.get("label", "")
        return (
            f"Seed text: {text[:300]}\n"
            f"Seed label: {label}\n\n"
            f"Generate a new text example with the same label '{label}'. "
            f"Return JSON: {{\"text\": \"...\", \"label\": \"{label}\"}}"
        )


def _parse_response(content: str, task: str) -> dict:
    """Parse LLM response into a structured row."""
    # Strip markdown code blocks
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        import re
        match = re.search(r'\{[^{}]+\}', content)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if task == "sft":
        if "instruction" in data and "output" in data:
            return {
                "instruction": str(data["instruction"]).strip(),
                "output": str(data["output"]).strip(),
                "tags": {},
            }
    elif task == "classification":
        if "text" in data and "label" in data:
            return {
                "text": str(data["text"]).strip(),
                "label": str(data["label"]).strip(),
                "tags": {},
            }

    return {}

"""Nested field handling — flatten dicts and expand chat turns.

Converts nested structures into flat rows suitable for Verifily's pipeline,
and expands chat-format ``messages`` into SFT ``(input, output)`` pairs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Keys that should never be flattened (they contain structured metadata)
_PRESERVE_KEYS = {"tags", "messages", "conversations"}


def flatten_row(
    row: Dict[str, Any],
    *,
    separator: str = ".",
    max_depth: int = 10,
    preserve_lists: bool = True,
    _prefix: str = "",
    _depth: int = 0,
) -> Dict[str, Any]:
    """Recursively flatten nested dicts.

    ``{"meta": {"author": "X"}}`` becomes ``{"meta.author": "X"}``.

    Lists are preserved as-is by default (``preserve_lists=True``).
    Keys in ``_PRESERVE_KEYS`` (tags, messages, conversations) are never flattened.
    """
    out: Dict[str, Any] = {}
    for k, v in row.items():
        key = f"{_prefix}{separator}{k}" if _prefix else k
        # Never flatten preserved keys at the top level
        if _depth == 0 and k in _PRESERVE_KEYS:
            out[key] = v
            continue
        if isinstance(v, dict) and _depth < max_depth:
            nested = flatten_row(
                v,
                separator=separator,
                max_depth=max_depth,
                preserve_lists=preserve_lists,
                _prefix=key,
                _depth=_depth + 1,
            )
            out.update(nested)
        elif isinstance(v, list) and not preserve_lists and _depth < max_depth:
            for i, item in enumerate(v):
                idx_key = f"{key}{separator}{i}"
                if isinstance(item, dict):
                    nested = flatten_row(
                        item,
                        separator=separator,
                        max_depth=max_depth,
                        preserve_lists=preserve_lists,
                        _prefix=idx_key,
                        _depth=_depth + 1,
                    )
                    out.update(nested)
                else:
                    out[idx_key] = item
        else:
            out[key] = v
    return out


def expand_chat_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a chat-format row into SFT ``(input, output)`` pairs.

    Input format::

        {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ]}

    Output::

        [
            {"input": "Hi", "output": "Hello!",
             "tags": {"turn": 0, "source_schema": "chat", "system_prompt": "You are helpful."}},
            {"input": "Bye", "output": "Goodbye!",
             "tags": {"turn": 1, "source_schema": "chat", "system_prompt": "You are helpful."}},
        ]

    If no valid user/assistant pairs are found, returns the original row unchanged.
    """
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list) or len(messages) == 0:
        return [row]

    # Normalise keys to lowercase
    normalised = []
    for msg in messages:
        if isinstance(msg, dict):
            normalised.append({k.lower(): v for k, v in msg.items()})

    # Extract system prompt
    system_prompt: Optional[str] = None
    filtered = []
    for msg in normalised:
        role = str(msg.get("role", "")).lower()
        if role == "system":
            system_prompt = str(msg.get("content", "")).strip()
        else:
            filtered.append(msg)

    # Pair up user/assistant turns
    pairs: List[Dict[str, Any]] = []
    turn_idx = 0
    i = 0
    while i < len(filtered) - 1:
        current = filtered[i]
        nxt = filtered[i + 1]
        c_role = str(current.get("role", "")).lower()
        n_role = str(nxt.get("role", "")).lower()

        if c_role == "user" and n_role == "assistant":
            user_text = str(current.get("content", "")).strip()
            assistant_text = str(nxt.get("content", "")).strip()
            tags: Dict[str, Any] = {"turn": turn_idx, "source_schema": "chat"}
            if system_prompt:
                tags["system_prompt"] = system_prompt

            # Carry forward any existing tags from the original row
            existing_tags = row.get("tags", {})
            if isinstance(existing_tags, dict):
                for k, v in existing_tags.items():
                    if k not in tags:
                        tags[k] = v

            pairs.append({
                "input": user_text,
                "output": assistant_text,
                "tags": tags,
            })
            turn_idx += 1
            i += 2
        else:
            i += 1

    return pairs if pairs else [row]


def flatten_rows(
    rows: List[Dict[str, Any]],
    *,
    expand_chat: bool = True,
    separator: str = ".",
    max_depth: int = 10,
) -> List[Dict[str, Any]]:
    """Flatten nested fields and optionally expand chat turns.

    This is the main entry point for the flatten module:
    1. Flatten nested dicts in each row
    2. If ``expand_chat`` is True, expand rows with ``messages``/``conversations``
       into SFT pairs
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        # Step 1: flatten nested dicts (preserve lists like messages)
        flat = flatten_row(row, separator=separator, max_depth=max_depth)

        # Step 2: expand chat turns if applicable
        if expand_chat and ("messages" in flat or "conversations" in flat):
            expanded = expand_chat_turns(flat)
            out.extend(expanded)
        else:
            out.append(flat)

    return out

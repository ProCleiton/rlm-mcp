"""Context loading for rlm sessions (DESIGN section 2 / C2).

The context is a symbolic handle: the supervisor only ever exposes
*metadata* (chars, lines, part list, head/tail preview) to the harness while
the full text lives inside the sandbox as the ``context`` variable.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from rlm_mcp.types import ContextMeta

_FILE_SEPARATOR = "\n\n===== FILE: {name} =====\n"

HEAD_TAIL_CHARS = 500

DEFAULT_CHUNK_SIZE = 4000
DEFAULT_CHUNK_OVERLAP = 200


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _line_count(text: str) -> int:
    return len(text.splitlines())


def load_context(
    text: str | None, paths: Sequence[str]
) -> tuple[str, list[dict[str, object]], ContextMeta]:
    """Combine ``text`` and file ``paths`` into one context payload.

    Returns ``(context, context_parts, meta)`` where ``context`` is the full
    text, ``context_parts`` is a list of ``{name, chars, lines, text}`` dicts
    (one per source unit, names are ``None`` for the bare text) and ``meta``
    is the text-free metadata the harness may see.
    """
    if text is None and not paths:
        raise ValueError("context requires either 'text' or at least one 'paths' entry")

    # (name, chars, lines, text) per source unit, in join order.  Keeping a
    # typed side-channel avoids object-typed dict lookups below.
    units: list[tuple[str | None, int, int, str]] = []
    if text is not None:
        units.append((None, len(text), _line_count(text), text))
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"context path not found: {path}")
        content = _read_text(str(path))
        units.append((str(path), len(content), _line_count(content), content))

    full_parts: list[dict[str, object]] = [
        {"name": name, "chars": chars, "lines": lines, "text": body}
        for name, chars, lines, body in units
    ]
    joined = units[0][3]
    for name, _, _, body in units[1:]:
        joined += _FILE_SEPARATOR.format(name=name) + body

    meta_parts: list[dict[str, object]] = [
        {"name": name, "chars": chars, "lines": line_count}
        for name, chars, line_count, _ in units
    ]

    total = len(joined)
    meta = ContextMeta(
        chars=total,
        lines=_line_count(joined),
        parts=meta_parts,
        head=joined[:HEAD_TAIL_CHARS],
        tail=joined[-HEAD_TAIL_CHARS:] if total > HEAD_TAIL_CHARS else "",
    )
    return joined, full_parts, meta


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``size`` chars.

    Consecutive chunks overlap by ``overlap`` chars, which lets recursive
    sub-queries over chunk boundaries see shared context (DESIGN section 4).
    """
    if size <= 0:
        raise ValueError(f"chunk size must be > 0, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"chunk overlap must be in [0, size), got {overlap}")
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks

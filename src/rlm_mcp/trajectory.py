"""Append-only JSONL trajectory of a session tree (DESIGN sections 5/G5 and 7).

One file per root session id lives under
``$RLM_MCP_TRAJECTORY_DIR`` or ``$XDG_STATE_HOME/rlm-mcp/trajectories`` or
``~/.local/state/rlm-mcp/trajectories``.  Every line carries the envelope
``{"v": 1, "type", "sid", "root", "depth", "seq", "ts"}`` plus event fields.
A write failure never breaks the session: it degrades to a warning.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("rlm_mcp.trajectory")

VERSION = 1
TRUNC_CAP = 4000

#: Event types recorded in the trajectory (aligned with DESIGN section 7).
EVENT_TYPES = frozenset(
    {"open", "exec", "llm_request", "llm_response", "final", "error", "exhausted", "close"}
)


def _default_dir() -> str:
    if env := os.environ.get("RLM_MCP_TRAJECTORY_DIR"):
        return env
    state_home = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state"
    )
    return os.path.join(state_home, "rlm-mcp", "trajectories")


def cut_text(text: str, cap: int = TRUNC_CAP) -> str:
    """Tail-cut ``text`` at ``cap`` chars with an elision marker."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"[... {len(text) - cap} chars elided ...]"


class TrajectoryWriter:
    """Thread-safe, append-only JSONL writer with one file per root session."""

    def __init__(self, directory: str | os.PathLike[str] | None = None):
        self.directory = Path(directory) if directory is not None else Path(_default_dir())
        self._lock = threading.Lock()
        self._seqs: dict[str, int] = {}
        self._counts: dict[str, int] = {}
        self._usable = True
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._usable = False
            logger.warning("trajectory directory unusable (%s); trajectory disabled", exc)

    def _path_for(self, root_id: str) -> Path:
        if not root_id.replace("_", "").isalnum():
            raise ValueError(f"invalid root id for trajectory: {root_id!r}")
        return self.directory / f"{root_id}.jsonl"

    def log(
        self,
        *,
        sid: str,
        root: str,
        depth: int,
        event: str,
        **fields: object,
    ) -> bool:
        """Append one event line; returns False (after a warning) on failure."""
        if event not in EVENT_TYPES:
            raise ValueError(f"unknown trajectory event type: {event!r}")
        if not self._usable:
            logger.warning("trajectory unavailable; dropping event %s", event)
            return False
        with self._lock:
            seq = self._seqs.get(root, 0) + 1
            self._seqs[root] = seq
            line = {
                "v": VERSION,
                "type": event,
                "sid": sid,
                "root": root,
                "depth": depth,
                "seq": seq,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            }
            line.update(fields)
            try:
                with self._path_for(root).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                self._counts[root] = self._counts.get(root, 0) + 1
                return True
            except OSError as exc:
                logger.warning("trajectory write failed (%s); dropping event %s", exc, event)
                return False

    def read(self, root_id: str) -> str:
        """Return the whole JSONL text for ``root_id`` ("" when absent)."""
        if not self._usable:
            return ""
        path = self._path_for(root_id)
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("trajectory read failed for %s: %s", root_id, exc)
            return ""

    def count(self, root_id: str) -> int:
        with self._lock:
            return self._counts.get(root_id, 0)

    def summary(self, root_id: str) -> dict[str, object]:
        """Small dict the harness sees in final/status payloads."""
        return {
            "root_id": root_id,
            "path": str(self._path_for(root_id)),
            "lines": self.count(root_id),
        }

"""Shared data types for the rlm-mcp core.

This module is the single source of truth for the payload shapes that the MCP
layer returns to the harness (DESIGN sections 5 and 8).  Every result
dataclass exposes ``to_dict()`` that produces JSON-pure data (only
dict/list/str/int/float/bool/None).  ``StepResult`` is a discriminated union
on ``status``: fields that are irrelevant for a given status are ``None`` and
are omitted by ``to_dict()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["llm", "rlm"]
Status = Literal["ok", "needs_llm", "final", "error", "exhausted"]
SessionState = Literal["idle", "running", "parked", "final", "dead", "closed"]
Mode = Literal["doc", "exec"]

#: Names injected into the sandbox namespace (DESIGN section 6). Rebinding
#: any of them is refused at ``rlm_exec`` time.
RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "context",
        "context_parts",
        "history",
        "llm_query",
        "llm_query_batched",
        "rlm_query",
        "rlm_query_batched",
        "FINAL",
        "FINAL_VAR",
        "SHOW_VARS",
        "chunk_text",
        "spawn_background",
        "BackgroundHandle",
    }
)


@dataclass(frozen=True)
class Limits:
    """Budget and containment limits of one session tree (DESIGN section 8)."""

    max_iterations: int = 30
    max_llm_calls: int = 60
    max_depth: int = 2
    max_wall_seconds: float = 900.0
    max_exec_seconds: float = 120.0
    max_output_chars: int = 8_000
    max_errors: int = 5

    def to_dict(self) -> dict[str, object]:
        return {
            "max_iterations": self.max_iterations,
            "max_llm_calls": self.max_llm_calls,
            "max_depth": self.max_depth,
            "max_wall_seconds": self.max_wall_seconds,
            "max_exec_seconds": self.max_exec_seconds,
            "max_output_chars": self.max_output_chars,
            "max_errors": self.max_errors,
        }


@dataclass(frozen=True)
class OpenSpec:
    """Everything ``SessionManager.open`` needs to start a session.

    ``mode`` selects the session channel: ``"doc"`` (default) is the
    document-processing sandbox with a fully scrubbed environment, while
    ``"exec"`` opts into controlled reinjection of ``trusted_env`` entries.
    ``trusted_env`` is ignored unless ``mode == "exec"`` (defense in depth;
    see ``SessionManager.open``).
    """

    text: str | None = None
    paths: tuple[str, ...] = ()
    parent_session_id: str | None = None
    limits: Limits = field(default_factory=Limits)
    label: str | None = None
    mode: Mode = "doc"
    trusted_env: dict[str, str] | None = None


@dataclass(frozen=True)
class SubRequest:
    """One pending sub-model request, as surfaced to the harness.

    ``prompt`` and ``context`` carry display copies (already truncated at the
    configured output cap, per DESIGN section 5 / C3). ``chars`` is the
    length of the *full* prompt so the harness can judge how much was elided.
    """

    id: str
    kind: Kind
    prompt: str
    context: str | None = None
    chars: int | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "chars": self.chars if self.chars is not None else len(self.prompt),
        }
        if self.context is not None:
            out["suggested_context"] = self.context
        return out


@dataclass(frozen=True)
class SubResult:
    """One answer to a pending sub-model request (``rlm_resume`` input)."""

    id: str
    text: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class VarInfo:
    """A user variable living in the sandbox namespace."""

    name: str
    type: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "type": self.type, "size": self.size}


@dataclass(frozen=True)
class ContextMeta:
    """Metadata about the loaded context; never carries the text itself."""

    chars: int
    lines: int
    parts: list[dict[str, object]]
    head: str
    tail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chars": self.chars,
            "lines": self.lines,
            "parts": self.parts,
            "head": self.head,
            "tail": self.tail,
        }


@dataclass(frozen=True)
class Spent:
    """Snapshot of the tree-wide ledger counters."""

    iterations: int
    llm_calls: int
    wall_seconds: float
    errors: int
    output_chars: int

    def to_dict(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "wall_seconds": self.wall_seconds,
            "errors": self.errors,
            "output_chars": self.output_chars,
        }


@dataclass(frozen=True)
class StepResult:
    """Discriminated result of ``rlm_open``/``rlm_exec``/``rlm_resume``."""

    status: Status
    # rlm_open payload
    session_id: str | None = None
    depth: int | None = None
    context: ContextMeta | None = None
    budget: dict[str, object] | None = None
    # status == "ok" payload
    stdout: str | None = None
    vars: list[VarInfo] | None = None
    # status == "needs_llm" payload
    requests: list[SubRequest] | None = None
    # status == "final" payload
    answer: str | None = None
    trajectory: dict[str, object] | None = None
    # status == "error" payload
    error: dict[str, str] | None = None
    # status == "exhausted" payload
    reason: str | None = None
    limits: Limits | None = None
    partial: dict[str, object] | None = None
    # spent is meaningful for every status except the open payload
    spent: Spent | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"status": self.status}
        if self.session_id is not None:
            out["session_id"] = self.session_id
        if self.depth is not None:
            out["depth"] = self.depth
        if self.context is not None:
            out["context"] = self.context.to_dict()
        if self.budget is not None:
            out["budget"] = self.budget
        if self.stdout is not None:
            out["stdout"] = self.stdout
        if self.vars is not None:
            out["vars"] = [v.to_dict() for v in self.vars]
        if self.spent is not None:
            out["spent"] = self.spent.to_dict()
        if self.requests is not None:
            out["requests"] = [r.to_dict() for r in self.requests]
        if self.answer is not None:
            out["answer"] = self.answer
        if self.trajectory is not None:
            out["trajectory"] = self.trajectory
        if self.error is not None:
            out["error"] = self.error
        if self.reason is not None:
            out["reason"] = self.reason
        if self.limits is not None:
            out["limits"] = self.limits.to_dict()
        if self.partial is not None:
            out["partial"] = self.partial
        return out


@dataclass(frozen=True)
class PeekResult:
    """One page of ``rlm_peek`` over a sandbox expression."""

    text: str
    offset: int
    returned: int
    total: int
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "offset": self.offset,
            "returned": self.returned,
            "total": self.total,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class StatusResult:
    """``rlm_status`` payload: tree view of one session."""

    depth: int
    spent: Spent
    limits: Limits
    state: SessionState
    trajectory: dict[str, object]
    jobs: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "spent": self.spent.to_dict(),
            "limits": self.limits.to_dict(),
            "state": self.state,
            "trajectory": self.trajectory,
            "jobs": self.jobs,
        }

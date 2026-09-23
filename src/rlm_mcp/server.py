"""MCP adapter layer for rlm-mcp.

Exposes the RLM paradigm (docs/DESIGN.md, section 5) as eight tools
(`rlm_open`, `rlm_exec`, `rlm_exec_async`, `rlm_wait`, `rlm_resume`,
`rlm_peek`, `rlm_status`, `rlm_close`), one prompt (`rlm_playbook`) and one
resource (`rlm://trajectory/{root_id}`).

This module is a thin, dumb adapter over ``SessionManager`` (the only port
the MCP layer uses, docs/DESIGN.md section 8): it keeps no state of its own
and re-implements no budget or truncation logic. The server never calls a
language model and never needs an API key -- sub-completions are answered by
the harness's own agent through ``rlm_resume``.

Tool results are ``result.to_dict()`` payloads (pure JSON). Usage errors
(unknown session id -> ``SessionNotFoundError``, refused ``open``/``peek`` ->
``SessionError``) come back as readable error payloads, never as raw
exceptions; wrong-state flows are returned by the core as ``status="error"``
step results and passed through unchanged.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from mcp.server.mcpserver import MCPServer

from rlm_mcp import __version__
from rlm_mcp.playbook import INSTRUCTIONS, PLAYBOOK
from rlm_mcp.sandbox.driver import KEEP_ENV
from rlm_mcp.session import SessionError, SessionManager
from rlm_mcp.types import Limits, Mode, OpenSpec, SubResult

#: Validated ``trusted_env`` names: uppercase, digits/underscores, no secret
#: values ever logged (only key names appear in error messages). Minimum
#: length 2, maximum 64 chars: ``^[A-Z][A-Z0-9_]{1,63}$``.
_TRUSTED_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")

#: ``trusted_env`` keys that would override the driver's own scrubbed
#: environment. Single source of truth: derived from ``KEEP_ENV`` in
#: ``sandbox/driver.py`` (keep both in sync by construction). Rejected
#: loudly before any session is created; the driver's silent drop stays
#: as defense in depth. Never log values, only key names.
_RESERVED_ENV_KEYS = frozenset(KEEP_ENV)


def _validate_trusted_env(
    mode: str, trusted_env: Mapping[str, object] | None
) -> dict[str, str] | None:
    """Validate ``mode``/``trusted_env`` before building ``OpenSpec``.

    ``mode`` must be ``"doc"`` or ``"exec"``; ``trusted_env`` is only
    accepted with ``mode == "exec"``. Keys must match
    ``^[A-Z][A-Z0-9_]{1,63}$`` (2-64 chars, uppercase start), must not
    start with ``RLM_`` (reserved for the driver's internal pipe-fd/rlimit
    params), and must not override a reserved sandbox variable (``KEEP_ENV``:
    ``PATH``, ``HOME``, ``LANG``, ``TZ``, ``TMPDIR``). Values must be
    strings. Never logs values, only key names.
    """
    if mode not in ("doc", "exec"):
        raise ValueError(f"invalid mode {mode!r}: expected 'doc' or 'exec'")
    if trusted_env is None:
        return None
    if not isinstance(trusted_env, Mapping):
        raise ValueError("trusted_env must be an object mapping names to values")
    if mode != "exec":
        raise ValueError("trusted_env requires mode='exec'")
    validated: dict[str, str] = {}
    pattern = r"^[A-Z][A-Z0-9_]{1,63}$"
    for key, value in trusted_env.items():
        bad = not isinstance(key, str) or not _TRUSTED_ENV_KEY_RE.match(key)
        if bad:
            raise ValueError(f"invalid trusted_env key {key!r}: must match {pattern}")
        if key.startswith("RLM_"):
            raise ValueError(f"invalid trusted_env key {key!r}: 'RLM_' prefix is reserved")
        if key in _RESERVED_ENV_KEYS:
            raise ValueError(
                f"invalid trusted_env key {key!r}: overrides a reserved sandbox variable"
            )
        if not isinstance(value, str):
            raise ValueError(f"invalid trusted_env entry {key!r}: value must be a string")
        validated[key] = value
    return validated


#: Log levels accepted by the SDK's MCPServer constructor.
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_TOOL_NAMES: tuple[str, ...] = (
    "rlm_open",
    "rlm_exec",
    "rlm_exec_async",
    "rlm_wait",
    "rlm_resume",
    "rlm_peek",
    "rlm_status",
    "rlm_close",
)


def _to_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return int(value)


def _to_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


# Limits fields are all numeric (int or float); dataclass annotations may be
# classes or (with `from __future__ import annotations`) their string names.
_LIMIT_CONVERTERS: dict[object, Callable[[object], int | float]] = {
    int: _to_int,
    float: _to_float,
    "int": _to_int,
    "float": _to_float,
}


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    """A readable error payload, mirroring the StepResult ``error`` shape."""
    return {"error": {"type": error_type, "message": message}}


def _session_error_payload(exc: SessionError) -> dict[str, Any]:
    message = str(exc) or type(exc).__name__
    return _error_payload(type(exc).__name__, message)


def _merge_limits(base: Limits, overrides: Mapping[str, object] | None) -> Limits:
    """Merge per-call budget overrides on top of the server default limits.

    Raises ValueError for unknown keys, non-numeric values or non-positive
    limits (``max_depth`` may be 0 but not negative); the tools turn that
    into a readable error payload.
    """
    if overrides is None:
        return base
    fields = {f.name: f for f in dataclasses.fields(Limits)}
    unknown = sorted(set(overrides) - set(fields))
    if unknown:
        raise ValueError(f"unknown budget override(s): {', '.join(unknown)}")
    updates: dict[str, int | float] = {}
    for key, value in overrides.items():
        field = fields[key]
        converter = _LIMIT_CONVERTERS.get(field.type)
        if converter is None or isinstance(value, bool):
            raise ValueError(f"budget override '{key}' must be a number")
        try:
            updates[key] = converter(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"budget override '{key}' must be a number") from None
        if key == "max_depth":
            if updates[key] < 0:
                raise ValueError(f"budget override 'max_depth' must be >= 0, got {updates[key]}")
        elif updates[key] <= 0:
            raise ValueError(f"budget override '{key}' must be > 0, got {updates[key]}")
    # dataclasses.replace cannot be typed through a ``**dict[str, int | float]``
    # (Limits mixes int and float fields). Rebuild field-by-field; each value
    # came from the converter chosen by that field's declared type above, so
    # the per-field casts only restate what the validation already proved.
    return Limits(
        max_iterations=cast(int, updates.get("max_iterations", base.max_iterations)),
        max_llm_calls=cast(int, updates.get("max_llm_calls", base.max_llm_calls)),
        max_depth=cast(int, updates.get("max_depth", base.max_depth)),
        max_wall_seconds=cast(float, updates.get("max_wall_seconds", base.max_wall_seconds)),
        max_exec_seconds=cast(float, updates.get("max_exec_seconds", base.max_exec_seconds)),
        max_output_chars=cast(int, updates.get("max_output_chars", base.max_output_chars)),
        max_errors=cast(int, updates.get("max_errors", base.max_errors)),
    )


def _register_tools(server: MCPServer, manager: SessionManager, base_limits: Limits) -> None:
    """Register the eight RLM tools, each delegating to ``manager``."""

    @server.tool(
        name="rlm_open",
        description=(
            "Open an RLM session: load `text` and/or `paths` into the sandbox "
            "variable `context`. Returns session_id, depth, context metadata "
            "(never the raw text) and the budget. Pass parent_session_id to open "
            "a child session for recursive (depth>1) work. Optional `limits` "
            "overrides per-session budgets."
        ),
    )
    async def rlm_open(
        text: str | None = None,
        paths: list[str] | None = None,
        parent_session_id: str | None = None,
        limits: dict[str, Any] | None = None,
        mode: Mode = "doc",
        trusted_env: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            validated_env = _validate_trusted_env(mode, trusted_env)
            spec = OpenSpec(
                text=text,
                paths=tuple(paths) if paths is not None else (),
                parent_session_id=parent_session_id,
                limits=_merge_limits(base_limits, limits),
                mode=mode,
                trusted_env=validated_env,
            )
            result = await manager.open(spec)
            return result.to_dict()
        except ValueError as exc:
            return _error_payload("invalid_arguments", str(exc))
        except SessionError as exc:
            return _session_error_payload(exc)

    @server.tool(
        name="rlm_exec",
        description=(
            "Run Python `code` in the session's persistent REPL namespace. "
            "Status ok: code finished (stdout + var metadata). needs_llm: code "
            "called llm_query and is suspended -- answer the requests with your "
            "own model and call rlm_resume. final: code called FINAL. error: "
            "sandbox exception (truncated traceback). exhausted: budget spent."
        ),
    )
    async def rlm_exec(session_id: str, code: str) -> dict[str, Any]:
        try:
            result = await manager.exec(session_id, code)
            return result.to_dict()
        except SessionError as exc:
            return _session_error_payload(exc)

    @server.tool(
        name="rlm_exec_async",
        description=(
            "Queue Python `code` in the session's persistent REPL. Returns a "
            "unique `{handle, state}` immediately; multiple jobs per session "
            "run FIFO because the sandbox is single-flight. Collect any job "
            "independently with `rlm_wait(handle)`. Syntax, reserved-name, "
            "session-state, and budget guards still apply."
        ),
    )
    async def rlm_exec_async(session_id: str, code: str) -> dict[str, Any]:
        try:
            return await manager.exec_async(session_id, code)
        except SessionError as exc:
            return _session_error_payload(exc)

    @server.tool(
        name="rlm_wait",
        description=(
            "Collect one dispatched `rlm_exec_async` handle. A completed job "
            "returns its terminal step result; a queued/running job past "
            "`timeout` returns `{status: 'pending', state, elapsed}` without "
            "cancelling. Collection order is independent of FIFO execution. "
            "Unknown or already collected handles return error payloads."
        ),
    )
    async def rlm_wait(handle: str, timeout: float = 30.0) -> dict[str, Any]:
        try:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                return _error_payload("invalid_arguments", "timeout must be a number")
            if timeout < 0:
                return _error_payload("invalid_arguments", "timeout must be >= 0")
            return await manager.wait(handle, float(timeout))
        except SessionError as exc:
            return _session_error_payload(exc)

    @server.tool(
        name="rlm_resume",
        description=(
            "Feed sub-answers back to a suspended session. results is a list of "
            "{id, text} or {id, error} covering every request from the previous "
            "needs_llm result; the sandbox llm_query calls return and the code "
            "resumes. Returns the next step result (ok/needs_llm/final/error/"
            "exhausted)."
        ),
    )
    async def rlm_resume(session_id: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            sub_results: list[SubResult] = []
            for item in results:
                if not isinstance(item, dict):
                    raise ValueError("each result must be an object")
                rid = item.get("id")
                if not isinstance(rid, str) or not rid:
                    raise ValueError("each result needs a non-empty string 'id'")
                text = item.get("text")
                error = item.get("error")
                if text is not None and not isinstance(text, str):
                    raise ValueError(f"result '{rid}' text must be a string")
                if error is not None and not isinstance(error, str):
                    raise ValueError(f"result '{rid}' error must be a string")
                if text is None and error is None:
                    raise ValueError(f"result '{rid}' needs 'text' or 'error'")
                sub_results.append(SubResult(id=rid, text=text, error=error))
            result = await manager.resume(session_id, sub_results)
            return result.to_dict()
        except ValueError as exc:
            return _error_payload("invalid_arguments", str(exc))
        except SessionError as exc:
            return _session_error_payload(exc)

    @server.tool(
        name="rlm_peek",
        description=(
            "Page through the value of an expression in the session namespace "
            "(for example context[5000:10000]). Returns truncated text plus "
            "offset/returned/total/truncated so you can page on; a page never "
            "exceeds the absolute per-call ceiling. Works while the session is "
            "parked, e.g. to read a request's full prompt from history before "
            "answering. Use this for large reads; never paste raw text into chat."
        ),
    )
    async def rlm_peek(
        session_id: str,
        expr: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> dict[str, Any]:
        try:
            result = await manager.peek(session_id, expr, offset, limit)
            return result.to_dict()
        except SessionError as exc:
            return _session_error_payload(exc)

    @server.tool(
        name="rlm_status",
        description=(
            "Report depth, budget, session state, outstanding job handles/states, "
            "and trajectory pointer. Check this before planning more work; "
            "budget exhaustion is a hard stop."
        ),
    )
    async def rlm_status(session_id: str) -> dict[str, Any]:
        try:
            result = manager.status(session_id)
            return result.to_dict()
        except SessionError as exc:
            return _session_error_payload(exc)

    @server.tool(
        name="rlm_close",
        description=(
            "Close the session and its whole child subtree, releasing budgets. "
            "Returns the list of closed session ids."
        ),
    )
    async def rlm_close(session_id: str) -> dict[str, Any]:
        try:
            closed = await manager.close(session_id)
            return {"closed": closed}
        except SessionError as exc:
            return _session_error_payload(exc)


def _register_prompt(server: MCPServer) -> None:
    """Register the ``rlm_playbook`` prompt with the root-LM instructions."""

    @server.prompt(
        name="rlm_playbook",
        description=(
            "RLM operating protocol for the root LM: context is a sandbox "
            "variable, sub-completions come from llm_query inside code loops, "
            "kind=rlm means delegate to a child session, finish with FINAL."
        ),
    )
    def playbook() -> str:
        return PLAYBOOK


def _register_resource(server: MCPServer, trajectory_reader: Callable[[str], str]) -> None:
    """Register the tree trajectory resource."""

    @server.resource(
        uri="rlm://trajectory/{root_id}",
        title="RLM trajectory (JSONL)",
        description=(
            "Versioned JSONL trajectory of a whole session tree, keyed by the "
            "root session id. Read it to audit what the sessions actually did."
        ),
        mime_type="application/x-ndjson",
    )
    def trajectory(root_id: str) -> str:
        return trajectory_reader(root_id)


def build_server(
    *,
    limits: Limits | None = None,
    trajectory_dir: str | os.PathLike[str] | None = None,
    log_level: LogLevel = "INFO",
    name: str = "rlm-mcp",
    manager: SessionManager | None = None,
) -> MCPServer:
    """Build a configured MCP server over one real ``SessionManager``.

    ``limits`` is the default budget for sessions opened without explicit
    overrides; ``trajectory_dir`` is where trajectory JSONL files are stored.
    Every tool call delegates to the same manager instance; the server itself
    holds no session state.  ``manager`` optionally injects an existing
    SessionManager (the CLI does this so it can run the background TTL sweep
    around the stdio loop); when omitted a manager is created and owned here.
    """
    base_limits = limits if limits is not None else Limits()
    if manager is None:
        manager = SessionManager(default_limits=base_limits, trajectory_dir=trajectory_dir)
    server = MCPServer(
        name=name,
        instructions=INSTRUCTIONS,
        version=__version__,
        log_level=log_level,
    )

    def read_trajectory(root_id: str) -> str:
        return manager.writer.read(root_id)

    _register_tools(server, manager, base_limits)
    _register_prompt(server)
    _register_resource(server, read_trajectory)
    return server

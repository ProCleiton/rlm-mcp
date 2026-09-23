"""Session manager: the ONLY door the MCP layer uses (DESIGN section 8).

One :class:`Session` wraps one sandbox process (persistent REPL) plus its
position in the tree.  :class:`SessionManager` owns the sessions, hands out
the tree-wide :class:`BudgetLedger` and owns the :class:`TrajectoryWriter`.

State machine per session: ``idle | running | parked | final | dead | closed``.
``exec`` only runs from ``idle``; a sub-model request parks the session until
``resume`` delivers exactly the pending ids.  A step (one ``exec`` or one
``resume`` call) has its own wall-clock deadline of ``max_exec_seconds`` of
*sandbox-active* time; blowing it kills the process group and marks the
session ``dead``.

Errors split in two channels:
  * programming misuse (unknown sid, peek on the wrong state, max_depth
    exceeded, unloadable context) raises :class:`SessionError`;
  * in-flow refusals on a real session (exec while parked, resume id
    mismatch, reserved-name rebind, code errors, timeout) come back as
    ``StepResult(status="error")`` with a ``{"type", "message"}`` payload.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import os
import secrets
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rlm_mcp.budget import BudgetLedger
from rlm_mcp.context import load_context
from rlm_mcp.sandbox import AGENT_SCRIPT, KEEP_ENV, LocalDriver, SandboxError, SandboxTimeout
from rlm_mcp.trajectory import TrajectoryWriter, cut_text
from rlm_mcp.types import (
    RESERVED_NAMES,
    Kind,
    Limits,
    OpenSpec,
    PeekResult,
    SessionState,
    Spent,
    StatusResult,
    StepResult,
    SubRequest,
    SubResult,
    VarInfo,
)

logger = logging.getLogger("rlm_mcp.session")

INIT_TIMEOUT = 30.0  # generous default for the sandbox "ready" frame
TRACEBACK_CAP = 4000  # display cap for error tracebacks
LOG_CAP = 4000  # per-field cap inside the trajectory
PEEK_DEFAULT_LIMIT = 4000

#: Hard per-call ceiling (chars) for one rlm_peek page (C3 / DESIGN 5).
#: Whatever ``limit`` the harness asks for, a single page never returns more
#: than this; ``total`` keeps reporting the full size so callers can page on.
#: 16 000 chars ~ 4-5k tokens: far below the ~25k-token result truncation of
#: Claude Code, yet large enough for full pages of typical chunk sizes.
PEEK_CHAR_CAP = 16_000

#: Absolute ceiling (chars) for the serialized ``requests`` list of one
#: needs_llm step result (C3 / DESIGN 5).  A batch of many large prompts
#: must never push the whole result over the client's truncation limit
#: (25k tokens ~ 100k chars) -- that would elide pending ids and make
#: rlm_resume impossible.  80 000 chars ~ 20k tokens leaves room for the
#: rest of the StepResult envelope.
REQUESTS_PAYLOAD_CAP = 80_000


def _ready_timeout() -> float:
    """Seconds to wait for the sandbox ready frame (env-tunable).

    ``RLM_SANDBOX_READY_TIMEOUT`` overrides the default so operators can
    loosen the budget on loaded machines without touching code.
    """
    raw = os.environ.get("RLM_SANDBOX_READY_TIMEOUT")
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return INIT_TIMEOUT


class SessionError(Exception):
    """Programmer-level API misuse (unknown sid, wrong state for peek...)."""


class SessionNotFoundError(SessionError):
    """The requested session id does not exist (or is already closed)."""


def _truncate(text: str, cap: int) -> str:
    """Head+tail truncation at ``cap`` chars with an elision marker.

    ``cap`` is a hard ceiling: a non-positive or degenerate cap never
    slices the text (a negative slice would hand almost everything back);
    the result is an explicit elision marker instead.
    """
    if cap < 0:
        return f"[... {len(text)} chars elided ...]"
    if len(text) <= cap:
        return text
    if cap < 32:
        return f"[... {len(text)} chars elided ...]"
    elided = len(text) - cap + 32
    head_n = tail_n = 0
    for _ in range(4):
        marker = f"[... {elided} chars elided ...]"
        avail = cap - len(marker)
        if avail <= 1:
            return text[:cap]
        head_n = avail // 2
        tail_n = avail - head_n
        new_elided = len(text) - avail
        if new_elided == elided:
            break
        elided = new_elided
    return text[:head_n] + marker + text[-tail_n:]


def _error_result(error_type: str, message: str, spent: Spent) -> StepResult:
    return StepResult(status="error", error={"type": error_type, "message": message}, spent=spent)


def _budget_payload(ledger: BudgetLedger) -> dict[str, object]:
    return {"limits": ledger.limits.to_dict(), "spent": ledger.snapshot().to_dict()}


# ---------------------------------------------------------------------------
# needs_llm batch ceiling (DESIGN section 5 / C3)
# ---------------------------------------------------------------------------


def _peek_pointer(rid: str) -> str | None:
    """rlm_peek expression returning the full prompt of ``rid``.

    Every submitted request appends exactly one entry to the sandbox
    ``history`` list and ids are issued sequentially (q1, q2, ...), so the
    entry of ``qN`` sits at ``history[N - 1]`` and is readable with
    rlm_peek even while the session is parked.
    """
    if rid.startswith("q") and rid[1:].isdigit():
        index = int(rid[1:]) - 1
        if index >= 0:
            return f"history[{index}]['prompt']"
    return None


def _fit_request_prompt(prompt: str, budget: int, pointer: str | None) -> str:
    """Head+tail fit of an over-budget prompt display copy (C3).

    The elision marker tells the harness which rlm_peek expression returns
    the full prompt, so a batch request that had to be cut stays answerable.
    """
    if len(prompt) <= budget:
        return prompt
    if budget <= 0:
        return ""
    hinted = f"[... {len(prompt)} chars elided; full prompt via rlm_peek: {pointer} ...]"
    plain = f"[... {len(prompt)} chars elided ...]"
    for marker in ((hinted if pointer else plain), plain):
        if budget >= len(marker):
            avail = budget - len(marker)
            head_n = avail // 2
            return prompt[:head_n] + marker + prompt[-(avail - head_n) :]
    return plain


def _fit_requests_payload(requests: list[SubRequest], cap: int) -> list[SubRequest]:
    """Shrink prompt display copies so the serialized requests fit ``cap`` chars.

    Only over-budget batches are touched and the widest prompts are trimmed
    first, so small prompts stay verbatim; every pending ``id`` is preserved
    1:1 (rlm_resume demands all of them).
    """

    def serialized(items: list[SubRequest]) -> int:
        return len(json.dumps([item.to_dict() for item in items], ensure_ascii=False))

    fitted = list(requests)
    while True:
        size = serialized(fitted)
        if size <= cap:
            return fitted
        widest = max(range(len(fitted)), key=lambda i: len(fitted[i].prompt))
        request = fitted[widest]
        prompt = request.prompt
        if len(prompt) <= 96:
            # Every prompt is already at its marker floor; ids are kept 1:1
            # and nothing further can be cut without losing ids.
            return fitted
        budget = max(96, len(prompt) - (size - cap))
        fitted[widest] = SubRequest(
            id=request.id,
            kind=request.kind,
            prompt=_fit_request_prompt(prompt, budget, _peek_pointer(request.id)),
            context=request.context,
            chars=request.chars,
        )


# ---------------------------------------------------------------------------
# Reserved-name rebinding detector (DESIGN section 6)
# ---------------------------------------------------------------------------

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    stack: list[ast.AST | None] = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, ast.Name):
            names.add(current.id)
        elif isinstance(current, (ast.Tuple, ast.List)):
            stack.extend(current.elts)
        elif isinstance(current, ast.Starred):
            stack.append(current.value)
    return names


def _binding_names(node: ast.AST) -> set[str]:
    """Names a single statement/pattern binds in its *enclosing* scope."""
    names: set[str] = set()
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        names |= _target_names(target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        names |= _target_names(node.target)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        # optional_vars lives on each withitem in the AST (it was removed
        # from the With node itself), so collect the per-item targets.
        for item in node.items:
            if item.optional_vars is not None:
                names |= _target_names(item.optional_vars)
    elif isinstance(node, ast.ExceptHandler):
        if node.name:
            names.add(node.name)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            names.add(alias.asname or alias.name)
    elif isinstance(node, ast.NamedExpr):
        names |= _target_names(node.target)
    elif isinstance(node, ast.Delete):
        for target in node.targets:
            names |= _target_names(target)
    elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
        if node.name:
            names.add(node.name)
    elif isinstance(node, ast.MatchMapping):
        if node.rest:
            names.add(node.rest)
    return names


def _reserved_rebindings(tree: ast.AST) -> list[str]:
    """Module-level bindings of reserved names, refusing before execution.

    Function/lambda/class bodies create their own scopes: bindings inside
    them never touch the sandbox namespace, so they are not walked.
    Comprehension loop targets are comprehension-local; only element
    expressions can leak a walrus target into the enclosing scope.
    """

    found: set[str] = set()

    def walk(node: ast.AST) -> None:
        nonlocal found
        if isinstance(node, _SCOPE_NODES):
            found |= _binding_names(node) & RESERVED_NAMES
            return
        if isinstance(node, _COMPREHENSIONS):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.comprehension):
                    continue
                walk(child)
            return
        found |= _binding_names(node) & RESERVED_NAMES
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return sorted(found)


_NAMESPACE_ACCESSORS = frozenset({"globals", "locals", "vars"})


def _is_namespace_accessor(node: ast.AST) -> bool:
    """True for a call to ``globals()``/``locals()``/``vars()`` (bare or
    ``builtins``-qualified) -- an expression yielding a namespace dict whose
    entries *are* live bindings."""
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _NAMESPACE_ACCESSORS
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _NAMESPACE_ACCESSORS
        and isinstance(func.value, ast.Name)
        and func.value.id == "builtins"
    )


def _constant_string(node: ast.AST) -> str | None:
    """The string value of a literal AST node, or ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _subscript_slot_names(target: ast.AST) -> set[str]:
    """Names written into namespace dicts through one store target.

    Walks ``globals()['x']``-style targets (including through tuple /
    starred / attribute chains): the subscript key is the *bound* slot only
    when the subscripted container is a namespace accessor.
    """
    slots: set[str] = set()
    stack: list[ast.AST] = [target]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Subscript):
            if _is_namespace_accessor(current.value):
                key = _constant_string(current.slice)
                if key is not None:
                    slots.add(key)
            stack.append(current.value)
        elif isinstance(current, (ast.Tuple, ast.List)):
            stack.extend(current.elts)
        elif isinstance(current, (ast.Starred, ast.Attribute)):
            stack.append(current.value)
    return slots


def _namespace_rebinding_names(tree: ast.AST) -> set[str]:
    """Reserved names rebound through namespace-dict stores, anywhere in the
    tree: a function body cannot bind plain names into the module namespace,
    but ``globals()['x'] = ...`` works from any depth, so scope nodes are
    walked for subscript stores.
    """

    def targets_of(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.Assign):
            return list(node.targets)
        if isinstance(node, ast.AugAssign):
            return [node.target]
        if isinstance(node, ast.AnnAssign):
            return [node.target] if node.value is not None else []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return [node.target]
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return [item.optional_vars for item in node.items if item.optional_vars is not None]
        if isinstance(node, ast.Delete):
            return list(node.targets)
        return []

    found: set[str] = set()
    stack = [tree]
    while stack:
        current = stack.pop()
        for target in targets_of(current):
            found |= _subscript_slot_names(target)
        stack.extend(ast.iter_child_nodes(current))
    return found & RESERVED_NAMES


def _reserved_rebinding_names(tree: ast.AST) -> list[str]:
    """Every reserved name a single ``exec`` would bind or rebind in the
    sandbox namespace: plain targets at module level plus namespace-dict
    store targets (``globals()['name'] = ...`` and friends) at any depth."""
    found = set(_reserved_rebindings(tree)) | _namespace_rebinding_names(tree)
    return sorted(found)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    """One accepted async exec, retained until collected with ``wait``."""

    handle: str
    code: str
    completion: asyncio.Future[StepResult]
    accepted: asyncio.Event
    state: str = "queued"
    enqueued: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None


@dataclass
class _Session:
    sid: str
    root_id: str
    parent_sid: str | None
    depth: int
    label: str | None
    ledger: BudgetLedger
    driver: LocalDriver
    state: SessionState = "idle"
    pending: list[SubRequest] | None = None
    vars_cache: list[VarInfo] = field(default_factory=list)
    stdout_cache: str = ""
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    # Phase 3 (async exec): accepted jobs live in a per-session table until
    # collected. A single runner drains ``job_queue`` FIFO because the real
    # sandbox protocol remains strictly request/response and can execute only
    # one frame at a time. Completed jobs remain independently collectable.
    jobs: dict[str, _Job] = field(default_factory=dict)
    job_queue: list[str] = field(default_factory=list)
    job_runner: asyncio.Task[None] | None = None
    job_wakeup: asyncio.Event = field(default_factory=asyncio.Event)


class SessionManager:
    """Owns sessions, tree ledgers and the trajectory writer."""

    def __init__(
        self,
        default_limits: Limits | None = None,
        trajectory_dir: str | os.PathLike[str] | None = None,
        session_ttl: float | None = None,
    ):
        self.default_limits = default_limits if default_limits is not None else Limits()
        self.writer = TrajectoryWriter(trajectory_dir)
        if session_ttl is None:
            raw = os.environ.get("RLM_MCP_SESSION_TTL")
            try:
                session_ttl = float(raw) if raw else 30 * 60.0
            except ValueError:
                session_ttl = 30 * 60.0
        self.session_ttl = session_ttl
        self._sessions: dict[str, _Session] = {}
        self._agent_script = AGENT_SCRIPT
        self._job_sessions: dict[str, str] = {}

    # -- helpers --------------------------------------------------------------

    def _require(self, sid: str) -> _Session:
        session = self._sessions.get(sid)
        if session is None:
            raise SessionNotFoundError(f"unknown session: {sid}")
        return session

    @staticmethod
    def _sandbox_died_message(driver: LocalDriver, exit_code: int | None, stderr: str = "") -> str:
        """Explain a sandbox that exited before reporting ready."""
        code = f" (exit code {exit_code})" if exit_code is not None else ""
        detail = f"; agent stderr: {stderr or '(empty)'}"
        return f"sandbox process died during startup{code}{detail}"

    def _touch(self, session: _Session) -> None:
        session.last_used = time.monotonic()

    def _log(self, session: _Session, event: str, **fields: object) -> None:
        self.writer.log(
            sid=session.sid, root=session.root_id, depth=session.depth, event=event, **fields
        )

    def _partial_payload(self, session: _Session) -> dict[str, object]:
        return {
            "vars": [v.to_dict() for v in session.vars_cache],
            "stdout": session.stdout_cache,
        }

    def _exhausted(
        self, session: _Session, reason: str, *, kill_sandbox: bool = False
    ) -> StepResult:
        if kill_sandbox:
            session.driver.kill()
            session.state = "dead"
        self._log(session, "exhausted", reason=reason)
        return StepResult(
            status="exhausted",
            reason=reason,
            spent=session.ledger.snapshot(),
            limits=session.ledger.limits,
            partial=self._partial_payload(session),
        )

    def _fatal(
        self, session: _Session, error_type: str, message: str, *, kill: bool = True
    ) -> StepResult:
        if kill:
            session.driver.kill()
        session.state = "dead"
        session.ledger.charge_error()
        self._log(
            session,
            "error",
            error_type=error_type,
            message=cut_text(message, LOG_CAP),
            traceback="",
        )
        return StepResult(
            status="error",
            error={"type": error_type, "message": message},
            spent=session.ledger.snapshot(),
        )

    def _timeout(self, session: _Session, limits: Limits) -> StepResult:
        message = (
            f"execution exceeded max_exec_seconds ({limits.max_exec_seconds:g}s); "
            f"sandbox killed and session state lost (state=dead)"
        )
        return self._fatal(session, "Timeout", message)

    # -- frame handling (sandbox is executing; step deadline is ticking) ------

    async def _drive(self, session: _Session, step_started: float) -> StepResult:
        limits = session.ledger.limits
        while True:
            remaining = limits.max_exec_seconds - (time.monotonic() - step_started)
            if remaining <= 0:
                return self._timeout(session, limits)
            try:
                frame = await session.driver.recv(remaining)
            except SandboxTimeout:
                return self._timeout(session, limits)
            except SandboxError as exc:
                return self._fatal(session, "ProtocolError", f"sandbox protocol failure: {exc}")
            if frame is None:
                code = session.driver.returncode
                suffix = f" (exit code {code})" if code is not None else ""
                return self._fatal(
                    session, "SandboxDied", f"sandbox process exited unexpectedly{suffix}"
                )
            op = frame.get("op")
            if op == "llm_request":
                return await self._on_llm_request(session, frame)
            if op == "exec_done":
                return self._on_exec_done(session, frame)
            if op == "final":
                return await self._on_final(session, frame)
            if op == "error":
                return self._on_error(session, frame)
            logger.debug("ignoring unexpected frame op %r from sandbox %s", op, session.sid)

    async def _on_llm_request(self, session: _Session, frame: dict[str, object]) -> StepResult:
        limits = session.ledger.limits
        items = frame.get("requests")
        if not isinstance(items, list) or not items:
            return self._fatal(
                session, "ProtocolError", "sandbox sent an empty or malformed llm_request"
            )
        cap = limits.max_output_chars
        degrade = session.depth >= limits.max_depth
        requests: list[SubRequest] = []
        for entry in items:
            if not isinstance(entry, dict):
                return self._fatal(session, "ProtocolError", "sandbox sent a malformed llm_request")
            rid = entry.get("id")
            prompt = entry.get("prompt")
            if not isinstance(rid, str) or not isinstance(prompt, str):
                return self._fatal(session, "ProtocolError", "sandbox sent a malformed llm_request")
            raw_kind = entry.get("kind")
            kind: Kind = "rlm" if raw_kind == "rlm" else "llm"
            effective: Kind = "llm" if (degrade and kind == "rlm") else kind
            raw_context = entry.get("context")
            suggested = _truncate(str(raw_context), cap) if isinstance(raw_context, str) else None
            requests.append(
                SubRequest(
                    id=rid,
                    kind=effective,
                    prompt=_truncate(prompt, cap),
                    context=suggested,
                    chars=len(prompt),
                )
            )
        if session.ledger.llm_calls + len(requests) > limits.max_llm_calls:
            reason = (
                f"max_llm_calls exceeded "
                f"({session.ledger.llm_calls + len(requests)}/{limits.max_llm_calls})"
            )
            return self._exhausted(session, reason, kill_sandbox=True)
        # Aggregate ceiling: a big batch must not push the serialized
        # payload over the client truncation limit (ids would be elided and
        # rlm_resume would become impossible).  No-op for small batches.
        requests = _fit_requests_payload(requests, REQUESTS_PAYLOAD_CAP)
        session.pending = requests
        self._log(
            session,
            "llm_request",
            requests=[
                {
                    "id": req.id,
                    "kind": req.kind,
                    "chars": req.chars,
                    "prompt": cut_text(req.prompt, LOG_CAP),
                }
                for req in requests
            ],
        )
        session.state = "parked"
        return StepResult(status="needs_llm", requests=requests, spent=session.ledger.snapshot())

    def _on_exec_done(self, session: _Session, frame: dict[str, object]) -> StepResult:
        stdout_raw = frame.get("stdout") or ""
        stderr_raw = frame.get("stderr") or ""
        if not isinstance(stdout_raw, str):
            stdout_raw = str(stdout_raw)
        if not isinstance(stderr_raw, str):
            stderr_raw = str(stderr_raw)
        if stderr_raw:
            separator = "" if (not stdout_raw or stdout_raw.endswith("\n")) else "\n"
            combined = stdout_raw + separator + stderr_raw
        else:
            combined = stdout_raw
        session.ledger.charge_output(len(combined))

        raw_vars = frame.get("vars")
        vars_items: list[object] = raw_vars if isinstance(raw_vars, list) else []
        vars_out: list[VarInfo] = []
        for entry in vars_items:
            if not isinstance(entry, dict):
                continue
            raw_size = entry.get("size")
            size = raw_size if isinstance(raw_size, int) and not isinstance(raw_size, bool) else 0
            vars_out.append(
                VarInfo(
                    name=str(entry.get("name") or ""),
                    type=str(entry.get("type") or ""),
                    size=size,
                )
            )
        visible = _truncate(combined, session.ledger.limits.max_output_chars)
        session.vars_cache = vars_out
        session.stdout_cache = visible
        session.state = "idle"
        reason = session.ledger.check()
        if reason:
            return self._exhausted(session, reason)
        return StepResult(
            status="ok", stdout=visible, vars=vars_out, spent=session.ledger.snapshot()
        )

    async def _on_final(self, session: _Session, frame: dict[str, object]) -> StepResult:
        answer = frame.get("answer")
        if not isinstance(answer, str):
            answer = str(answer or "")
        self._log(session, "final", answer=cut_text(answer, LOG_CAP))
        session.state = "final"
        await session.driver.close()
        return StepResult(
            status="final",
            answer=_truncate(answer, session.ledger.limits.max_output_chars),
            spent=session.ledger.snapshot(),
            trajectory=self.writer.summary(session.root_id),
        )

    def _on_error(self, session: _Session, frame: dict[str, object]) -> StepResult:
        raw = frame.get("error")
        if isinstance(raw, dict):
            error_type = str(raw.get("type") or "Error")
            message = str(raw.get("message") or "")
            traceback_text = str(raw.get("traceback") or "")
        else:
            error_type, message, traceback_text = "Error", "", ""
        session.ledger.charge_error()
        self._log(
            session,
            "error",
            error_type=error_type,
            message=cut_text(message, LOG_CAP),
            traceback=cut_text(traceback_text, LOG_CAP),
        )
        session.state = "idle"
        reason = session.ledger.check()
        if reason:
            return self._exhausted(session, reason)
        error_payload: dict[str, str] = {
            "type": error_type,
            "message": _truncate(message, session.ledger.limits.max_output_chars),
        }
        if traceback_text:
            error_payload["traceback"] = _truncate(traceback_text, TRACEBACK_CAP)
        return StepResult(status="error", error=error_payload, spent=session.ledger.snapshot())

    # -- public API (DESIGN section 8) ----------------------------------------

    async def open(self, spec: OpenSpec) -> StepResult:
        """Start a session; ``trusted_env`` applies only when ``mode="exec"``.

        In ``"doc"`` mode (default) ``trusted_env`` is ignored even when
        provided (defense in depth: the scrubbed environment is preserved).
        In ``"exec"`` mode, ``trusted_env`` keys with the ``RLM_`` prefix or
        matching a reserved sandbox variable (``KEEP_ENV``) are refused with
        ``SessionError`` before any sandbox process is spawned — the same
        rejection ``server._validate_trusted_env`` applies at the ``rlm_open``
        boundary, replicated here so direct ``SessionManager``/``OpenSpec``
        callers cannot bypass it. The driver's silent drop of such keys stays
        as a final defense layer.
        """
        if spec.mode == "exec" and spec.trusted_env:
            for key in spec.trusted_env:
                if isinstance(key, str) and (key.startswith("RLM_") or key in KEEP_ENV):
                    raise SessionError(
                        f"invalid trusted_env key {key!r}: overrides a reserved sandbox variable"
                    )
        parent: _Session | None = None
        if spec.parent_session_id is not None:
            parent = self._require(spec.parent_session_id)
            depth = parent.depth + 1
            if depth > parent.ledger.limits.max_depth:
                raise SessionError(
                    f"cannot open child session under {spec.parent_session_id}: "
                    f"max_depth {parent.ledger.limits.max_depth} exceeded (depth {depth})"
                )
        try:
            context, context_parts, meta = load_context(spec.text, spec.paths)
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise SessionError(f"cannot load context: {exc}") from exc

        while True:
            # 128-bit ids make collisions vanishingly rare; the existence
            # check still guarantees a fresh id never overwrites a live one.
            sid = "rlm_" + secrets.token_hex(8)
            if sid not in self._sessions:
                break
        if parent is None:
            root_id = sid
            limits = self.default_limits if spec.limits == Limits() else spec.limits
            ledger = BudgetLedger(sid, limits)
            depth = 0
        else:
            root_id = parent.root_id
            limits = parent.ledger.limits
            ledger = parent.ledger

        # Startup protocol: spawn the interpreter, send init, and wait for the
        # ready frame.  A failed *spawn* (process never existed) is retried
        # once: fork/exec can transiently fail under load.  Once the child
        # exists we do NOT blind-retry -- a process that dies or stalls before
        # ready is reported with its exit code and captured stderr so a real
        # boot crash stays visible.
        timeout = _ready_timeout()
        extra_env = spec.trusted_env if spec.mode == "exec" else None
        driver = LocalDriver(self._agent_script, limits, extra_env=extra_env, mode=spec.mode)
        spawn_failure: Exception | None = None
        for _attempt in range(2):
            try:
                await driver.start()
            except (SandboxError, OSError) as exc:
                spawn_failure = exc
                await driver.close()
                driver = LocalDriver(
                    self._agent_script, limits, extra_env=extra_env, mode=spec.mode
                )
                continue
            spawn_failure = None
            break
        if spawn_failure is not None:
            raise SessionError(f"cannot spawn sandbox: {spawn_failure}") from spawn_failure

        try:
            await driver.send({"op": "init", "context": context, "context_parts": context_parts})
            ready = await driver.recv(timeout)
        except SandboxTimeout:
            # Deadline hit.  If the child is still alive it is booting slowly
            # (machine under load); if it died meanwhile, surface that.
            alive = driver.alive
            stderr = driver.stderr_tail(1000)
            await driver.close()
            if alive:
                raise SessionError(
                    f"sandbox did not become ready within {timeout:g}s and is still running; "
                    f"agent stderr: {stderr or '(empty)'}"
                ) from None
            exit_code = await driver.wait_exit(3.0)
            raise SessionError(self._sandbox_died_message(driver, exit_code, stderr)) from None
        except SandboxError as exc:
            stderr = driver.stderr_tail(1000)
            exit_code = await driver.wait_exit(3.0)
            await driver.close()
            raise SessionError(
                f"sandbox startup failed: {exc}; "
                f"{self._sandbox_died_message(driver, exit_code, stderr)}"
            ) from exc
        if ready is None:
            # EOF before the ready frame: the child exited during boot.
            stderr = driver.stderr_tail(1000)
            exit_code = await driver.wait_exit(3.0)
            await driver.close()
            raise SessionError(self._sandbox_died_message(driver, exit_code, stderr)) from None
        if ready.get("op") != "ready":
            await driver.close()
            raise SessionError(f"sandbox sent an unexpected first frame: {ready!r}") from None

        session = _Session(
            sid=sid,
            root_id=root_id,
            parent_sid=spec.parent_session_id,
            depth=depth,
            label=spec.label,
            ledger=ledger,
            driver=driver,
        )
        self._sessions[sid] = session
        trusted_keys = sorted(spec.trusted_env.keys()) if spec.trusted_env else []
        logger.info("opened session %s mode=%s trusted_env_keys=%s", sid, spec.mode, trusted_keys)
        self._log(
            session,
            "open",
            label=spec.label,
            context_chars=meta.chars,
            context_lines=meta.lines,
            parts=len(meta.parts),
            mode=spec.mode,
            trusted_env_keys=trusted_keys,
        )
        return StepResult(
            status="ok",
            session_id=sid,
            depth=depth,
            context=meta,
            budget=_budget_payload(ledger),
        )

    def _validate_exec_code(self, session: _Session, code: object) -> ast.AST | StepResult:
        """Validate code without consulting the session scheduling state."""
        spent = session.ledger.snapshot()
        if not isinstance(code, str):
            return _error_result("InvalidState", "exec code must be a string", spent)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return _error_result("SyntaxError", f"code does not parse: {exc}", spent)
        names = _reserved_rebinding_names(tree)
        if names:
            return _error_result(
                "RebindRefused",
                "rebinding reserved sandbox names is not allowed: " + ", ".join(names),
                spent,
            )
        return tree

    def _state_refusal(self, session: _Session, sid: str) -> StepResult | None:
        """Return the common terminal/parked refusal, if any."""
        spent = session.ledger.snapshot()
        if session.state == "parked":
            pending_ids = ", ".join(req.id for req in (session.pending or [])) or "?"
            return _error_result(
                "InvalidState",
                f"session {sid} is parked waiting for sub-model answers "
                f"(pending: {pending_ids}); call rlm_resume, not rlm_exec",
                spent,
            )
        if session.state == "final":
            return _error_result(
                "InvalidState",
                f"session {sid} is final: FINAL was already delivered; open a new session",
                spent,
            )
        if session.state == "dead":
            return _error_result(
                "InvalidState",
                f"session {sid} is dead: its sandbox was lost; open a new session",
                spent,
            )
        return None

    def _prepare_exec(self, session: _Session, sid: str, code: object) -> ast.AST | StepResult:
        """Validate a synchronous exec, which cannot overlap uncollected jobs."""
        refusal = self._state_refusal(session, sid)
        if refusal is not None:
            return refusal
        if session.jobs:
            return _error_result(
                "InvalidState",
                f"session {sid} has pending async exec jobs; "
                "collect them with rlm_wait before starting a synchronous exec",
                session.ledger.snapshot(),
            )
        if session.state != "idle":
            return _error_result(
                "InvalidState",
                f"session {sid} is not idle (state={session.state})",
                session.ledger.snapshot(),
            )
        return self._validate_exec_code(session, code)

    def _begin_exec_step(self, session: _Session, code: str) -> StepResult | None:
        """Charge one iteration, check the tree budget and log the step.

        Returns the ``exhausted`` result when the charge blows the budget,
        else ``None`` -- the caller then opens the ledger window and
        dispatches (sync ``await _drive`` or async background task).
        """
        session.ledger.charge_iteration()
        reason = session.ledger.check()
        if reason:
            return self._exhausted(session, reason)
        self._log(session, "exec", code=cut_text(code, LOG_CAP))
        return None

    async def exec(self, sid: str, code: str) -> StepResult:
        session = self._require(sid)
        self._touch(session)
        prepared = self._prepare_exec(session, sid, code)
        if isinstance(prepared, StepResult):
            return prepared
        assert isinstance(code, str)
        exhausted = self._begin_exec_step(session, code)
        if exhausted is not None:
            return exhausted

        session.state = "running"
        session.ledger.resume()
        started = time.monotonic()
        try:
            try:
                await session.driver.send({"op": "exec", "code": code})
            except SandboxError as exc:
                return self._fatal(session, "SandboxDied", f"cannot start execution: {exc}")
            return await self._drive(session, started)
        except Exception as exc:
            logger.exception("supervisor failure while executing session %s", session.sid)
            session.driver.kill()
            session.state = "dead"
            return _error_result(
                "InternalError",
                f"unexpected supervisor failure: {type(exc).__name__}: {exc}",
                session.ledger.snapshot(),
            )
        finally:
            session.ledger.pause()
            if session.state == "running":
                session.state = "dead"

    async def _run_async_job(self, session: _Session, started: float) -> StepResult:
        """Drive one running queue entry and close only its active budget window."""
        try:
            return await self._drive(session, started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("supervisor failure in async job of session %s", session.sid)
            session.driver.kill()
            session.state = "dead"
            return _error_result(
                "InternalError",
                f"unexpected supervisor failure: {type(exc).__name__}: {exc}",
                session.ledger.snapshot(),
            )
        finally:
            session.ledger.pause()
            if session.state == "running":
                session.state = "dead"

    @staticmethod
    def _finish_job(job: _Job, result: StepResult) -> None:
        job.state = "completed"
        job.finished = time.monotonic()
        job.accepted.set()
        if not job.completion.done():
            job.completion.set_result(result)

    def _finish_queued_jobs(self, session: _Session, message: str) -> None:
        """Settle entries that can no longer reach the terminal sandbox."""
        while session.job_queue:
            handle = session.job_queue.pop(0)
            job = session.jobs.get(handle)
            if job is None or job.state != "queued":
                continue
            self._finish_job(
                job,
                _error_result("InvalidState", message, session.ledger.snapshot()),
            )

    async def _run_job_queue(self, session: _Session) -> None:
        """Drain accepted jobs FIFO; the sandbox itself remains single-flight."""
        try:
            while session.job_queue:
                handle = session.job_queue[0]
                job = session.jobs.get(handle)
                if job is None or job.state != "queued":
                    session.job_queue.pop(0)
                    continue

                if session.state != "idle":
                    if session.state in ("final", "dead"):
                        self._finish_queued_jobs(
                            session,
                            f"session {session.sid} became {session.state} before the job ran",
                        )
                        return
                    # A preceding job may have parked on llm_query. The queue
                    # resumes only after rlm_resume leaves the sandbox idle.
                    session.job_wakeup.clear()
                    await session.job_wakeup.wait()
                    continue

                exhausted = self._begin_exec_step(session, job.code)
                if exhausted is not None:
                    session.job_queue.pop(0)
                    self._finish_job(job, exhausted)
                    self._finish_queued_jobs(
                        session,
                        f"session {session.sid} exhausted its budget before the job ran",
                    )
                    return

                session.state = "running"
                session.ledger.resume()
                job.started = time.monotonic()
                job.state = "running"
                try:
                    await session.driver.send({"op": "exec", "code": job.code})
                except SandboxError as exc:
                    result = self._fatal(session, "SandboxDied", f"cannot start execution: {exc}")
                    session.ledger.pause()
                else:
                    job.accepted.set()
                    result = await self._run_async_job(session, job.started)

                session.job_queue.pop(0)
                self._finish_job(job, result)
                if session.state in ("final", "dead"):
                    self._finish_queued_jobs(
                        session,
                        f"session {session.sid} became {session.state} before the job ran",
                    )
                    return
        except asyncio.CancelledError:
            raise
        finally:
            session.job_runner = None

    async def exec_async(self, sid: str, code: str) -> dict[str, object]:
        """Accept an async exec and enqueue it behind this session's running job.

        Handles are unique per dispatch. A single FIFO runner serializes real
        sandbox executions, while completed entries stay in ``jobs`` until
        individually collected. Queued time does not open a budget window.
        """
        session = self._require(sid)
        self._touch(session)
        while True:
            handle = "job_" + secrets.token_hex(12)
            if handle not in self._job_sessions:
                break

        refusal = self._state_refusal(session, sid)
        if refusal is None and session.state == "running" and session.job_runner is None:
            refusal = _error_result(
                "InvalidState",
                f"session {sid} is running a synchronous step",
                session.ledger.snapshot(),
            )
        if refusal is None and session.state not in ("idle", "running"):
            refusal = _error_result(
                "InvalidState",
                f"session {sid} cannot accept jobs in state {session.state}",
                session.ledger.snapshot(),
            )
        prepared = refusal or self._validate_exec_code(session, code)
        if isinstance(prepared, StepResult):
            return {"handle": handle, "state": session.state, "error": prepared.to_dict()}
        assert isinstance(code, str)

        loop = asyncio.get_running_loop()
        job = _Job(
            handle=handle,
            code=code,
            completion=loop.create_future(),
            accepted=asyncio.Event(),
        )
        was_empty = not session.job_queue
        session.jobs[handle] = job
        session.job_queue.append(handle)
        self._job_sessions[handle] = sid
        if session.job_runner is None or session.job_runner.done():
            session.job_runner = asyncio.create_task(self._run_job_queue(session))
        if was_empty and session.state == "idle":
            # Preserve Phase-2's normal single-job observation: by return the
            # first frame is dispatched (or synchronously failed/exhausted).
            await job.accepted.wait()
        return {"handle": handle, "state": job.state}

    async def wait(self, handle: str, timeout: float = 30.0) -> dict[str, Any]:
        """Collect one job by handle, independently of dispatch/collection order."""
        sid = self._job_sessions.get(handle)
        if sid is None:
            raise SessionNotFoundError(f"unknown or already collected async job handle: {handle}")
        session = self._sessions.get(sid)
        if session is None:
            raise SessionNotFoundError(f"session for async job {handle} is closed")
        self._touch(session)
        job = session.jobs.get(handle)
        if job is None:
            raise SessionError(f"async job {handle} was already collected")
        if not job.completion.done():
            try:
                result = await asyncio.wait_for(asyncio.shield(job.completion), timeout)
            except asyncio.TimeoutError:
                elapsed = 0.0 if job.started is None else time.monotonic() - job.started
                return {"status": "pending", "state": job.state, "elapsed": elapsed}
        else:
            result = job.completion.result()
        session.jobs.pop(handle, None)
        self._job_sessions.pop(handle, None)
        return result.to_dict()

    async def resume(self, sid: str, results: Sequence[SubResult]) -> StepResult:
        session = self._require(sid)
        self._touch(session)
        if session.state != "parked":
            return _error_result(
                "InvalidState",
                f"session {sid} is not parked (state={session.state}); "
                "there are no pending sub-model requests to resume",
                session.ledger.snapshot(),
            )
        normalized: list[SubResult] = []
        for result in results:
            if isinstance(result, SubResult):
                normalized.append(result)
            elif isinstance(result, dict):
                normalized.append(
                    SubResult(
                        id=str(result.get("id") or ""),
                        text=result.get("text"),
                        error=result.get("error"),
                    )
                )
            else:
                raise SessionError(
                    f"resume results must be SubResult objects or dicts, "
                    f"got {type(result).__name__}"
                )
        pending = list(session.pending or [])
        expected = Counter(req.id for req in pending)
        actual = Counter(result.id for result in normalized)
        missing = sorted((expected - actual).elements())
        extra = sorted((actual - expected).elements())
        if missing or extra:
            message = (
                f"resume ids do not match the pending requests: "
                f"missing={missing or None} extra={extra or None}; "
                f"expected={sorted(expected.elements())}"
            )
            return _error_result("IdMismatch", message, session.ledger.snapshot())

        session.ledger.charge_llm(len(normalized))
        reason = session.ledger.check()
        if reason:
            exhausted = self._exhausted(session, reason, kill_sandbox=True)
            session.job_wakeup.set()
            return exhausted
        self._log(
            session,
            "llm_response",
            results=[
                {
                    "id": result.id,
                    "ok": result.error is None,
                    "text": cut_text(result.text, LOG_CAP) if result.text is not None else None,
                    "error": cut_text(result.error, LOG_CAP) if result.error else None,
                }
                for result in normalized
            ],
        )

        session.state = "running"
        session.ledger.resume()
        started = time.monotonic()
        payload = [
            {"id": result.id, "text": result.text, "error": result.error} for result in normalized
        ]
        try:
            try:
                await session.driver.send({"op": "llm_response", "results": payload})
            except SandboxError as exc:
                return self._fatal(session, "SandboxDied", f"cannot deliver responses: {exc}")
            return await self._drive(session, started)
        except Exception as exc:
            logger.exception("supervisor failure while resuming session %s", session.sid)
            session.driver.kill()
            session.state = "dead"
            return _error_result(
                "InternalError",
                f"unexpected supervisor failure: {type(exc).__name__}: {exc}",
                session.ledger.snapshot(),
            )
        finally:
            session.ledger.pause()
            if session.state == "running":
                session.state = "dead"
            session.job_wakeup.set()

    async def peek(
        self, sid: str, expr: str, offset: int = 0, limit: int = PEEK_DEFAULT_LIMIT
    ) -> PeekResult:
        session = self._require(sid)
        self._touch(session)
        if not isinstance(expr, str) or not expr.strip():
            raise SessionError("peek requires a non-empty expression")
        if offset < 0 or limit < 0:
            raise SessionError(
                f"peek offset and limit must be >= 0 (offset={offset}, limit={limit})"
            )
        if session.state not in ("idle", "parked"):
            raise SessionError(f"cannot peek while session {sid} is {session.state}")
        # Absolute per-call ceiling (C3 / DESIGN section 5): whatever page
        # size was asked for, one call never returns more than PEEK_CHAR_CAP
        # chars; ``total`` still reports the full size so callers can page on.
        limit = min(limit, PEEK_CHAR_CAP)
        limits = session.ledger.limits
        session.ledger.resume()
        try:
            try:
                await session.driver.send(
                    {"op": "peek", "expr": expr, "offset": offset, "limit": limit}
                )
                frame = await session.driver.recv(limits.max_exec_seconds)
            except SandboxTimeout:
                session.driver.kill()
                session.state = "dead"
                raise SessionError(
                    f"peek timed out after {limits.max_exec_seconds:g}s; "
                    f"sandbox killed and session {sid} is dead"
                ) from None
            except SandboxError as exc:
                raise SessionError(f"peek failed: {exc}") from exc
            if frame is None:
                session.state = "dead"
                raise SessionError("peek failed: sandbox process exited")
            if frame.get("op") == "peek_result":
                raw_offset = frame.get("offset")
                raw_returned = frame.get("returned")
                raw_total = frame.get("total")
                return PeekResult(
                    text=str(frame.get("text") or ""),
                    offset=(
                        raw_offset
                        if isinstance(raw_offset, int) and not isinstance(raw_offset, bool)
                        else offset
                    ),
                    returned=(
                        raw_returned
                        if isinstance(raw_returned, int) and not isinstance(raw_returned, bool)
                        else 0
                    ),
                    total=(
                        raw_total
                        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
                        else 0
                    ),
                    truncated=bool(frame.get("truncated") or False),
                )
            if frame.get("op") == "error":
                raw = frame.get("error")
                if isinstance(raw, dict):
                    raise SessionError(
                        f"peek failed: {raw.get('type') or 'Error'}: {raw.get('message') or ''}"
                    )
                raise SessionError("peek failed")
            raise SessionError(f"peek failed: unexpected frame op {frame.get('op')!r}")
        finally:
            session.ledger.pause()

    def status(self, sid: str) -> StatusResult:
        session = self._require(sid)
        self._touch(session)
        now = time.monotonic()
        jobs = [
            {
                "handle": job.handle,
                "state": job.state,
                "elapsed": (
                    0.0 if job.started is None else max(0.0, (job.finished or now) - job.started)
                ),
            }
            for job in session.jobs.values()
        ]
        return StatusResult(
            depth=session.depth,
            spent=session.ledger.snapshot(),
            limits=session.ledger.limits,
            state=session.state,
            trajectory=self.writer.summary(session.root_id),
            jobs=jobs,
        )

    async def _cancel_async_jobs(self, session: _Session) -> None:
        """Cancel the queue runner, forget every handle, and reap the sandbox."""
        runner = session.job_runner
        if runner is not None and not runner.done():
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await runner
        for handle, job in list(session.jobs.items()):
            self._job_sessions.pop(handle, None)
            job.accepted.set()
            if not job.completion.done():
                job.completion.cancel()
        session.jobs.clear()
        session.job_queue.clear()
        session.job_runner = None
        with contextlib.suppress(Exception):
            await session.driver.close()

    async def close(self, sid: str) -> list[str]:
        """Close ``sid`` and its whole subtree; returns the closed ids.

        Phase 3 policy: every queued/running/uncollected async job in the
        subtree is cancelled and the corresponding sandbox process group is
        killed and reaped, so no orphan survives. A session in a synchronous
        ``exec``/``resume`` step still refuses close for the whole subtree.
        """
        self._require(sid)
        to_close: list[str] = []
        queue = [sid]
        while queue:
            current = queue.pop(0)
            if current in to_close:
                continue
            to_close.append(current)
            for other in self._sessions.values():
                if (
                    other.parent_sid == current
                    and other.sid not in to_close
                    and other.sid not in queue
                ):
                    queue.append(other.sid)
        sync_running = [
            current
            for current in to_close
            if self._sessions[current].state == "running"
            and not any(job.state == "running" for job in self._sessions[current].jobs.values())
        ]
        if sync_running:
            raise SessionError(
                "cannot close mid-execution: running session(s): " + ", ".join(sorted(sync_running))
            )
        # With synchronous steps ruled out, cancel every async queue first.
        for current in to_close:
            session = self._sessions.get(current)
            if session is not None and session.jobs:
                await self._cancel_async_jobs(session)
        closed: list[str] = []
        for current in to_close:
            session = self._sessions.pop(current, None)
            if session is None:
                continue
            self._log(session, "close")
            await session.driver.close()
            closed.append(current)
        return closed

    async def sweep(self) -> None:
        """Evict sessions unused for longer than the TTL."""
        now = time.monotonic()
        for sid in list(self._sessions):
            session = self._sessions.get(sid)
            if session is None or session.state == "running":
                continue
            if now - session.last_used > self.session_ttl:
                with contextlib.suppress(SessionError):
                    await self.close(sid)

    async def shutdown(self) -> None:
        """Close every session (server teardown helper)."""
        for sid in list(self._sessions):
            with contextlib.suppress(SessionError):
                await self.close(sid)

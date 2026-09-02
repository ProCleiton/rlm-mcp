"""Sandbox agent: executes user RLM code inside a confined subprocess.

This is the child half of the sandbox protocol.  It runs under
``python3 -I -S`` (no site-packages, isolated interpreter) and intentionally
imports only the standard library -- it must never import the ``rlm_mcp``
package.

Protocol: newline-delimited JSON frames over two dedicated pipes.  The agent
reads inbound frames from fd 3 and writes outbound frames to fd 4.  The
driver passes the pipes via ``pass_fds`` (they may arrive at other numbers;
``RLM_IN_FD``/``RLM_OUT_FD`` tell us which and this module rewires them to
3/4).  stdin/stdout of the process stay unused; user-code stdout/stderr are
captured in-process with ``redirect_stdout``/``redirect_stderr``.

The central mechanism (DESIGN C4): ``llm_query`` emits an ``llm_request``
frame and then *blocks reading fd 3* until the matching ``llm_response``
arrives, so the interpreter stack -- and therefore arbitrary user loops -- is
preserved across suspension points with zero rewriting of user code.
"""

from __future__ import annotations

import collections
import contextlib
import io
import json
import operator
import os
import resource
import sys
import traceback
from collections.abc import Callable, Iterable

# Canonical protocol fds (the driver may pass them at other numbers).
_IN_FD = 3
_OUT_FD = 4

#: Duplicate of types.RESERVED_NAMES: this module cannot import the package.
RESERVED_NAMES = frozenset(
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
    }
)

#: Largest prompt shipped over the pipe; the harness only ever sees the
#: display-truncated copy anyway (DESIGN section 5 / C3).
PROMPT_WIRE_CAP = 1_000_000


class SubCallError(Exception):
    """A sub-model call failed (visible to -- and catchable by -- user code)."""


class _FinalSignal(BaseException):
    """Internal sentinel raised by FINAL/FINAL_VAR (not caught by user code
    that filters on ``Exception``)."""

    def __init__(self, answer: str):
        super().__init__()
        self.answer = answer


class _StreamClosed(Exception):
    """The inbound pipe reached EOF while a frame was expected."""


def _truncate(text: str, cap: int) -> str:
    """Head+tail truncation at ``cap`` chars with an elision marker.

    ``cap`` is a hard ceiling: a non-positive or degenerate cap never
    slices the text (a negative slice would hand almost everything back),
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


def chunk_text(text: str, size: int = 4000, overlap: int = 200) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``size`` chars."""
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


class _Capture(io.TextIOBase):
    """Bounded stream capture with head/tail retention.

    While the total stays at or below the cap everything is kept verbatim.
    Once it exceeds the cap only the first and last windows are retained, so
    a runaway ``print`` cannot balloon memory or the pipe; ``value()``
    reassembles the windows around an elision marker.
    """

    def __init__(self, cap: int):
        super().__init__()
        self._cap = max(64, cap)
        self.total = 0
        self._cut = False
        self._buf: list[str] = []
        self._buf_len = 0
        self._head = ""
        self._tail: collections.deque[str] = collections.deque()
        self._tail_len = 0
        self._keep = self._cap - 64
        self._head_limit = self._keep // 2
        self._tail_limit = self._keep - self._head_limit

    # io.TextIOBase
    def write(self, s: str) -> int:  
        if not isinstance(s, str):
            s = str(s)
        if not s:
            return len(s)
        n = len(s)
        self.total += n
        if not self._cut:
            self._buf.append(s)
            self._buf_len += n
            if self._buf_len > self._cap:
                whole = "".join(self._buf)
                self._buf = []
                self._buf_len = 0
                self._cut = True
                self._head = whole[: self._head_limit]
                self._tail.append(whole[-self._tail_limit :])
                self._tail_len = min(len(whole), self._tail_limit)
        else:
            chunk = s if n <= self._tail_limit else s[-self._tail_limit :]
            self._tail.append(chunk)
            self._tail_len += len(chunk)
            while len(self._tail) > 1 and self._tail_len - len(self._tail[0]) >= self._tail_limit:
                first = self._tail.popleft()
                self._tail_len -= len(first)
        return len(s)

    def value(self) -> str:
        if not self._cut:
            return "".join(self._buf)
        head = self._head
        tail = "".join(self._tail)
        elided = self.total - (len(head) + len(tail))
        return head + f"[... {elided} chars elided ...]" + tail


def _var_list(ns: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for key in sorted(ns):
        if key.startswith("__") or key in RESERVED_NAMES:
            continue
        value = ns[key]
        # length_hint accepts arbitrary user objects and falls back to 0 when
        # the value has no usable length, matching the previous try/except.
        try:
            size = operator.length_hint(value, 0)
        except (TypeError, AttributeError):
            size = 0
        out.append({"name": key, "type": type(value).__name__, "size": size})
    return out


def _build_namespace(
    context: str,
    context_parts: list[dict[str, object]],
    send: Callable[[dict[str, object]], None],
    read_frame: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Build the persistent user namespace with the injected/reserved names."""
    seq = 0

    history: list[dict[str, object]] = []
    #: Populated at the end of this builder; the functions below read the
    #: same dict object at call time (runtime closure, not a copy).
    ns: dict[str, object] = {}

    def _next_id() -> str:
        nonlocal seq
        seq += 1
        return f"q{seq}"

    def _submit(kind: str, prompts: list[str], contexts: list[str | None]) -> list[str | None]:
        if not prompts:
            return []
        ids: list[str] = []
        entries: list[dict[str, object]] = []
        requests: list[dict[str, object]] = []
        for index, prompt in enumerate(prompts):
            rid = _next_id()
            entry: dict[str, object] = {
                "id": rid,
                "kind": kind,
                "prompt": prompt,
                "result": None,
                "error": None,
            }
            history.append(entry)
            entries.append(entry)
            ids.append(rid)
            ctx = contexts[index] if index < len(contexts) else None
            requests.append(
                {
                    "id": rid,
                    "kind": kind,
                    "prompt": _truncate(prompt, PROMPT_WIRE_CAP),
                    "context": ctx,
                }
            )
        send({"op": "llm_request", "requests": requests})

        answered: dict[str, dict[str, object]] = {}
        while len(answered) < len(ids):
            try:
                frame = read_frame()
            except (_StreamClosed, json.JSONDecodeError) as exc:
                raise SubCallError(
                    f"connection to supervisor lost while waiting for llm_response: {exc}"
                ) from exc
            if not isinstance(frame, dict):
                continue
            if frame.get("op") == "peek":
                # rlm_peek while the session is parked (execution suspended
                # inside llm_query): answer from the live namespace, e.g.
                # so the harness can page a request's full prompt from
                # ``history`` before answering it.
                _handle_peek(ns, frame, send)
                continue
            if frame.get("op") != "llm_response":
                # Defensive: only a response can unblock a suspended call.
                continue
            results = frame.get("results")
            items = results if isinstance(results, list) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_rid = item.get("id")
                answer_rid = raw_rid if isinstance(raw_rid, str) else None
                if answer_rid is not None and answer_rid in ids and answer_rid not in answered:
                    answered[answer_rid] = item

        ordered: list[str | None] = []
        first_error: tuple[str, str] | None = None
        for index, rid in enumerate(ids):
            item = answered[rid]
            entry = entries[index]
            if item.get("error"):
                entry["error"] = item["error"]
                if first_error is None:
                    first_error = (rid, str(item["error"]))
                ordered.append(None)
                continue
            raw_text = item.get("text")
            text: str | None = raw_text if isinstance(raw_text, str) else None
            entry["result"] = text
            ordered.append(text)
        if first_error is not None:
            rid, message = first_error
            raise SubCallError(f"{kind} sub-call {rid} failed: {message}")
        return ordered

    def llm_query(prompt: str) -> str | None:
        return _submit("llm", [prompt], [None])[0]

    def llm_query_batched(prompts: Iterable[str]) -> list[str | None]:
        items = list(prompts)
        return _submit("llm", items, [None] * len(items))

    def rlm_query(prompt: str, context: str | None = None) -> str | None:
        return _submit("rlm", [prompt], [context])[0]

    def rlm_query_batched(prompts: Iterable[str]) -> list[str | None]:
        items = list(prompts)
        return _submit("rlm", items, [None] * len(items))

    def FINAL(answer: object) -> None:
        raise _FinalSignal(str(answer))

    def FINAL_VAR(name: str) -> None:
        if not isinstance(name, str) or name not in ns:
            raise NameError(f"FINAL_VAR: no variable named {name!r} in the sandbox namespace")
        raise _FinalSignal(str(ns[name]))

    def SHOW_VARS() -> list[dict[str, object]]:
        return _var_list(ns)

    ns.update(
        {
            "context": context,
            "context_parts": context_parts,
            "history": history,
            "llm_query": llm_query,
            "llm_query_batched": llm_query_batched,
            "rlm_query": rlm_query,
            "rlm_query_batched": rlm_query_batched,
            "FINAL": FINAL,
            "FINAL_VAR": FINAL_VAR,
            "SHOW_VARS": SHOW_VARS,
            "chunk_text": chunk_text,
        }
    )
    return ns


def _handle_exec(
    ns: dict[str, object],
    code: str,
    send: Callable[[dict[str, object]], None],
    cap_out: int,
) -> bool:
    """Run one exec frame; returns True when FINAL terminated the session."""
    out_cap = _Capture(cap_out)
    err_cap = _Capture(cap_out)
    try:
        with contextlib.redirect_stdout(out_cap), contextlib.redirect_stderr(err_cap):
            try:
                exec(code, ns)
            except _FinalSignal as sig:
                send({"op": "final", "answer": _truncate(sig.answer, cap_out)})
                return True
            except BaseException as exc:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                stderr_text = err_cap.value()
                if stderr_text:
                    tb = stderr_text.rstrip("\n") + "\n" + tb
                send(
                    {
                        "op": "error",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc) or type(exc).__name__,
                            "traceback": tb,
                        },
                    }
                )
                return False
    except BaseException as exc:  # capture plumbing failure: report and stop
        send(
            {
                "op": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc) or type(exc).__name__,
                    "traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                },
            }
        )
        return False
    send(
        {
            "op": "exec_done",
            "stdout": out_cap.value(),
            "stderr": err_cap.value(),
            "vars": _var_list(ns),
        }
    )
    return False


def _as_int(value: object, default: int) -> int:
    # JSON/env values are str/int/float; anything else (None, bool, dicts)
    # falls back to ``default`` exactly like the previous int()/except did.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _handle_peek(
    ns: dict[str, object],
    frame: dict[str, object],
    send: Callable[[dict[str, object]], None],
) -> None:
    expr = frame.get("expr")
    if not isinstance(expr, str) or not expr.strip():
        send(
            {
                "op": "error",
                "error": {
                    "type": "ValueError",
                    "message": "peek requires a non-empty 'expr'",
                    "traceback": "",
                },
            }
        )
        return
    offset = max(_as_int(frame.get("offset"), 0), 0)
    limit = max(_as_int(frame.get("limit"), 0), 0)
    try:
        value = eval(expr, ns)
    except BaseException as exc:
        send(
            {
                "op": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc) or type(exc).__name__,
                    "traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                },
            }
        )
        return
    text = value if isinstance(value, str) else repr(value)
    total = len(text)
    start = min(offset, total)
    end = min(start + limit, total)
    send(
        {
            "op": "peek_result",
            "text": text[start:end],
            "offset": start,
            "returned": end - start,
            "total": total,
            "truncated": end < total,
        }
    )


def _apply_rlimits() -> None:
    """Apply resource limits coming from the driver via the environment."""
    specs = (
        (resource.RLIMIT_AS, "RLM_RLIMIT_AS_BYTES"),
        (resource.RLIMIT_CPU, "RLM_RLIMIT_CPU_SECONDS"),
        (resource.RLIMIT_FSIZE, "RLM_RLIMIT_FSIZE_BYTES"),
        (resource.RLIMIT_NPROC, "RLM_RLIMIT_NPROC"),
    )
    for rlimit, env_name in specs:
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0:
            continue
        # Containment is best-effort per-resource; keep going.
        with contextlib.suppress(OSError, ValueError):
            resource.setrlimit(rlimit, (value, value))


def _rewire_fds() -> tuple[int, int]:
    """Move the inherited pipe fds to the canonical 3 (in) / 4 (out)."""
    r = _as_int(os.environ.get("RLM_IN_FD"), _IN_FD)
    w = _as_int(os.environ.get("RLM_OUT_FD"), _OUT_FD)
    if r == w:
        raise OSError(f"RLM_IN_FD and RLM_OUT_FD are the same fd ({r})")
    if r == _IN_FD and w == _OUT_FD:
        return r, w
    if r == _OUT_FD and w == _IN_FD:  # swapped: bounce through a scratch fd
        os.dup2(r, 5)
        os.dup2(w, _OUT_FD)
        os.dup2(5, _IN_FD)
    elif r == _IN_FD:
        os.dup2(w, _OUT_FD)
    elif r == _OUT_FD:
        os.dup2(r, _IN_FD)
        os.dup2(w, _OUT_FD)
    elif w == _IN_FD:
        os.dup2(w, _OUT_FD)
        os.dup2(r, _IN_FD)
    elif w == _OUT_FD:
        os.dup2(r, _IN_FD)
    else:
        os.dup2(r, _IN_FD)
        os.dup2(w, _OUT_FD)
    for fd in (r, w):
        if fd not in (_IN_FD, _OUT_FD):
            with contextlib.suppress(OSError):
                os.close(fd)
    return _IN_FD, _OUT_FD


def _main() -> int:
    _apply_rlimits()
    try:
        in_fd, out_fd = _rewire_fds()
    except OSError as exc:
        print(f"rlm sandbox: cannot rewire fds: {exc}", file=sys.stderr)
        return 2

    cap_out = _as_int(os.environ.get("RLM_MAX_OUTPUT_CHARS"), 8_000)
    inp = os.fdopen(in_fd, "r", encoding="utf-8", errors="replace")
    outp = os.fdopen(out_fd, "w", encoding="utf-8", errors="replace")

    def send(obj: dict[str, object]) -> None:
        outp.write(json.dumps(obj, ensure_ascii=False))
        outp.write("\n")
        outp.flush()

    def read_frame() -> dict[str, object]:
        while True:
            line = inp.readline()
            if not line:
                raise _StreamClosed("sandbox input pipe closed")
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
            # Non-object JSON is protocol garbage: skip it, mirroring the
            # `if not isinstance(frame, dict): continue` guards upstream.

    ns: dict[str, object] | None = None
    try:
        while True:
            try:
                frame = read_frame()
            except _StreamClosed:
                return 0
            except json.JSONDecodeError:
                send(
                    {
                        "op": "error",
                        "error": {
                            "type": "ProtocolError",
                            "message": "sandbox received a malformed frame",
                            "traceback": "",
                        },
                    }
                )
                continue
            if not isinstance(frame, dict):
                continue
            op = frame.get("op")
            if op == "init":
                raw_context = frame.get("context")
                context = raw_context if isinstance(raw_context, str) else str(raw_context or "")
                parts = frame.get("context_parts")
                context_parts = parts if isinstance(parts, list) else []
                ns = _build_namespace(context, context_parts, send, read_frame)
                send({"op": "ready"})
            elif op == "exec":
                if ns is None:
                    send(
                        {
                            "op": "error",
                            "error": {
                                "type": "NotInitialized",
                                "message": "exec received before init",
                                "traceback": "",
                            },
                        }
                    )
                    continue
                code = frame.get("code")
                if _handle_exec(ns, code if isinstance(code, str) else "", send, cap_out):
                    return 0  # FINAL delivered: the trajectory is over
            elif op == "peek":
                if ns is None:
                    send(
                        {
                            "op": "error",
                            "error": {
                                "type": "NotInitialized",
                                "message": "peek received before init",
                                "traceback": "",
                            },
                        }
                    )
                    continue
                _handle_peek(ns, frame, send)
            elif op == "shutdown":
                return 0
            # Stray frames (e.g. an llm_response nobody is waiting for) are
            # dropped: the protocol keeps strict request/response alternation.
    finally:
        with contextlib.suppress(OSError, ValueError):
            outp.flush()
    return 0


if __name__ == "__main__":
    sys.exit(_main())

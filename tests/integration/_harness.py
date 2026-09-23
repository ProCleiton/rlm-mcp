"""Real end-to-end MCP harness for rlm-mcp (shared by the integration tests).

Spawns the actual ``rlm-mcp`` console script (located via ``shutil.which``)
as a stdio MCP server and drives it through the official Python client SDK
(``mcp.client.stdio`` + ``ClientSession``). Every tool answer is the
server's plain JSON payload (``result.to_dict()``), parsed back into a dict:
nothing here touches the core package in-process, so these tests exercise
the exact transport a real harness (Claude Code, Cursor, a custom loop, ...)
would use.

The server never calls a language model: sub-completions requested by
sandbox code (``needs_llm``) are answered deterministically by the helpers
below (identifier extraction over a synthetic corpus), which lets the tests
drive the full suspend/resume loop without any network or model.

Teardown closes the MCP session and then the stdio transport; the SDK's
shutdown sequence (close stdin, bounded wait, process-tree kill) guarantees
no orphan ``rlm-mcp`` process survives the context.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

EXPECTED_TOOLS: frozenset[str] = frozenset(
    {
        "rlm_open",
        "rlm_exec",
        "rlm_exec_async",
        "rlm_wait",
        "rlm_resume",
        "rlm_peek",
        "rlm_status",
        "rlm_close",
    }
)

#: Chunk geometry used by the DOC-extraction sandbox code below.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

_DOC_TEMPLATE = "Document {i:05d} reports identifier DOC-{i:05d} with measured value {v}.\n"


def make_doc(i: int) -> str:
    """One deterministic corpus document (constant width except the value)."""
    return _DOC_TEMPLATE.format(i=i, v=i * 37 % 100000)


def make_corpus(n: int) -> str:
    """Deterministic corpus of ``n`` documents, one DOC-xxxxx id each."""
    return "".join(make_doc(i) for i in range(n))


def make_corpus_for_chars(target: int) -> tuple[str, int]:
    """Whole documents until the corpus reaches ``target`` chars.

    Returns ``(corpus, n_docs)`` where ``n_docs`` is the ground-truth id
    count (the last document may push the total past ``target``).
    """
    parts: list[str] = []
    total = 0
    i = 0
    while total < target:
        doc = make_doc(i)
        parts.append(doc)
        total += len(doc)
        i += 1
    return "".join(parts), i


def answer_doc_ids(prompt: str) -> str:
    """Deterministic stand-in for the harness's model on the extraction task."""
    found = ";".join(re.findall(r"DOC-\d+", prompt))
    return found or "NONE"


def found_doc_ids(answer: str) -> set[str]:
    """Complete 5-digit identifiers present in an answer.

    An identifier bisected by a chunk cut can only surface as a partial
    token (``DOC-0005``), which never matches; with chunk overlap >= the
    identifier width every cut id is re-read in full by the next chunk.
    """
    return set(re.findall(r"DOC-\d{5}", answer))


def doc_extraction_code() -> str:
    """The sandbox code the harness drives: chunk the context, one llm_query
    per chunk, then finalize with the collected per-chunk answers.

    The overlap (server default 200 chars) guarantees identifiers straddling
    a chunk cut are never lost; fixed-size cuts without overlap can bisect
    an id, which would silently drop it from every per-chunk prompt.
    """
    return (
        "parts = []\n"
        f"for chunk in chunk_text(context, size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}):\n"
        '    parts.append(llm_query("Extract every DOC-xxxxx identifier from this text. '
        'Answer semicolon-separated or NONE: " + chunk))\n'
        'FINAL_VAR("parts")'
    )


class Harness:
    """One ``rlm-mcp`` stdio server process plus a connected MCP client.

    ``async with Harness(max_llm_calls=...) as h:`` spawns the server with a
    high ``--max-llm-calls`` default (so large extraction loops never hit an
    early per-session ``exhausted`` unless a test overrides the limit in
    ``rlm_open``) and a private trajectory dir removed at teardown.
    """

    def __init__(self, *, max_llm_calls: int = 100_000, log_level: str = "ERROR") -> None:
        self._max_llm_calls = max_llm_calls
        self._log_level = log_level
        self._transport_cm: Any | None = None
        self._session: ClientSession | None = None
        self._trajectory_dir: Path | None = None
        self._entered = False

    async def __aenter__(self) -> Harness:
        if self._entered:
            raise RuntimeError("Harness is not re-entrant")
        binary = shutil.which("rlm-mcp")
        if binary is None:
            raise RuntimeError(
                "rlm-mcp executable not found on PATH; is the package installed in this venv?"
            )
        trajectory_dir = Path(tempfile.mkdtemp(prefix="rlm-mcp-harness-"))
        params = StdioServerParameters(
            command=binary,
            args=[
                "--max-llm-calls",
                str(self._max_llm_calls),
                "--trajectory-dir",
                str(trajectory_dir),
                "--log-level",
                self._log_level,
            ],
        )
        self._transport_cm = stdio_client(params)
        try:
            read_stream, write_stream = await self._transport_cm.__aenter__()
        except BaseException:
            self._transport_cm = None
            shutil.rmtree(trajectory_dir, ignore_errors=True)
            raise
        self._trajectory_dir = trajectory_dir
        session = ClientSession(read_stream, write_stream)
        try:
            await session.__aenter__()
            await session.initialize()
        except BaseException:
            await session.__aexit__(None, None, None)
            await self._transport_cm.__aexit__(None, None, None)
            self._transport_cm = None
            shutil.rmtree(trajectory_dir, ignore_errors=True)
            raise
        self._session = session
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._entered = False
        session, self._session = self._session, None
        transport_cm, self._transport_cm = self._transport_cm, None
        try:
            if session is not None:
                await session.__aexit__(exc_type, exc, tb)
        finally:
            try:
                if transport_cm is not None:
                    await transport_cm.__aexit__(exc_type, exc, tb)
            finally:
                if self._trajectory_dir is not None:
                    shutil.rmtree(self._trajectory_dir, ignore_errors=True)
                    self._trajectory_dir = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Harness must be used as an async context manager")
        return self._session

    async def list_tools(self) -> list[str]:
        """Tool names as registered by the running server."""
        listing = await self._require_session().list_tools()
        return [tool.name for tool in listing.tools]

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one tool and return its parsed JSON payload."""
        result = await self._require_session().call_tool(name, arguments)
        text = result.content[0].text  # type: ignore[union-attr]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise AssertionError(f"tool {name} returned a non-object payload: {payload!r}")
        return payload


@dataclass
class LlmCycle:
    """One ``needs_llm`` step: the pending requests and the answers given."""

    requests: list[dict[str, Any]]
    answers: list[str]


@dataclass
class ExtractionRun:
    """Everything observed while driving one extraction session to completion."""

    opened: dict[str, Any]
    session_id: str
    code: str
    cycles: list[LlmCycle] = field(default_factory=list)
    final: dict[str, Any] | None = None


async def run_doc_extraction(
    harness: Harness,
    *,
    corpus: str,
    n_docs: int,
    extra_limits: dict[str, Any] | None = None,
) -> ExtractionRun:
    """Open a session over ``corpus`` and drive the DOC-extraction loop to
    completion (DESIGN section 4, exactly what a real harness does).

    ``rlm_open`` -> ``rlm_exec``; while the server reports ``needs_llm``,
    answer every pending request with ``answer_doc_ids`` and call
    ``rlm_resume``. Each resume unblocks the suspended ``llm_query`` inside
    the sandbox and the loop continues from the exact suspension point.
    """
    limits: dict[str, Any] = {"max_llm_calls": n_docs + 100}
    if extra_limits:
        limits.update(extra_limits)
    opened = await harness.call("rlm_open", {"text": corpus, "limits": limits})
    if opened.get("status") != "ok":
        raise AssertionError(f"rlm_open failed: {opened}")
    session_id = opened["session_id"]
    code = doc_extraction_code()
    run = ExtractionRun(opened=opened, session_id=session_id, code=code)
    response = await harness.call("rlm_exec", {"session_id": session_id, "code": code})
    while response.get("status") == "needs_llm":
        requests = list(response.get("requests") or [])
        answers = [answer_doc_ids(str(req.get("prompt", ""))) for req in requests]
        run.cycles.append(LlmCycle(requests=requests, answers=answers))
        results = [
            {"id": req["id"], "text": text} for req, text in zip(requests, answers, strict=True)
        ]
        response = await harness.call("rlm_resume", {"session_id": session_id, "results": results})
    run.final = response
    return run

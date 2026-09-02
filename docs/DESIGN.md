# RLM over MCP — design and protocol

Status: v0 (draft, implemented by `feat/rlm-over-mcp-core`)

## 1. What this is

`rlm-mcp` is a **harness-agnostic MCP server** that gives any MCP-capable agent
(Claude Code, Cursor, VS Code/Copilot, Zed, Cline, LibreChat, custom loops) the
**Recursive Language Model** inference paradigm.

The server **never calls a language model** and **never needs an API key**. The
harness's agent plays the role of the *root LM*; the server is the *environment*:
it holds the context as a variable, runs code against it in a persistent REPL,
suspends execution when the code asks for a sub-completion, and enforces budgets.

Reference paradigm: Recursive Language Models (Zhang & Khattab, arXiv 2512.24601
v3) — the long prompt becomes part of an external environment; the model
programmatically examines, decomposes, and recursively calls a model over
snippets of it.

## 2. Paradigm conformance (non-negotiable requirements)

| Id | Requirement | How `rlm-mcp` satisfies it |
|----|-------------|----------------------------|
| C1 | Drop-in completion interface | The harness keeps its own loop; `rlm_open`/`rlm_exec` are the interface |
| C2 | Context is a **symbolic handle**, not tokens | Context is loaded into the sandbox as `context`; the caller only ever receives *metadata* (type, length, line count, head/tail preview) |
| C3 | Protected window | Every payload returned to the agent is truncated (head+tail with an elision marker); large reads only through `rlm_peek` pagination |
| C4 | **Symbolic recursion** | `llm_query(...)` is callable *from inside running code*, in arbitrary loops. Execution suspends mid-loop and resumes with the answer. Two independent tools (`exec` + `sub_llm`) are explicitly the anti-pattern (paper, Algorithm 2) and are rejected |
| C5 | Explicit finalization | `FINAL(text)` / `FINAL_VAR(name)` terminate the trajectory |
| G1 | Budgets with fallback | Per-session and **tree-wide** ledgers: iterations, LLM calls, depth, wall time, output chars, errors |
| G2 | Persistent REPL | One long-lived sandbox process per session; namespace survives across `rlm_exec` calls |
| G3 | No credentials in the environment | The sandbox never sees provider keys — the model call is performed by the harness, outside the sandbox. Env is scrubbed |
| G4 | Model-family independence | The root playbook is served as an MCP *prompt*, overridable |
| G5 | Loggable trajectory | Versioned JSONL, exposed as an MCP resource |
| G6 | Default answer on exhaustion | Budget exhaustion returns `status="exhausted"` with the accumulated state, never a bare error |

## 3. Why continuation and not MCP `sampling`

`sampling/createMessage` would let a server request a completion from the client.
It is **not usable**:

- Unsupported by Claude Code (issue anthropics/claude-code#1785, open since
  2025-06-08), Cursor, Claude Desktop, Cline, Continue, Zed, Windsurf, LibreChat.
  Supported only by VS Code/Copilot and the official SDKs.
- **Deprecated** in MCP revision `2026-07-28` (SEP-2577): "New implementations
  SHOULD NOT adopt it."

So recursion is driven by the client, through tool calls: `rlm_exec` returns a
`needs_llm` state carrying the pending sub-queries; the agent answers them with
its own model (or its own subagent facility) and calls `rlm_resume`. The
harness's model and billing are used by construction.

Session state lives behind an **explicit handle** (`session_id` argument), the
pattern MCP itself moved to when `Mcp-Session-Id` and the `initialize` handshake
were removed in `2026-07-28` (SEP-2567). The server therefore works unchanged on
both `2025-11-25` and `2026-07-28`, over stdio and streamable HTTP.

## 4. Suspension mechanism (the core)

The sandbox is a **separate process** speaking newline-delimited JSON frames over
two pipes. Because it is a real process, `llm_query` can simply **block on a pipe
read** in the middle of an arbitrary loop; the interpreter's stack, and therefore
the loop, is preserved with zero rewriting of user code.

```
agent → rlm_exec(sid, code)
  supervisor → sandbox: {"op":"exec","code":...}
  sandbox: exec(code, ns)
    user code: for chunk in chunks: answers.append(llm_query(f"...{chunk}"))
      llm_query → sandbox → supervisor: {"op":"llm_request","requests":[...]}
      llm_query blocks reading stdin
  supervisor → agent: {"status":"needs_llm","requests":[...]}
agent (runs the sub-completions with ITS OWN model) → rlm_resume(sid, results)
  supervisor → sandbox: {"op":"llm_response","results":[...]}
  llm_query returns the string; the loop continues where it stopped
  ... repeats until:
  sandbox → supervisor: {"op":"exec_done","stdout":...} | {"op":"final","answer":...}
  supervisor → agent: {"status":"ok"|"final"|"error"|"exhausted", ...}
```

`llm_query_batched(prompts)` emits a single request frame with N items so the
agent can fan them out to parallel subagents. `rlm_query(prompt, context=...)`
marks a request as `kind="rlm"`, signalling the agent to delegate to a subagent
that opens a **child session** (`parent_session_id`), which is how depth > 1 is
achieved. When `depth >= max_depth` the server degrades `kind` to `"llm"`
(paper's base case) instead of failing.

## 5. MCP surface

Tool descriptions and server instructions must stay under **2 KB** each (Claude
Code truncates both at 2 KB). Tool results must stay well under the client
truncation limit (Claude Code: 25 000 tokens by default).

### Tools

`rlm_open` budget overrides are validated: every numeric limit must be
strictly positive (`max_depth` may be 0); anything else is refused with a
usage-error payload before any session starts. `rlm_peek` pages are capped
at an absolute per-call ceiling (`PEEK_CHAR_CAP`, 16 000 chars) no matter
what `limit` the caller asks for; `total`/`returned`/`truncated` keep
reporting the full value so larger reads page through in several calls.

`StepResult.status` ∈ `ok | needs_llm | final | error | exhausted`:

- `ok` → `{stdout (truncated), vars: [{name, type, size}], spent}`
- `needs_llm` → `{requests: [{id, kind: "llm"|"rlm", prompt (truncated for display), chars, suggested_context?}], spent}`
- `final` → `{answer, spent, trajectory}`
- `error` → `{error: {type, message, traceback (truncated)}, spent}`
- `exhausted` → `{reason, spent, limits, partial: {vars, stdout}}`

### Prompt

`rlm_playbook` — the root-LM instruction set: context is a variable named
`context`; you only see metadata; write code; call `llm_query`/`llm_query_batched`
inside loops; never paste large slices into your own reasoning; finish with
`FINAL`/`FINAL_VAR`.

### Resource

`rlm://trajectory/{root_id}` — JSONL of the whole tree.

## 6. Sandbox namespace

Injected, reserved names: `context`, `context_parts`, `history`, `llm_query`,
`llm_query_batched`, `rlm_query`, `rlm_query_batched`, `FINAL`, `FINAL_VAR`,
`SHOW_VARS`, `chunk_text`. Rebinding a reserved name is refused at `rlm_exec`
time with a clear message.

`FINAL` raises an internal sentinel exception so execution stops immediately.

## 7. Security posture (stated honestly)

Default driver = local subprocess: `python3 -I -S`, `RLIMIT_AS`, `RLIMIT_CPU`,
`RLIMIT_FSIZE`, `RLIMIT_NPROC`, dedicated temp cwd, environment scrubbed of
`*_KEY`/`*_TOKEN`/`*_SECRET`/`*_PASSWORD` and of provider variables. This is
**containment, not isolation**: the code can still read files the user can read.
Untrusted context or untrusted code must use the Docker driver
(`--driver docker`, no network, read-only mounts), planned right after v0.
`RestrictedPython` is deliberately **not** used: it is not a security boundary,
and it breaks the very idiom the paradigm depends on.

## 8. Internal API (module contract)

```python
# rlm_mcp/types.py
Kind = Literal["llm", "rlm"]
Status = Literal["ok", "needs_llm", "final", "error", "exhausted"]

@dataclass(frozen=True)
class Limits:
    max_iterations: int = 30
    max_llm_calls: int = 60
    max_depth: int = 2
    max_wall_seconds: float = 900.0
    max_exec_seconds: float = 120.0
    max_output_chars: int = 8_000
    max_errors: int = 5

@dataclass(frozen=True)
class OpenSpec:
    text: str | None = None
    paths: tuple[str, ...] = ()
    parent_session_id: str | None = None
    limits: Limits = Limits()
    label: str | None = None

@dataclass(frozen=True)
class SubRequest:  id: str; kind: Kind; prompt: str; context: str | None
@dataclass(frozen=True)
class SubResult:   id: str; text: str | None; error: str | None = None
@dataclass(frozen=True)
class StepResult:  status: Status; ... (fields per §5)

# rlm_mcp/session.py
class SessionManager:
    async def open(self, spec: OpenSpec) -> StepResult          # status="ok"
    async def exec(self, sid: str, code: str) -> StepResult
    async def resume(self, sid: str, results: Sequence[SubResult]) -> StepResult
    async def peek(self, sid: str, expr: str, offset: int, limit: int) -> PeekResult
    def status(self, sid: str) -> StatusResult
    async def close(self, sid: str) -> list[str]                # closes subtree
    async def sweep(self) -> None                               # TTL eviction
```

`SessionManager` owns the tree-wide `BudgetLedger` (keyed by root session id) and
the `TrajectoryWriter`. The MCP layer is a thin, dumb adapter over it: it does
no state handling of its own.

## 9. Out of scope for v0

Docker driver, streamable HTTP transport, prefix caching, async sub-call
pipelining, trajectory visualizer, non-Python REPLs.

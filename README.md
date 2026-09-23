# rlm-mcp

> **Installing and wiring a harness:** see
> [`docs/install-and-usage.md`](docs/install-and-usage.md).
> **Offloading a harness's own initial context:** see
> [`docs/harness-context-offload.md`](docs/harness-context-offload.md).

Harness-agnostic MCP server that brings the **Recursive Language Model (RLM)**
paradigm to any agent that speaks MCP — Claude Code, Cursor, VS Code/Copilot,
Zed, Cline, LibreChat, custom loops.

The server **never calls a language model and never needs an API key**. Your
harness's agent is the *root LM*; the server is the *environment*: it holds a
long context as a variable inside a persistent Python REPL, runs code against
it, suspends execution when the code asks for a sub-completion, and enforces
budgets. Billing and model choice stay with your harness, by construction.

The paradigm is that of *Recursive Language Models* (Zhang & Khattab,
[arXiv 2512.24601](https://arxiv.org/abs/2512.24601)): instead of stuffing a
long prompt into the context window, the model treats it as an external
resource — it examines it programmatically, decomposes it, and recursively
processes the pieces with short, focused sub-completions. **This project is an
independent implementation of the paradigm, not affiliated with the paper's
authors.**

## Why continuation, and not MCP `sampling`

`schema`-era `sampling/createMessage` would let a server ask the client for a
completion, but it is unusable here:

- Claude Code, Cursor, Claude Desktop, Cline, Continue, Zed, Windsurf and
  LibreChat do not support it (only VS Code/Copilot and the official SDKs do).
- It was **deprecated** in the MCP revision of 2026-07-28 (SEP-2577): new
  implementations SHOULD NOT adopt it.

So recursion is driven by the *client*: `rlm_exec` returns a `needs_llm` state
carrying the pending sub-questions; you answer them with your own model (or
your own subagent facility) and call `rlm_resume`. Your model and your budget
are used — the server has neither.

## The cycle

```
 agent (root LM)                    server (sandbox REPL + budgets)
   │
   │  rlm_open(text?, paths?, parent_session_id?)
   ├──────────────────────────────────────────────► load `context`, open session
   │  {session_id, depth, context: <metadata>, budget}
   │
   │  rlm_exec(session_id, code)     code calls llm_query(...) inside a loop
   ├──────────────────────────────────────────────► execution suspends mid-loop
   │  {status: "needs_llm", requests: [{id, kind, prompt}]}
   │
   │  ── you answer each request with YOUR OWN model ────────────────┐
   │  ── (kind="llm": inline or parallel subagents)                  │
   │  ── (kind="rlm": a subagent opens a child session)              │
   │                                                                 │
   │  rlm_resume(session_id, results=[{id, text}, ...])              │
   ├──────────────────────────────────────────────► llm_query returns;      │
   │  {status: "ok" | "needs_llm" | ...}          the loop continues ──────┘
   │  ... until your code calls FINAL / FINAL_VAR ...
   │  rlm_close(session_id)
```

The context lives as the variable `context` in the sandbox. You only ever see
*metadata* about it (type, length, head/tail preview); payloads are truncated,
and large reads go through `rlm_peek` pagination. Child sessions
(`parent_session_id`) give you recursion depth > 1: a `kind="rlm"` request is
the signal to delegate to a subagent that opens a child session for that slice
of work.

## Tools

| Tool | Input | Output |
|------|-------|--------|
| `rlm_open` | `text?`, `paths?`, `parent_session_id?`, budget overrides | `{session_id, depth, context: <metadata>, budget}` |
| `rlm_exec` | `session_id`, `code` | step result: `ok` / `needs_llm` / `final` / `error` / `exhausted` |
| `rlm_resume` | `session_id`, `results: [{id, text, error?}]` | step result (see `rlm_exec`) |
| `rlm_peek` | `session_id`, `expr`, `offset?`, `limit?` | `{text, offset, returned, total, truncated}` |
| `rlm_status` | `session_id` | `{depth, spent, limits, state, trajectory}` |
| `rlm_close` | `session_id` | `{closed: [session_id, ...]}` |

`rlm_exec` runs Python in the sandbox namespace; the namespace persists across
calls. Reserved names (`context`, `llm_query`, `llm_query_batched`, `rlm_query`,
`rlm_query_batched`, `FINAL`, `FINAL_VAR`, ...) cannot be rebound. Ask the
`rlm_playbook` prompt for the full protocol and an idiomatic code example.
Trajectories are exposed as the resource `rlm://trajectory/{root_id}` (JSONL of
the whole session tree).

## Installing and configuring

Requires Python >= 3.10 and a client that supports MCP over stdio. The package
is distributed **from GitHub only** — there is no PyPI release. Install it
with `uv tool install --from git+https://github.com/ProCleiton/rlm-mcp
rlm-mcp`, or clone the repository and run `./install.sh` (prefers `uv`, then
`pipx`, then `pip`; never `sudo`). The examples below launch the installed
`rlm-mcp` binary directly. See
[`docs/install-and-usage.md`](docs/install-and-usage.md) for the full guide
and ready-to-paste config blocks for omp, Claude Code, Cursor and VS Code.

Claude Code (`.mcp.json` in the project root, or `claude mcp add`):

```json
{
  "mcpServers": {
    "rlm-mcp": {
      "command": "rlm-mcp",
      "args": []
    }
  }
}
```

Cursor (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "rlm-mcp": {
      "command": "rlm-mcp",
      "args": []
    }
  }
}
```

VS Code / Copilot (`.vscode/mcp.json`):

```json
{
  "servers": {
    "rlm-mcp": {
      "command": "rlm-mcp",
      "args": []
    }
  }
}
```

The server speaks stdio only in v0. Useful options: `--max-depth`,
`--max-iterations`, `--max-llm-calls`, `--trajectory-dir`, `--log-level`
(logs go to stderr; stdout is the MCP channel). Run `rlm-mcp --help` for
defaults.

## Security posture (honest)

The default driver is a **local subprocess**: `python3 -I -S` with
`RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE` and `RLIMIT_NPROC`, a dedicated temp
working directory, and an environment scrubbed of `*_KEY` / `*_TOKEN` /
`*_SECRET` / `*_PASSWORD` and provider variables.

That is **containment, not isolation**: code you run can still read files your
user can read. Treat the sandbox as an extension of your own shell — never feed
it context or code you would not run yourself. A Docker driver (no network,
read-only mounts) is planned as the isolation boundary. `RestrictedPython` is
deliberately **not** used: it is not a security boundary and it breaks the
suspension idiom the paradigm depends on.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run rlm-mcp --help
```

Layout: `src/rlm_mcp/server.py` is the thin MCP adapter (eight tools, one
prompt, one resource); the core (`session.py`, `sandbox/`, `budget.py`,
`trajectory.py`, `types.py`) implements sessions, the REPL driver and the
budget ledgers. Read `docs/DESIGN.md` for the protocol.

## License

Apache-2.0. This project is an independent implementation of the RLM paradigm;
it is not affiliated with the authors of the paper, and any reference to
"Recursive Language Models" is a citation of the paradigm, not an endorsement.

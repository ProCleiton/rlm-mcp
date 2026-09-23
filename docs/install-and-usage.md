# Installing and using rlm-mcp

See the [project README](../README.md) for the overview, and
[`docs/DESIGN.md`](DESIGN.md) for the protocol design.

rlm-mcp is an MCP server that runs as a **stdio child process** of your
harness (oh-my-pi/omp, Claude Code, Cursor, VS Code/Copilot, or any MCP
client). The server never calls a language model and never needs an API key:
your harness's own agent answers the sub-completions, so model choice and
billing stay with the harness.

## 1. Install

**Distribution is GitHub-only.** rlm-mcp is not published to PyPI — every
command below pulls the package from the git repository
`https://github.com/ProCleiton/rlm-mcp`. Requires Python >= 3.10.

Pick one:

**(a) Clone and run `install.sh` (recommended)**

```bash
git clone https://github.com/ProCleiton/rlm-mcp
cd rlm-mcp
./install.sh
```

`install.sh` installs from GitHub and prefers `uv`, then `pipx`, then `pip`
(never `sudo`). When it finishes it prints the installed binary and
ready-to-paste MCP config blocks for omp, Claude Code, Cursor and VS Code.
Options: `--branch <ref>` (install a git ref instead of `main`), `--dev`
(editable install from the checkout), `--force` (reinstall), `-h/--help`.

**(b) One-liner with `uv`**

```bash
uv tool install --from git+https://github.com/ProCleiton/rlm-mcp rlm-mcp
```

**(c) `pipx`**

```bash
pipx install git+https://github.com/ProCleiton/rlm-mcp
```

**(d) Development / editable install**

```bash
git clone https://github.com/ProCleiton/rlm-mcp
cd rlm-mcp
uv sync --extra dev    # editable install + dev deps into .venv
```

Without `uv`: `python3 -m pip install --user -e '.[dev]'`. The dev binary is
`.venv/bin/rlm-mcp`; run it directly or via `uv run rlm-mcp --help`.

Verify the install:

```bash
rlm-mcp --version    # -> rlm-mcp 0.1.0
```

## 2. Wire a harness

The server speaks MCP over **stdio only** in v0: the harness launches the
binary as a child process and speaks JSON-RPC over stdin/stdout. Logs go to
stderr; stdout is the MCP channel. Because the server has no model of its
own, there is **no API key, endpoint or model configuration** anywhere in
these blocks.

In the JSON below, `<bin>` is either:

- `rlm-mcp` — if the install directory is on the harness's PATH, or
- the absolute path to the binary — `$HOME/.local/bin/rlm-mcp` is the
  default target of `uv tool`, `pipx` and `pip install --user`. JSON config
  files do not expand `~`, so write the full path (for example
  `/home/you/.local/bin/rlm-mcp`).

Restart the harness after adding an entry. Server flags such as
`--max-depth`, `--max-iterations`, `--max-llm-calls`, `--trajectory-dir` or
`--log-level` go into `"args"` (for example
`"args": ["--max-depth", "3", "--log-level", "DEBUG"]`).

### oh-my-pi / omp

File: `~/.omp/agent/mcp.json`. Add the entry under the top-level
`"mcpServers"` object (merge it in if the file already lists other servers):

```json
{
  "mcpServers": {
    "rlm-mcp": {
      "type": "stdio",
      "command": "<bin>",
      "args": []
    }
  }
}
```

### Claude Code

Either create `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "rlm-mcp": {
      "command": "<bin>",
      "args": []
    }
  }
}
```

or register it with the CLI:

```bash
claude mcp add rlm-mcp -- <bin>
```

### Cursor

File: `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "rlm-mcp": {
      "command": "<bin>",
      "args": []
    }
  }
}
```

### VS Code / Copilot

File: `.vscode/mcp.json` (note the `"servers"` top-level key):

```json
{
  "servers": {
    "rlm-mcp": {
      "command": "<bin>",
      "args": []
    }
  }
}
```

### Generic stdio JSON

Any MCP client that launches stdio servers accepts this shape:

```json
{
  "mcpServers": {
    "rlm-mcp": {
      "command": "<bin>",
      "args": []
    }
  }
}
```

## 3. How the harness drives the loop

rlm-mcp implements the RLM (Recursive Language Model) paradigm. A long
document is loaded into a sandbox REPL as the variable `context`; the
harness's agent — the *root LM* — works on it by running Python code, never
by having the raw text pasted into the conversation. The server exposes eight
tools:

| Tool | Input | Output |
| --- | --- | --- |
| `rlm_open` | `text?`, `paths?`, `parent_session_id?`, `limits?`, `mode?` (`"doc"` default, `"exec"` opt-in), `trusted_env?` (only with `mode="exec"`) | `{session_id, depth, context: <metadata>, budget}` |
| `rlm_exec` | `session_id`, `code` | step result: `ok` / `needs_llm` / `final` / `error` / `exhausted` |
| `rlm_exec_async` | `session_id`, `code` | unique `{handle, state}`; multiple jobs queue FIFO per session |
| `rlm_wait` | `handle`, `timeout?` | terminal step result, or `{status: "pending", state, elapsed}` |
| `rlm_resume` | `session_id`, `results: [{id, text, error?}]` | step result (see `rlm_exec`) |
| `rlm_peek` | `session_id`, `expr`, `offset?`, `limit?` | truncated page `{text, offset, returned, total, truncated}` |
| `rlm_status` | `session_id` | `{depth, spent, limits, state, jobs: [{handle, state, elapsed}], trajectory}` |
| `rlm_close` | `session_id` | `{closed: [...]}` |

The cycle is driven by the harness agent:

1. `rlm_open` loads the document (from `text` and/or `paths`) into the
   sandbox. The agent only receives **metadata** — type, length, head/tail
   preview — plus `session_id`, `depth` and the budget. Passing
   `parent_session_id` opens a *child* session, which is how recursion depth
   > 1 works. `mode` defaults to `"doc"` (current behavior, unchanged); pass
   `mode="exec"` plus `trusted_env={...}` for builds/long shell runs
   (key names `^[A-Z][A-Z0-9_]{1,63}$`, no `RLM_` prefix, no `KEEP_ENV`
   overrides; only names are logged). Raised ceilings use the existing
   `limits` overrides, e.g. `limits={"max_exec_seconds": 600,
   "max_wall_seconds": 900}`.
2. `rlm_exec` runs Python in the sandbox namespace, which persists across
   calls. The code sees the document as `context` and can call the reserved
   helpers `llm_query(prompt)`, `llm_query_batched([...])`,
   `rlm_query(...)`, and the terminals `FINAL(text)` / `FINAL_VAR("name")`.
3. When the code calls `llm_query`, the sandbox **suspends mid-loop** and
   `rlm_exec` returns `{status: "needs_llm", requests: [{id, kind, prompt},
   ...]}`. The agent answers each request with its own model:
   - `kind: "llm"` — answer inline, or fan the ids out to its own parallel
     subagents, keeping each answer short;
   - `kind: "rlm"` — recursion: delegate to a subagent that opens a child
     session with `rlm_open(parent_session_id=<sid>)` for that slice and
     returns its final answer.
   Large batches trim prompts to a per-call budget and carry an elision
   marker; the full prompt can be read with `rlm_peek`, which also works
   while the session is parked.
4. The agent calls `rlm_resume(session_id, results=[...])` covering **every**
   pending id (omitted, unknown or duplicated ids are rejected). The code
   resumes where it stopped; the loop repeats until the code hits a
   terminal.
5. `rlm_status` reports the real budget (depth, iterations, LLM calls,
   output size, wall time) against the limits — including pending async jobs
   as `jobs: [{handle, state, elapsed}]`; the agent should plan the
   decomposition up front and let child sessions do the heavy lifting.
6. Long runs use the async flow: `rlm_exec_async(session_id, code)` returns a
   unique `{handle, state}` immediately and several jobs per session queue
   FIFO (the sandbox stays single-flight). `rlm_wait(handle, timeout=30)`
   collects one job — finished jobs return the terminal step result
   (`needs_llm` still resumes via the synchronous `rlm_resume`); a
   queued/running job past `timeout` returns `{status: "pending", state,
   elapsed}` without cancelling, so poll again later. Collection order is
   independent of FIFO execution order.
7. The code must end with `FINAL(text)` or `FINAL_VAR("name")`; the agent
   then closes finished sessions with `rlm_close` so their budgets release
   (closing cancels queued/running/uncollected async jobs; a synchronous
   step in flight still refuses `close`).

### Minimal example

This is the kind of code the agent sends to `rlm_exec` (summarizing a long
`context`, chunk by chunk). Each `llm_query` suspends the sandbox and asks
the harness's agent for a completion:

```python
STEP = 4000
summaries = []
for start in range(0, len(context), STEP):
    chunk = context[start:start + STEP]
    note = llm_query("Summarize this slice in at most 3 sentences:\n" + chunk)
    summaries.append(note)

merged = llm_query(
    "Merge these section summaries into one coherent whole:\n"
    + "\n".join(summaries)
)
FINAL(merged)   # or FINAL_VAR("merged") to end with a variable's value
```

Every time the code calls `llm_query`, the harness sees `needs_llm` with one
or more requests, answers them through `rlm_resume`, and the loop continues
where it stopped — the answers are handed back to the code as strings, so
what crosses the session boundary stays small.

### Exec-mode example (real command via subprocess)

```python
# sid = rlm_open(paths=[...], mode="exec", trusted_env={"CI": "1"},
#                limits={"max_exec_seconds": 600, "max_wall_seconds": 900})
# rlm_exec(sid, "import subprocess; "
#     "out = subprocess.run(['make', '-j4'], capture_output=True, text=True, timeout=500); "
#     "print(out.stdout[-4000:])")
```

Exec sessions raise the `RLIMIT_NPROC` default to 2048 (`doc` stays at 256);
an explicit `RLM_RLIMIT_NPROC` operator env var always wins, and heavily
loaded hosts may still need to raise it manually.

The server advertises the prompt **`rlm_playbook`** (the short version is
also served as the server `instructions` field): it teaches the agent the
full protocol — reserved names, chunking idiom, subagent fan-out for
`kind="rlm"`, budget rules — and is the canonical source for agent-facing
behavior. Its text lives in `src/rlm_mcp/playbook.py`.

## 4. Security

The default driver runs your code in a **local subprocess** (`python3 -I -S`)
with `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE` and `RLIMIT_NPROC` (default
256 for `mode="doc"`, 2048 for `mode="exec"`; explicit `RLM_RLIMIT_NPROC`
always wins), a dedicated temp working directory, and an environment scrubbed
of `*_KEY`/`*_TOKEN`/`*_SECRET`/`*_PASSWORD` and provider variables.
`mode="exec"` sessions may additionally reinject a whitelisted `trusted_env`
(validated before the session is created; values never logged).

That is **containment, not isolation**: code you run can still read files
your user can read. Treat the sandbox as an extension of your own shell —
never feed it context or code you would not run yourself. A Docker driver
(no network, read-only mounts) is planned as the isolation boundary.

## 5. More documentation

- [Project README](../README.md) — paradigm, cycle diagram, tools and CLI
  options.
- [`docs/DESIGN.md`](DESIGN.md) — protocol design and implementation notes.
- `src/rlm_mcp/playbook.py` — the `rlm_playbook` prompt text.

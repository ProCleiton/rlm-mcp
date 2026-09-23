# Harness context offload (RLM as the harness's context provider)

This document describes how a coding harness (Claude Code, Cursor, VS Code
Copilot, omp, a custom loop) can use `rlm-mcp` to stop injecting its whole
initial context into the model's context window, and instead hold that context
**inside an RLM session** and hand the model only metadata + a handle.

Status: the **agent-side** part works today. The **harness-side** part (auto-load
at session open) is a feature spec — it requires a small change in the harness,
not in `rlm-mcp`.

## 1. The problem

Every harness loads a fixed preamble into the model window at session start:
system prompt, workspace instructions (`AGENTS.md`/`RULES.md`/`SYSTEM.md`),
skill descriptions, MCP tool descriptions, and project context. That preamble is
present in every turn, and the window then **grows dynamically** through the
session (conversation history + tool outputs + file reads). Context rot is
cumulative: the fuller the window, the more the model degrades.

The RLM paradigm fixes exactly this: hold the context as a **variable** outside
the window, show the model only **metadata**, and let it query the context
programmatically. `rlm-mcp` implements that as a plain MCP tool set.

## 2. What works today (agent-side discipline)

No harness change is required for this part:

- Mount `rlm-mcp` as an MCP server (stdio). The model gets eight tools
  (`rlm_open`/`rlm_exec`/`rlm_exec_async`/`rlm_wait`/`rlm_resume`/`rlm_peek`/`rlm_status`/`rlm_close`).
- Instruct the agent to offload real context **early** — the moment a task has
  real input (files, docs, codebase, corpus), load it into `rlm_open` and work it
  with `rlm_exec`/`llm_query` over chunks instead of reading it into the window.
  The `rlm_playbook` prompt served by the server encodes the full protocol, and
  the `skill://rlm-usage` skill is the reusable form of the same guidance.

The efficiency is measurable (see `benchmarks/efficiency.py`): the root-LM
working window stays **constant** (~500 tokens: metadata + code + short answers)
while the input grows from 2.4k to 479k tokens (198x), and past a ~200k-token
window the naive baseline no longer fits at all while the RLM still completes.

Limitation: this is opt-in discipline. The harness still injects its own fixed
preamble; RLM only offloads the *working material* the agent chooses to route
there, not the preamble itself, and not the conversation history.

## 3. The harness feature (spec)

To make the offload automatic — the harness's initial context lives in RLM from
the very first turn — the harness needs one small capability: a **context
provider** that replaces the full-text preamble with an RLM handle + metadata.

Reference implementation sketch:

1. On session start, the harness collects the preamble sources (its system
   prompt, workspace `AGENTS.md`/`RULES.md`/`SYSTEM.md`, skill bodies, MCP tool
   descriptions, project files) and calls `rlm_open(paths=[...])` (or
   `rlm_open(text=...)`), producing a `session_id`. (`rlm_open` also accepts
   `mode` (`"doc"` default, `"exec"` opt-in) and `trusted_env` (only with
   `mode="exec"`) for builds/long shell runs — see
   `docs/install-and-usage.md` §3; preamble offload itself stays `mode="doc"`.)
2. It injects into the model window, instead of the full text, a compact block:
   - per source: name + char/line count + a short head/tail preview (the
     `context` metadata `rlm_open` already returns — never the raw text);
   - the `session_id` handle;
   - a one-line instruction: "your context is in RLM session `<id>`; query it
     with `rlm_exec`/`llm_query` rather than asking for the raw text."
3. The agent then uses the eight RLM tools to inspect/process the context on
   demand. Subagents inherit the same handle (or open child sessions with
   `parent_session_id`).

This is exactly the RLM requirements C2 (context is a symbolic handle, caller
only sees metadata) and C3 (protected window), applied to the harness's own
preamble.

### Where the harness hook lives (per harness)

- **omp**: a `session_start`/`context` hook in the extension API (today only
  `tool_call` is exposed), or a startup context-provider key in `config.yml`.
- **Claude Code / Cursor / VS Code**: a startup hook / context provider (MCP
  servers are mounted as tools; the preamble is assembled by the client runtime,
  so this is a client-side feature).

None of these exist today; the spec above is the shape of the change.

## 4. Why "early" and not "when it is large"

The context does not need to be large to start. The window grows dynamically and
rot compounds, so the earlier the working material lives in RLM instead of the
window, the less degradation accumulates. Load early, keep it lean throughout.

## 5. What RLM does not offload

RLM offloads the **environment** (files, documents, codebase, corpus). It does
not offload the agent's own reasoning or the conversation history — those still
accumulate in the window. Full protection of the reasoning history is a separate
problem (compaction/summarization), orthogonal to RLM.

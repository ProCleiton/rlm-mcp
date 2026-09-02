# Context-efficiency benchmark

`efficiency.py` measures the reason the Recursive Language Model paradigm
exists: what a naive long-context model would pay to process a corpus, vs
what the root model actually puts in its window when the context is an
external resource held by `rlm-mcp`.

It drives the **real** MCP server over stdio (spawned by the same harness
as `tests/integration/`), runs the DOC-extraction loop (Test 2): the sandbox
chunks the context with `chunk_text(size=2000, overlap=200)` and issues one
`llm_query` per chunk; the harness answers every `needs_llm` deterministically
(regex extraction, no model, no network) and calls `rlm_resume` until the
code finalizes. Nothing runs inside a model: the numbers are pure token
accounting of what each side would send to / hold in a context window.

## What is measured, per corpus size N (chars)

| Column | Meaning |
| --- | --- |
| `N_chars` | actual corpus size (whole generated documents until >= target) |
| `context_tokens` | `cl100k_base` tokens of the full corpus |
| `baseline_peak` | the naive baseline: window = whole context in one shot (`= context_tokens`) |
| `rlm_root_peak` | the root model's window at any instant: `metadata_tokens + code_tokens + max(answer_tokens)` — metadata from `rlm_open` (`context` meta, head/tail previews), the code sent to `rlm_exec`, and the sub-answers it submits. **The context itself never enters this window.** |
| `rlm_sub_peak` | the largest sub-call window: `max(tokenize(request.prompt))` = one chunk (~2000 chars) |
| `rlm_total` | total tokens moved: `code_tokens + sum(prompt_tokens) + sum(answer_tokens)` — the cost of *reading* the context |
| `root_peak_reduction_x` | `context_tokens / rlm_root_peak` |
| `beyond_window` | `yes` when `context_tokens > 200_000` (a typical long-context window) |

Each run is verified for correctness before being reported: the final
answer must contain every ground-truth `DOC-xxxxx` id.

## Tokenizer caveat

Tokens are counted with `tiktoken`'s `cl100k_base` as a proxy (~4 chars per
token on this corpus). Absolute numbers are proxy numbers; what matters is
the **relative** ratio, which is insensitive to the exact tokenizer.

## Honest interpretation

RLM does not promise fewer **total** tokens: every chunk still has to be
read (`rlm_total` grows ~linearly with N, same as the baseline's one-shot
read). It promises (1) a **protected working window** — the root model only
ever holds metadata + code + short sub-answers, ~constant no matter how
large the context grows — and (2) the ability to **work past the window**:
when `context_tokens` exceeds the window the baseline cannot run at all,
while the RLM still concludes with sub-call windows of a single chunk.

## Running

```bash
uv sync --extra dev          # installs tiktoken (dev dependency)
uv run python benchmarks/efficiency.py
```

No external network is used (corpus is generated; the tokenizer is local;
the server is spawned locally). A run takes roughly a minute and prints the
table plus three reading lines explaining (a) the flat root window, (b) the
~linear total, (c) the beyond-window comparison.

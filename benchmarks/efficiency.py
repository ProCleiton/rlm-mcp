"""Context-efficiency benchmark for rlm-mcp (the reason the paradigm exists).

Drives the REAL ``rlm-mcp`` server over MCP (same loop as Test 2) at
increasing context sizes and reports the token economics: what a naive
long-context baseline would pay (the whole corpus in the window once) vs
what the RLM actually puts in the root model's window (metadata + code +
sub-answers), the sub-call window (one chunk), and the total token flow
(read all chunks).  Full methodology in benchmarks/README.md.

Standalone, offline: ``uv run python benchmarks/efficiency.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.integration._harness import (
    Harness,
    found_doc_ids,
    make_corpus_for_chars,
    run_doc_extraction,
)

#: Corpus targets in chars. 2_000_000 chars is ~1000 chunks of 2000.
TARGET_CHARS = (10_000, 100_000, 500_000, 1_000_000, 2_000_000)

#: Typical long-context window used for the "beyond window" verdict.
WINDOW_TOKENS = 200_000

TOKENIZER = "cl100k_base"

HEADERS = (
    "N_chars",
    "context_tokens",
    "baseline_peak",
    "rlm_root_peak",
    "rlm_sub_peak",
    "rlm_total",
    "root_peak_reduction_x",
    "beyond_window",
)


async def _measure(harness: Harness, tok: tiktoken.Encoding, target: int) -> dict[str, object]:
    corpus, n_docs = make_corpus_for_chars(target)
    # Baseline: a naive long-context model ingests the corpus once.
    context_tokens = len(tok.encode(corpus))

    run = await run_doc_extraction(
        harness,
        corpus=corpus,
        n_docs=n_docs,
        # The final answer is the repr of every per-chunk answer; the cap
        # must fit it whole so the correctness check below sees every id.
        extra_limits={"max_output_chars": max(8_000, len(corpus))},
    )
    if run.final is None or run.final.get("status") != "final":
        raise RuntimeError(f"run at {target} chars did not conclude: {run.final}")

    # Ground truth: every document id must be present in the final answer.
    ground = {f"DOC-{i:05d}" for i in range(n_docs)}
    found = found_doc_ids(str(run.final.get("answer", "")))
    missing = ground - found
    if missing:
        raise RuntimeError(
            f"run at {target} chars ({len(corpus)} chars, {n_docs} ids) missed "
            f"{len(missing)} ids, e.g. {sorted(missing)[:3]!r}"
        )

    # Root model, per stage: what it actually sees and sends.
    metadata_tokens = len(tok.encode(json.dumps(run.opened["context"])))
    code_tokens = len(tok.encode(run.code))
    total_prompt_tokens = 0
    sub_peak = 0
    total_answer_tokens = 0
    max_answer_tokens = 0
    for cycle in run.cycles:
        for req in cycle.requests:
            prompt_tokens = len(tok.encode(str(req["prompt"])))
            total_prompt_tokens += prompt_tokens
            sub_peak = max(sub_peak, prompt_tokens)
        for answer in cycle.answers:
            answer_tokens = len(tok.encode(answer))
            total_answer_tokens += answer_tokens
            max_answer_tokens = max(max_answer_tokens, answer_tokens)

    # Root window at any instant: metadata + code + the sub-answers; the
    # context never enters it. Sub window: the largest single chunk prompt.
    rlm_root_peak = metadata_tokens + code_tokens + max_answer_tokens
    rlm_sub_peak = sub_peak
    rlm_total = code_tokens + total_prompt_tokens + total_answer_tokens

    return {
        "N_chars": len(corpus),
        "n_docs": n_docs,
        "context_tokens": context_tokens,
        "baseline_peak": context_tokens,
        "rlm_root_peak": rlm_root_peak,
        "rlm_sub_peak": rlm_sub_peak,
        "rlm_total": rlm_total,
        "reduction": context_tokens / rlm_root_peak,
        "beyond_window": "yes" if context_tokens > WINDOW_TOKENS else "no",
        "found": len(found),
    }


def _render(rows: list[dict[str, object]]) -> list[str]:
    cells: list[list[str]] = []
    for row in rows:
        cells.append(
            [
                f"{row['N_chars']}",
                f"{row['context_tokens']}",
                f"{row['baseline_peak']}",
                f"{row['rlm_root_peak']}",
                f"{row['rlm_sub_peak']}",
                f"{row['rlm_total']}",
                f"{row['reduction']:.1f}",
                str(row["beyond_window"]),
            ]
        )
    widths = [len(header) for header in HEADERS]
    for cell in cells:
        for index, value in enumerate(cell):
            widths[index] = max(widths[index], len(value))
    lines = [
        " | ".join(header.rjust(widths[index]) for index, header in enumerate(HEADERS)),
        "-+-".join("-" * width for width in widths),
    ]
    for cell in cells:
        lines.append(" | ".join(value.rjust(widths[index]) for index, value in enumerate(cell)))
    return lines


async def main() -> int:
    tok = tiktoken.get_encoding(TOKENIZER)
    print(f"tokenizer: {TOKENIZER} (proxy; the RELATIVE ratios are what matter)")
    print(f"window threshold for beyond_window: {WINDOW_TOKENS:,} tokens")
    print()
    rows: list[dict[str, object]] = []
    async with Harness(max_llm_calls=200_000) as harness:
        for target in TARGET_CHARS:
            print(f"measuring ~{target:,} chars ...", file=sys.stderr)
            rows.append(await _measure(harness, tok, target))
    print("\n".join(_render(rows)))
    print()

    first, last = rows[0], rows[-1]
    root_peaks = [int(row["rlm_root_peak"]) for row in rows]  # type: ignore[arg-type]
    totals = [int(row["rlm_total"]) for row in rows]  # type: ignore[arg-type]
    contexts = [int(row["context_tokens"]) for row in rows]  # type: ignore[arg-type]
    sub_peaks = [int(row["rlm_sub_peak"]) for row in rows]  # type: ignore[arg-type]
    beyond = [int(row["N_chars"]) for row in rows if row["beyond_window"] == "yes"]  # type: ignore[arg-type]

    for row in rows:
        print(
            f"correctness: N_chars={row['N_chars']:,} status=final "
            f"ids={row['found']}/{row['n_docs']} in the final answer"
        )
    print()

    print("reading the table")
    print(
        f"(a) the root-LM window stays ~constant as the context grows: rlm_root_peak "
        f"ranges {min(root_peaks)}..{max(root_peaks)} tokens while context_tokens grow "
        f"{contexts[0]:,} -> {contexts[-1]:,} ({contexts[-1] // contexts[0]}x)"
    )
    print(
        f"(b) the RLM total grows ~linearly with the context (it reads every chunk): "
        f"rlm_total {totals[0]:,} -> {totals[-1]:,} ({totals[-1] / totals[0]:.0f}x) over "
        f"N_chars {first['N_chars']:,} -> {last['N_chars']:,} "
        f"({last['N_chars'] / first['N_chars']:.0f}x)"
    )
    print(
        f"(c) beyond a {WINDOW_TOKENS:,}-token window the baseline simply does not fit "
        f"(context_tokens > {WINDOW_TOKENS:,} at N_chars={beyond}) while the RLM still "
        f"concludes, with a sub-call window of at most {max(sub_peaks)} tokens"
    )
    return 0


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    except RuntimeError as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)

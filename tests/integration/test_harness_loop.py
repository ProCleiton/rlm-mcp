"""Test 2: the harness drives the full symbolic-recursion loop over MCP.

The task: extract every ``DOC-xxxxx`` identifier from a deterministic
corpus. The harness answers every ``needs_llm`` request with a pure
deterministic function (no model involved anywhere) and calls
``rlm_resume``; the sandbox code suspends inside its chunk loop and resumes
from the exact suspension point, once per chunk. This proves the core RLM
mechanics (DESIGN C4: symbolic recursion through ``llm_query`` inside
arbitrary loops) over the real MCP transport.
"""

from __future__ import annotations

from tests.integration._harness import (
    Harness,
    found_doc_ids,
    make_corpus,
    run_doc_extraction,
)

#: 60 docs x ~69 chars = ~4.2k chars -> 3 chunks of 2000 with overlap 200.
N_DOCS = 60


async def test_extraction_loop_resumes_until_final_and_covers_every_chunk() -> None:
    corpus = make_corpus(N_DOCS)
    async with Harness() as harness:
        run = await run_doc_extraction(harness, corpus=corpus, n_docs=N_DOCS)

    # The loop ended with an explicit FINAL_VAR, not exhaustion or an error.
    assert run.final is not None
    assert run.final["status"] == "final"

    # >= 3 chunks means the code suspended and resumed mid-loop at least
    # twice; the resumed execution kept going from the exact suspension
    # point instead of restarting (each cycle is one llm_query/one chunk).
    requests = [req for cycle in run.cycles for req in cycle.requests]
    assert len(run.cycles) >= 2
    assert len(requests) >= 3
    assert run.final["spent"]["llm_calls"] == len(requests)

    # The final answer carries the identifiers of every chunk, not just the
    # first one: the whole ground truth is present...
    answer = str(run.final.get("answer", ""))
    ground = {f"DOC-{i:05d}" for i in range(N_DOCS)}
    assert ground <= found_doc_ids(answer)
    # ...including identifiers that only occur in the last chunk.
    assert "DOC-00059" in answer


async def test_open_without_text_or_paths_returns_an_error_payload() -> None:
    async with Harness() as harness:
        payload = await harness.call("rlm_open", {})
    # A usage error surfaces as a readable JSON error payload (never a raw
    # exception over the transport). The core wraps the empty-context
    # ValueError into a SessionError, so the payload type is SessionError,
    # not invalid_arguments.
    assert "error" in payload
    error = payload["error"]
    assert error["message"]
    assert "text" in error["message"] and "paths" in error["message"]

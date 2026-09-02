"""End-to-end tests for SessionManager over the real sandbox process."""

import asyncio
import json

import pytest

from rlm_mcp.context import chunk_text
from rlm_mcp.session import (
    PEEK_CHAR_CAP,
    REQUESTS_PAYLOAD_CAP,
    SessionError,
    SessionManager,
)
from rlm_mcp.types import Limits, OpenSpec, SubResult


def long_text(length: int = 10000) -> str:
    line = "lorem ipsum dolor sit amet, consectetur adipiscing elit 0123456789 "
    return (line * (length // len(line) + 1))[:length]


@pytest.fixture
async def mgr(tmp_path):
    manager = SessionManager(trajectory_dir=str(tmp_path / "trajectories"))
    yield manager
    await manager.shutdown()


async def _open(mgr, text="x", **overrides):
    result = await mgr.open(OpenSpec(text=text, **overrides))
    return result.to_dict()["session_id"]


# ---------------------------------------------------------------------------
# The core paradigm: suspension inside a loop, resumption from the middle
# ---------------------------------------------------------------------------


async def test_loop_over_chunks_resumes_inside_and_aggregates(mgr):
    text = long_text()
    chunks = chunk_text(text)
    assert len(chunks) == 3

    sid = await _open(mgr, text=text)
    code = (
        'acc = ""\n'
        "for i, c in enumerate(chunk_text(context)):\n"
        '    r = llm_query(f"[{i}|{len(acc)}] " + c[:8])\n'
        "    acc += r\n"
        "FINAL(acc)"
    )
    step = await mgr.exec(sid, code)
    assert step.status == "needs_llm"
    assert mgr.status(sid).to_dict()["state"] == "parked"

    first = step.to_dict()["requests"][0]
    assert first["id"] == "q1"
    assert first["kind"] == "llm"
    assert first["prompt"] == f"[0|0] {chunks[0][:8]}"
    assert first["chars"] == len(first["prompt"])

    answer = ""
    for i in range(1, 4):
        answer += f"A{i - 1}"
        step = await mgr.resume(sid, [SubResult(id=f"q{i}", text=f"A{i - 1}")])
        if i < 3:
            # The run resumed INSIDE the loop: iteration i sees the answers
            # accumulated by iterations 0..i-1 (len(acc) == 2*i).
            assert step.status == "needs_llm"
            nxt = step.to_dict()["requests"][0]
            assert nxt["id"] == f"q{i + 1}"
            assert nxt["prompt"] == f"[{i}|{2 * i}] {chunks[i][:8]}"
        else:
            assert step.status == "final"
            final = step.to_dict()
            assert final["answer"] == answer == "A0A1A2"
            assert "trajectory" in final
    assert mgr.status(sid).to_dict()["state"] == "final"


# ---------------------------------------------------------------------------
# Budgets and refusals
# ---------------------------------------------------------------------------


async def test_llm_query_batched_single_frame_preserves_order(mgr):
    sid = await _open(mgr)
    code = 'rs = llm_query_batched(["one", "two", "three"])\nFINAL("|".join(rs))'
    step = await mgr.exec(sid, code)
    assert step.status == "needs_llm"
    payload = step.to_dict()
    requests = payload["requests"]
    assert [r["id"] for r in requests] == ["q1", "q2", "q3"]
    assert [r["prompt"] for r in requests] == ["one", "two", "three"]

    step = await mgr.resume(
        sid,
        [
            SubResult(id="q2", text="B"),
            SubResult(id="q1", text="A"),
            SubResult(id="q3", text="C"),
        ],
    )
    assert step.status == "final"
    assert step.to_dict()["answer"] == "A|B|C"


async def test_rebind_of_reserved_names_is_refused(mgr):
    sid = await _open(mgr)
    cases = {
        "context = 5": "context",
        "context, x = 1, 2": "context",
        "llm_query = lambda p: p": "llm_query",
        "(context := 5)": "context",
        "del context": "context",
        "import json as context": "context",
        "def FINAL(): pass": "FINAL",
        "def chunk_text(): pass": "chunk_text",
        "globals()['llm_query'] = lambda p: p": "llm_query",
        "globals()['context'] = 5": "context",
        "globals()['llm_query'] += ''": "llm_query",
        "del globals()['context']": "context",
        "locals()['llm_query'] = 1": "llm_query",
        "builtins.globals()['FINAL'] = 1": "FINAL",
        "def f():\n    globals()['llm_query'] = 1": "llm_query",
    }
    for code, token in cases.items():
        step = await mgr.exec(sid, code)
        assert step.status == "error", code
        error = step.to_dict()["error"]
        assert error["type"] == "RebindRefused", code
        assert token in error["message"], code

    # The session survives the refusals and stays usable.
    step = await mgr.exec(sid, "x = 1")
    assert step.status == "ok"
    # A plain dict whose key happens to be a reserved name is not a rebind.
    step = await mgr.exec(sid, "d = {}\nd['llm_query'] = 1\nx = d['llm_query']")
    assert step.status == "ok"


async def test_rebind_inside_function_is_allowed(mgr):
    # Bindings inside function bodies never touch the sandbox namespace.
    sid = await _open(mgr)
    code = "def f(context):\n    return context + 1\nFINAL(str(f(41)))"
    step = await mgr.exec(sid, code)
    assert step.status == "final"
    assert step.to_dict()["answer"] == "42"


async def test_max_iterations_exhausted_with_partial(mgr):
    sid = await _open(mgr, limits=Limits(max_iterations=2))
    assert (await mgr.exec(sid, "x = 1")).status == "ok"
    assert (await mgr.exec(sid, "x = 2")).status == "ok"
    step = await mgr.exec(sid, "x = 3")
    assert step.status == "exhausted"
    payload = step.to_dict()
    assert "max_iterations" in payload["reason"]
    assert set(payload["partial"]) == {"vars", "stdout"}
    assert payload["partial"]["vars"] == [{"name": "x", "type": "int", "size": 0}]
    assert payload["limits"]["max_iterations"] == 2
    assert mgr.status(sid).to_dict()["state"] == "idle"


async def test_max_llm_calls_exhausted_kills_sandbox(mgr):
    sid = await _open(mgr, limits=Limits(max_llm_calls=2, max_iterations=10))
    code = 'out = []\nfor i in range(3):\n    out.append(llm_query(f"q{i}"))\nFINAL("|".join(out))'
    step = await mgr.exec(sid, code)
    assert step.status == "needs_llm"
    step = await mgr.resume(sid, [SubResult(id="q1", text="A")])
    assert step.status == "needs_llm"
    step = await mgr.resume(sid, [SubResult(id="q2", text="B")])
    assert step.status == "exhausted"
    payload = step.to_dict()
    assert "max_llm_calls" in payload["reason"]
    assert mgr.status(sid).to_dict()["state"] == "dead"

    step = await mgr.exec(sid, "pass")
    assert step.status == "error"
    assert step.to_dict()["error"]["type"] == "InvalidState"


async def test_resume_with_wrong_ids_is_refused_and_session_stays_parked(mgr):
    sid = await _open(mgr)
    step = await mgr.exec(sid, 'FINAL(llm_query("hi"))')
    assert step.status == "needs_llm"
    step = await mgr.resume(sid, [SubResult(id="q9", text="x")])
    assert step.status == "error"
    error = step.to_dict()["error"]
    assert error["type"] == "IdMismatch"
    assert "q1" in error["message"] and "q9" in error["message"]
    assert mgr.status(sid).to_dict()["state"] == "parked"

    step = await mgr.resume(sid, [SubResult(id="q1", text="hello")])
    assert step.status == "final"
    assert step.to_dict()["answer"] == "hello"

    # Nothing pending anymore: a late resume is refused, not matched.
    step = await mgr.resume(sid, [SubResult(id="q1", text="again")])
    assert step.status == "error"
    assert step.to_dict()["error"]["type"] == "InvalidState"


async def test_exec_while_parked_mentions_resume(mgr):
    sid = await _open(mgr)
    step = await mgr.exec(sid, "FINAL(llm_query('p'))")
    assert step.status == "needs_llm"
    step = await mgr.exec(sid, "print(1)")
    assert step.status == "error"
    assert step.to_dict()["error"]["type"] == "InvalidState"
    assert "rlm_resume" in step.to_dict()["error"]["message"]


async def test_resume_error_surfaces_as_subcall_error(mgr):
    sid = await _open(mgr)
    step = await mgr.exec(sid, 'FINAL(llm_query("p"))')
    assert step.status == "needs_llm"
    step = await mgr.resume(sid, [SubResult(id="q1", text=None, error="model refused")])
    assert step.status == "error"
    error = step.to_dict()["error"]
    assert error["type"] == "SubCallError"
    assert "model refused" in error["message"]
    # Sandbox survives the error and stays usable.
    step = await mgr.exec(sid, 'FINAL("fine")')
    assert step.status == "final"


async def test_code_syntax_error_returns_error_status(mgr):
    sid = await _open(mgr)
    step = await mgr.exec(sid, "def broken(:\n    pass")
    assert step.status == "error"
    assert step.to_dict()["error"]["type"] == "SyntaxError"


async def test_exec_timeout_kills_sandbox_and_marks_dead(mgr):
    sid = await _open(mgr, limits=Limits(max_exec_seconds=0.8))
    step = await mgr.exec(sid, "while True:\n    pass")
    assert step.status == "error"
    error = step.to_dict()["error"]
    assert error["type"] == "Timeout"
    assert "lost" in error["message"]
    assert mgr.status(sid).to_dict()["state"] == "dead"

    step = await mgr.exec(sid, "x = 1")
    assert step.status == "error"
    assert "dead" in step.to_dict()["error"]["message"]


# ---------------------------------------------------------------------------
# Tree semantics: shared ledger, depth, kind degradation
# ---------------------------------------------------------------------------


async def test_shared_ledger_between_parent_and_child(mgr):
    parent_sid = await _open(mgr, limits=Limits(max_iterations=10))
    assert (await mgr.exec(parent_sid, "x = 1")).status == "ok"

    child = await mgr.open(OpenSpec(text="child context", parent_session_id=parent_sid))
    child_payload = child.to_dict()
    assert child_payload["depth"] == 1
    assert child_payload["session_id"] != parent_sid
    child_sid = child_payload["session_id"]

    assert mgr.status(child_sid).to_dict()["spent"] == mgr.status(parent_sid).to_dict()["spent"]
    assert mgr.status(child_sid).to_dict()["spent"]["iterations"] == 1

    assert (await mgr.exec(child_sid, "y = 2")).status == "ok"
    assert mgr.status(parent_sid).to_dict()["spent"]["iterations"] == 2
    assert mgr.status(child_sid).to_dict()["depth"] == 1


async def test_depth_limit_refuses_deeper_opens(mgr):
    parent_sid = await _open(mgr, limits=Limits(max_depth=2))
    child = await mgr.open(OpenSpec(text="c", parent_session_id=parent_sid))
    child_sid = child.to_dict()["session_id"]
    grandchild = await mgr.open(OpenSpec(text="g", parent_session_id=child_sid))
    assert grandchild.to_dict()["depth"] == 2
    with pytest.raises(SessionError, match="max_depth"):
        await mgr.open(OpenSpec(text="x", parent_session_id=grandchild.to_dict()["session_id"]))


async def test_rlm_kind_and_suggested_context(mgr):
    sid = await _open(mgr)
    code = 'FINAL(str(rlm_query("explain", context="snippet content")))'
    step = await mgr.exec(sid, code)
    assert step.status == "needs_llm"
    request = step.to_dict()["requests"][0]
    assert request["kind"] == "rlm"
    assert request["suggested_context"] == "snippet content"
    step = await mgr.resume(sid, [SubResult(id=request["id"], text="R")])
    assert step.status == "final"
    assert step.to_dict()["answer"] == "R"


async def test_rlm_kind_degrades_at_max_depth(mgr):
    parent_sid = await _open(mgr, limits=Limits(max_depth=1))
    child = await mgr.open(OpenSpec(text="child", parent_session_id=parent_sid))
    child_sid = child.to_dict()["session_id"]
    code = 'FINAL(str(rlm_query("deep")))'
    step = await mgr.exec(child_sid, code)
    assert step.status == "needs_llm"
    request = step.to_dict()["requests"][0]
    assert request["id"] == "q1"
    assert request["kind"] == "llm"  # degraded: a child would exceed max_depth
    step = await mgr.resume(child_sid, [SubResult(id="q1", text="D")])
    assert step.status == "final"
    assert step.to_dict()["answer"] == "D"


# ---------------------------------------------------------------------------
# Context: metadata only, files, chunking helpers
# ---------------------------------------------------------------------------


async def test_open_retries_transient_sandbox_start_failure(mgr, monkeypatch):
    from rlm_mcp.sandbox import LocalDriver, SandboxError

    original_start = LocalDriver.start
    calls = {"n": 0}

    async def flaky_start(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SandboxError("transient spawn failure")
        return await original_start(self)

    monkeypatch.setattr(LocalDriver, "start", flaky_start)
    payload = (await mgr.open(OpenSpec(text="x"))).to_dict()
    assert payload["status"] == "ok"
    assert calls["n"] == 2
    assert (await mgr.exec(payload["session_id"], "x = 1")).status == "ok"


async def test_open_returns_metadata_only(mgr):
    text = long_text(12000)
    payload = (await mgr.open(OpenSpec(text=text, label="big"))).to_dict()
    assert payload["status"] == "ok"
    assert payload["session_id"].startswith("rlm_")
    assert payload["depth"] == 0
    meta = payload["context"]
    assert set(meta) == {"chars", "lines", "parts", "head", "tail"}
    assert meta["chars"] == len(text)
    assert meta["lines"] == 1
    assert meta["parts"] == [{"name": None, "chars": len(text), "lines": 1}]
    assert meta["head"] == text[:500]
    assert meta["tail"] == text[-500:]
    assert "text" not in payload
    assert set(payload["budget"]) == {"limits", "spent"}


async def test_open_loads_files_with_file_separators(mgr, tmp_path):
    # The joined context is longer than the per-call peek ceiling, so the
    # separator has to be verified by paging -- a single absurd limit must
    # never return the whole context at once.
    file_a = tmp_path / "a.txt"
    file_a.write_text(long_text(20_000), encoding="utf-8")
    file_b = tmp_path / "b.txt"
    file_b.write_text("beta\nline2", encoding="utf-8")
    sid = await _open(mgr, text=None, paths=(str(file_a), str(file_b)))

    expected = long_text(20_000) + f"\n\n===== FILE: {file_b} =====\n" + "beta\nline2"

    page = (await mgr.peek(sid, "context", offset=0, limit=10**6)).to_dict()
    assert page["text"] == expected[:PEEK_CHAR_CAP]
    assert page["returned"] == PEEK_CHAR_CAP
    assert page["truncated"] is True
    assert page["total"] == len(expected)
    assert "===== FILE:" not in page["text"]  # the ceiling cuts before it

    seen = page["text"]
    offset = page["returned"]
    while page["truncated"]:
        page = (await mgr.peek(sid, "context", offset=offset, limit=10**6)).to_dict()
        seen += page["text"]
        offset += page["returned"]
    assert seen == expected  # the separator lands exactly where expected

    meta = (await mgr.open(OpenSpec(paths=(str(file_a), str(file_b))))).to_dict()["context"]
    assert [p["name"] for p in meta["parts"]] == [str(file_a), str(file_b)]


async def test_open_refuses_missing_path_or_empty_spec(mgr, tmp_path):
    with pytest.raises(SessionError, match="not found"):
        await mgr.open(OpenSpec(paths=(str(tmp_path / "nope.txt"),)))
    with pytest.raises(SessionError, match="context requires"):
        await mgr.open(OpenSpec())


async def test_peek_pagination(mgr):
    text = long_text(40_000)
    sid = await _open(mgr, text=text)

    page = (await mgr.peek(sid, "context", offset=1000, limit=500)).to_dict()
    assert page["text"] == text[1000:1500]
    assert page["offset"] == 1000
    assert page["returned"] == 500
    assert page["total"] == len(text)
    assert page["truncated"] is True

    # An absurd requested limit is capped: one call never returns more than
    # the per-call ceiling, and total/truncated keep paging possible.
    page = (await mgr.peek(sid, "context", offset=0, limit=10**6)).to_dict()
    assert page["text"] == text[:PEEK_CHAR_CAP]
    assert page["returned"] == PEEK_CHAR_CAP
    assert page["truncated"] is True
    assert page["total"] == len(text)

    seen = page["text"]
    offset = page["returned"]
    while page["truncated"]:
        page = (await mgr.peek(sid, "context", offset=offset, limit=10**6)).to_dict()
        assert page["text"] == text[offset : offset + len(page["text"])]
        seen += page["text"]
        offset += page["returned"]
    assert seen == text
    assert offset == len(text)

    page = (await mgr.peek(sid, "context", offset=len(text) + 5, limit=10)).to_dict()
    assert page["text"] == ""
    assert page["returned"] == 0
    assert page["truncated"] is False

    with pytest.raises(SessionError):
        await mgr.peek(sid, "context", offset=-1, limit=10)
    with pytest.raises(SessionError):
        await mgr.peek(sid, "no_such_variable_xyz")
    with pytest.raises(SessionError):
        await mgr.peek(sid, "")


async def test_final_answer_truncated_at_output_cap(mgr):
    sid = await _open(mgr, limits=Limits(max_output_chars=2000))
    step = await mgr.exec(sid, 'FINAL("F" * 5000)')
    assert step.status == "final"
    answer = step.to_dict()["answer"]
    assert len(answer) <= 2000
    assert "chars elided" in answer
    assert answer.startswith("F")
    assert answer.endswith("F")


async def test_stderr_merged_into_stdout(mgr):
    sid = await _open(mgr)
    step = await mgr.exec(sid, "import sys\nsys.stderr.write('oops')\nprint('ok')")
    assert step.status == "ok"
    stdout = step.to_dict()["stdout"]
    assert "oops" in stdout and "ok" in stdout


# ---------------------------------------------------------------------------
# Environment and trajectory
# ---------------------------------------------------------------------------


async def test_sandbox_env_never_sees_parent_secrets(mgr, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "hunter2-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    sid = await _open(mgr)
    code = (
        "import os\n"
        "print('FAKE_API_KEY' in os.environ, 'OPENAI_API_KEY' in os.environ, "
        "'GITHUB_TOKEN' in os.environ, 'PATH' in os.environ)"
    )
    step = await mgr.exec(sid, code)
    assert step.status == "ok"
    assert step.to_dict()["stdout"].split() == ["False", "False", "False", "True"]


async def test_trajectory_records_tree_events(mgr):
    sid = await _open(mgr, text="t", label="traj-test")
    step = await mgr.exec(sid, 'FINAL(llm_query("p"))')
    assert step.status == "needs_llm"
    rid = step.to_dict()["requests"][0]["id"]
    step = await mgr.resume(sid, [SubResult(id=rid, text="ok")])
    assert step.status == "final"

    text = mgr.writer.read(sid)
    lines = [json.loads(line) for line in text.splitlines()]
    types = [line["type"] for line in lines]
    assert types[0] == "open"
    assert "exec" in types
    assert "llm_request" in types
    assert "llm_response" in types
    assert types[-1] == "final"
    for line in lines:
        assert line["v"] == 1
        assert line["sid"] == sid
        assert line["root"] == sid
        assert line["depth"] == 0
        assert {"seq", "ts"} <= set(line)

    final_trajectory = step.to_dict()["trajectory"]
    assert final_trajectory["root_id"] == sid
    assert final_trajectory["lines"] == len(lines)
    assert str(mgr.writer.directory / f"{sid}.jsonl") == final_trajectory["path"]
    assert mgr.writer.read("rlm_00000000") == ""  # unknown root: empty, no raise


async def test_close_returns_subtree_ids_and_unknown_raises(mgr):
    parent_sid = await _open(mgr)
    child = await mgr.open(OpenSpec(text="c", parent_session_id=parent_sid))
    child_sid = child.to_dict()["session_id"]
    grandchild = await mgr.open(OpenSpec(text="g", parent_session_id=child_sid))
    grandchild_sid = grandchild.to_dict()["session_id"]

    closed = await mgr.close(parent_sid)
    assert sorted(closed) == sorted([parent_sid, child_sid, grandchild_sid])
    with pytest.raises(SessionError, match="unknown session"):
        mgr.status(parent_sid)
    with pytest.raises(SessionError, match="unknown session"):
        mgr.status(child_sid)


async def test_sweep_evicts_stale_sessions(tmp_path):
    manager = SessionManager(
        trajectory_dir=str(tmp_path / "trajectories"), session_ttl=0.05
    )
    try:
        sid = (await manager.open(OpenSpec(text="x"))).to_dict()["session_id"]
        assert (await manager.exec(sid, "x = 1")).status == "ok"
        await asyncio_sleep(0.1)
        await manager.sweep()
        with pytest.raises(SessionError, match="unknown session"):
            manager.status(sid)
    finally:
        await manager.shutdown()


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# Review findings: truncation proofs, peek ceiling, batch payload, lifecycle
# ---------------------------------------------------------------------------


async def test_error_message_with_giant_context_is_truncated(mgr):
    """C3 proof (finding 1): a ValueError whose message carries the whole
    context must come back truncated, never verbatim to the root LM."""
    secret = long_text(30_000)
    sid = await _open(mgr, text="x")
    step = await mgr.exec(sid, f"raise ValueError({secret!r})")
    assert step.status == "error"
    error = step.to_dict()["error"]
    assert error["type"] == "ValueError"
    message = error["message"]
    assert len(message) <= 8_000  # default max_output_chars cap
    assert "chars elided" in message
    assert secret not in message
    assert "traceback" in error


async def test_invalid_output_cap_cannot_leak_context(mgr):
    """Finding 2 backstop at the manager level: even a programmatic negative
    cap (bypassing the MCP-level validation) can never hand the context
    back -- the budget gate stops the run and _truncate degrades to markers."""
    text = long_text(20_000)
    sid = await _open(mgr, text=text, limits=Limits(max_output_chars=-1))
    step = await mgr.exec(sid, "FINAL(context)")
    payload = step.to_dict()
    assert payload["status"] == "exhausted"
    assert "max_output_chars" in payload["reason"]
    assert text[:200] not in json.dumps(payload)


async def test_peek_absurd_limit_capped_and_pagination_complete(mgr):
    """C3 proof (finding 3): a peek page never exceeds the per-call ceiling
    and the full value is still reachable by paging with huge limits."""
    text = long_text(50_000)
    sid = await _open(mgr, text=text)

    page = (await mgr.peek(sid, "context", offset=0, limit=10**9)).to_dict()
    assert page["text"] == text[:PEEK_CHAR_CAP]
    assert page["returned"] == PEEK_CHAR_CAP
    assert page["truncated"] is True
    assert page["total"] == len(text)

    seen = page["text"]
    offset = page["returned"]
    pages = 1
    while page["truncated"]:
        page = (await mgr.peek(sid, "context", offset=offset, limit=10**9)).to_dict()
        assert len(page["text"]) <= PEEK_CHAR_CAP
        seen += page["text"]
        offset += page["returned"]
        pages += 1
    assert seen == text
    assert pages == 4  # 50 000 chars at 16 000 per page
    assert offset == len(text)


async def test_batched_requests_payload_capped_and_full_prompt_peekable(mgr):
    """Finding 5: the serialized requests payload of one needs_llm result is
    capped as a whole, ids stay 1:1, trimmed prompts point at the sandbox
    history, and rlm_peek works while the session is parked so the full
    prompt can be fetched before answering."""
    sid = await _open(mgr)
    count = 60
    code = (
        f"items = [f'job {{i}}: ' + 'P' * 9000 for i in range({count})]\n"
        "rs = llm_query_batched(items)\n"
        'FINAL("|".join(rs))'
    )
    step = await mgr.exec(sid, code)
    assert step.status == "needs_llm"
    payload = step.to_dict()
    requests = payload["requests"]
    assert [r["id"] for r in requests] == [f"q{i}" for i in range(1, count + 1)]
    assert len(json.dumps(requests)) <= REQUESTS_PAYLOAD_CAP

    fifth = requests[4]  # id q5 lives at history[4] inside the sandbox
    assert fifth["id"] == "q5"
    assert fifth["chars"] == 9007
    assert "rlm_peek" in fifth["prompt"]
    assert "history[4]['prompt']" in fifth["prompt"]
    assert "P" * 9000 not in fifth["prompt"]  # the prompt copy was cut

    # While the session is parked the harness can page the full prompt.
    assert mgr.status(sid).to_dict()["state"] == "parked"
    page = (await mgr.peek(sid, "history[4]['prompt']", offset=0, limit=10**9)).to_dict()
    assert page["text"] == "job 4: " + "P" * 9000
    assert page["truncated"] is False
    assert mgr.status(sid).to_dict()["state"] == "parked"  # peek is read-only

    step = await mgr.resume(
        sid, [SubResult(id=f"q{i}", text=f"A{i}") for i in range(1, count + 1)]
    )
    assert step.status == "final"
    assert step.to_dict()["answer"] == "|".join(f"A{i}" for i in range(1, count + 1))


async def test_resume_after_external_kill_while_parked_is_clean(mgr):
    """Finding 12: an externally killed sandbox while the session is parked
    surfaces as a clean SandboxDied error on resume -- no hang, session dead."""
    sid = await _open(mgr)
    step = await mgr.exec(sid, "FINAL(llm_query('p'))")
    assert step.status == "needs_llm"
    assert mgr.status(sid).to_dict()["state"] == "parked"

    mgr._sessions[sid].driver.kill()  # external death while parked

    step = await mgr.resume(sid, [SubResult(id="q1", text="late answer")])
    assert step.status == "error"
    error = step.to_dict()["error"]
    assert error["type"] == "SandboxDied"
    assert mgr.status(sid).to_dict()["state"] == "dead"
    step = await mgr.exec(sid, "x = 1")
    assert step.status == "error"
    assert "dead" in step.to_dict()["error"]["message"]


async def test_frame_over_reader_limit_kills_session_cleanly(mgr):
    """Finding 13: a frame above the StreamReader limit (16 MiB) raises the
    ValueError -> SandboxError path; the session dies cleanly instead of
    hanging the supervisor."""
    sid = await _open(mgr)
    code = 'llm_query_batched(["A" * 1_000_000] * 20)'  # ~20 MB single frame
    step = await mgr.exec(sid, code)
    assert step.status == "error"
    error = step.to_dict()["error"]
    assert error["type"] == "ProtocolError"
    assert "reader limit" in error["message"]
    assert mgr.status(sid).to_dict()["state"] == "dead"


async def test_close_refused_while_child_is_running(mgr):
    """Finding 7: closing a parent whose child is mid-execution is refused
    for the whole subtree -- a running child is never SIGKILLed silently."""
    parent_sid = await _open(mgr)
    child = await mgr.open(OpenSpec(text="c", parent_session_id=parent_sid))
    child_sid = child.to_dict()["session_id"]

    task = asyncio.create_task(mgr.exec(child_sid, "import time\ntime.sleep(1)"))
    for _ in range(200):
        if mgr.status(child_sid).to_dict()["state"] == "running":
            break
        await asyncio.sleep(0.01)
    assert mgr.status(child_sid).to_dict()["state"] == "running"

    with pytest.raises(SessionError, match="cannot close mid-execution"):
        await mgr.close(parent_sid)
    assert (await task).status == "ok"

    closed = await mgr.close(parent_sid)
    assert sorted(closed) == sorted([parent_sid, child_sid])

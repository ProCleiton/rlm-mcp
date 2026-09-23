"""Exec-mode tests: Phase 1 (doc default, trusted_env opt-in) + Phase 2 (async exec).

Phase 1 covers the blocking requirements: (1) doc-default regression,
(2) exec + trusted_env visible via a real ``rlm_exec`` reading
``os.environ``, (3) ``RLM_`` keys rejected, (4) trusted_env with mode=doc
rejected. Error payloads must never leak secret values, only key names.

Phase 2 covers ``exec_async``/``wait`` over the REAL sandbox (no mocks
where subprocess/timing is involved): fast dispatch+collect parity with
sync ``exec``, ``pending`` on a slow job + later collection, ``close``
cancelling a pending job with no orphan process, ``needs_llm`` via async
followed by sync ``resume``, and wall-clock accounting across spaced polls.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rlm_mcp.sandbox.driver import DEFAULT_RLIMIT_NPROC, _build_sandbox_env
from rlm_mcp.server import _validate_trusted_env, build_server
from rlm_mcp.session import SessionManager
from rlm_mcp.types import Limits, OpenSpec

SECRET = "s3cr3t-should-never-appear-in-errors"


@pytest.fixture
async def mgr(tmp_path: Path):
    manager = SessionManager(trajectory_dir=str(tmp_path / "trajectories"))
    yield manager
    await manager.shutdown()


@pytest.fixture()
def server(tmp_path: Path) -> MCPServer:
    return build_server(trajectory_dir=tmp_path)


async def _call(server: MCPServer, name: str, arguments: dict[str, object]) -> dict[str, object]:
    result = await server.call_tool(name, arguments)
    text = result.content[0].text  # type: ignore[union-attr]
    payload = json.loads(text)
    json.dumps(payload)
    return payload


def test_open_spec_defaults_preserve_doc_behavior() -> None:
    spec = OpenSpec()
    assert spec.mode == "doc"
    assert spec.trusted_env is None


def test_build_sandbox_env_ignores_rlm_prefixed_extra() -> None:
    env = _build_sandbox_env(Limits(), 5, 6, {"FOO_BAR": "x", "RLM_IN_FD": "999", "RLM_EVIL": "1"})
    assert env["FOO_BAR"] == "x"
    assert env["RLM_IN_FD"] == "5"
    assert env.get("RLM_EVIL") != "1"


def test_validate_trusted_env_rejects_rlm_prefix_without_value_leak() -> None:
    with pytest.raises(ValueError, match="RLM_"):
        _validate_trusted_env("exec", {"RLM_FOO": SECRET})
    try:
        _validate_trusted_env("exec", {"RLM_FOO": SECRET})
    except ValueError as exc:
        assert SECRET not in str(exc)
    else:  # pragma: no cover - the raises above already proves rejection
        raise AssertionError("RLM_ key was accepted")


def test_validate_trusted_env_with_doc_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode='exec'"):
        _validate_trusted_env("doc", {"MY_VAR": SECRET})


async def test_doc_default_session_has_scrubbed_env(mgr: SessionManager) -> None:
    result = await mgr.open(OpenSpec(text="hello"))
    assert result.status == "ok"
    sid = result.session_id
    assert sid is not None
    try:
        step = await mgr.exec(sid, "import os; print(repr(os.environ.get('FOO_BAR')))")
        assert step.status == "ok"
        assert "None" in (step.stdout or "")
    finally:
        await mgr.close(sid)


async def test_doc_mode_ignores_trusted_env_defense_in_depth(mgr: SessionManager) -> None:
    result = await mgr.open(OpenSpec(text="hi", mode="doc", trusted_env={"FOO_BAR": "NOP"}))
    assert result.status == "ok"
    sid = result.session_id
    assert sid is not None
    try:
        step = await mgr.exec(sid, "import os; print(repr(os.environ.get('FOO_BAR')))")
        assert step.status == "ok"
        assert "None" in (step.stdout or "")
    finally:
        await mgr.close(sid)


async def test_exec_trusted_env_visible_via_real_rlm_exec(mgr: SessionManager) -> None:
    result = await mgr.open(OpenSpec(text="hi", mode="exec", trusted_env={"FOO_BAR": "hello123"}))
    assert result.status == "ok"
    sid = result.session_id
    assert sid is not None
    try:
        step = await mgr.exec(sid, "import os; print(os.environ.get('FOO_BAR'))")
        assert step.status == "ok"
        assert "hello123" in (step.stdout or "")
        internals = await mgr.exec(
            sid, "import os; print(os.environ.get('RLM_IN_FD'), os.environ.get('RLM_OUT_FD'))"
        )
        assert internals.status == "ok"
        assert "None" not in (internals.stdout or "")
    finally:
        await mgr.close(sid)


async def test_server_rejects_rlm_prefixed_key(server: MCPServer) -> None:
    payload = await _call(
        server, "rlm_open", {"text": "x", "mode": "exec", "trusted_env": {"RLM_FOO": SECRET}}
    )
    assert "error" in payload
    assert "RLM_" in payload["error"]["message"]
    assert SECRET not in payload["error"]["message"]


async def test_server_rejects_trusted_env_with_doc_mode(server: MCPServer) -> None:
    payload = await _call(
        server, "rlm_open", {"text": "x", "mode": "doc", "trusted_env": {"MY_VAR": SECRET}}
    )
    assert "error" in payload
    assert "exec" in payload["error"]["message"]
    assert SECRET not in payload["error"]["message"]


async def test_server_rejects_invalid_mode(server: MCPServer) -> None:
    # The MCP framework validates the ``mode`` literal before our handler
    # runs, so an invalid mode surfaces as ToolError (still a clear
    # rejection naming the offending value), not an error payload.
    with pytest.raises(ToolError, match="bogus"):
        await _call(server, "rlm_open", {"text": "x", "mode": "bogus"})


def test_validate_trusted_env_strict_key_shape() -> None:
    # valid: 2..64 chars, uppercase start
    assert _validate_trusted_env("exec", {"AB": "1"}) == {"AB": "1"}
    assert _validate_trusted_env("exec", {"A" * 64: "1"}) == {"A" * 64: "1"}
    for bad in ["", "A", "aB", "ABc", "A-B", "A B", "*", "A" * 65, "1AB", "_AB"]:
        with pytest.raises(ValueError):
            _validate_trusted_env("exec", {bad: "1"})
    # value leak check on shape rejection
    try:
        _validate_trusted_env("exec", {"bad-key": SECRET})
    except ValueError as exc:
        assert SECRET not in str(exc)
    else:
        raise AssertionError("bad key was accepted")


def test_build_sandbox_env_drops_keep_env_and_rlm() -> None:
    import os

    baseline = dict(os.environ)
    try:
        os.environ["PATH"] = "/orig-path"
        os.environ["HOME"] = "/orig-home"
        env = _build_sandbox_env(
            Limits(),
            5,
            6,
            {
                "PATH": "/evil",
                "HOME": "/evil",
                "LANG": "evil",
                "TZ": "evil",
                "TMPDIR": "/evil",
                "RLM_EVIL": "1",
                "GOOD_VAR": "ok",
            },
        )
        assert env["PATH"] == "/orig-path"
        assert env["HOME"] == "/orig-home"
        assert env.get("RLM_EVIL") != "1"
        assert env["GOOD_VAR"] == "ok"
    finally:
        os.environ.clear()
        os.environ.update(baseline)


async def test_open_trajectory_records_mode_and_keys(mgr: SessionManager) -> None:
    import json

    result = await mgr.open(
        OpenSpec(text="hi", mode="exec", trusted_env={"ZZ_VAR": "v", "AA_VAR": "w"})
    )
    assert result.status == "ok"
    sid = result.session_id
    assert sid is not None
    try:
        raw = mgr.writer.read(sid)
        lines = [json.loads(line) for line in raw.splitlines()]
        opens = [e for e in lines if e["type"] == "open"]
        assert opens, "expected an open event in trajectory"
        evt = opens[0]
        assert evt["mode"] == "exec"
        assert evt["trusted_env_keys"] == ["AA_VAR", "ZZ_VAR"]
        assert json.dumps(evt["trusted_env_keys"]) == '["AA_VAR", "ZZ_VAR"]'
    finally:
        await mgr.close(sid)


def test_validate_trusted_env_rejects_reserved_keep_env_keys_without_value_leak() -> None:
    for key in ("PATH", "HOME", "LANG", "TZ", "TMPDIR"):
        with pytest.raises(ValueError, match="reserved"):
            _validate_trusted_env("exec", {key: SECRET})
        try:
            _validate_trusted_env("exec", {key: SECRET})
        except ValueError as exc:
            assert SECRET not in str(exc)
            assert key in str(exc)
        else:  # pragma: no cover - the raises above already proves rejection
            raise AssertionError(f"reserved key {key} was accepted")


async def test_server_rejects_reserved_keep_env_key_before_session_created(
    server: MCPServer,
) -> None:
    for key in ("PATH", "TMPDIR"):
        payload = await _call(
            server, "rlm_open", {"text": "x", "mode": "exec", "trusted_env": {key: "/tmp/evil"}}
        )
        assert "error" in payload
        assert payload["error"]["type"] == "invalid_arguments"
        assert "reserved" in payload["error"]["message"]
        assert "session_id" not in payload
        assert "/tmp/evil" not in payload["error"]["message"]


async def test_session_manager_direct_open_rejects_reserved_key_no_spawn(
    mgr: SessionManager,
) -> None:
    from rlm_mcp.session import SessionError

    before = len(mgr._sessions)
    with pytest.raises(SessionError, match="reserved"):
        await mgr.open(OpenSpec(text="hi", mode="exec", trusted_env={"PATH": "/tmp/evil"}))
    assert len(mgr._sessions) == before


def test_build_sandbox_env_nproc_default_exec_higher_doc_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm_mcp.sandbox.driver import DEFAULT_RLIMIT_NPROC_EXEC

    monkeypatch.delenv("RLM_RLIMIT_NPROC", raising=False)
    doc_env = _build_sandbox_env(Limits(), 5, 6, None)
    exec_env = _build_sandbox_env(Limits(), 5, 6, None, mode="exec")
    assert doc_env["RLM_RLIMIT_NPROC"] == str(DEFAULT_RLIMIT_NPROC) == "256"
    assert exec_env["RLM_RLIMIT_NPROC"] == str(DEFAULT_RLIMIT_NPROC_EXEC)
    assert int(exec_env["RLM_RLIMIT_NPROC"]) > int(doc_env["RLM_RLIMIT_NPROC"])
    monkeypatch.setenv("RLM_RLIMIT_NPROC", "4096")
    assert _build_sandbox_env(Limits(), 5, 6, None)["RLM_RLIMIT_NPROC"] == "4096"
    assert _build_sandbox_env(Limits(), 5, 6, None, mode="exec")["RLM_RLIMIT_NPROC"] == "4096"


async def test_effective_nproc_rlimit_exec_vs_doc_via_real_sandbox(
    mgr: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rlm_mcp.sandbox.driver import DEFAULT_RLIMIT_NPROC_EXEC

    monkeypatch.delenv("RLM_RLIMIT_NPROC", raising=False)
    doc_open = await mgr.open(OpenSpec(text="doc-nproc"))
    exec_open = await mgr.open(OpenSpec(text="exec-nproc", mode="exec"))
    assert doc_open.status == "ok" and exec_open.status == "ok"
    assert doc_open.session_id is not None and exec_open.session_id is not None
    try:
        doc_step = await mgr.exec(
            doc_open.session_id, "import os; print(os.environ.get('RLM_RLIMIT_NPROC'))"
        )
        exec_step = await mgr.exec(
            exec_open.session_id, "import os; print(os.environ.get('RLM_RLIMIT_NPROC'))"
        )
        assert doc_step.status == "ok"
        assert exec_step.status == "ok"
        assert (doc_step.stdout or "").strip() == str(DEFAULT_RLIMIT_NPROC) == "256"
        assert (exec_step.stdout or "").strip() == str(DEFAULT_RLIMIT_NPROC_EXEC)
    finally:
        await mgr.close(doc_open.session_id)
        await mgr.close(exec_open.session_id)


async def _open_sid(mgr: SessionManager, **overrides: object) -> str:
    result = await mgr.open(OpenSpec(text="phase2", **overrides))  # type: ignore[arg-type]
    assert result.status == "ok"
    assert result.session_id is not None
    return result.session_id


async def test_async_single_job_roundtrip_remains_phase2_compatible(
    mgr: SessionManager,
) -> None:
    """The common one-job flow still dispatches, polls, and collects identically."""
    sid = await _open_sid(mgr)
    try:
        dispatched = await mgr.exec_async(sid, 'FINAL("ok-async")')
        handle = dispatched["handle"]
        assert isinstance(handle, str) and handle.startswith("job_")
        assert dispatched["state"] == "running"
        status = mgr.status(sid).to_dict()
        assert status["state"] == "running"
        assert status["jobs"] == [
            {"handle": handle, "state": "running", "elapsed": pytest.approx(0, abs=1)}
        ]
        collected = await mgr.wait(handle, timeout=30.0)
        assert collected["status"] == "final"
        assert collected["answer"] == "ok-async"
        assert mgr.status(sid).to_dict()["state"] == "final"
    finally:
        await mgr.close(sid)


async def test_async_slow_job_pending_then_complete(mgr: SessionManager) -> None:
    sid = await _open_sid(mgr)
    try:
        dispatched = await mgr.exec_async(sid, "import time\ntime.sleep(2)\nprint('slow-done')")
        handle = dispatched["handle"]
        first = await mgr.wait(handle, timeout=0.3)
        assert first["status"] == "pending"
        assert first["state"] == "running"
        assert first["elapsed"] >= 0.3
        done = await mgr.wait(handle, timeout=30.0)
        assert done["status"] == "ok"
        assert "slow-done" in (done.get("stdout") or "")
        assert mgr.status(sid).to_dict()["jobs"] == []
        again = await mgr.exec(sid, "print('after-async')")
        assert again.status == "ok"
    finally:
        await mgr.close(sid)


async def test_async_fifo_accepts_two_jobs_and_collects_out_of_order(
    mgr: SessionManager,
) -> None:
    """Real sandbox: B queues behind A, then B can be collected before A."""
    sid = await _open_sid(mgr)
    try:
        first = await mgr.exec_async(
            sid,
            "import time\ntime.sleep(1)\nsequence = ['a']\nprint('A-done')",
        )
        second = await mgr.exec_async(
            sid,
            "sequence.append('b')\nprint(','.join(sequence))",
        )
        assert first["handle"] != second["handle"]
        assert first["state"] == "running"
        assert second["state"] == "queued"

        pending_b = await mgr.wait(second["handle"], timeout=0.2)
        assert pending_b == {"status": "pending", "state": "queued", "elapsed": 0.0}
        jobs = mgr.status(sid).to_dict()["jobs"]
        assert [job["handle"] for job in jobs] == [first["handle"], second["handle"]]
        assert [job["state"] for job in jobs] == ["running", "queued"]

        # Waiting B drives no alternate execution path: the FIFO runner first
        # finishes A, then executes B, while A remains uncollected in the table.
        done_b = await mgr.wait(second["handle"], timeout=30.0)
        assert done_b["status"] == "ok"
        assert "a,b" in (done_b.get("stdout") or "")
        done_a = await mgr.wait(first["handle"], timeout=0.0)
        assert done_a["status"] == "ok"
        assert "A-done" in (done_a.get("stdout") or "")
        assert mgr.status(sid).to_dict()["jobs"] == []
    finally:
        await mgr.close(sid)


async def test_sync_exec_refused_while_async_results_uncollected(
    mgr: SessionManager,
) -> None:
    sid = await _open_sid(mgr)
    try:
        dispatched = await mgr.exec_async(sid, "print('one')")
        await asyncio.sleep(0.2)
        refused = await mgr.exec(sid, "print('sync')")
        assert refused.status == "error"
        assert "pending async exec jobs" in refused.error["message"]  # type: ignore[index]
        assert (await mgr.wait(dispatched["handle"], timeout=30.0))["status"] == "ok"
    finally:
        await mgr.close(sid)


async def test_close_cancels_multiple_jobs_no_orphan(mgr: SessionManager) -> None:
    """One running plus two queued jobs are all cancelled and the OS child is reaped."""
    sid = await _open_sid(mgr)
    driver = mgr._sessions[sid].driver
    proc = driver._proc
    assert proc is not None
    pid = proc.pid
    jobs = [
        await mgr.exec_async(sid, "import time\ntime.sleep(30)\nprint('never')"),
        await mgr.exec_async(sid, "print('queued-2')"),
        await mgr.exec_async(sid, "print('queued-3')"),
    ]
    assert [job["state"] for job in jobs] == ["running", "queued", "queued"]
    assert len(mgr.status(sid).to_dict()["jobs"]) == 3
    await asyncio.sleep(0.3)
    assert await mgr.close(sid) == [sid]
    assert not driver.alive
    assert proc.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    for job in jobs:
        with pytest.raises(Exception, match=r"unknown|collected"):
            await mgr.wait(job["handle"], timeout=1.0)


async def test_close_parent_cancels_child_async_jobs(mgr: SessionManager) -> None:
    from rlm_mcp.session import SessionError

    parent = await _open_sid(mgr)
    child_open = await mgr.open(OpenSpec(text="kid", parent_session_id=parent))
    assert child_open.session_id is not None
    child = child_open.session_id
    await mgr.exec_async(child, "import time\ntime.sleep(30)")
    await mgr.exec_async(child, "print('queued')")
    await asyncio.sleep(0.2)
    assert sorted(await mgr.close(parent)) == sorted([parent, child])
    with pytest.raises(SessionError):
        await mgr.close(parent)


async def test_async_parked_job_holds_queue_until_sync_resume(
    mgr: SessionManager,
) -> None:
    from rlm_mcp.types import SubResult

    sid = await _open_sid(mgr)
    first = await mgr.exec_async(sid, "ans = llm_query('async-q')\nprint('got:' + ans)")
    second = await mgr.exec_async(sid, "print('after-resume')")
    parked = await mgr.wait(first["handle"], timeout=30.0)
    assert parked["status"] == "needs_llm"
    pending = await mgr.wait(second["handle"], timeout=0.1)
    assert pending["state"] == "queued"
    resumed = await mgr.resume(sid, [SubResult(id="q1", text="hello")])
    assert resumed.status == "ok"
    done = await mgr.wait(second["handle"], timeout=30.0)
    assert done["status"] == "ok"
    assert "after-resume" in (done.get("stdout") or "")
    await mgr.close(sid)


async def test_async_unknown_and_double_collect_raise(mgr: SessionManager) -> None:
    from rlm_mcp.session import SessionError

    with pytest.raises(Exception, match="unknown"):
        await mgr.wait("job_does_not_exist", timeout=1.0)
    sid = await _open_sid(mgr)
    dispatched = await mgr.exec_async(sid, 'FINAL("once")')
    handle = dispatched["handle"]
    assert (await mgr.wait(handle, timeout=30.0))["status"] == "final"
    with pytest.raises(SessionError, match=r"already collected|unknown"):
        await mgr.wait(handle, timeout=1.0)
    await mgr.close(sid)


async def test_queued_time_is_not_charged_to_wall_budget(mgr: SessionManager) -> None:
    sid = await _open_sid(mgr)
    try:
        before = mgr.status(sid).to_dict()["spent"]["wall_seconds"]
        first = await mgr.exec_async(sid, "import time\ntime.sleep(2)\nprint('first')")
        second = await mgr.exec_async(sid, "print('second')")
        pending = await mgr.wait(second["handle"], timeout=0.4)
        assert pending == {"status": "pending", "state": "queued", "elapsed": 0.0}
        assert (await mgr.wait(second["handle"], timeout=30.0))["status"] == "ok"
        assert (await mgr.wait(first["handle"], timeout=0.0))["status"] == "ok"
        wall = mgr.status(sid).to_dict()["spent"]["wall_seconds"] - before
        assert 1.5 <= wall <= 5.0, f"wall {wall} should count execution, not queue wait twice"
    finally:
        await mgr.close(sid)


async def test_server_exec_async_and_wait_roundtrip(server: MCPServer) -> None:
    opened = await _call(server, "rlm_open", {"text": "srv-async"})
    sid = opened["session_id"]
    dispatched = await _call(
        server, "rlm_exec_async", {"session_id": sid, "code": 'FINAL("srv-ok")'}
    )
    handle = dispatched["handle"]
    assert handle.startswith("job_")
    collected = await _call(server, "rlm_wait", {"handle": handle, "timeout": 30})
    assert collected["status"] == "final"
    assert collected["answer"] == "srv-ok"
    pending_probe = await _call(server, "rlm_wait", {"handle": handle, "timeout": 1})
    assert "error" in pending_probe
    assert await _call(server, "rlm_close", {"session_id": sid}) == {"closed": [sid]}

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


async def test_async_fast_roundtrip_matches_sync_exec(mgr: SessionManager) -> None:
    """Dispatch ``FINAL("ok")`` async; ``wait`` returns the sync-identical result."""
    sid = await _open_sid(mgr)
    try:
        dispatched = await mgr.exec_async(sid, 'FINAL("ok-async")')
        assert dispatched["handle"] == sid
        assert dispatched["state"] == "running"
        assert mgr.status(sid).to_dict()["state"] == "running"
        collected = await mgr.wait(sid, timeout=30.0)
        assert collected["status"] == "final"
        assert collected["answer"] == "ok-async"
        assert mgr.status(sid).to_dict()["state"] == "final"
    finally:
        await mgr.close(sid)


async def test_async_slow_job_pending_then_complete(mgr: SessionManager) -> None:
    """``time.sleep(2)`` in the sandbox: first ``wait(0.3)`` is pending, later poll completes."""
    sid = await _open_sid(mgr)
    try:
        dispatched = await mgr.exec_async(sid, "import time\ntime.sleep(2)\nprint('slow-done')")
        assert dispatched["state"] == "running"
        first = await mgr.wait(sid, timeout=0.3)
        assert first["status"] == "pending"
        assert first["elapsed"] >= 0.3
        # Session untouched by the pending poll: still running, re-waitable.
        assert mgr.status(sid).to_dict()["state"] == "running"
        done = await mgr.wait(sid, timeout=30.0)
        assert done["status"] == "ok"
        assert "slow-done" in (done.get("stdout") or "")
        assert mgr.status(sid).to_dict()["state"] == "idle"
        # Second exec works normally after collection (slot cleared).
        again = await mgr.exec(sid, "print('after-async')")
        assert again.status == "ok"
        assert "after-async" in (again.stdout or "")
    finally:
        await mgr.close(sid)


async def test_async_double_dispatch_refused_until_collected(mgr: SessionManager) -> None:
    """A second ``exec``/``exec_async`` while a job is outstanding is refused in-flow."""
    sid = await _open_sid(mgr)
    try:
        first = await mgr.exec_async(sid, "import time\ntime.sleep(2)\nprint('one')")
        assert "error" not in first
        second = await mgr.exec_async(sid, "print('two')")
        assert "error" in second
        assert second["error"]["status"] == "error"  # type: ignore[index]
        sync_refused = await mgr.exec(sid, "print('three')")
        assert sync_refused.status == "error"
        done = await mgr.wait(sid, timeout=30.0)
        assert done["status"] == "ok"
    finally:
        await mgr.close(sid)


async def test_close_cancels_pending_async_job_no_orphan(mgr: SessionManager) -> None:
    """``close`` on a session with a pending job cancels + kills the sandbox (no orphan)."""
    sid = await _open_sid(mgr)
    driver = mgr._sessions[sid].driver
    proc = driver._proc
    assert proc is not None
    pid = proc.pid
    dispatched = await mgr.exec_async(sid, "import time\ntime.sleep(30)\nprint('never')")
    assert dispatched["state"] == "running"
    await asyncio.sleep(0.5)  # let the sandbox actually enter the sleep
    closed = await mgr.close(sid)
    assert closed == [sid]
    assert sid not in mgr._sessions
    assert not driver.alive
    # The OS process is really gone (reaped): no live pid, no zombie.
    assert proc.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(Exception, match="unknown session"):
        await mgr.wait(sid, timeout=1.0)


async def test_close_parent_cancels_child_async_job(mgr: SessionManager) -> None:
    """Parent ``close`` cancels a child's pending async job and closes the subtree."""
    from rlm_mcp.session import SessionError

    parent = await _open_sid(mgr)
    child_open = await mgr.open(OpenSpec(text="kid", parent_session_id=parent))
    assert child_open.session_id is not None
    child = child_open.session_id
    try:
        dispatched = await mgr.exec_async(child, "import time\ntime.sleep(30)")
        assert dispatched["state"] == "running"
        await asyncio.sleep(0.5)
        closed = await mgr.close(parent)
        assert sorted(closed) == sorted([parent, child])
    finally:
        with pytest.raises(SessionError):
            await mgr.close(parent)


async def test_async_needs_llm_then_sync_resume_completes(mgr: SessionManager) -> None:
    """``needs_llm`` via async collects normally; sync ``resume`` finishes the session."""
    from rlm_mcp.types import SubResult

    sid = await _open_sid(mgr)
    dispatched = await mgr.exec_async(sid, "ans = llm_query('async-q')\nFINAL('got:' + ans)")
    assert dispatched["state"] == "running"
    parked = await mgr.wait(sid, timeout=30.0)
    assert parked["status"] == "needs_llm"
    assert parked["requests"][0]["id"] == "q1"
    assert mgr.status(sid).to_dict()["state"] == "parked"
    done = await mgr.resume(sid, [SubResult(id="q1", text="hello")])
    assert done.status == "final"
    assert done.answer == "got:hello"
    await mgr.close(sid)


async def test_async_unknown_and_double_collect_raise(mgr: SessionManager) -> None:
    """``wait`` on an unknown handle or an already-collected job raises clearly."""
    from rlm_mcp.session import SessionError

    with pytest.raises(Exception, match="unknown session"):
        await mgr.wait("rlm_does_not_exist", timeout=1.0)
    sid = await _open_sid(mgr)
    await mgr.exec_async(sid, 'FINAL("once")')
    done = await mgr.wait(sid, timeout=30.0)
    assert done["status"] == "final"
    with pytest.raises(SessionError, match=r"already collected|no pending"):
        await mgr.wait(sid, timeout=1.0)
    await mgr.close(sid)

async def test_async_wall_clock_counts_execution_not_poll_gaps(mgr: SessionManager) -> None:
    """Wall clock banks the ~2s sandbox sleep once, not the idle gap between polls."""
    sid = await _open_sid(mgr)
    try:
        before = mgr.status(sid).to_dict()["spent"]["wall_seconds"]
        await mgr.exec_async(sid, "import time\ntime.sleep(2)\nprint('w')")
        first = await mgr.wait(sid, timeout=0.3)
        assert first["status"] == "pending"
        await asyncio.sleep(1.0)  # idle harness gap: must NOT inflate the ledger
        mid_wall = mgr.status(sid).to_dict()["spent"]["wall_seconds"]
        done = await mgr.wait(sid, timeout=30.0)
        assert done["status"] == "ok"
        after = mgr.status(sid).to_dict()["spent"]["wall_seconds"]
        slept = after - before
        assert 1.5 <= slept <= 6.0, f"wall {slept} should be ~2s of sandbox sleep"
        # The 1s harness-side gap added less than ~0.9s extra beyond the sleep.
        assert (after - mid_wall) <= 2.5
    finally:
        await mgr.close(sid)


async def test_server_exec_async_and_wait_roundtrip(server: MCPServer) -> None:
    """MCP surface: ``rlm_exec_async`` + ``rlm_wait`` through the real server machinery."""
    tools = {tool.name for tool in await server.list_tools()}
    assert "rlm_exec_async" in tools
    assert "rlm_wait" in tools
    opened = await _call(server, "rlm_open", {"text": "srv-async"})
    sid = opened["session_id"]
    dispatched = await _call(
        server, "rlm_exec_async", {"session_id": sid, "code": 'FINAL("srv-ok")'}
    )
    assert dispatched["handle"] == sid
    assert dispatched["state"] == "running"
    collected = await _call(server, "rlm_wait", {"handle": sid, "timeout": 30})
    assert collected["status"] == "final"
    assert collected["answer"] == "srv-ok"
    pending_probe = await _call(server, "rlm_wait", {"handle": sid, "timeout": 1})
    assert "error" in pending_probe
    closed = await _call(server, "rlm_close", {"session_id": sid})
    assert closed == {"closed": [sid]}

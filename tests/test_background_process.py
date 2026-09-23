"""Phase 4 background-process tests over the REAL sandbox (no mocks).

Covers the Contract item 3: real ``sleep``/loop processes, ``poll() is
None`` while alive, survival across execs in the same session, real
``read_output()``, real ``kill()`` (``os.kill(pid, 0)`` raising
``ProcessLookupError``), ``close()`` killing orphans, plus a regression
smoke test for sync ``exec`` / ``exec_async`` + ``wait`` (Phase 3).
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import pytest

from rlm_mcp.session import SessionManager
from rlm_mcp.types import OpenSpec


@pytest.fixture
async def mgr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # This dev host owns >2048 tasks for our UID, so the exec default still
    # starves fork(); raise the ceiling via the operator knob (env is read
    # at spawn time in _build_sandbox_env).
    monkeypatch.setenv("RLM_RLIMIT_NPROC", "12000")
    manager = SessionManager(trajectory_dir=str(tmp_path / "trajectories"))
    yield manager
    await manager.shutdown()


async def _open(mgr: SessionManager) -> str:
    # mode="exec" raises the sandbox RLIMIT_NPROC default (256 -> 2048):
    # background children count as one process each against the agent's
    # rlimit, and shared dev hosts already own hundreds of tasks/threads.
    result = await mgr.open(OpenSpec(text="phase4", mode="exec"))
    assert result.status == "ok"
    assert result.session_id is not None
    return result.session_id


async def _exec_ok(mgr: SessionManager, sid: str, code: str) -> str:
    step = await mgr.exec(sid, code)
    assert step.status == "ok", f"exec failed: {step.to_dict()}"
    return step.stdout or ""


def _first_int(text: str) -> int:
    match = re.search(r"-?\d+", text)
    assert match is not None, f"no integer in {text!r}"
    return int(match.group(0))


async def _wait_poll(mgr: SessionManager, sid: str, var: str = "h") -> int:
    """Poll until the background handle exits; return its exit code."""
    for _ in range(200):
        out = await _exec_ok(mgr, sid, f"print({var}.poll())")
        if "None" not in out:
            return _first_int(out)
        await asyncio.sleep(0.05)
    raise AssertionError("background process did not exit in time")


SPAWN_SLEEP = (
    "import sys; h = spawn_background([sys.executable, '-c', "
    "'import time; time.sleep(60)']); print(h.pid)"
)


async def test_spawn_poll_none_and_survives_between_execs(mgr: SessionManager) -> None:
    sid = await _open(mgr)
    try:
        out = await _exec_ok(mgr, sid, SPAWN_SLEEP)
        pid = _first_int(out)
        assert pid > 0
        # Still running: poll() is None ...
        assert "None" in await _exec_ok(mgr, sid, "print(h.poll())")
        # ... and the handle survives an unrelated exec in the same session.
        assert "foreground-ok" in await _exec_ok(mgr, sid, "print('foreground-ok')")
        assert "None" in await _exec_ok(mgr, sid, "print(h.poll())")
        os.kill(pid, 0)  # alive from the supervisor side too
    finally:
        await mgr.close(sid)


async def test_read_output_real(mgr: SessionManager) -> None:
    sid = await _open(mgr)
    try:
        await _exec_ok(
            mgr,
            sid,
            "import sys; h = spawn_background([sys.executable, '-c', "
            "\"print('hello-bg'); print('line2')\"])",
        )
        assert await _wait_poll(mgr, sid) == 0
        out = await _exec_ok(mgr, sid, "out = h.read_output(); print(repr(out))")
        assert "hello-bg" in out
        assert "line2" in out
    finally:
        await mgr.close(sid)


async def test_kill_real_no_orphan(mgr: SessionManager) -> None:
    sid = await _open(mgr)
    try:
        out = await _exec_ok(mgr, sid, SPAWN_SLEEP)
        pid = _first_int(out)
        killed = await _exec_ok(mgr, sid, "h.kill(); print(h.poll())")
        assert "None" not in killed
        assert _first_int(killed) == -9
        await asyncio.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        await mgr.close(sid)


async def test_close_kills_background_tree_no_orphans(mgr: SessionManager) -> None:
    sid = await _open(mgr)
    driver = mgr._sessions[sid].driver
    out = await _exec_ok(
        mgr,
        sid,
        "import sys; h = spawn_background([sys.executable, '-c', "
        "'import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, \\'-c\\', "
        "\\'import time; time.sleep(60)\\']); "
        "print(p.pid, flush=True); time.sleep(60)']); print(h.pid)",
    )
    parent_pid = _first_int(out)
    grandchild_pid = -1
    await asyncio.sleep(1.5)  # one sleep instead of a poll loop: each exec costs an iteration
    child_out = await _exec_ok(mgr, sid, "print(h.read_output(), flush=True)")
    found = re.findall(r"\d+", child_out)
    assert found, f"grandchild never printed its pid: {child_out!r}"
    grandchild_pid = int(found[-1])
    assert await mgr.close(sid) == [sid]
    assert not driver.alive
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(parent_pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)


async def test_regression_exec_and_async_wait_phase3(mgr: SessionManager) -> None:
    """Sync exec + exec_async/wait still work alongside spawn_background."""
    sid = await _open(mgr)
    try:
        assert "plain-ok" in await _exec_ok(mgr, sid, "print('plain-ok')")
        dispatched = await mgr.exec_async(sid, "print('async-ok')")
        collected = await mgr.wait(dispatched["handle"], timeout=30.0)
        assert collected["status"] == "ok"
        assert "async-ok" in (collected.get("stdout") or "")
        assert "plain-ok" not in await _exec_ok(mgr, sid, "print('after-async')")
    finally:
        await mgr.close(sid)

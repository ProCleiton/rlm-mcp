"""Phase 1 exec-mode tests: doc default preserved, trusted_env opt-in.

Covers the blocking requirements: (1) doc-default regression, (2) exec +
trusted_env visible via a real ``rlm_exec`` reading ``os.environ``,
(3) ``RLM_`` keys rejected, (4) trusted_env with mode=doc rejected.
Error payloads must never leak secret values, only key names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rlm_mcp.sandbox.driver import _build_sandbox_env
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
    env = _build_sandbox_env(
        Limits(), 5, 6, {"FOO_BAR": "x", "RLM_IN_FD": "999", "RLM_EVIL": "1"}
    )
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

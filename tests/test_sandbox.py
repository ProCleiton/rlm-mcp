"""Tests for the sandbox subprocess: LocalDriver + agent frames."""

import asyncio

from rlm_mcp.sandbox import AGENT_SCRIPT, LocalDriver, scrub_env
from rlm_mcp.types import Limits

INIT = {
    "op": "init",
    "context": "hello world",
    "context_parts": [{"name": None, "chars": 11, "lines": 1, "text": "hello world"}],
}


def test_scrub_env_keeps_only_whitelist(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "hunter2")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ant")
    monkeypatch.setenv("GITHUB_TOKEN", "gh")
    monkeypatch.setenv("DATABASE_PASSWORD", "pw")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/tmp")
    monkeypatch.setenv("PYTHONPATH", "/some/venv")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")

    env = scrub_env()
    assert env.get("PATH") == "/usr/bin:/bin"
    assert env.get("HOME") == "/tmp"
    for secret in (
        "FAKE_API_KEY",
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "ANTHROPIC_AUTH_TOKEN",
        "GITHUB_TOKEN",
        "DATABASE_PASSWORD",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ):
        assert secret not in env


async def _started_driver(limits: Limits | None = None):
    driver = LocalDriver(AGENT_SCRIPT, limits or Limits())
    await driver.start()
    await driver.send(INIT)
    frame = await driver.recv(5)
    assert frame == {"op": "ready"}
    return driver


async def test_ready_exec_done_and_final_persistence():
    driver = await _started_driver()
    try:
        await driver.send({"op": "exec", "code": "x = 41\nprint('mark', x)"})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "exec_done"
        assert "mark 41" in frame["stdout"]
        names = {v["name"] for v in frame["vars"]}
        assert "x" in names
        assert "context" not in names  # reserved names are never listed
        assert "print" not in names

        # Namespace persists across exec frames; FINAL interrupts mid-code.
        await driver.send({"op": "exec", "code": "side = 1\nFINAL(str(x + 1))\nside = 999"})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "final"
        assert frame["answer"] == "42"

        for _ in range(100):
            if not driver.alive:
                break
            await asyncio.sleep(0.02)
        assert not driver.alive
    finally:
        await driver.close()


async def test_error_frame_for_uncaught_exception():
    driver = await _started_driver()
    try:
        await driver.send({"op": "exec", "code": "raise ValueError('boom')"})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "error"
        error = frame["error"]
        assert error["type"] == "ValueError"
        assert "boom" in error["message"]
        assert "ValueError" in error["traceback"]
    finally:
        await driver.close()


async def test_llm_request_response_roundtrip_and_history():
    driver = await _started_driver()
    try:
        code = (
            'rs = llm_query_batched(["one", "two"])\n'
            'FINAL(str(len(history)) + "|" + history[0]["id"] + "|" '
            '+ str(history[1]["result"]) + "|" + "|".join(rs))'
        )
        await driver.send({"op": "exec", "code": code})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "llm_request"
        requests = frame["requests"]
        assert [r["id"] for r in requests] == ["q1", "q2"]
        assert [r["prompt"] for r in requests] == ["one", "two"]
        assert all(r["kind"] == "llm" for r in requests)

        # One frame answers both; ids may arrive in any order.
        await driver.send(
            {
                "op": "llm_response",
                "results": [
                    {"id": "q2", "text": "B", "error": None},
                    {"id": "q1", "text": "A", "error": None},
                ],
            }
        )
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "final"
        assert frame["answer"] == "2|q1|B|A|B"
    finally:
        await driver.close()


async def test_llm_query_blocks_then_continues_loop():
    driver = await _started_driver()
    try:
        code = 'acc = ""\nfor i in range(3):\n    acc += llm_query(f"c{i}")\nFINAL(acc)'
        await driver.send({"op": "exec", "code": code})
        for i in range(3):
            frame = await driver.recv(5)
            assert frame is not None and frame["op"] == "llm_request"
            rid = frame["requests"][0]["id"]
            assert rid == f"q{i + 1}"
            assert frame["requests"][0]["prompt"] == f"c{i}"
            await driver.send(
                {"op": "llm_response", "results": [{"id": rid, "text": f"R{i}", "error": None}]}
            )
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "final"
        assert frame["answer"] == "R0R1R2"
    finally:
        await driver.close()


async def test_error_result_raises_subcall_error_in_user_code():
    driver = await _started_driver()
    try:
        await driver.send({"op": "exec", "code": "FINAL(llm_query('p'))"})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "llm_request"
        rid = frame["requests"][0]["id"]
        await driver.send(
            {"op": "llm_response", "results": [{"id": rid, "text": None, "error": "model refused"}]}
        )
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "error"
        assert frame["error"]["type"] == "SubCallError"
        assert "model refused" in frame["error"]["message"]
    finally:
        await driver.close()


async def test_peek_frame_pagination():
    driver = await _started_driver()
    try:
        await driver.send({"op": "peek", "expr": "context", "offset": 6, "limit": 5})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "peek_result"
        assert frame["text"] == "world"
        assert frame["total"] == 11
        assert frame["returned"] == 5
        assert frame["offset"] == 6
        assert frame["truncated"] is False

        await driver.send({"op": "peek", "expr": "no_such_var", "offset": 0, "limit": 10})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "error"
        assert frame["error"]["type"] == "NameError"
    finally:
        await driver.close()


async def test_sandbox_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "hunter2")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/tmp")
    driver = await _started_driver()
    try:
        code = (
            "import os\n"
            "print('FAKE_API_KEY' in os.environ, 'OPENAI_API_KEY' in os.environ, "
            "'PATH' in os.environ, 'HOME' in os.environ, 'RLM_MAX_OUTPUT_CHARS' in os.environ)"
        )
        await driver.send({"op": "exec", "code": code})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "exec_done"
        assert frame["stdout"].split() == ["False", "False", "True", "True", "True"]
    finally:
        await driver.close()


async def test_shutdown_frame_exits_cleanly():
    driver = await _started_driver()
    try:
        await driver.send({"op": "shutdown"})
        frame = await driver.recv(5)
        assert frame is None
        for _ in range(100):
            if not driver.alive:
                break
            await asyncio.sleep(0.02)
        assert not driver.alive
    finally:
        await driver.close()


async def test_output_capture_truncates_with_marker():
    driver = LocalDriver(AGENT_SCRIPT, Limits(max_output_chars=2000))
    await driver.start()
    await driver.send(INIT)
    frame = await driver.recv(5)
    assert frame == {"op": "ready"}
    try:
        await driver.send({"op": "exec", "code": "print('A' * 5000)"})
        frame = await driver.recv(5)
        assert frame is not None and frame["op"] == "exec_done"
        stdout = frame["stdout"]
        assert len(stdout) <= 2000
        assert "chars elided" in stdout
        assert stdout.startswith("A")
        assert stdout.rstrip("\n").endswith("A")
    finally:
        await driver.close()

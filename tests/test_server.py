"""Tests for the MCP adapter layer (server.py / cli.py).

These exercise the public MCP surface through the real ``MCPServer``
machinery: registration, description-size limits, end-to-end tool calls that
delegate to a real ``SessionManager`` (including the sandbox REPL), the
needs_llm -> resume -> final cycle, and readable usage-error payloads.

The module is skipped wholesale when the core package is not importable yet
(core and MCP layers land in parallel).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rlm_mcp.session", reason="core (session.py) not landed yet")

from mcp.server.mcpserver import MCPServer

from rlm_mcp import __version__, cli
from rlm_mcp.playbook import INSTRUCTIONS
from rlm_mcp.server import build_server

MAX_LIMIT_CHARS = 2000

EXPECTED_TOOLS = {
    "rlm_open",
    "rlm_exec",
    "rlm_exec_async",
    "rlm_wait",
    "rlm_resume",
    "rlm_peek",
    "rlm_status",
    "rlm_close",
}


@pytest.fixture()
def server(tmp_path: Path) -> MCPServer:
    # A real trajectory dir keeps test JSONL out of the user's state dir.
    return build_server(trajectory_dir=tmp_path)


async def _call(server: MCPServer, name: str, arguments: dict[str, object]) -> dict[str, object]:
    """Invoke a tool through the real server and parse its JSON text payload."""
    result = await server.call_tool(name, arguments)
    text = result.content[0].text  # type: ignore[union-attr]
    payload = json.loads(text)
    # The payload must be JSON-serializable in both directions.
    json.dumps(payload)
    return payload


async def test_eight_tools_registered_with_exact_names(server: MCPServer) -> None:
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert names == EXPECTED_TOOLS
    assert len(tools) == len(EXPECTED_TOOLS)


async def test_instructions_and_tool_descriptions_stay_under_2000_chars(server: MCPServer) -> None:
    assert len(INSTRUCTIONS) <= MAX_LIMIT_CHARS
    assert len(server.instructions) <= MAX_LIMIT_CHARS
    tools = await server.list_tools()
    assert tools, "expected the eight tools to be registered"
    for tool in tools:
        assert len(tool.description) <= MAX_LIMIT_CHARS, (
            f"description of {tool.name!r} is {len(tool.description)} chars "
            f"(limit {MAX_LIMIT_CHARS})"
        )


async def test_rlm_playbook_prompt_and_trajectory_resource_registered(server: MCPServer) -> None:
    prompt_names = {prompt.name for prompt in await server.list_prompts()}
    assert "rlm_playbook" in prompt_names
    templates = [t.uri_template for t in await server.list_resource_templates()]
    assert "rlm://trajectory/{root_id}" in templates


async def test_open_and_exec_end_to_end_with_json_payload(server: MCPServer) -> None:
    opened = await _call(server, "rlm_open", {"text": "hello rlm world"})
    assert opened["session_id"]
    # The core reports the tree depth of the new session; a root is depth 0.
    assert opened["depth"] == 0
    assert set(opened) >= {"session_id", "depth", "context", "budget"}
    context_meta = opened["context"]
    assert isinstance(context_meta, dict)
    assert set(context_meta) >= {"chars", "lines", "head", "tail"}

    executed = await _call(
        server,
        "rlm_exec",
        {"session_id": opened["session_id"], "code": "doubled = len(context) * 2"},
    )
    assert executed["status"] == "ok"
    assert set(executed) >= {"status", "stdout", "vars", "spent"}
    assert "doubled" in {var["name"] for var in executed["vars"]}

    peeked = await _call(
        server,
        "rlm_peek",
        {"session_id": opened["session_id"], "expr": "doubled", "limit": 100},
    )
    assert set(peeked) >= {"text", "offset", "returned", "total", "truncated"}

    status = await _call(server, "rlm_status", {"session_id": opened["session_id"]})
    assert set(status) >= {"depth", "spent", "limits", "state", "trajectory"}

    closed = await _call(server, "rlm_close", {"session_id": opened["session_id"]})
    assert opened["session_id"] in closed["closed"]


async def test_needs_llm_resume_cycle_ends_in_final(server: MCPServer) -> None:
    opened = await _call(server, "rlm_open", {"text": "count the letters"})
    sid = opened["session_id"]
    executed = await _call(
        server,
        "rlm_exec",
        {
            "session_id": sid,
            "code": (
                "answer = llm_query('What is 6 times 7? Reply with just the number.')\n"
                "FINAL(answer)"
            ),
        },
    )
    assert executed["status"] == "needs_llm"
    assert set(executed) >= {"status", "requests", "spent"}
    requests = executed["requests"]
    assert requests and all("id" in req and "kind" in req for req in requests)

    resumed = await _call(
        server,
        "rlm_resume",
        {"session_id": sid, "results": [{"id": requests[0]["id"], "text": "42"}]},
    )
    assert resumed["status"] == "final"
    assert set(resumed) >= {"status", "answer", "spent", "trajectory"}
    assert "42" in resumed["answer"]


async def test_usage_error_for_unknown_session_is_readable_payload(server: MCPServer) -> None:
    for tool, arguments in (
        ("rlm_open", {}),  # neither text nor paths: core refuses the open
        ("rlm_status", {"session_id": "no-such-session"}),
        ("rlm_exec", {"session_id": "no-such-session", "code": "1"}),
        ("rlm_peek", {"session_id": "no-such-session", "expr": "x"}),
        (
            "rlm_resume",
            {"session_id": "no-such-session", "results": [{"id": "q1", "text": "x"}]},
        ),
    ):
        payload = await _call(server, tool, arguments)  # must not raise
        assert "error" in payload, f"{tool} should return an error payload"
        assert payload["error"]["message"]


async def test_invalid_budget_override_is_readable_payload(server: MCPServer) -> None:
    payload = await _call(server, "rlm_open", {"text": "x", "limits": {"max_deph": 3}})
    assert "error" in payload and "max_deph" in payload["error"]["message"]
    payload = await _call(server, "rlm_open", {"text": "x", "limits": {"max_depth": "deep"}})
    assert "error" in payload


async def test_nonpositive_budget_override_is_refused_and_cannot_leak(server: MCPServer) -> None:
    """Finding 2 proof: the rlm_open(limits={"max_output_chars": -1}) vector
    is refused up front with a usage error -- before the fix the session
    opened with a negative cap and FINAL(context) handed the whole context
    to the root LM."""
    secret = "secret-context-" * 2000  # 30k chars, way over any legit cap
    for bad in (
        {"max_output_chars": -1},
        {"max_output_chars": 0},
        {"max_iterations": 0},
        {"max_iterations": -3},
        {"max_llm_calls": 0},
        {"max_errors": -1},
        {"max_wall_seconds": 0.0},
        {"max_exec_seconds": -0.5},
        {"max_depth": -1},
    ):
        payload = await _call(server, "rlm_open", {"text": secret, "limits": bad})
        assert "error" in payload, bad
        error = payload["error"]
        assert error["type"] == "invalid_arguments", bad
        assert "must be" in error["message"], bad
        assert secret not in payload.get("session_id", "")  # never opened

    # max_depth may legitimately be 0 (a root-only tree).
    opened = await _call(server, "rlm_open", {"text": "x", "limits": {"max_depth": 0}})
    assert "session_id" in opened
    status = await _call(server, "rlm_status", {"session_id": opened["session_id"]})
    assert status["limits"]["max_depth"] == 0
    closed = await _call(server, "rlm_close", {"session_id": opened["session_id"]})
    assert opened["session_id"] in closed["closed"]


def test_cli_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    flags = (
        "--max-depth",
        "--max-iterations",
        "--max-llm-calls",
        "--trajectory-dir",
        "--log-level",
    )
    for flag in flags:
        assert flag in out

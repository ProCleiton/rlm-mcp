"""Test 1: the MCP transport itself, tool registration, and basic execution.

Every assertion runs against the real ``rlm-mcp`` server spawned over stdio
by :class:`tests.integration._harness.Harness`, through the official MCP
client SDK -- the same channel any MCP-capable harness would use.
"""

from __future__ import annotations

import json

from tests.integration._harness import EXPECTED_TOOLS, Harness

TEXT = "hello world, this is context"


def _walk(node: object) -> tuple[list[str], list[tuple[str, str]]]:
    """All dict keys and all ``(key, str value)`` pairs of a JSON payload."""
    keys: list[str] = []
    strings: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            keys.append(key)
            if isinstance(value, str):
                strings.append((key, value))
            sub_keys, sub_strings = _walk(value)
            keys.extend(sub_keys)
            strings.extend(sub_strings)
    elif isinstance(node, list):
        for item in node:
            sub_keys, sub_strings = _walk(item)
            keys.extend(sub_keys)
            strings.extend(sub_strings)
    return keys, strings


async def test_list_tools_exposes_exactly_the_six_rlm_tools() -> None:
    async with Harness() as harness:
        names = await harness.list_tools()
    assert len(names) == len(EXPECTED_TOOLS)
    assert set(names) == EXPECTED_TOOLS


async def test_open_returns_metadata_only_for_a_short_context() -> None:
    async with Harness() as harness:
        payload = await harness.call("rlm_open", {"text": TEXT})
    assert payload["status"] == "ok"
    assert str(payload["session_id"]).startswith("rlm_")
    assert payload["depth"] == 0

    context = payload["context"]
    assert set(context) == {"chars", "lines", "parts", "head", "tail"}
    assert context["chars"] == len(TEXT) == 28
    assert context["lines"] == 1
    assert context["parts"] == [{"name": None, "chars": 28, "lines": 1}]

    # C2: the context is a symbolic handle. No field anywhere is a raw-text
    # key, and every string value is a bounded preview at most. The 500-char
    # head preview legitimately starts at char 0, so for a short context it
    # equals the text; nothing else in the payload may echo the text.
    keys, strings = _walk(payload)
    assert "text" not in keys
    assert all(len(value) <= 500 for _, value in strings)
    for key, value in strings:
        if key == "head":
            continue
        assert TEXT not in value, f"field {key!r} leaks the raw context: {value!r}"
    assert context["head"] == TEXT  # bounded head preview of the short text
    assert context["tail"] == ""


async def test_open_never_leaks_a_large_context_outside_the_previews() -> None:
    marker = "S3N71N3L-MIDDLE"
    big = ("a" * 9_750) + marker + ("b" * 10_235)
    assert len(big) == 20_000
    async with Harness() as harness:
        payload = await harness.call("rlm_open", {"text": big})
    context = payload["context"]
    assert context["chars"] == 20_000
    # Of 20k chars the harness receives at most two bounded previews (500
    # chars each, DESIGN C3); the middle of the text must never appear.
    serialized = json.dumps(payload)
    assert marker not in serialized
    assert len(serialized) < 2_000
    assert context["head"] == big[:500]
    assert context["tail"] == big[-500:]


async def test_exec_runs_code_in_the_session_and_returns_stdout() -> None:
    async with Harness() as harness:
        opened = await harness.call("rlm_open", {"text": TEXT})
        executed = await harness.call(
            "rlm_exec",
            {"session_id": opened["session_id"], "code": "x = 1 + 1\nprint(x)"},
        )
    assert executed["status"] == "ok"
    assert "2" in executed["stdout"]
    assert executed["spent"]["iterations"] == 1
    names = {var["name"] for var in executed["vars"]}
    assert "x" in names


async def test_status_and_close_report_and_end_the_session() -> None:
    async with Harness() as harness:
        opened = await harness.call("rlm_open", {"text": TEXT})
        session_id = opened["session_id"]
        status = await harness.call("rlm_status", {"session_id": session_id})
        closed = await harness.call("rlm_close", {"session_id": session_id})
    assert set(status) >= {"depth", "spent", "limits", "state", "trajectory"}
    assert "spent" in status and "iterations" in status["spent"]
    assert session_id in closed["closed"]

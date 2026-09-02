"""Command-line entry point for rlm-mcp.

Runs the MCP server over stdio with configurable default budgets and
trajectory directory. stdout is the MCP transport and must never carry log
output; all logging goes to stderr. Failures exit non-zero with the reason on
stderr -- never a raw traceback.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import Sequence
from typing import cast

from mcp.server.mcpserver import MCPServer

from rlm_mcp import __version__
from rlm_mcp.server import LogLevel, build_server
from rlm_mcp.session import SessionManager
from rlm_mcp.types import Limits

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlm-mcp",
        description=(
            "Harness-agnostic MCP server that brings the Recursive Language "
            "Model (RLM) paradigm to any agent. The server never calls a "
            "language model and never needs an API key: the harness's own "
            "agent answers the sub-completions via rlm_resume."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=Limits().max_depth,
        metavar="N",
        help="maximum recursion depth for new sessions (default: %(default)s).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=Limits().max_iterations,
        metavar="N",
        help="maximum rlm_exec steps per session (default: %(default)s).",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=Limits().max_llm_calls,
        metavar="N",
        help="maximum llm_query sub-completions per session (default: %(default)s).",
    )
    parser.add_argument(
        "--trajectory-dir",
        metavar="DIR",
        default=None,
        help="directory for trajectory JSONL files (default: a temp dir).",
    )
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="logging level, applied to stderr only (default: %(default)s).",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level),
        format=_LOG_FORMAT,
        force=True,
    )


def _sweep_interval() -> float:
    """Seconds between TTL sweeps (env ``RLM_MCP_SWEEP_INTERVAL``).

    Parked sessions abandoned by a harness that never calls rlm_resume would
    otherwise hold their sandbox process forever; the periodic sweep evicts
    them once their TTL expires.  Invalid/non-positive values fall back to
    60s.
    """
    raw = os.environ.get("RLM_MCP_SWEEP_INTERVAL")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return 60.0


async def _sweep_loop(manager: SessionManager, interval: float) -> None:
    """Background TTL sweep; runs until cancelled."""
    while True:
        await asyncio.sleep(interval)
        try:
            await manager.sweep()
        except Exception:
            # One failed pass must not kill the periodic task.
            logging.getLogger("rlm_mcp.cli").exception("periodic session sweep failed")


async def _serve(server: MCPServer, manager: SessionManager, sweep_interval: float) -> None:
    """Run the stdio server with a background TTL-sweep task.

    The sweeper is cancelled (and awaited) in ``finally`` so shutdown is
    clean: no orphan task, no 'Task was destroyed' warning, sessions closed.
    """
    sweeper = asyncio.create_task(_sweep_loop(manager, sweep_interval), name="rlm-mcp-sweeper")
    try:
        await server.run_stdio_async()
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        await manager.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the server; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    limits = Limits(
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_depth=args.max_depth,
    )
    log_level: LogLevel = cast(LogLevel, args.log_level)
    try:
        manager = SessionManager(default_limits=limits, trajectory_dir=args.trajectory_dir)
        server = build_server(
            limits=limits,
            trajectory_dir=args.trajectory_dir,
            log_level=log_level,
            manager=manager,
        )
        asyncio.run(_serve(server, manager, _sweep_interval()))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # The CLI boundary reports a one-line reason and exits non-zero.
        logging.getLogger("rlm_mcp.cli").error("fatal: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

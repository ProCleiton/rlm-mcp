"""Sandbox subpackage: the confined subprocess that runs RLM user code.

``agent.py`` is the child process (stdlib only, never imports this package);
``driver.py`` is the supervisor side (``LocalDriver``) that spawns it over
two dedicated pipes.  DESIGN section 7.
"""

from __future__ import annotations

from pathlib import Path

from rlm_mcp.sandbox.driver import (
    DEFAULT_RLIMIT_AS_BYTES,
    DEFAULT_RLIMIT_CPU_SECONDS,
    DEFAULT_RLIMIT_FSIZE_BYTES,
    DEFAULT_RLIMIT_NPROC,
    KEEP_ENV,
    LocalDriver,
    SandboxError,
    SandboxTimeout,
    scrub_env,
)

#: Absolute path of the agent entry script (``python3 -I -S <this>``).
AGENT_SCRIPT = str(Path(__file__).resolve().parent / "agent.py")

__all__ = [
    "AGENT_SCRIPT",
    "DEFAULT_RLIMIT_AS_BYTES",
    "DEFAULT_RLIMIT_CPU_SECONDS",
    "DEFAULT_RLIMIT_FSIZE_BYTES",
    "DEFAULT_RLIMIT_NPROC",
    "KEEP_ENV",
    "LocalDriver",
    "SandboxError",
    "SandboxTimeout",
    "scrub_env",
]

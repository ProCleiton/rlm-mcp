"""Local sandbox driver: spawns and supervises one agent subprocess.

The agent runs as ``python3 -I -S <agent.py>`` in its own process group
(``start_new_session=True``) with a scrubbed environment and a dedicated
temporary cwd.  Frames are newline-delimited JSON over two dedicated pipes
passed with ``pass_fds``; stdin/stdout of the process are unused, which keeps
the child's stdio free.

The child's stderr is not discarded: it is drained into a bounded tail
buffer so startup failures (fd rewiring errors, interpreter tracebacks) can
be reported to the caller instead of surfacing as a bare EOF.

The environment is scrubbed by *whitelist*: only ``PATH``/``HOME``/``LANG``/
``TZ``/``TMPDIR`` (when present) survive, plus the ``RLM_*`` parameters this
driver injects.  Provider variables (``OPENAI_*``, ``ANTHROPIC_*``, ``AWS_*``,
``GH_*``, ``GITHUB_*``) and anything whose name contains KEY/TOKEN/SECRET/
PASSWORD/CREDENTIAL therefore never reach the sandbox (DESIGN G3).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import Mapping

from rlm_mcp.types import Limits

KEEP_ENV = ("PATH", "HOME", "LANG", "TZ", "TMPDIR")

DEFAULT_RLIMIT_AS_BYTES = 1 << 30  # 1 GiB of address space
DEFAULT_RLIMIT_CPU_SECONDS = 600
DEFAULT_RLIMIT_FSIZE_BYTES = 64 << 20  # 64 MiB of file writes
DEFAULT_RLIMIT_NPROC = 256
#: Higher ``RLIMIT_NPROC`` default for ``mode="exec"`` sessions only. Exec
#: sessions fork real subprocesses; 256 (the doc default, unchanged) is too
#: low on shared hosts where the operator's UID already owns hundreds of
#: tasks/threads. The explicit ``RLM_RLIMIT_NPROC`` operator knob always wins
#: over both defaults. Hosts under extreme load may still need to raise
#: ``RLM_RLIMIT_NPROC`` manually even with this higher default.
DEFAULT_RLIMIT_NPROC_EXEC = 2048

#: How much of the agent's captured stderr to keep for diagnostics.
_STDERR_TAIL_BYTES = 64 * 1024
#: Upper bound for a single inbound frame (batch prompts, big reprs...).
_READER_LIMIT = 16 * 1024 * 1024


class SandboxError(Exception):
    """Protocol-level sandbox failure (malformed frame, dead process...)."""


class SandboxTimeout(TimeoutError):
    """No frame arrived within the requested deadline."""


def scrub_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the whitelisted subset of an environment mapping.

    Keeping only a small allow-list subsumes the removals DESIGN section 7
    demands: provider variables and secret-looking names never even enter the
    sandbox environment.
    """
    src = os.environ if source is None else source
    return {key: src[key] for key in KEEP_ENV if key in src}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _build_sandbox_env(
    limits: Limits,
    in_fd: int,
    out_fd: int,
    extra_env: Mapping[str, str] | None = None,
    mode: str = "doc",
) -> dict[str, str]:
    """Build the scrubbed sandbox environment plus internal ``RLM_*`` params.

    ``extra_env`` (opt-in, exec mode only) is applied AFTER the scrub and the
    internal ``RLM_*`` entries. Entries whose key starts with ``RLM_`` or
    that matches a ``KEEP_ENV`` name (``PATH``, ``HOME``, ``LANG``, ``TZ``,
    ``TMPDIR``) are silently dropped so callers can never override the pipe
    fds, output cap, rlimits, or the driver's own scrubbed values. That drop
    is defense in depth: ``server._validate_trusted_env`` already rejects
    such keys loudly before any session is created when the call goes
    through ``rlm_open``.

    ``mode`` only selects the ``RLIMIT_NPROC`` default (``"exec"`` gets the
    higher ``DEFAULT_RLIMIT_NPROC_EXEC``); an explicit ``RLM_RLIMIT_NPROC``
    operator env var always wins, and ``mode="doc"`` keeps the 256 default.
    """
    nproc_default = DEFAULT_RLIMIT_NPROC_EXEC if mode == "exec" else DEFAULT_RLIMIT_NPROC
    env = scrub_env()
    env.update(
        {
            "RLM_IN_FD": str(in_fd),
            "RLM_OUT_FD": str(out_fd),
            "RLM_MAX_OUTPUT_CHARS": str(limits.max_output_chars),
            "RLM_RLIMIT_AS_BYTES": str(_env_int("RLM_RLIMIT_AS_BYTES", DEFAULT_RLIMIT_AS_BYTES)),
            "RLM_RLIMIT_CPU_SECONDS": str(
                _env_int("RLM_RLIMIT_CPU_SECONDS", DEFAULT_RLIMIT_CPU_SECONDS)
            ),
            "RLM_RLIMIT_FSIZE_BYTES": str(
                _env_int("RLM_RLIMIT_FSIZE_BYTES", DEFAULT_RLIMIT_FSIZE_BYTES)
            ),
            "RLM_RLIMIT_NPROC": str(_env_int("RLM_RLIMIT_NPROC", nproc_default)),
        }
    )
    if extra_env:
        for key, value in extra_env.items():
            if key.startswith("RLM_") or key in KEEP_ENV:
                continue
            env[key] = value
    return env


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"short write to sandbox pipe (fd {fd})")
        view = view[written:]


class LocalDriver:
    """Supervises one sandbox agent process speaking JSON lines over pipes.

    ``extra_env`` carries opt-in reinjected variables (exec mode only,
    already validated upstream). It is applied after the scrub and the
    internal ``RLM_*`` params; ``RLM_*``-prefixed keys and ``KEEP_ENV`` names
    are ignored, never overlaid (defense in depth behind the server-side
    pre-open rejection). ``mode`` only selects the ``RLIMIT_NPROC`` default.
    """

    def __init__(
        self,
        agent_script: str | os.PathLike[str],
        limits: Limits,
        extra_env: Mapping[str, str] | None = None,
        mode: str = "doc",
    ):
        self._agent_script = os.path.abspath(os.fspath(agent_script))
        self._limits = limits
        self._extra_env = dict(extra_env) if extra_env else None
        self._mode = mode
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._read_transport: asyncio.BaseTransport | None = None
        self._stderr_reader: asyncio.StreamReader | None = None
        self._stderr_transport: asyncio.BaseTransport | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = bytearray()
        self._in_w: int | None = None
        self._sandbox_dir: str | None = None
        self._closed = False

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def sandbox_dir(self) -> str | None:
        return self._sandbox_dir

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc is not None else None

    async def _drain_stderr(self, reader: asyncio.StreamReader) -> None:
        """Continuously drain stderr into a bounded tail buffer.

        Never stops the child from writing (no pipe back-pressure) and keeps
        only the last ``_STDERR_TAIL_BYTES`` for diagnostics.
        """
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > _STDERR_TAIL_BYTES:
                    del self._stderr_tail[: len(self._stderr_tail) - _STDERR_TAIL_BYTES]
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass  # transport closed by close(): keep whatever we already read

    def stderr_tail(self, max_chars: int = 2000) -> str:
        """Decoded tail of the child's captured stderr ("" when none)."""
        data = bytes(self._stderr_tail).decode("utf-8", "replace")
        return data if len(data) <= max_chars else data[-max_chars:]

    async def wait_exit(self, timeout: float = 5.0) -> int | None:
        """Wait for the child to exit and return its exit code.

        Returns ``None`` when the deadline expires while the process is
        still alive, letting callers tell "crashed" from "not yet done".
        """
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return proc.returncode if proc is not None else None
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        return proc.returncode

    async def start(self) -> None:
        """Spawn the agent process and wire the frame + stderr readers."""
        if self._closed or self._proc is not None:
            raise SandboxError("driver already started or closed")
        self._sandbox_dir = tempfile.mkdtemp(prefix="rlm-sandbox-")
        in_r, in_w = os.pipe()  # supervisor -> agent
        out_r, out_w = os.pipe()  # agent -> supervisor
        err_r, err_w = os.pipe()  # agent stderr -> supervisor (diagnostics)
        env = _build_sandbox_env(self._limits, in_r, out_w, self._extra_env, self._mode)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                self._agent_script,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=err_w,
                cwd=self._sandbox_dir,
                env=env,
                pass_fds=(in_r, out_w),
                start_new_session=True,
            )
        except OSError as exc:
            for fd in (in_r, in_w, out_r, out_w, err_r, err_w):
                with contextlib.suppress(OSError):
                    os.close(fd)
            self._cleanup_dir()
            self._proc = None
            raise SandboxError(f"cannot spawn sandbox process: {exc}") from exc

        # Parent keeps only the ends it uses: write end of the in-pipe and
        # read ends of the out-pipe and the stderr pipe.
        os.close(in_r)
        os.close(out_w)
        os.close(err_w)
        self._in_w = in_w

        loop = asyncio.get_running_loop()
        os.set_blocking(out_r, False)
        reader = asyncio.StreamReader(limit=_READER_LIMIT)
        transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            os.fdopen(out_r, "rb", closefd=True),
        )
        self._read_transport = transport
        self._reader = reader

        os.set_blocking(err_r, False)
        err_reader = asyncio.StreamReader(limit=_STDERR_TAIL_BYTES)
        err_transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(err_reader),
            os.fdopen(err_r, "rb", closefd=True),
        )
        self._stderr_reader = err_reader
        self._stderr_transport = err_transport
        self._stderr_task = asyncio.create_task(self._drain_stderr(err_reader))

    async def send(self, frame: dict[str, object]) -> None:
        """Serialize and write one frame to the agent."""
        if not self.alive or self._in_w is None:
            raise SandboxError("sandbox is not alive")
        payload = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _write_all, self._in_w, payload)
        except OSError as exc:
            raise SandboxError(f"cannot write to sandbox: {exc}") from exc

    async def recv(self, timeout: float) -> dict[str, object] | None:
        """Read one frame; ``None`` on EOF, :class:`SandboxTimeout` on expiry."""
        if self._reader is None:
            raise SandboxError("driver not started")
        while True:
            try:
                line = await asyncio.wait_for(self._reader.readline(), timeout)
            except asyncio.TimeoutError:
                raise SandboxTimeout(f"no frame from sandbox within {timeout:g}s") from None
            except (ConnectionError, asyncio.IncompleteReadError):
                return None
            except ValueError as exc:  # StreamReader limit exceeded
                raise SandboxError(f"frame from sandbox exceeds the reader limit: {exc}") from exc
            if not line:
                return None
            try:
                obj = json.loads(line.decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise SandboxError(f"malformed frame from sandbox: {exc}") from exc
            if not isinstance(obj, dict):
                raise SandboxError("sandbox frame is not a JSON object")
            return obj

    def kill(self) -> None:
        """Kill the whole process group (children included), best effort."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()

    async def close(self) -> None:
        """Kill the process group, reap, and remove the sandbox cwd."""
        if self._closed:
            return
        self._closed = True
        self.kill()
        proc = self._proc
        if proc is not None:
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                self.kill()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), 2)
        for transport in (self._read_transport, self._stderr_transport):
            if transport is not None:
                transport.close()
        self._read_transport = None
        self._stderr_transport = None
        task = self._stderr_task
        if task is not None:
            task.cancel()
            self._stderr_task = None
        if self._in_w is not None:
            with contextlib.suppress(OSError):
                os.close(self._in_w)
            self._in_w = None
        self._reader = None
        self._stderr_reader = None
        self._cleanup_dir()

    def _cleanup_dir(self) -> None:
        if self._sandbox_dir:
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            self._sandbox_dir = None

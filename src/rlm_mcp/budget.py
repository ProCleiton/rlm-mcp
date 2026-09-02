"""Tree-wide budget ledger (DESIGN sections 1/G1 and 8).

One :class:`BudgetLedger` exists per *root* session and is shared by the
whole tree (parent and children).  The wall clock only counts time during
which the sandbox is actually executing: ``pause()``/``resume()`` bracket the
windows where the session is parked waiting for the harness to answer
sub-model requests, so answering time never drains the wall budget.
"""

from __future__ import annotations

import threading
import time

from rlm_mcp.types import Limits, Spent


class BudgetLedger:
    """Mutable per-tree counters plus the frozen :class:`Limits` of the tree."""

    def __init__(self, root_id: str, limits: Limits):
        self.root_id = root_id
        self.limits = limits
        self.iterations = 0
        self.llm_calls = 0
        self.errors = 0
        self.output_chars = 0
        self._lock = threading.Lock()
        self._active_windows = 0
        self._wall_base = 0.0
        self._window_started: float | None = None

    # -- wall clock (active windows only) -----------------------------------

    def resume(self) -> None:
        """Open an active window (a sandbox is executing)."""
        with self._lock:
            if self._active_windows == 0:
                self._window_started = time.monotonic()
            self._active_windows += 1

    def pause(self) -> None:
        """Close an active window; time is banked into the wall counter."""
        with self._lock:
            if self._active_windows <= 0:
                return
            self._active_windows -= 1
            if self._active_windows == 0 and self._window_started is not None:
                self._wall_base += time.monotonic() - self._window_started
                self._window_started = None

    def elapsed(self) -> float:
        """Wall seconds banked so far (parked/harness time not included)."""
        with self._lock:
            total = self._wall_base
            if self._active_windows > 0 and self._window_started is not None:
                total += time.monotonic() - self._window_started
            return total

    # -- counters ------------------------------------------------------------

    def charge_iteration(self, n: int = 1) -> None:
        with self._lock:
            self.iterations += max(n, 0)

    def charge_llm(self, n: int = 1) -> None:
        with self._lock:
            self.llm_calls += max(n, 0)

    def charge_error(self, n: int = 1) -> None:
        with self._lock:
            self.errors += max(n, 0)

    def charge_output(self, n: int = 1) -> None:
        with self._lock:
            self.output_chars += max(n, 0)

    # -- checks ---------------------------------------------------------------

    def check(self) -> str | None:
        """Return the reason the tree budget is exhausted, or ``None``."""
        limits = self.limits
        with self._lock:
            if self.iterations > limits.max_iterations:
                return f"max_iterations exceeded ({self.iterations}/{limits.max_iterations})"
            if self.llm_calls > limits.max_llm_calls:
                return f"max_llm_calls exceeded ({self.llm_calls}/{limits.max_llm_calls})"
            if self.errors > limits.max_errors:
                return f"max_errors exceeded ({self.errors}/{limits.max_errors})"
            if self.output_chars > limits.max_output_chars:
                return f"max_output_chars exceeded ({self.output_chars}/{limits.max_output_chars})"
        if self.elapsed() > limits.max_wall_seconds:
            return (
                f"max_wall_seconds exceeded "
                f"({self.elapsed():.1f}/{limits.max_wall_seconds:g})"
            )
        return None

    def snapshot(self) -> Spent:
        return Spent(
            iterations=self.iterations,
            llm_calls=self.llm_calls,
            wall_seconds=round(self.elapsed(), 6),
            errors=self.errors,
            output_chars=self.output_chars,
        )

"""Tests for the tree-wide BudgetLedger (budget.py)."""

import time

from rlm_mcp.budget import BudgetLedger
from rlm_mcp.types import Limits


def test_counters_and_snapshot():
    ledger = BudgetLedger("rlm_root", Limits())
    assert ledger.snapshot().to_dict() == {
        "iterations": 0,
        "llm_calls": 0,
        "wall_seconds": 0.0,
        "errors": 0,
        "output_chars": 0,
    }
    ledger.charge_iteration()
    ledger.charge_iteration()
    ledger.charge_llm(3)
    ledger.charge_error()
    ledger.charge_output(1200)
    spent = ledger.snapshot().to_dict()
    assert spent["iterations"] == 2
    assert spent["llm_calls"] == 3
    assert spent["errors"] == 1
    assert spent["output_chars"] == 1200


def test_check_reports_each_limit():
    ledger = BudgetLedger("r", Limits(max_iterations=2, max_errors=1))
    assert ledger.check() is None
    ledger.charge_iteration(3)
    assert ledger.check() == "max_iterations exceeded (3/2)"

    ledger = BudgetLedger("r", Limits(max_llm_calls=2))
    ledger.charge_llm(2)
    assert ledger.check() is None
    ledger.charge_llm(1)
    assert ledger.check() == "max_llm_calls exceeded (3/2)"

    ledger = BudgetLedger("r", Limits(max_errors=2, max_iterations=30))
    ledger.charge_error(3)
    assert ledger.check() == "max_errors exceeded (3/2)"

    ledger = BudgetLedger("r", Limits(max_output_chars=100))
    ledger.charge_output(99)
    assert ledger.check() is None
    ledger.charge_output(2)
    assert ledger.check() == "max_output_chars exceeded (101/100)"


def test_wall_clock_only_counts_active_windows():
    ledger = BudgetLedger("r", Limits())
    ledger.resume()
    time.sleep(0.06)
    ledger.pause()
    elapsed = ledger.elapsed()
    assert elapsed >= 0.06

    time.sleep(0.06)  # parked: nothing accumulates
    assert ledger.elapsed() == elapsed

    ledger.resume()
    time.sleep(0.04)
    ledger.pause()
    assert ledger.elapsed() >= elapsed + 0.03


def test_wall_limit_check():
    ledger = BudgetLedger("r", Limits(max_wall_seconds=0.05))
    ledger.resume()
    time.sleep(0.09)
    ledger.pause()
    assert "max_wall_seconds exceeded" in (ledger.check() or "")


def test_snapshot_shares_limits():
    limits = Limits(max_iterations=7)
    ledger = BudgetLedger("r", limits)
    assert ledger.limits is limits
    assert ledger.limits.max_iterations == 7

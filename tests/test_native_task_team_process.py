"""Signals and fork during proven native execution cannot bypass team cleanup."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["main", "main-twice", "background", "fork"])
def test_signal_and_fork_contract_in_bounded_processes(mode):
    if mode == "fork" and not hasattr(os, "fork"):
        pytest.skip("requires fork")
    result = subprocess.run(
        [sys.executable, "-m", "support.native_team_process", mode],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    print(json.dumps(receipt))
    assert receipt["balanced"]
    if mode == "fork":
        assert receipt["child"]
        assert receipt["baseline"] == 3
    elif mode == "background":
        assert receipt["background_result"]
        assert receipt["interrupt"]
    else:
        assert receipt["interrupt"]
        assert receipt["latency"] >= 0


@pytest.mark.parametrize("mode", ["main", "main-twice", "background", "interrupt-error", "fork"])
def test_copied_tasks_account_before_interrupt_can_raise(mode):
    if mode == "fork" and not hasattr(os, "fork"):
        pytest.skip("requires fork")
    result = subprocess.run(
        [sys.executable, "-m", "support.native_team_process", mode, "copied"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["balanced"]
    if mode == "fork":
        assert receipt["child"]
        assert receipt["baseline"] == 3
    else:
        assert receipt["interrupt"]
    assert receipt["copies_accounted"]
    if mode == "background":
        assert receipt["background_result"]

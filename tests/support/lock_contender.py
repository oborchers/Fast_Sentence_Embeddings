"""A separate Unix process for shared development-lock coordination tests."""

import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from _development_budget import stopped
from _development_locks import coordinated
from _harness.budget import WallBudgetExpiredError


def run():
    stopping = False
    ledger = SimpleNamespace(wall_deadline=None, stopping=False)

    def receive(_number, _frame):
        nonlocal stopping
        stopping = True

    def notify(event):
        print(json.dumps(event), flush=True)
        if event["event"] == "waiting_for_lock" and sys.argv[2] == "budget":
            ledger.wall_deadline = time.monotonic() - 1

    signal.signal(signal.SIGTERM, receive)
    try:
        with coordinated(
            sys.argv[1],
            "test-contender",
            "coordination test",
            notify,
            lambda: stopping or stopped(ledger),
            poll=0.01,
        ):
            sys.stdin.buffer.read(1)
    except (InterruptedError, WallBudgetExpiredError) as error:
        print(json.dumps({"event": "stopped", "reason": type(error).__name__}), flush=True)


if __name__ == "__main__":
    run()

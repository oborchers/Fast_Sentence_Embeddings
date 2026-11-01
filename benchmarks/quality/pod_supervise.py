"""Bound a fresh workload process group and retain logs without an inherited tee pipe."""

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path


def kill_group(pid, sig):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, sig)


def run(command, output, seconds):
    if seconds <= 0:
        return 124
    with output.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            status = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            status = 124
        finally:
            kill_group(child.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=10)
            kill_group(child.pid, signal.SIGKILL)
            child.wait()
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    raise SystemExit(
        run(
            ["/bin/bash", "/dr151-work.sh"],
            Path("/dr151-results/bootstrap.log"),
            float(sys.argv[1]),
        )
    )

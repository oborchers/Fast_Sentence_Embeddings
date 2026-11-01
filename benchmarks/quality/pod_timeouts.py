"""Abort an isolated pod workload before another stage can overlap timed-out descendants."""

import subprocess


def abort(error, enabled=True):
    if enabled and isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        raise error

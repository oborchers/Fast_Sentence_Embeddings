"""Opt-in host admission, before installation, using a bounded observation window."""

from __future__ import annotations

import math
import time
from pathlib import Path

FIELDS = {"duration_seconds", "cpu_some_percent_max", "load1_max"}


def validate(config):
    """Validate original inputs before observing or converting anything."""
    if not isinstance(config, dict) or set(config) - FIELDS:
        raise ValueError("host_quiet must be an object with only " + ", ".join(sorted(FIELDS)))
    if not {"cpu_some_percent_max", "load1_max"}.issubset(config):
        raise ValueError("host_quiet requires cpu_some_percent_max and load1_max")
    for name, value in {"duration_seconds": 180, **config}.items():
        if type(value) not in (int, float):
            raise ValueError(f"host_quiet {name} must be a positive finite number")
        try:
            valid = math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid or (name == "cpu_some_percent_max" and value > 100):
            raise ValueError(f"host_quiet {name} is outside its positive finite range")


def _check_deadline(deadline, clock):
    if deadline is not None and clock() >= deadline:
        raise ValueError("host quiet observation exhausted the allocation work budget")


def _sample(proc, clock, deadline):
    """Read host PSI and load from one proc source, including synthetic probe sources."""
    _check_deadline(deadline, clock)
    rows = [line.split() for line in (proc / "pressure/cpu").read_text().splitlines()]
    some = [row[1:] for row in rows if row and row[0] == "some"]
    if len(some) != 1:
        raise ValueError("host CPU pressure needs exactly one some row")
    totals = [field[6:] for field in some[0] if field.startswith("total=")]
    if len(totals) != 1 or not totals[0].isascii() or not totals[0].isdecimal():
        raise ValueError("host CPU pressure some.total must be a nonnegative integer")
    total = int(totals[0])
    monotonic = clock()
    load = float((proc / "loadavg").read_text().split()[0])
    if not math.isfinite(load) or load < 0:
        raise ValueError("host one-minute load must be finite and nonnegative")
    _check_deadline(deadline, clock)
    return {"monotonic": monotonic, "some_total_microseconds": total, "load1": load}


def observe(config, *, budget=None, proc=Path("/proc"), clock=time.monotonic, sleep=time.sleep):
    """Keep raw samples and fail closed on missing, malformed or reset observations."""
    validate(config)
    duration = config.get("duration_seconds", 180)
    record = {
        "duration_seconds": duration,
        "samples": [],
        "cpu_some_percent": None,
        "load1_max": None,
    }
    if budget is not None and not budget.admit(duration):
        record["unavailable_reason"] = "host quiet observation does not fit allocation work budget"
        return record
    deadline = None if budget is None else clock() + budget.remaining()
    try:
        first = _sample(proc, clock, deadline)
        record["samples"].append(first)
        last = first
        while last["monotonic"] - first["monotonic"] < duration:
            _check_deadline(deadline, clock)
            wait = min(1, duration - (last["monotonic"] - first["monotonic"]))
            sleep(wait if deadline is None else min(wait, max(0, deadline - clock())))
            current = _sample(proc, clock, deadline)
            record["samples"].append(current)
            if current["monotonic"] <= last["monotonic"]:
                raise ValueError("host quiet observation clock did not advance")
            if current["some_total_microseconds"] < last["some_total_microseconds"]:
                raise ValueError("host CPU pressure some.total decreased")
            last = current
        elapsed = last["monotonic"] - first["monotonic"]
        delta = last["some_total_microseconds"] - first["some_total_microseconds"]
        percent = 100 * delta / (elapsed * 1_000_000)
        if not math.isfinite(percent) or percent > 100:
            raise ValueError("host CPU pressure interval is outside 0 to 100 percent")
        record.update(
            elapsed_seconds=elapsed,
            some_delta_microseconds=delta,
            cpu_some_percent=percent,
            load1_max=max(sample["load1"] for sample in record["samples"]),
        )
    except (OSError, UnicodeError, ValueError, IndexError, OverflowError) as error:
        record["unavailable_reason"] = str(error)
    return record


def deviations(observed, config):
    if "unavailable_reason" in observed:
        return ["host_quiet: unavailable: " + observed["unavailable_reason"]]
    found = []
    for name in ("cpu_some_percent", "load1"):
        field = name if name == "cpu_some_percent" else "load1_max"
        limit = config[name + "_max"]
        if observed[field] >= limit:
            found.append(f"host_quiet {field}: observed {observed[field]}, required < {limit}")
    return found

"""Fail-closed execution guards and a persistent, crash-conservative gate budget."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from mlx_gate_data import BASE


class StopError(RuntimeError):
    """An operational stop, preserved in the raw journal, never a successful measurement."""


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def append(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def threshold(receipt: dict) -> float:
    baseline = receipt["baseline"]
    limit = receipt["max_load"]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (baseline, limit)):
        raise ValueError("idle receipt requires finite numeric baseline and threshold")
    if receipt["valid"] is not True or baseline < 0:
        raise ValueError("idle receipt is invalid")
    if "threshold_override" in receipt:
        return _decision_threshold(receipt)
    if limit != min(baseline + 1.5, 5.0):
        raise ValueError("idle receipt is invalid or has the wrong threshold formula")
    return limit


def _decision_threshold(receipt: dict) -> float:
    """Accept only Oliver's explicit DR-147 limit-5 decision, preserving real idle data."""
    decision = receipt["threshold_override"]
    if not isinstance(decision, dict) or decision.keys() != {
        "policy",
        "authorized_by",
        "decided_at",
    }:
        raise ValueError("threshold override requires exactly policy, authorized_by and decided_at")
    if (
        decision["policy"] != "dr147-oliver-limit-5@1"
        or decision["authorized_by"] != "Oliver"
        or receipt["baseline"] != 2.39
        or receipt["max_load"] != 5
    ):
        raise ValueError("threshold override does not match Oliver's DR-147 limit-5 decision")
    stamp = decision["decided_at"]
    if not isinstance(stamp, str):
        raise ValueError("threshold override requires a timestamp with timezone")
    try:
        decided = datetime.fromisoformat(stamp)
    except ValueError as error:
        raise ValueError("threshold override requires a timestamp with timezone") from error
    if decided.utcoffset() is None:
        raise ValueError("threshold override requires a timestamp with timezone")
    return receipt["max_load"]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def identity(repo: Path, commit: str) -> dict:
    import fse
    from fse import _native
    from fse._cpus import usable_cpus

    if git(repo, "rev-parse", "HEAD") != commit or git(repo, "status", "--porcelain"):
        raise StopError("checkout changed or contains uncommitted files")
    subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE, commit], check=True
    )
    files = {
        "fse": str(Path(fse.__file__).resolve()),
        "native": str(Path(_native.__file__).resolve()),
    }
    if any(not Path(p).is_relative_to(repo / "src") for p in files.values()):
        raise StopError("imports do not come from this checkout")
    from fse._container._hash import Hasher

    native_digest = Hasher()
    native_digest.update(Path(files["native"]).read_bytes())

    hardware = {}
    if sys.platform == "darwin":
        keys = (
            "machdep.cpu.brand_string",
            "hw.perflevel0.physicalcpu",
            "hw.perflevel1.physicalcpu",
            "hw.memsize",
        )
        hardware = {
            k: subprocess.check_output(["sysctl", "-n", k], text=True).strip() for k in keys
        }
    return {
        "commit": commit,
        "base": BASE,
        "files": files,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "compiler": platform.python_compiler(),
        "default_workers": usable_cpus(),
        "hardware": hardware,
        "native_checksum": native_digest.checksum(),
        "native_isa": _native.default_isa(),
        "versions": {p: version(p) for p in ("numpy", "mlx", "memray")},
    }


def services() -> list[dict]:
    """Observe process names and CPU only. Never collect process arguments or contents."""
    lines = subprocess.check_output(
        ["ps", "-axo", "pid=,pcpu=,time=,comm="], text=True
    ).splitlines()
    out = []
    for line in lines:
        pid, cpu, total, name = line.split(None, 3)
        name = Path(name).name
        if float(cpu) >= 1 or any(s in name.lower() for s in ("wdav", "defender", "mdatp")):
            out.append(
                {"pid": int(pid), "name": name, "cpu_percent": float(cpu), "cpu_time": total}
            )
    return out


def charged(events: list[dict]) -> float:
    reservations: dict[str, float] = {}
    settled = set()
    for event in events:
        key = event["id"]
        seconds = event["seconds"]
        if event.get("base", BASE) != BASE or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid budget event")
        if event["kind"] == "reserve" and key not in reservations:
            reservations[key] = seconds
        elif event["kind"] == "settle" and key in reservations and key not in settled:
            reservations[key] = seconds
            settled.add(key)
        else:
            raise ValueError("invalid or duplicate budget event")
    return sum(reservations.values())


@dataclass
class Guard:
    scratch: Path
    lock: dict
    limit: float
    deadline: float
    journal: Path

    def check(self, label: str, *, sample: float | None = None) -> dict:
        if (self.scratch / "pause-local.marker").exists():
            raise StopError("pause-local.marker is present")
        if read(self.scratch / "bench.lock" / "owner.json") != self.lock:
            raise StopError("measurement lock owner changed")
        os.kill(self.lock["pid"], 0)
        if time.monotonic() >= self.deadline:
            raise StopError("measurement budget exhausted")
        value = os.getloadavg()[0] if sample is None else sample
        event = {
            "event": "guard",
            "label": label,
            "at": time.time(),
            "load": value if math.isfinite(value) else None,
            "finite_load": math.isfinite(value),
        }
        append(self.journal, event)
        if not math.isfinite(value) or value >= self.limit:
            raise StopError(f"load {value} is not strictly below {self.limit}")
        return event


def acquire(scratch: Path, commit: str, branch: str) -> dict:
    if (scratch / "pause-local.marker").exists():
        raise StopError("pause-local.marker is present")
    folder = scratch / "bench.lock"
    folder.mkdir()  # atomic, never break a pre-existing lock
    lock = {
        "pid": os.getpid(),
        "branch": branch,
        "commit": commit,
        "started": time.time(),
        "token": uuid.uuid4().hex,
    }
    write(folder / "owner.json", lock)
    return lock


class Budget:
    """Reserve before starting; a killed process consumes its reservation on the next run."""

    def __init__(self, scratch: Path, seconds: float):
        self.path = scratch / "mlx-gate-budget.jsonl"
        if not math.isfinite(seconds) or not 0 < seconds <= 1800:
            raise ValueError("budget must be positive and at most 1800 seconds")
        events = (
            [json.loads(line) for line in self.path.read_text().splitlines()]
            if self.path.exists()
            else []
        )
        if charged(events) + seconds > 1800:
            raise StopError("reservation exceeds the shared 1800-second budget")
        self.token, self.seconds, self.start = uuid.uuid4().hex, seconds, time.monotonic()
        append(self.path, {"id": self.token, "kind": "reserve", "seconds": seconds, "base": BASE})

    def settle(self) -> float:
        elapsed = time.monotonic() - self.start
        append(
            self.path,
            {
                "id": self.token,
                "kind": "settle",
                "seconds": elapsed,
                "overrun": elapsed > self.seconds,
            },
        )
        return elapsed

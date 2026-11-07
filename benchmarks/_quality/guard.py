"""Private resource guard of a local quality run: caches, disk floor, memory and deadline.

A stop signal ends the run only at the next task boundary, so the guard has two levels:
low disk space asks for that boundary stop; a resident set above the limit cannot wait
for the running task and ends the process at once, after the rows are written."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

CACHE_VARIABLES = ("HF_HOME", "HF_DATASETS_CACHE")
GIB = 1024**3
INTERVAL_SECONDS = 10


def caches():
    """The Hugging Face cache folders, which a local run requires to be set explicitly: the
    datasets of a multilingual benchmark take tens of GB and must land where they are watched."""
    paths = {}
    for name in CACHE_VARIABLES:
        value = os.environ.get(name)
        if not value or not Path(value).is_absolute():
            raise ValueError(f"a local run needs an absolute {name}")
        paths[name] = Path(value)
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def size_bytes(path):
    result = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=False)
    if result.returncode:
        return None
    return int(result.stdout.split()[0]) * 1024


def resident_bytes(pid):
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    text = result.stdout.strip()
    return int(text) * 1024 if result.returncode == 0 and text.isdecimal() else None


def free_memory_percent():
    """The share of memory the system reports as available (macOS memorystatus level,
    Linux MemAvailable)."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, check=False
        )
        value = result.stdout.strip()
        return int(value) if value.isdecimal() else None
    fields = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
    )
    total = int(fields["MemTotal"].split()[0])
    return int(100 * int(fields["MemAvailable"].split()[0]) / total)


class Guard:
    """Samples disk, memory and the deadline every ten seconds into ``telemetry.jsonl``."""

    def __init__(self, results, folders, limits, abort):
        self.results, self.folders, self.limits, self.abort = results, folders, limits, abort
        self.stop_reason = None
        self.peak_rss = 0
        self.min_free = None
        self.ended = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def start(self):
        self.sample()
        self.thread.start()

    def watch(self):
        while not self.ended.wait(INTERVAL_SECONDS):
            self.sample()

    def sample(self):
        free = {str(p): shutil.disk_usage(p).free for p in self.folders}
        rss = resident_bytes(os.getpid()) or 0
        self.peak_rss = max(self.peak_rss, rss)
        lowest = min(free.values())
        self.min_free = lowest if self.min_free is None else min(self.min_free, lowest)
        row = {
            "unix": time.time(),
            "free_bytes": free,
            "rss_bytes": rss,
            "free_memory_percent": free_memory_percent(),
        }
        with (self.results / "telemetry.jsonl").open("a") as stream:  # no console line
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        if lowest < self.limits["min_free_bytes"] and self.stop_reason is None:
            self.stop_reason = f"free disk below {self.limits['min_free_bytes'] // GIB} GiB"
        if rss > self.limits["rss_limit_bytes"]:
            self.abort(f"resident set {rss // GIB} GiB above the limit")

    def close(self):
        self.ended.set()
        self.thread.join()
        self.sample()
        return {
            "peak_rss_bytes": self.peak_rss,
            "min_free_bytes": self.min_free,
            "limits": self.limits,
            "folders": [str(p) for p in self.folders],
        }


class RefusedError(ValueError):
    """A task refused before loading because its memory estimate exceeds the limit."""


def admit(task, dim, limit):
    """The memory estimate of ``task``; a task above ``limit`` is refused before loading."""
    estimate = memory_estimate(task, dim)
    if limit is not None and estimate is not None and estimate > limit:
        raise RefusedError(f"memory estimate {estimate // GIB} GiB above the limit")
    return estimate


def memory_estimate(task, dim):
    """A coarse upper estimate of the bytes one task holds: its texts (four bytes per
    character in the worst Python string case) and one float32 vector per text, both twice
    (MTEB copies inputs into datasets and results into arrays). Bitext mining counts every
    pair, so its estimate is far above the unique sentences. Unknown statistics give None."""
    stats = task.metadata.descriptive_stats or {}
    parts = [p for split in task.eval_splits for p in evaluated(stats.get(split) or {}, task)]
    samples = sum(int(p.get("num_samples") or 0) for p in parts)
    characters = sum(int(p.get("number_of_characters") or 0) for p in parts)
    if not samples:
        return None
    return 2 * (4 * characters + 4 * dim * samples)


def evaluated(entry, task):
    """The statistics of the subsets the benchmark evaluates, where the statistics list them."""
    subsets = entry.get("hf_subset_descriptive_stats") or {}
    if not subsets:
        return [entry]
    return [subsets[s] for s in task.hf_subsets if s in subsets]

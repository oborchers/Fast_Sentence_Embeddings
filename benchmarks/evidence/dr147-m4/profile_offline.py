"""Read-only replay of existing six profiles; never records or changes raw files."""

import bisect
import csv
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

from memray import FileReader

REPO = Path(__file__).resolve().parents[3]
ROOT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("/Users/oliver.borchers/Desktop/Code/fse-m4-scratch/dr147-profiles-3101e4a7fdd8")
)
sys.path.insert(0, str(REPO / "tools"))
profile_top = importlib.import_module("profile_top")
Symbols = profile_top.Symbols
thread_counts = profile_top.thread_counts

MATCH = {
    "pool": lambda n: n == "fse_loop_f32_neon",
    "cvwait": lambda n: n == "__psynch_cvwait",
    "wordpiece": lambda n: n == "__pyx_f_3fse_7_native__wp_piece",
    "workq": lambda n: n == "__workq_kernreturn",
    "memmove": lambda n: n == "_platform_memmove",
    "IOKit": lambda n: n.startswith("IOKit+"),
    "IOKit_dominant": lambda n: n == "IOKit+0x1ecc0",
    "FLOAT_absolute": lambda n: n == "FLOAT_absolute",
}


def read(p):
    return json.loads(p.read_text())


def counts(threads, libs, symbols):
    own, total, n = Counter(), Counter(), 0
    per_thread = []
    for thread in threads:
        a, b, k = thread_counts(thread, libs, symbols)
        own.update(a)
        total.update(b)
        n += k
        per_thread.append(
            {
                "name": thread.get("name"),
                "tid": thread.get("tid"),
                "samples": k,
                "memmove_self_samples": a["_platform_memmove"],
                "cvwait_self_samples": a["__psynch_cvwait"],
            }
        )
    return {
        "samples": n,
        "categories": {
            key: {
                "self_samples": sum(v for name, v in own.items() if test(name)),
                "self_sample_percent": 100 * sum(v for name, v in own.items() if test(name)) / n,
                "matching_symbols": [name for name in own if test(name)],
            }
            for key, test in MATCH.items()
        },
        "top_self": [
            {
                "function": name,
                "self_samples": v,
                "self_sample_percent": 100 * v / n,
                "total_stack_samples": total[name],
            }
            for name, v in own.most_common(30)
        ],
        "per_thread": per_thread,
    }


result = {
    "profile_root": str(ROOT),
    "profile_commit": read(ROOT / "state.json")["identity"]["commit"],
    "method": {
        "mapping": "sample_unix_ms = meta.startTime + samples.time[i]; window_unix_ms = 1000 * call_windows_unix_seconds",
        "containment": "start <= sample_unix_ms <= stop, exact float comparisons, no padding/rounding",
        "denominator": "all non-null thread-stack samples in scope, summed over threads; NOT end-to-end wall time or CPU-time percentage",
        "inclusive": "whole recording including imports/loading/guards/checks",
        "filtered": "sample timestamps contained in encode calls only, retaining all threads; concurrency remains",
        "memory": "memray.FileReader(...).metadata.peak_memory exact bytes; cumulative allocation is not a peak or per-call comparison",
    },
    "cpu_profiles": {},
    "memory": {},
    "guards": {},
    "worker_checks": {},
    "status": {"extra_measurements": False, "policy": "undecided", "pr": "Draft"},
}
for backend in ("cpu", "mlx"):
    d = ROOT / (backend + "-cpu")
    p = read(d / "cpu.json")
    symbols = Symbols(read(d / "cpu.syms.json"))
    worker = read(d / "worker-result.json")
    windows = worker["call_windows_unix_seconds"]
    starts = [a * 1000 for a, b in windows]
    ends = [b * 1000 for a, b in windows]
    assert all(a <= b for a, b in zip(starts, ends)) and all(
        b <= a for b, a in zip(ends, starts[1:])
    )
    filtered = []
    for thread in p["threads"]:
        table = thread["samples"]
        n = len(table["stack"])
        keep = []
        for i, t in enumerate(table["time"]):
            absolute = p["meta"]["startTime"] + t
            k = bisect.bisect_right(starts, absolute) - 1
            if k >= 0 and absolute <= ends[k]:
                keep.append(i)
        t = dict(thread)
        t["samples"] = {
            key: [value[i] for i in keep] if isinstance(value, list) and len(value) == n else value
            for key, value in table.items()
        }
        t["samples"]["length"] = len(keep)
        filtered.append(t)
    result["cpu_profiles"][backend] = {
        "inclusive": counts(p["threads"], p["libs"], symbols),
        "inference_only": counts(filtered, p["libs"], symbols),
        "encode_windows": len(windows),
        "meta_startTime_unix_ms": p["meta"]["startTime"],
        "all_encode_wall_seconds_in_profile_only": sum(b - a for a, b in windows),
        "wall_percent_of_specific_causes": None,
    }
    mem = ROOT / (backend + "-mem")
    rss = ROOT / (backend + "-rss")
    with FileReader(str(mem / "mem.bin")) as reader:
        peak = reader.metadata.peak_memory
    with (rss / "rss.csv").open() as f:
        rows = list(csv.DictReader(f))
    high = max(rows, key=lambda r: int(r["rss_bytes"]))
    result["memory"][backend] = {
        "native_memray_peak_bytes": peak,
        "native_peak_scope": "all traced process allocations including imports/model/verification; not a per-call steady-state value",
        "rss_tree_peak_bytes": int(high["rss_bytes"]),
        "rss_peak_at_seconds": float(high["seconds"]),
        "rss_samples": len(rows),
        "mlx_allocator_peak_bytes_by_recording": {
            purpose: read(ROOT / (backend + "-" + purpose) / "worker-result.json").get(
                "mlx_peak_bytes"
            )
            for purpose in ("cpu", "mem", "rss")
        },
        "do_not_sum": "native allocations, RSS and MLX allocator overlap and have different scope",
    }
for d in ROOT.iterdir():
    if not d.is_dir():
        continue
    job = read(d / "job.json")
    events = [json.loads(line) for line in (d / "guards.jsonl").read_text().splitlines()]
    loads = [e["load"] for e in events if e["event"] == "guard"]
    worker = read(d / "worker-result.json")
    assert worker["first_last_bits_match_accepted_reference"] and not worker["fallback_events"]
    assert max(loads) < job["profile_guard_policy"]["max_load"]
    result["guards"][d.name] = {
        "min": min(loads),
        "max": max(loads),
        "count": len(loads),
        "authorized_limit": job["profile_guard_policy"]["max_load"],
        "observed_ge5": any(v >= 5 for v in loads),
    }
    result["worker_checks"][d.name] = {
        "bits_match": True,
        "fallback_events": {},
        "calls": worker["calls"],
        "workers": job["workers"],
    }
result["accounting"] = read(ROOT / "state.json")
print(json.dumps(result, indent=2))

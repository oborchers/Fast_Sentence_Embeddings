"""Profiler commands and explicit observed coverage, without optional imports."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

from _harness.records import document

SOURCES = {
    "py-spy": "https://github.com/benfred/py-spy/blob/v0.4.2/src/speedscope.rs",
    "samply": "https://github.com/mstange/samply/tree/samply-v0.13.1",
    "memray": "https://github.com/bloomberg/memray/blob/v1.20.0/src/memray/_metadata.py",
}
LIMITS = {
    "py-spy": "Python threads excluding idle frames; native frames on Linux only",
    "samply": "Tool records a process tree; project acceptance scope is child",
    "memray": "All threads of this process; no forks and no following exec",
}


def command(spec, executable, folder, target):
    tool = spec["tool"]
    if tool == "memray":
        return [
            spec["python"],
            "-m",
            "memray",
            "run",
            "--native",
            "-o",
            str(folder / "mem.bin"),
            *target[1:],
        ]
    if tool == "samply":
        return [
            executable,
            "record",
            "--save-only",
            "--unstable-presymbolicate",
            "-o",
            str(folder / "cpu.json"),
            "--",
            *target,
        ]
    children = (
        ["--subprocesses"] if spec["requested"]["processes"] == "child-and-subprocesses" else []
    )
    return [
        executable,
        "record",
        "--native",
        "--threads",
        *children,
        "-f",
        "speedscope",
        "-o",
        str(folder / "cpu.speedscope.json"),
        "--",
        *target,
    ]


def _spy(folder, pid, subprocesses):
    profiles = document(folder / "cpu.speedscope.json")["profiles"]
    processes, threads, count = set(), [], 0
    for profile in profiles:
        match = re.search(r"Process (\d+) Thread", profile["name"])
        process = int(match[1]) if match else None
        if process is not None:
            processes.add(process)
        threads.append(profile["name"])
        if not subprocesses or process == pid:
            count += len(profile["samples"])
    return {"processes": sorted(processes), "threads": threads, "samples": count}


def _samply(folder, pid):
    path = folder / "cpu.json"
    if path.is_file():
        data = document(path)
    else:
        with gzip.open(Path(str(path) + ".gz"), "rt") as stream:
            import json

            data = json.load(stream)
    threads = data["threads"]
    return {
        "processes": sorted({int(t["pid"]) for t in threads}),
        "threads": [{k: t[k] for k in ("pid", "tid", "name")} for t in threads],
        "samples": sum(t["samples"]["length"] for t in threads if int(t["pid"]) == pid),
    }


def coverage(spec, folder, receipt, leader_pid, profiler):
    pid = receipt["diagnostic_pid"]
    parent = receipt["diagnostic_parent_pid"]
    tool = spec["tool"]
    if tool == "py-spy":
        subprocesses = spec["requested"]["processes"] == "child-and-subprocesses"
        observed = _spy(folder, pid, subprocesses)
        assigned = pid in observed["processes"] if subprocesses else parent == leader_pid
        evidence = "profile names" if subprocesses else "child parent equals profiler leader"
    elif tool == "samply":
        observed = _samply(folder, pid)
        assigned, evidence = pid in observed["processes"], "threads[].pid"
    else:
        meta = document(folder / "memray-metadata.json")
        observed = {
            "processes": [meta["pid"]],
            "threads": None,
            "samples": meta["total_allocations"],
        }
        assigned = pid == leader_pid == meta["pid"]
        evidence = "same-process leader and memray metadata.pid"
    result = {
        "requested": spec["requested"],
        "observed": observed,
        "assured": {
            "limits": LIMITS[tool],
            "source": SOURCES[tool],
            "version": profiler["version"],
        },
        "child_pid": pid,
        "child_parent_pid": parent,
        "leader_pid": leader_pid,
        "assignment_evidence": evidence,
        "scope": "whole child including preparation and warmup",
        "valid": bool(assigned and observed["samples"] > 0),
    }
    if spec["action"] == "profile-head" and (folder / "profile.log").exists():
        result["profile_ready"] = "profile_ready" in (folder / "profile.log").read_text()
    return result

"""Small installed-interpreter probes for profile provenance and profiler metadata."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from _harness.records import atomic, document


def child(file):
    job = document(file)
    action = job["action"]
    if action == "identity":
        from _development_process import identity

        result = {"identity": identity()}
    elif action == "memray-metadata":
        import memray

        reader = memray.FileReader(job["profile"])
        result = {
            "pid": reader.metadata.pid,
            "total_allocations": reader.metadata.total_allocations,
            "version": memray.__version__,
        }
        if job.get("search_threads"):
            result["search_threads"] = sorted(
                {
                    r.thread_name
                    for r in reader.get_allocation_records()
                    if "fse-search" in r.thread_name
                }
            )
    elif action == "version":
        import memray

        binary = importlib.import_module("memray._memray")
        result = {
            "version": memray.__version__,
            "path": binary.__file__,
            "module_path": memray.__file__,
        }
    else:
        raise ValueError("unknown profile probe action")
    atomic(
        Path(job["receipt"]),
        {
            **result,
            "diagnostic_profile": True,
            "valid_for_comparison": False,
            "diagnostic_pid": os.getpid(),
            "diagnostic_parent_pid": os.getppid(),
        },
    )


def access_program(path, receipt):
    """Use a standalone script so isolated Python does not need repository imports."""
    path.write_text(
        "import json, os, time\n"
        "until = time.monotonic() + 1.5\n"
        "while time.monotonic() < until:\n"
        "    values = [str(i) for i in range(1000)]\n"
        f"with open({str(receipt)!r}, 'x') as stream:\n"
        "    json.dump({'diagnostic_pid': os.getpid(), 'diagnostic_parent_pid': os.getppid(), 'diagnostic_profile': True, 'valid_for_comparison': False}, stream)\n"
    )

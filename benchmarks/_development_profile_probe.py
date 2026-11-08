"""Small installed-interpreter probes for profile provenance and profiler metadata."""

from __future__ import annotations

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
    elif action == "version":
        import memray

        result = {"version": memray.__version__, "path": memray.__file__}
    else:
        raise ValueError("unknown profile probe action")
    atomic(Path(job["receipt"]), result)


def access_program(path, receipt):
    """Use a standalone script so isolated Python does not need repository imports."""
    path.write_text(
        "import json, os, time\n"
        f"with open({str(receipt)!r}, 'x') as stream:\n"
        "    json.dump({'diagnostic_pid': os.getpid(), 'diagnostic_parent_pid': os.getppid()}, stream)\n"
        "until = time.monotonic() + 1.5\n"
        "while time.monotonic() < until:\n"
        "    values = [str(i) for i in range(1000)]\n"
    )

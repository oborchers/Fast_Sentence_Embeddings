"""Prepare profiler tools after measurement without modifying the executing environment."""

from __future__ import annotations

from pathlib import Path

from _harness.profile import identity, lock_packages, same_identity
from _harness.records import atomic, sha256
from _measurement_config import CONFIG

COPY_PACKAGES = """
import pathlib, shutil, sys, sysconfig
source = pathlib.Path(sysconfig.get_path('purelib'))
target = pathlib.Path(sys.argv[1]) / source.relative_to(sys.prefix)
shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
"""


def install(runner, uv, python, lock):
    runner.launch(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--require-hashes",
            "-r",
            str(lock),
        ],
        "install-tools",
    )
    runner.launch([str(uv), "pip", "check", "--python", str(python)], "check-tools")


def prepare(runner, origin, *, allocation, selected):
    """Runs inside the profile heavy lock and deadline, including environment preparation."""
    allocation.check_budget()
    atomic(runner.root / "selection.json", selected)
    runner.open_evidence = selected["open_evidence"]
    original = runner.spec["python"]
    before = identity(runner.probe("identity"), "development")
    same_identity(origin["identity"], before, {})
    uv = allocation.root / "speed-env/tools/bin/uv"
    destination = Path(runner.spec["work"]) / "environment"
    runner.launch([str(uv), "venv", "--python", original, str(destination)], "create-environment")
    python = destination / "bin/python"
    if runner.spec["tool"] == "memray":
        lock = CONFIG / "profile.lock.txt"
        extra = lock_packages(lock)
        for name, version in extra.items():
            if name in before["dependencies"] and before["dependencies"][name] != version:
                raise ValueError("profile lock would change an origin package")
        runner.launch([original, "-c", COPY_PACKAGES, str(destination)], "copy-packages")
        runner.spec["python"] = str(python)
        same_identity(before, identity(runner.probe("identity"), "development"), {})
    else:
        lock = CONFIG / "profile-cpu.lock.txt"
    install(runner, uv, python, lock)
    if runner.spec["tool"] == "py-spy":
        runner.tool_executable = str(destination / "bin/py-spy")
    # The origin is checked again after installation, before access probes and execution.
    current = runner.spec["python"]
    runner.spec["python"] = original
    try:
        same_identity(before, identity(runner.probe("identity"), "development"), {})
    finally:
        runner.spec["python"] = current
    atomic(
        runner.root / "preparation.json",
        {
            "billing": allocation.budget.billing,
            "planning_minutes_source": allocation.budget.planning_minutes_source,
            "origin_python": original,
            "profile_python": current,
            "tool_environment": str(destination),
            "lock_sha256": sha256(lock),
            "elapsed_allocation_seconds": allocation.budget.elapsed(),
            "remaining_allocation_seconds": allocation.budget.remaining(),
        },
    )
    atomic(runner.root / "profile.json", runner.spec)
    allocation.check_budget()
    if not allocation.budget.admit(60):
        raise ValueError("allocation budget exhausted during profile preparation")

"""Strict profile@1 contract and content identities, independent of fse."""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

from _harness.records import sha256

FIELDS = {
    "format",
    "name",
    "entry",
    "source_results",
    "job",
    "nominal_arm",
    "action",
    "python",
    "tool",
    "requested",
    "iterations",
    "wall_time_seconds",
    "output",
    "work",
    "extra_lock",
    "datasets",
}


def validate(raw, check_only=False):
    if not isinstance(raw, dict) or raw.keys() != FIELDS or raw["format"] != "profile@1":
        raise ValueError("profile@1 requires exactly its documented fields")
    for key in ("name", "job"):
        if not isinstance(raw[key], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", raw[key]
        ):
            raise ValueError(f"{key} must be a safe file stem")
    if raw["entry"] not in ("development", "speed") or raw["tool"] not in (
        "samply",
        "py-spy",
        "memray",
    ):
        raise ValueError("unknown profile entry or tool")
    _matrix(raw, check_only)
    _action_budget(raw)
    _paths(raw, check_only)
    return raw


def _matrix(raw, check_only):
    if raw["requested"] not in (
        {"processes": "child", "threads": "all"},
        {"processes": "child-and-subprocesses", "threads": "all"},
    ):
        raise ValueError("unsupported requested coverage")
    if raw["tool"] != "py-spy" and raw["requested"]["processes"] != "child":
        raise ValueError("samply and memray support only the project scope child")
    if sys.platform != "linux" and (
        raw["tool"] == "py-spy" or (raw["entry"] == "speed" and not check_only)
    ):
        raise ValueError("py-spy and the speed entry require Linux")


def _action_budget(raw):
    if raw["action"] not in (
        ("measure", "profile-head") if raw["entry"] == "development" else ("child",)
    ):
        raise ValueError("unsupported profile action")
    iterations = raw["iterations"]
    if iterations is not None and (
        type(iterations) is not int or iterations <= 0 or raw["action"] != "profile-head"
    ):
        raise ValueError("iterations is null or a positive integer for profile-head")
    seconds = raw["wall_time_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite wall_time_seconds required")


def _paths(raw, check_only):
    for key in ("python", "output", "work", "source_results"):
        if not isinstance(raw[key], str) or not Path(raw[key]).is_absolute():
            raise ValueError(f"absolute {key} path required")
    if not Path(raw["python"]).is_file() or not os.access(raw["python"], os.X_OK):
        raise ValueError("executable profile Python required")
    _optional_paths(raw, check_only)
    paths = [Path(raw[key]).resolve() for key in ("output", "work")]
    if any(
        p.exists() or Path(raw[key]).is_symlink()
        for p, key in zip(paths, ("output", "work"), strict=True)
    ):
        raise ValueError("profile output and work must be fresh")
    source = Path(raw["source_results"]).resolve()
    if any(p.is_relative_to(source) or source.is_relative_to(p) for p in paths):
        raise ValueError("profile folders overlap the origin results")
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise ValueError("profile output and work must not overlap")


def _optional_paths(raw, check_only):
    if (raw["extra_lock"] is not None) != (raw["tool"] == "memray"):
        raise ValueError("extra_lock is required exactly for memray")
    if (
        raw["tool"] == "memray"
        and Path(raw["extra_lock"]).resolve()
        != Path(__file__).resolve().parents[1] / "config/profile.lock.txt"
    ):
        raise ValueError("extra_lock must name the repository profile.lock.txt")
    if raw["tool"] == "memray" and not Path(raw["extra_lock"]).is_file():
        raise ValueError("profile package lock is missing")
    if raw["entry"] == "development" and raw["datasets"] is not None:
        raise ValueError("datasets is null for development")
    if (
        raw["entry"] == "speed"
        and not check_only
        and (not isinstance(raw["datasets"], str) or not Path(raw["datasets"]).is_absolute())
    ):
        raise ValueError("absolute datasets path required for speed")


def hashed_inputs(value):
    """Find all nested content-addressed inputs, including pooling batches."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for nested in value.values():
            yield from hashed_inputs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from hashed_inputs(nested)


def verify_inputs(job):
    for item in hashed_inputs(job):
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("origin input hash differs")


def sources(root):
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*.py"))
        if "baseline" not in p.parts
    }


def normal_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def lock_packages(path):
    """Read hashed uv requirements, rejecting unpinned requirements."""
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--hash=")):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ;\\]+)(?:\s*\\)?", line)
        if match is None:
            raise ValueError("profile lock requires unconditional exact package pins")
        name = normal_name(match[1])
        if name in result:
            raise ValueError("duplicate profile package pin")
        result[name] = match[2]
    if result.get("memray") != "1.20.0":
        raise ValueError("profile lock must pin memray 1.20.0")
    return result


def identity(record, entry):
    if entry == "development":
        value = record["identity"]
        execution, package, deps = value, value["fse"], value["dependencies"]
    else:
        execution, package, deps = record["execution"], record["identity"], record["versions"]
    dependencies = {}
    for name, version in deps.items():
        key = normal_name(name)
        if key in dependencies:
            raise ValueError("duplicate normalized distribution name")
        dependencies[key] = version
    return {
        **{k: package[k] for k in ("package", "version", "sha256", "path")},
        "python": execution["python"],
        "executable": execution["executable"],
        "dependencies": dependencies,
    }


def same_identity(origin, current, extra):
    for key in ("python", "package", "version", "sha256"):
        if origin[key] != current[key]:
            raise ValueError(f"executing {key} differs from origin")
    expected = dict(origin["dependencies"])
    for name, version in extra.items():
        if name in expected and expected[name] != version:
            raise ValueError("profile lock would change an origin package")
        expected[name] = version
    if expected != current["dependencies"]:
        raise ValueError("profile dependencies differ from origin plus lock")


def rewrite(spec, original, deadline, origin_hash):
    job = {
        k: v
        for k, v in original.items()
        if k not in ("bench_owner", "session_last_operation_at_unix")
    }
    job.update(
        work=spec["work"],
        tag="profile-" + spec["name"],
        receipt=str(Path(spec["output"]) / "receipt.json"),
        wall_deadline=deadline - 60,
        functional_only=True,
        diagnostic_profile=True,
        origin={"job": original, "job_sha256": origin_hash, "nominal_arm": spec["nominal_arm"]},
    )
    if spec["entry"] == "development":
        job["action"] = spec["action"]
        if spec["iterations"] is not None:
            job["iterations"] = spec["iterations"]
        if spec["action"] == "profile-head":
            if not original.get("profile", "").startswith("linear-heads"):
                raise ValueError("profile-head requires a linear-heads profile")
            if (
                original["profile"] in ("linear-heads-multi@1", "linear-heads-scores@1")
                and "heads_mode" not in original
            ):
                raise ValueError("profile-head requires the frozen heads_mode")
    return job

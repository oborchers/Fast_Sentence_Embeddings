"""Run serial kernel conformance under a pinned Intel SDE, including followed child interpreters.

The workflow supplies FSE_SDE_EXE. The pinned SDE launcher follows child execs by default (see -no-follow-child in its help). Successful child imports check
the emulated feature profile itself, under both Sapphire Rapids and Haswell.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = (
    "tests/test_native_isa.py",
    "tests/test_native.py",
    "tests/test_native_head.py",
    "tests/test_native_search.py",
    "tests/test_native_prefilter.py",
    "tests/test_conformance.py",
    "tests/test_conformance_tiles.py",
    "tests/test_conformance_tiles_meta.py",
    "tests/test_conformance_tiles_dispatch.py",
    "tests/test_conformance_prefetch.py",
    "tests/test_conformance_steps.py",
    "tests/test_conformance_scores.py",
    "tests/test_conformance_store.py",
    "tests/test_conformance_search.py",
    "tests/test_conformance_prefilter.py",
    "tests/test_conformance_head.py",
    "tests/test_conformance_heads.py",
    "tests/test_conformance_multilabel.py",
    "tests/test_conformance_linear2.py",
)


REQUIRED = {
    "test_required_isa_is_executed",
    "test_cpu_registers_are_read_once_for_all_dispatch_paths",
}
CHILD_PREFIXES = (
    "test_environment_choice_and_explicit_precedence[",
    "test_invalid_environment_fails_at_extension_import[",
)


def emulator_prefix(profile: str) -> list[str]:
    prefix = [os.environ["FSE_SDE_EXE"], f"-{profile}"]
    if sys.platform == "win32":
        # Windows CRT queries OS feature flags that SDE does not intercept.
        # Keep emulation process-wide, but check the exact fse DLL, not that CRT.
        spec = importlib.util.find_spec("fse._native")
        if spec is None or not spec.origin:
            raise RuntimeError("cannot identify the native image for SDE chip checking")
        image = Path(spec.origin).name
        if not image.endswith(".pyd"):
            raise RuntimeError(f"expected the installed native image, got {image}")
        prefix.extend(["-chip-check-image", image])
        print(json.dumps({"sde_chip_check_image": image}), flush=True)
    return [*prefix, "--"]


def require_named_tests(cases, names):
    for name in names:
        matches = [case for case in cases if case.get("name") == name]
        if len(matches) != 1 or matches[0].find("skipped") is not None:
            raise RuntimeError(f"required SDE test did not execute: {name}")


def require_cases(cases, profile):
    native_cases = [case for case in cases if case.get("classname", "").endswith("test_native_isa")]
    required = REQUIRED | (
        {"test_avx512_selection_reaches_every_vector_dispatch_site"} if profile == "spr" else set()
    )
    require_named_tests(native_cases, required)
    children = [case for case in native_cases if case.get("name", "").startswith(CHILD_PREFIXES)]
    if len(children) != 12 or any(case.find("skipped") is not None for case in children):
        raise RuntimeError("SDE receipt lacks all twelve child-import tests")
    return {
        "profile": profile,
        "tests": len(cases),
        "required": sorted(required),
        "child_imports": len(children),
    }


def execution_cases(report):
    """A successful launcher alone is not evidence: -echo also returns zero."""
    if not report.is_file():
        raise RuntimeError("SDE returned without a pytest execution receipt")
    root = ET.parse(report).getroot()
    if root.findall(".//failure") or root.findall(".//error"):
        raise RuntimeError("SDE receipt contains failed tests")
    return root.findall(".//testcase")


def require_execution(report: Path, profile: str) -> dict:
    return require_cases(execution_cases(report), profile)


def require_execution(report: Path, profile: str) -> dict:
    return require_cases(execution_cases(report), profile)


def junit_identity(nodeid):
    module, *parts = nodeid.split("::")
    return (".".join([module.removesuffix(".py").replace("/", "."), *parts[:-1]]), parts[-1])


def validate_partition(manifest):
    nodeids, selected, shard = manifest["all"], manifest["selected"], manifest["shard"]
    if (
        shard not in (0, 1)
        or not nodeids
        or nodeids != sorted(set(nodeids))
        or selected != nodeids[shard::2]
        or {nodeid.split("::", 1)[0] for nodeid in nodeids} != set(TESTS)
    ):
        raise RuntimeError("SDE shard does not partition the complete test manifest")


def shard_receipt(report, manifest, profile, compiler, layout, source_ref):
    validate_partition(manifest)
    cases = execution_cases(report)
    observed = collections.Counter((case.get("classname"), case.get("name")) for case in cases)
    expected = collections.Counter(junit_identity(nodeid) for nodeid in manifest["selected"])
    if observed != expected:
        raise RuntimeError("SDE execution receipt does not match its selected node IDs")
    return {
        **manifest,
        "profile": profile,
        "compiler": compiler,
        "layout": layout,
        "source_ref": source_ref,
        "cases": [
            {
                "classname": case.get("classname"),
                "name": case.get("name"),
                "skipped": case.find("skipped") is not None,
            }
            for case in cases
        ],
    }


def merge_receipts(directory, source_ref):
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]
    expected = {
        (compiler, layout, profile, shard)
        for compiler in ("gcc", "clang", "msvc")
        for layout in (0, 1)
        for profile in ("spr", "hsw")
        for shard in (0, 1)
    }
    indexed = {(row["compiler"], row["layout"], row["profile"], row["shard"]): row for row in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise RuntimeError("SDE requires exactly all twenty-four shard receipts")
    summaries = []
    for compiler, layout, profile, shard in sorted(expected):
        if shard:
            continue
        pair = [indexed[compiler, layout, profile, index] for index in (0, 1)]
        summaries.append(merge_pair(pair, source_ref))
    return summaries


def merge_pair(pair, source_ref):
    identities = {(row["compiler"], row["layout"], row["profile"]) for row in pair}
    if len(pair) != 2 or {row["shard"] for row in pair} != {0, 1} or len(identities) != 1:
        raise RuntimeError("SDE requires two distinct shards of the same compiler/layout/profile")
    cases = []
    for row in pair:
        validate_partition(row)
        if row["source_ref"] != source_ref or row["all"] != pair[0]["all"]:
            raise RuntimeError("SDE shards disagree about source or full collection")
        observed = collections.Counter((case["classname"], case["name"]) for case in row["cases"])
        if observed != collections.Counter(junit_identity(nodeid) for nodeid in row["selected"]):
            raise RuntimeError("SDE shard receipt has missing or duplicate execution")
        for case in row["cases"]:
            element = ET.Element("testcase", classname=case["classname"], name=case["name"])
            if case["skipped"]:
                ET.SubElement(element, "skipped")
            cases.append(element)
    return {
        "compiler": pair[0]["compiler"],
        "layout": pair[0]["layout"],
        **require_cases(cases, pair[0]["profile"]),
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-smoke", action="store_true")
    parser.add_argument("--profile", choices=("spr", "hsw"))
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--compiler", choices=("gcc", "clang", "msvc"))
    parser.add_argument("--layout", type=int, choices=(0, 1))
    parser.add_argument("--source-ref")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--merge-receipts", type=Path)
    args = parser.parse_args()
    if args.merge_receipts:
        if not args.source_ref:
            parser.error("merging receipts requires --source-ref")
    if args.shard is not None and any(
        value is None
        for value in (args.profile, args.compiler, args.layout, args.source_ref, args.receipts)
    ):
        parser.error("a shard requires profile, compiler, layout, source-ref and receipts")
    return args


def main() -> None:
    args = arguments()
    if args.merge_receipts:
        print(
            json.dumps(
                {"sde_complete_manifest": merge_receipts(args.merge_receipts, args.source_ref)}
            )
        )
        return
    if args.wheel_smoke:
        subprocess.run(
            [
                *emulator_prefix("hsw"),
                sys.executable,
                "-I",
                str(Path(__file__).with_name("wheel_smoke.py")),
            ],
            check=True,
        )
        return
    with tempfile.TemporaryDirectory(prefix="fse-sde-proof-") as directory:
        for profile in (args.profile,) if args.profile else ("spr", "hsw"):
            run_profile(args, profile, Path(directory))


def run_profile(args, profile, directory):
    env = os.environ.copy()
    env["FSE_SDE_PROFILE"] = profile
    env["FSE_REQUIRE_ISA"] = "avx512" if profile == "spr" else "avx2"
    report = directory / f"{profile}.xml"
    manifest = directory / "collection.json"
    options = shard_options(args, manifest, env)
    # SDE follows children by default. Its -echo prints commands WITHOUT
    # executing them, so it must never appear in this execution prefix.
    subprocess.run(
        [
            *emulator_prefix(profile),
            sys.executable,
            "-m",
            "pytest",
            "-x",
            *options,
            *TESTS,
            "-m",
            "not network",
            f"--junitxml={report}",
        ],
        env=env,
        check=True,
    )
    record_execution(args, profile, report, manifest)


def shard_options(args, manifest, env):
    if args.shard is None:
        return []
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    return [
        "-p",
        "tools.sde_shard",
        f"--fse-sde-shard={args.shard}",
        f"--fse-sde-manifest={manifest}",
    ]


def record_execution(args, profile, report, manifest):
    if args.shard is None:
        receipt = require_execution(report, profile)
    else:
        receipt = shard_receipt(
            report,
            json.loads(manifest.read_text(encoding="utf-8")),
            profile,
            args.compiler,
            args.layout,
            args.source_ref,
        )
        args.receipts.mkdir(parents=True, exist_ok=True)
        name = f"{args.compiler}-{args.layout}-{profile}-{args.shard}.json"
        (args.receipts / name).write_text(json.dumps(receipt), encoding="utf-8")
        receipt = {
            key: value for key, value in receipt.items() if key not in ("all", "selected", "cases")
        }
    print(json.dumps({"sde_execution_receipt": receipt}), flush=True)


if __name__ == "__main__":
    main()

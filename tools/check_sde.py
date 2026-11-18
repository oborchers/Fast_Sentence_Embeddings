"""Run serial kernel conformance under a pinned Intel SDE, including followed child interpreters.

The workflow supplies FSE_SDE_EXE. The pinned SDE launcher follows child execs by default (see -no-follow-child in its help). Successful child imports check
the emulated feature profile itself, under both Sapphire Rapids and Haswell.
"""

from __future__ import annotations

import argparse
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


def require_execution(report: Path, profile: str) -> dict:
    """A successful launcher alone is not evidence: -echo also returns zero."""
    if not report.is_file():
        raise RuntimeError("SDE returned without a pytest execution receipt")
    root = ET.parse(report).getroot()
    if root.findall(".//failure") or root.findall(".//error"):
        raise RuntimeError("SDE receipt contains failed tests")
    cases = root.findall(".//testcase")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-smoke", action="store_true")
    args = parser.parse_args()
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
        for profile in ("spr", "hsw"):
            prefix = emulator_prefix(profile)
            env = os.environ.copy()
            env["FSE_SDE_PROFILE"] = profile
            env["FSE_REQUIRE_ISA"] = "avx512" if profile == "spr" else "avx2"
            report = Path(directory) / f"{profile}.xml"
            # SDE follows children by default. Its -echo prints commands WITHOUT
            # executing them, so it must never appear in this execution prefix.
            subprocess.run(
                [
                    *prefix,
                    sys.executable,
                    "-m",
                    "pytest",
                    "-x",
                    *TESTS,
                    "-m",
                    "not network",
                    f"--junitxml={report}",
                ],
                env=env,
                check=True,
            )
            print(
                json.dumps({"sde_execution_receipt": require_execution(report, profile)}),
                flush=True,
            )


if __name__ == "__main__":
    main()

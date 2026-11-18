"""Run serial kernel conformance under a pinned Intel SDE, including followed child interpreters.

The workflow supplies FSE_SDE_EXE. The pinned SDE launcher follows child execs by default (see -no-follow-child in its help). Successful child imports check
the emulated feature profile itself, under both Sapphire Rapids and Haswell.
"""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import sys

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
    for name in required:
        matches = [case for case in native_cases if case.get("name") == name]
        if len(matches) != 1 or matches[0].find("skipped") is not None:
            raise RuntimeError(f"required SDE test did not execute: {name}")
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
    with tempfile.TemporaryDirectory(prefix="fse-sde-proof-") as directory:
        for profile in ("spr", "hsw"):
            prefix = [os.environ["FSE_SDE_EXE"], f"-{profile}", "--"]
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

"""Run serial kernel conformance under a pinned Intel SDE, including child interpreters.

The workflow supplies FSE_SDE_EXE. Every Python exec in the environment-choice tests
uses the same explicit prefix and checks the emulated feature profile itself.
"""

from __future__ import annotations

import json
import os
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


def main() -> None:
    for profile in ("spr", "hsw"):
        prefix = [os.environ["FSE_SDE_EXE"], f"-{profile}", "--"]
        env = os.environ.copy()
        env["FSE_SDE_PROFILE"] = profile
        env["FSE_TEST_PYTHON_PREFIX"] = json.dumps(prefix)
        env["FSE_REQUIRE_ISA"] = "avx512" if profile == "spr" else "avx2"
        # Serial pytest prevents unwrapped xdist execs. Environment-choice children
        # explicitly enter SDE, so their feature checks cannot accidentally see the host.
        subprocess.run(
            [*prefix, sys.executable, "-m", "pytest", "-x", *TESTS, "-m", "not network"],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()

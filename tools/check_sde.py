"""Run serial kernel conformance under a pinned Intel SDE, including followed child interpreters.

The workflow supplies FSE_SDE_EXE. The pinned SDE launcher follows child execs by default (see -no-follow-child in its help). Each child checks
the emulated feature profile itself, under both Sapphire Rapids and Haswell.
"""

from __future__ import annotations

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
        prefix = [os.environ["FSE_SDE_EXE"], f"-{profile}", "-echo", "--"]
        env = os.environ.copy()
        env["FSE_SDE_PROFILE"] = profile
        env["FSE_REQUIRE_ISA"] = "avx512" if profile == "spr" else "avx2"
        # SDE enables Pin child following by default; -echo records its actual Pin
        # command. Pin's own follow option is not an SDE tool knob. Do not nest Pin.
        # Child assertions under BOTH profiles catch accidental host execution.
        subprocess.run(
            [*prefix, sys.executable, "-m", "pytest", "-x", *TESTS, "-m", "not network"],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()

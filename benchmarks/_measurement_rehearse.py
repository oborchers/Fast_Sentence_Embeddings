"""Private built-in control and restart rehearsal, never publishable timing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from _harness import controls
from _harness.records import atomic, sha256
from _measurement_config import CONFIG
from _measurement_setup import bootstrap


def control_proof(root, entry):
    """The lifecycle controls of the harness (normal, SIGTERM, budget end, rejected machine,
    step 2 after a failed step 1, allocation SIGTERM and budget), then the bootstrap restart."""
    cases = controls.run_all(root)
    script = root / "bootstrap.sh"
    script.write_text(bootstrap(entry, "0" * 40))
    # The probe names its own steps: an allocation file of the caller must not leak into it.
    env = {
        **{k: v for k, v in os.environ.items() if k != "MEASURE_ALLOCATION"},
        "MEASURE_CONTROL_PROBE": "1",
        "MEASURE_ROOT": str((root / "restart").resolve()),
    }
    with (root / "bootstrap.log").open("w") as stream:
        first = subprocess.run(
            ["bash", str(script)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
        done = Path(env["MEASURE_ROOT"]) / (entry + "-results/DONE")
        before = sha256(done)
        second = subprocess.run(
            ["bash", str(script)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if first.returncode != 0 or second.returncode != 77 or sha256(done) != before:
        raise ValueError("bootstrap restart did not preserve the existing completion")
    result = {
        "functional_only": True,
        "cases": cases,
        "restart_refused": True,
        "DONE_unchanged": True,
        "bootstrap_sha256": sha256(CONFIG / "bootstrap.sh"),
        "scope": "real result writer, allocation and signals; bootstrap restart gate with setup leaf disabled",
    }
    atomic(root / "proof.json", result)
    return result

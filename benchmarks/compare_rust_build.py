"""Build the locked Rust comparison worker and bind its binary to build inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from compare_artifacts import file_digest
from quality_coordination import task_reservation

ROOT = Path(__file__).resolve().parent / "compare_rust"
BINARY = ROOT / "target/release/fse-model2vec-rs-compare"
MANIFEST = BINARY.with_name("build-identity.json")


def build_inputs() -> dict:
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", *sorted((ROOT / "src").rglob("*.rs"))]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def verified_binary() -> tuple[Path, dict]:
    record = json.loads(MANIFEST.read_text())
    if record["inputs"] != build_inputs() or record["binary_digest"] != file_digest(BINARY):
        raise ValueError("Rust worker differs from its recorded locked build; rebuild it")
    return BINARY, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordination", type=Path, required=True)
    args = parser.parse_args()
    with task_reservation(args.coordination):
        inputs = build_inputs()
        subprocess.run(["cargo", "build", "--release", "--locked"], cwd=ROOT, check=True)
        if inputs != build_inputs():
            raise ValueError("Rust sources changed during the build")
        record = {
            "inputs": inputs,
            "binary_digest": file_digest(BINARY),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
            "command": ["cargo", "build", "--release", "--locked"],
        }
        MANIFEST.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

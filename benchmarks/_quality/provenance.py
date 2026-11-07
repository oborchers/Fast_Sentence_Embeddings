"""Private provenance of a local quality run: interpreter, packages, runner and caches."""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path

from _measurement import package_identity
from _measurement_config import PINS

PACKAGES = (
    "mteb",
    "sentence-transformers",
    "model2vec",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "huggingface_hub",
    "numpy",
    "scikit-learn",
    "pytrec_eval-terrier",
)


def versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def runner():
    """The commit of the checkout that runs this harness and whether it has local changes."""
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def provenance(folders):
    from _quality.guard import size_bytes

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": versions(),
        "fse": package_identity("fse"),
        "runner": runner(),
        "pins": {"mteb": PINS["versions"]["mteb"], "fse_revision": PINS["fse_revision"]},
        "pins_note": "local run: fse comes from the environment above, not fse_revision",
        "caches": {name: str(path) for name, path in folders.items()},
        "cache_bytes_before": {name: size_bytes(path) for name, path in folders.items()},
    }

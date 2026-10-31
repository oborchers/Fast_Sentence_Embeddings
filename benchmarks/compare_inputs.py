"""Immutable comparison inputs and full execution-environment provenance."""

from __future__ import annotations

import gzip
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

from fse._container._hash import Hasher

DATASETS = {
    "wikitext": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"),
    "massive": ("mteb/amazon_massive_intent", "4672e20407010da34463acc759c162ca9734bca6"),
}


def texts(corpus: str) -> list[str]:
    repo, revision = DATASETS[corpus]
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    folder = hub / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revision
    if corpus == "wikitext":
        import pyarrow.parquet as pq

        rows = []
        for split in ("validation", "test"):
            files = sorted((folder / "wikitext-103-raw-v1").glob(f"{split}-*"))
            if not files:
                raise FileNotFoundError(f"missing pinned {corpus} {revision} {split}")
            for path in files:
                rows.extend(pq.read_table(path).column("text").to_pylist())
        return [row for row in rows if row.strip()]
    files = sorted((folder / "test").glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"missing pinned {corpus} {revision} test")
    rows = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line)["text"] for line in stream)
    return rows


def corpus_identity(name: str, rows: list[str]) -> dict:
    hasher = Hasher()
    for row in rows:
        data = row.encode("utf-8")
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    repo, revision = DATASETS[name]
    return {
        "dataset": repo,
        "dataset_revision": revision,
        "splits": ["validation", "test"] if name == "wikitext" else ["test"],
        "input_digest": hasher.checksum(),
    }


def cpu_name() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def identity() -> dict:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "machine": platform.platform(),
        "cpu": cpu_name(),
        "python": sys.version,
        "versions": {dist.metadata["Name"]: dist.version for dist in distributions()},
    }

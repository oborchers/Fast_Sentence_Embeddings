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

DATASETS = {
    "wikitext": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"),
    "massive": ("mteb/amazon_massive_intent", "4672e20407010da34463acc759c162ca9734bca6"),
}

MASSIVE_COUNTS = {
    f"{language}.json.gz": 2974
    for language in (
        "af am ar az bn cy da de el en es fa fi fr he hi hu hy id is it ja jv ka km kn ko "
        "lv ml mn ms my nb nl pl pt ro ru sl sq sv sw ta te th tl tr ur vi zh-CN zh-TW"
    ).split()
}


def texts(corpus: str) -> list[str]:
    repo, revision = DATASETS[corpus]
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    folder = hub / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revision
    return _wikitext(folder) if corpus == "wikitext" else _massive(folder)


def _wikitext(folder: Path) -> list[str]:
    import pyarrow.parquet as pq

    rows = []
    for split in ("validation", "test"):
        files = sorted((folder / "wikitext-103-raw-v1").glob(f"{split}-*"))
        if not files:
            raise FileNotFoundError(f"missing pinned WikiText split {split} in {folder}")
        for path in files:
            rows.extend(pq.read_table(path).column("text").to_pylist())
    return [row for row in rows if row.strip()]


def _massive(folder: Path) -> list[str]:
    files = sorted((folder / "test").glob("*.json.gz"))
    if {path.name for path in files} != MASSIVE_COUNTS.keys():
        raise FileNotFoundError(f"incomplete pinned MASSIVE test file set in {folder}")
    rows = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            batch = [json.loads(line)["text"] for line in stream]
        if len(batch) != MASSIVE_COUNTS[path.name]:
            raise ValueError(f"unexpected row count in pinned MASSIVE file {path.name}")
        rows.extend(batch)
    return rows


def corpus_identity(name: str, rows: list[str]) -> dict:
    from fse._container._hash import Hasher

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

"""Pod disk budgets and narrowly owned preparation-cache cleanup, without changing MTEB."""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from pathlib import Path

from classic_sources import SOURCES
from pod_artifacts import emit
from pod_models import HUB

GIB = 2**30


def tree_bytes(folder):
    total, files = 0, 0
    if not folder.exists():
        return {"regular_file_bytes": 0, "files": 0}
    for root, _, names in os.walk(folder, followlinks=False):
        for name in names:
            info = (Path(root) / name).lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                files += 1
    return {"regular_file_bytes": total, "files": files}


def snapshot(cache, models, output, stage):
    usage = shutil.disk_usage(cache)
    roots = {
        "finished_models": models,
        "hub": cache / "huggingface/hub",
        "datasets_arrow": cache / "huggingface/datasets",
        "xet": cache / "huggingface/xet",
        "uv": cache / "uv",
        "temporary": cache / "tmp",
    }
    record = {
        "stage": stage,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "filesystem": {"total": usage.total, "used": usage.used, "free": usage.free},
        "categories": {
            name: {"path": str(path), **tree_bytes(path)} for name, path in roots.items()
        },
        "scope": "regular-file apparent bytes, symlinks excluded; filesystem usage includes all allocated data",
    }
    emit("disk-" + stage, record)
    try:
        (output / ("disk-" + stage + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    except OSError as error:
        # A full disk must not hide the loader's original failure behind diagnostics.
        emit("disk-diagnostic-write-error-" + stage, {"errno": error.errno, "stage": stage})
    return record


def require_free(cache, minimum_gib, stage):
    usage = shutil.disk_usage(cache)
    if usage.free < minimum_gib * GIB:
        raise RuntimeError(
            f"MIND disk preflight ({stage}): requires at least {minimum_gib} GiB free, "
            f"found {usage.free / GIB:.2f} GiB. Launch with a 200 GB container disk. "
            "No task, split, or row will be removed to fit the disk."
        )


def cleanup_preparation_sources(cache, models, output):
    from fse import Encoder

    finished = sorted(models.glob("*-float32.safetensors"))
    if len(finished) != 10:
        raise ValueError("cleanup requires all ten saved static float32 artifacts")
    for path in finished:
        encoder = Encoder.load(path, mmap=True)
        encoder.verify()
        del encoder
    hub = cache / "huggingface/hub"
    repos = [*HUB, *("fse/" + name for name in SOURCES)]
    targets = [hub / ("models--" + repo.replace("/", "--")) for repo in repos]
    targets.append(hub / "datasets--Salesforce--wikitext")
    receipt = []
    for target in targets:
        if not target.exists():
            continue
        if target.parent.resolve() != hub.resolve() or target.is_symlink():
            raise ValueError("preparation cleanup target escaped the owned pod cache")
        row = {
            "path": str(target),
            **tree_bytes(target),
            "reason": "finished artifacts verified; raw preparation source no longer needed",
        }
        emit("cleanup-" + target.name.replace(".", "_"), row)
        shutil.rmtree(target)
        if target.exists():
            raise RuntimeError("owned preparation cache cleanup failed")
        receipt.append(row)
    corpus = cache / "wikitext-103-b08601e0.txt"
    if corpus.exists():
        if corpus.is_symlink() or not corpus.is_file():
            raise ValueError("fit corpus cleanup target is not an owned regular file")
        row = {
            "path": str(corpus),
            "regular_file_bytes": corpus.stat().st_size,
            "reason": "fit corpus hash and fitted state retained in every recipe receipt",
        }
        emit("cleanup-fit-corpus", row)
        corpus.unlink()
        receipt.append(row)
    (output / "preparation-source-cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n")

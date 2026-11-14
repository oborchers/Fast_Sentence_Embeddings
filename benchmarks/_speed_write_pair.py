"""Sequential write arms, retained sample bits and streaming resource admission."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from _harness import resources
from _harness.records import append, sha256
from _speed_versions import pair_rounds


class SampleMismatchError(ValueError):
    """The two versions produced different persisted code or scale bits."""


def validate(cell):
    """Check original resource inputs before any arithmetic or conversion."""
    fields = ["texts", "workers", "dimensions"]
    if cell["reference"] == "fse-base" or "planned_input_bytes" in cell:
        fields.append("planned_input_bytes")
    for name in fields:
        if type(cell.get(name)) is not int or cell[name] <= 0:
            raise ValueError(f"write {name} must be a positive integer")
    pair_rounds(cell)


DEFAULT_DIRTY_RATIO = 0.20  # vm.dirty_ratio default: share of available memory that may be dirty
DISK_RESERVE = 5 * resources.GIB


def _sysctl(name):
    try:
        return int(Path("/proc/sys/vm", name).read_text())
    except (OSError, ValueError):
        return None


def _filesystem(work):
    """The filesystem type of the longest mount point that holds ``work``, or None."""
    best = (-1, None)
    try:
        resolved = str(Path(work).resolve())
        for line in Path("/proc/mounts").read_text().splitlines():
            _, mount, kind, *_ = line.split()
            mount = mount.replace("\\040", " ")
            if (resolved == mount or resolved.startswith(mount.rstrip("/") + "/")) and len(
                mount
            ) > best[0]:
                best = (len(mount), kind)
    except OSError:
        return None
    return best[1]


def dirty_ceiling(limit_bytes, store_bytes, work):
    """Bytes of one store that may be dirty at once, with the settings it was read from.

    The kernel throttles writers at ``vm.dirty_bytes`` or ``vm.dirty_ratio`` of the available
    memory, per cgroup under cgroup v2 writeback. tmpfs pages never become clean disk pages, so
    on tmpfs, or when the settings or the filesystem cannot be read, the whole store counts."""
    fs, dirty_bytes, ratio = (
        _filesystem(work) if work else None,
        _sysctl("dirty_bytes"),
        _sysctl("dirty_ratio"),
    )
    seen = dict(filesystem=fs, vm_dirty_bytes=dirty_bytes, vm_dirty_ratio=ratio)
    if work is None:
        return min(store_bytes, int(limit_bytes * DEFAULT_DIRTY_RATIO)), dict(
            seen, source="default"
        )
    if fs is None or fs in ("tmpfs", "ramfs") or (not dirty_bytes and ratio is None):
        return store_bytes, dict(seen, source="whole store: writeback not verifiable")
    ceiling = dirty_bytes if dirty_bytes else int(limit_bytes * ratio / 100)
    return min(store_bytes, ceiling), dict(seen, source="kernel settings")


def memory_plan(cell, model_bytes, limit, retained_source_bytes=0, work=None):
    """Hard (anonymous and dirty-bounded) peak; clean file pages are recorded, not counted.

    The input file and the shards written with durability none are file-backed. Clean pages
    are reclaimed by the kernel under memory.max (the process is slowed, not killed), so they
    are listed as ``reclaimable_bytes``. Dirty pages are not reclaimable until written back;
    the kernel throttles writers at ``vm.dirty_ratio`` of the available memory, so one store
    counts only up to that ceiling. The disk must hold the input and one store (stores never
    overlap) plus a reserve."""
    validate(cell)
    n, d, w = cell["texts"], cell["dimensions"], cell["workers"]
    input_bytes = cell.get("planned_input_bytes", n * 140)
    store_bytes = n * (d + 8)
    dirty, dirty_source = dirty_ceiling(limit[0], store_bytes, work)
    hard = {
        "controller_libraries_reserve": 8 * resources.GIB,
        "retained_corpus": retained_source_bytes,
        "preparation_positions_and_temporaries": n * 8 * 4,
        "dirty_store_pages_ceiling": dirty,
        "model_copies": model_bytes * (w + 1),
        "worker_buffers": w * 256 * 1024**2,
        "sample_buffers": min(n, 10000) * (d + 4) * 4,
    }
    plan = resources.assess(
        hard,
        limit=limit,
        work=work,
        disk=None if work is None else input_bytes + store_bytes + DISK_RESERVE,
        reclaimable_bytes={
            "input_page_cache": input_bytes,
            "store_pages_after_writeback": store_bytes,
        },
        hard_components=sorted(hard),
        dirty_ceiling_source=dirty_source,
    )
    return plan


def compare_samples(rows):
    import numpy as np

    samples = []
    for row in rows:
        path = Path(row["code_sample_file"])
        if sha256(path) != row["code_sample_sha256"]:
            raise SampleMismatchError("headline sample file changed")
        with np.load(path, allow_pickle=False) as saved:
            samples.append({key: saved[key] for key in ("indices", "codes", "scales")})
    for key in samples[0]:
        a, b = samples[0][key], samples[1][key]
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or not np.array_equal(a.view(np.uint8), b.view(np.uint8))
        ):
            raise SampleMismatchError(f"headline cross-arm {key} bits differ")


def write_round(context, cell, number, labels, path, sample, identity, launch, rows):
    from _speed_cells import remove_headline

    current, ledger = [], context["ledger"]
    for label in labels if number % 2 == 0 else labels[::-1]:
        row = launch(context, cell, label, number, path, sample, identity)
        rows.append(row)
        current.append(row)
        remove_headline(context["work"], Path(row["sample_file"]), row["files"])
        append(
            ledger.root / "cleanup.jsonl",
            dict(
                cell_id=cell["id"],
                label=label,
                round=number,
                completed_at=time.time(),
                disk_after=shutil.disk_usage(context["work"])._asdict(),
            ),
        )
    if len(labels) == 2:
        compare_samples(current)
        append(
            ledger.root / "checks.jsonl",
            dict(
                cell_id=cell["id"],
                round=number,
                sample_bitwise_equal=True,
                rows=len(identity["sample_rows"]),
                check="cross-arm codes and scales",
            ),
        )


def check_identities(rows, labels):
    for label in labels:
        identities = {
            json.dumps(row["identity"], sort_keys=True) for row in rows if row["label"] == label
        }
        if len(identities) > 1:
            raise ValueError("headline runtime identity changed between rounds")


def rounds(context, cell, path, sample, identity, launch, cleanup):
    rows, ledger = [], context["ledger"]
    paired = cell["reference"] == "fse-base"
    labels = ["fse", "fse-base"] if paired else ["fse"]
    try:
        for number in range(pair_rounds(cell)):
            if not ledger.admit(cell["estimated_seconds"] / cell["rounds"]):
                break
            write_round(context, cell, number, labels, path, sample, identity, launch, rows)
        cleanup(
            context,
            [path, path.with_suffix(".json"), sample, path.with_suffix(".sample-texts.json")],
        )
        check_identities(rows, labels)
    except SampleMismatchError as error:
        return rows, dict(status="invalid", reason=str(error))
    except Exception as error:
        return rows, dict(status="failed", reason=repr(error)) if paired else repr(error)
    return rows, None


def result(cell, rows, error, identity, summarize):
    if error:
        return dict(
            error,
            protocol="version-write@1",
            input_identity=identity,
            rounds_completed=len(rows) // 2,
            sample_checked=False,
        )
    if len(rows) != 2 * pair_rounds(cell):
        return dict(
            status="partial",
            protocol="version-write@1",
            input_identity=identity,
            rounds_completed=len(rows) // 2,
            sample_checked=bool(rows),
            reason="stop signal or budget before next pair",
        )
    return dict(summarize(cell, rows, identity), rounds_completed=pair_rounds(cell))

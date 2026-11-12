"""Search model groups: shared sizing, rotating rounds, admission and per-cell outcomes."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _harness.records import append, atomic, document
from _harness.search import order, sequence_digest, summarize
from _measurement import spread
from _measurement_pressure import report
from _speed_search import admission, prepare
from _speed_search_identity import identity, verify

ENTRY = Path(__file__).resolve().with_name("speed.py")


def launch(context, job, phase, number, passes):
    folder = Path(job["group_root"]) / f"{job['id']}-{phase}-{number}"
    folder.mkdir()
    job = dict(
        job,
        phase=phase,
        round=number,
        pool_passes=passes,
        output=str(folder),
        receipt=str(folder / "receipt.json"),
    )
    spec = context["ledger"].root / f"{job['id']}-{phase}-{number}.job.json"
    verify(job)
    atomic(spec, job)
    with spec.with_suffix(".log").open("x") as log:
        subprocess.run(
            [sys.executable, str(ENTRY), "--child-job", str(spec)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    row = document(Path(job["receipt"]))
    append(
        context["ledger"].root / "operations.jsonl",
        dict(cell_id=job["id"], phase=phase, round=number, model=job["model"], **row),
    )
    if row.get("diagnostic_profile") or row.get("valid_for_comparison") is False:
        raise ValueError("diagnostic search output cannot enter a summary")
    if row["identity"] != job["search_identity"] or row["identity_after"] != job["search_identity"]:
        raise ValueError("search receipt provenance differs from the sealed job")
    expected = sequence_digest(
        job["search_identity"]["queries"], job["pool_size"], passes, job["queries_per_call"]
    )
    if row["sequence_sha256"] != expected or row["pool_passes"] != passes:
        raise ValueError("measured query sequence changed")
    verify(job)
    check(context, job, row)
    return dict(row, round=number)


def check(context, job, row):
    import numpy as np
    from _speed_cells import check_output

    binary = job["mode"] == "binary"
    prefix = "binary" if binary else job["store_dtype"]
    actual = {
        key: np.load(p, allow_pickle=False) for key, p in row["measured_output_files"].items()
    }
    expected = {
        key: np.load(job["reference_files"][f"{prefix}-{key}"], allow_pickle=False)
        for key in ("ids", "scores")
    }
    for key, value in expected.items():
        check_output(value, actual[key], True)
    evidence = row["path_evidence"]
    if job["prefilter"] == "off" and evidence["entered"]:
        raise ValueError("prefilter-off search entered the prefilter")
    if (
        job["prefilter"] == "shipped"
        and evidence["prefilter_variant"] is not None
        and not evidence["entered"]
    ):
        raise ValueError("enabled prefilter was not entered")
    checked = dict(
        cell_id=job["id"],
        phase=job["phase"],
        round=job["round"],
        measured_final_pass_checked=True,
        path_evidence_is_separate=True,
    )
    if binary:
        candidates = np.load(job["reference_files"]["candidates"], allow_pickle=False)
        if not all(
            np.isin(ids, selected).all()
            for ids, selected in zip(actual["ids"], candidates, strict=True)
        ):
            raise ValueError("binary result lies outside independent candidate set")
        exact = np.load(job["reference_files"]["float32-ids"], allow_pickle=False)
        recalls = [
            len(set(a) & set(b)) / job["k"] for a, b in zip(actual["ids"], exact, strict=True)
        ]
        row["recall_at_k"] = {"mean": float(np.mean(recalls)), "min": min(recalls), "k": job["k"]}
        checked["recall_at_k"] = row["recall_at_k"]
    append(context["ledger"].root / "checks.jsonl", checked)


def jobs(context, cells, root):
    stores, queries, references = prepare(context, cells, root)
    result = {}
    for cell in cells:
        job = dict(
            cell,
            backend="fse",
            artifact=context["models"][cell["model"]]["artifact"],
            store=stores[cell["store_dtype"]],
            queries_file=queries,
            reference_files=references,
            group_root=str(root),
        )
        job["search_identity"] = identity(job)
        result[cell["id"]] = job
    return result


def calibrate(context, cells, specs, errors):
    admission(context, "calibration", 120 * len(cells))
    passes, seconds = [], {}
    for cell in cells:
        try:
            row = launch(context, specs[cell["id"]], "calibration", 0, 1)
            passes.append(row["calibrated_passes"])
            seconds[cell["id"]] = row["seconds"]
        except Exception as error:
            errors[cell["id"]] = repr(error)
    if not passes:
        raise ValueError("no search cell calibrated successfully")
    result = max(passes)
    atomic(
        Path(cells_root(specs)) / "calibration.json",
        {"common_pool_passes": result, "per_cell_passes": passes},
    )
    return result, seconds


def cells_root(specs):
    return next(iter(specs.values()))["group_root"]


def rounds(context, cells, specs, passes, seconds, rows, errors):
    active = [cell for cell in cells if cell["id"] not in errors]
    for number in range(cells[0]["rounds"]):
        # Warmup, complete shared sequence, separate path pass, verification and startup.
        estimate = sum(15 + (passes + 2) * seconds[c["id"]] for c in active)
        if not context["ledger"].admit(estimate):
            break
        run_round(context, order(active, number), specs, passes, number, rows, errors)
        active = [cell for cell in active if cell["id"] not in errors]
        if not active:
            break


def run_round(context, cells, specs, passes, number, rows, errors):
    for cell in cells:
        key = cell["id"]
        try:
            rows[key].append(launch(context, specs[key], "measurement", number, passes))
        except Exception as error:
            errors[key] = repr(error)


def stable_path(rows):
    return len({json.dumps(r["path_evidence"], sort_keys=True) for r in rows}) == 1


def relative(cell, rows, all_rows, errors, short):
    baseline = cell.get("relative_to")
    if not baseline:
        return {}
    reference = all_rows[baseline]
    complete = (
        baseline not in errors and len(reference) == cell["rounds"] and stable_path(reference)
    )
    long_enough = cell.get("functional_only") or all(r["seconds"] >= 2 for r in reference)
    if complete and long_enough and not short:
        if [r["sequence_sha256"] for r in reference] != [r["sequence_sha256"] for r in rows]:
            raise ValueError("relative search comparison has different query sequences")
        return {
            "relative_speedup": spread(
                [b["seconds"] / a["seconds"] for a, b in zip(rows, reference, strict=True)]
            )
        }
    return {"publishable": False, "relative_comparison": "unavailable"}


def summary(cell, rows, all_rows, errors, passes):
    if cell["id"] in errors:
        return dict(status="failed", reason=errors[cell["id"]])
    if len(rows) != cell["rounds"]:
        return dict(
            status="partial" if rows else "not_started",
            reason="search budget or stop signal",
            completed_rounds=len(rows),
        )
    if not stable_path(rows):
        return dict(status="failed", reason="search path changed between rounds")
    short = any(r["seconds"] < 2 for r in rows) and not cell.get("functional_only")
    result = dict(
        cpu_pressure=report(rows),
        protocol="search-time@1",
        status="invalid" if short else "valid",
        reason="search block below stability target" if short else None,
        publishable=not cell.get("functional_only") and not short,
        measured_final_pass_checked=True,
        identity=rows[0]["identity"],
        path_evidence=rows[0]["path_evidence"],
        prefilter_enabled_by_adapter=any(r["prefilter_enabled_by_adapter"] for r in rows),
        **summarize(rows, cell["pool_size"], passes),
    )
    if "recall_at_k" in rows[0]:
        result["recall_at_k"] = rows[0]["recall_at_k"]
    result.update(relative(cell, rows, all_rows, errors, short))
    return result


def group(context, cells):
    ledger = context["ledger"]
    started = time.monotonic()
    root = context["work"] / ("search-" + cells[0]["model"].replace("/", "--"))
    root.mkdir()
    rows, errors, passes = {c["id"]: [] for c in cells}, {}, None
    ledger.stage, ledger.current = "search", cells[0]["id"]
    try:
        specs = jobs(context, cells, root)
        passes, seconds = calibrate(context, cells, specs, errors)
        rounds(context, cells, specs, passes, seconds, rows, errors)
    except TimeoutError:
        pass
    except Exception as error:
        errors.update({c["id"]: repr(error) for c in cells})
    for cell in cells:
        ledger.result(cell, started, **summary(cell, rows[cell["id"]], rows, errors, passes))
    cleanup_stores(context, root, rows)


def cleanup_stores(context, root, rows):
    # Keep all raw files on failed or partial groups. Successful stores are owned here.
    outcomes = [r for r in context["ledger"].finished if r["id"] in rows]
    if not outcomes or any(r["status"] != "valid" for r in outcomes):
        return
    for path in root.iterdir():
        if path.name not in ("float32", "int8"):
            continue
        if path.is_symlink() or path.parent != root or any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError("search store cleanup ownership changed")
        shutil.rmtree(path)


def run(context, cells):
    for model in dict.fromkeys(c["model"] for c in cells):
        group(context, [c for c in cells if c["model"] == model])

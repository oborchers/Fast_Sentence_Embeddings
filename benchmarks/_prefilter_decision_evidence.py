"""Read-only semantic evidence checks for the sealed AVX2 allocation decision."""

from __future__ import annotations

import re
from pathlib import Path

from _development_config import validate_cell
from _development_decision import judge
from _development_path_evidence import validate as validate_path
from _development_statistics import paired
from _harness.decide import number
from _harness.profile import identity, same_identity
from _harness.records import document, read_rows, sha256


def production(value):
    if value.get("functional_only") or value.get("diagnostic_profile") or value.get("sizing_only"):
        raise ValueError("nonproduction timing evidence")


def build(row, entry):
    value = identity(row, entry)
    return {key: value[key] for key in ("python", "package", "version", "sha256")}


def search(root, contract):
    cells = contract["cells"]
    if [c["id"] for c in cells] != ["o1-d256-q8", "o2-d1024-q8"]:
        raise ValueError("both search workloads required")
    operations = read_rows(root, "operations.jsonl")
    if any(r["cell_id"] not in {c["id"] for c in cells} for r in operations):
        raise ValueError("unexpected search operation")
    builds, decisions = {}, []
    for cell in cells:
        validate_cell(cell, {})
        control = document(root / (cell["id"] + "-aa.json"))
        frozen = _frozen(cell, control)
        _workload(frozen)
        own = [r for r in operations if r["cell_id"] == cell["id"]]
        _sizing(own)
        _paths(root, own, builds, frozen, control)
        blocks = _blocks(frozen, own, control)
        decisions.append({"cell_id": cell["id"], **judge(frozen, *blocks)})
    return decisions, builds


def _frozen(cell, control):
    frozen = control["cell"]
    validate_cell(frozen, {})
    if (
        frozen["iterations"] < cell["iterations"]
        or not cell["target_seconds"] <= frozen["target_seconds"] <= cell["max_target_seconds"]
    ):
        raise ValueError("frozen iterations or target is outside calibration bounds")
    if any(frozen[k] != v for k, v in cell.items() if k not in ("iterations", "target_seconds")):
        raise ValueError("calibration changed the declared workload")
    if (
        frozen["profile"] != "int8-prefilter-search@1"
        or frozen["pairs"] != 8
        or frozen["target_aa_iqr"] > 0.02
    ):
        raise ValueError("eight-pair prefilter profile required")
    return frozen


def _workload(cell):
    required = {
        "stage": "store-search",
        "corpus": "wikitext-train",
        "workers": 2,
        "queries": 8,
        "k": 10,
        "output_dtype": "int8",
        "table_dtype": "float32",
        "filter_kind": "none",
        "base_doc_ids": "none",
        "candidate_doc_ids": "none",
        "resident_cycles": 1,
        "model": "minishlab/potion-base-8M"
        if cell["id"] == "o1-d256-q8"
        else "sentence-transformers/static-retrieval-mrl-en-v1",
    }
    if any(cell[key] != value for key, value in required.items()):
        raise ValueError("search workload differs from the AVX2 enabling study")


def _paths(root, rows, builds, cell, control):
    if not rows:
        raise ValueError("missing path observations")
    common = None
    for row in rows:
        job, candidate = _path_job(root, row, cell)
        witness = _inputs(row, job, control, candidate)
        common = witness if common is None else common
        if witness != common:
            raise ValueError("search inputs changed between arms or phases")
        validate_path(row, job, candidate=candidate)
        key = "candidate" if candidate else "base"
        current = build(row, "development")
        if builds.setdefault(key, current) != current:
            raise ValueError("search executing build changed")


def _inputs(row, job, control, candidate):
    context = control["context"]
    if (
        job["artifact"] != context["models"]["artifacts"][job["table_dtype"]]
        or job["corpus"] != context["corpus"]
    ):
        raise ValueError("search inputs differ from the calibrated model or corpus")
    if row["identity"] != context["identities"][int(candidate)]:
        raise ValueError("search build differs from its quality control")
    if (
        row["texts"] != job["texts"]
        or job["texts"] != control["texts"]
        or control["texts"] != 859346
        or row["requested_workers"] != job["workers"]
    ):
        raise ValueError("search corpus row count differs")
    pool, digest = row["query_pool"], row["query_digest"]
    if (
        pool["selected"] != 256
        or pool["disjoint_checked"] is not True
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("complete disjoint query pool identity required")
    return {
        "artifact": job["artifact"],
        "corpus": job["corpus"],
        "queries": job["query_corpora"],
        "query_pool": pool,
        "query_digest": digest,
        "texts": job["texts"],
        "workers": row["requested_workers"],
    }


def _sizing(rows):
    phases = {row["phase"] for row in rows}
    names = sorted(name for name in phases if name.startswith("size-"))
    expected = {f"size-0-{n}" for n in range(len(names))}
    if (
        not names
        or set(names) != expected
        or phases - expected - {"aa-tune-0", "aa-validate-0", "comparison", "confirmation"}
    ):
        raise ValueError("missing sizing attempts or unknown calibration phase")
    for phase in names:
        own = [row for row in rows if row["phase"] == phase]
        _sizing_pair(own)


def _sizing_pair(own):
    if len(own) != 2 or [(r["pair"], r["position"], r["arm"]) for r in own] != [
        (0, 0, "base"),
        (0, 1, "candidate"),
    ]:
        raise ValueError("incomplete sizing pair")
    if own[0]["output"] != own[1]["output"] or any(
        number(r["seconds"], "sizing duration") <= 0 for r in own
    ):
        raise ValueError("sizing pair output or duration differs")


def _path_job(root, row, cell):
    phase = row["phase"]
    control = phase.startswith(("aa-tune-", "aa-validate-"))
    if (
        not control
        and phase not in ("comparison", "confirmation")
        and not phase.startswith("size-")
    ):
        raise ValueError("unexpected search phase")
    tag = f"{row['cell_id']}-{phase}-{row['pair']}-{row['position']}"
    job = document(root / (tag + ".job.json"))
    receipt = document(root / (tag + ".json"))
    if any(row.get(k) != v for k, v in receipt.items()):
        raise ValueError("search operation differs from child receipt")
    if job["tag"] != tag or job["id"] != row["cell_id"] or job["action"] != "measure":
        raise ValueError("search operation address differs from job")
    _job_workload(row, job, cell, phase)
    return job, not control and row["arm"] == "candidate"


def _job_workload(row, job, cell, phase):
    expected = {**cell, "warmup_seconds": 0} if phase.startswith("size-") else cell
    if any(
        job[k] != v
        for k, v in expected.items()
        if k not in ("iterations", "target_seconds", "corpus")
    ):
        raise ValueError("search job differs from frozen cell")
    if (
        row["operation_repetitions"] != job["iterations"]
        or row["input"]["sha256"] != job["corpus"]["sha256"]
    ):
        raise ValueError("search receipt differs from frozen job workload")


def _block(cell, rows, phase):
    own = [r for r in rows if r["phase"] == phase]
    if not own:
        raise ValueError("required search block is missing")
    for row in own:
        production(row)
        if row["valid_for_comparison"] is not True or number(
            row["seconds"], "group duration"
        ) < max(2, cell["target_seconds"]):
            raise ValueError("invalid or short search group")
        if row["operation_repetitions"] != cell["iterations"] or row["output"] != own[0]["output"]:
            raise ValueError("search workload or bits differ")
    return {**paired(cell, own), "output": own[0]["output"]}


def _blocks(cell, rows, control):
    production(control)
    attempts = control["attempts"]
    if not isinstance(attempts, list) or len(attempts) != 1 or cell["max_calibration_steps"] != 1:
        raise ValueError("AVX2 study requires exactly one calibration attempt")
    index = len(attempts) - 1
    tune = _block(cell, rows, f"aa-tune-{index}")
    aa = _block(cell, rows, f"aa-validate-{index}")
    if tune["factor"]["iqr"] >= cell["target_aa_iqr"] or tune["output"] != aa["output"]:
        raise ValueError("A/A tuning is noisy or changed output")
    return aa, _block(cell, rows, "comparison"), _block(cell, rows, "confirmation")


def float32(root, builds):
    contract = document(root / "environment.json")["contract"]
    production(contract)
    rows = read_rows(root)
    if len(rows) != 1 or len(contract["cells"]) != 1:
        raise ValueError("exactly one float32 control cell required")
    cell = rows[0]
    production(cell)
    if (cell["status"], cell["protocol"], cell["dtype"], cell["reference"]) != (
        "valid",
        "version-compare@1",
        "float32",
        "fse-base",
    ) or cell["sample_checked"] is not True:
        raise ValueError("valid bit-exact float32 cell required")
    rounds = cell.get("rounds", 3)
    if type(rounds) is not int or rounds < 3:
        raise ValueError("three float32 rounds required")
    checks = read_rows(root, "checks.jsonl")
    if len(checks) != rounds or [r["round"] for r in checks] != list(range(rounds)):
        raise ValueError("missing float32 bit checks")
    if any(
        r["cell_id"] != cell["cell_id"] or r["max_abs_difference"] != 0 or r["rows"] <= 0
        for r in checks
    ):
        raise ValueError("float32 outputs differ")
    declared = contract["cells"][0]
    if cell["cell_id"] != declared["id"] or any(
        cell[k] != declared[k] for k in ("kind", "dtype", "reference")
    ):
        raise ValueError("float32 cell differs from its contract")
    _float32_operations(root, cell, rounds, builds)


def _float32_operations(root, cell, rounds, builds):
    operations = read_rows(root, "operations.jsonl")
    for label, arm in (("fse-base", "base"), ("fse", "candidate")):
        own = [r for r in operations if r["label"] == label]
        if [r["round"] for r in own] != list(range(rounds)):
            raise ValueError("missing float32 rounds")
        for row in own:
            production(row)
            if (
                row["cell_id"] != cell["cell_id"]
                or number(row["seconds"], "float32 duration") < 2
                or build(row, "speed") != builds[arm]
            ):
                raise ValueError("float32 build or duration differs")
    if len(operations) != rounds * 2:
        raise ValueError("unexpected float32 observations")


def profiles(root, context, origin):
    for step, tool in (("profile-cpu", "py-spy"), ("profile-memory", "memray")):
        folder = root / (step + "-results")
        environment = document(folder / "environment.json")
        spec = environment["profile"]
        if (
            spec["entry"] != "development"
            or spec["nominal_arm"] != "candidate"
            or spec["tool"] != tool
        ):
            raise ValueError("wrong profile arm or tool")
        if spec["source_results"] != str(Path(context["root"]) / "prefilter-results"):
            raise ValueError("profile originates from a different search run")
        if (
            environment["measurement_source_sha256"]
            != document(origin / "environment.json")["measurement_source_sha256"]
        ):
            raise ValueError("profile runner differs from search runner")
        rows = read_rows(folder)
        if len(rows) != 1:
            raise ValueError("exactly one complete profile required")
        row = rows[0]
        if (
            row["status"] != "valid"
            or row["phase"] != "execution"
            or row.get("causes")
            or row.get("open_evidence")
        ):
            raise ValueError("profile is incomplete")
        _coverage(folder, spec, row)
        _profile_origin(folder, origin, spec, row)


def _coverage(folder, spec, row):
    for key in ("coverage", "access_coverage"):
        if row[key]["valid"] is not True or row[key]["requested"] != spec["requested"]:
            raise ValueError("profile child coverage is incomplete")
    if spec["requested"] != {"processes": "child", "threads": "all"}:
        raise ValueError("complete child and thread coverage required")
    if (
        document(folder / "coverage.json") != row["coverage"]
        or document(folder / "check.json") != row["access_coverage"]
    ):
        raise ValueError("profile coverage differs from verified records")
    data = folder / ("cpu.speedscope.json" if spec["tool"] == "py-spy" else "mem.bin")
    if not data.is_file() or not data.stat().st_size:
        raise ValueError("profile data is missing")


def _profile_origin(folder, origin, spec, row):
    tag = spec["job"]
    job = document(origin / (tag + ".job.json"))
    matches = [
        r
        for r in read_rows(origin, "operations.jsonl")
        if f"{r['cell_id']}-{r['phase']}-{r['pair']}-{r['position']}" == tag
    ]
    if len(matches) != 1:
        raise ValueError("profile child is missing or duplicated")
    operation = matches[0]
    if (operation["cell_id"], operation["phase"], operation["arm"]) != (
        "o2-d1024-q8",
        "comparison",
        "candidate",
    ):
        raise ValueError("profile does not address the deciding candidate child")
    provenance = row["provenance"]
    if provenance["nominal_arm"] != "candidate" or provenance["job_sha256"] != sha256(
        origin / (tag + ".job.json")
    ):
        raise ValueError("profile job hash differs")
    if provenance["origin"] != identity(operation, "development"):
        raise ValueError("profile build differs from candidate")
    # memray adds only the hashed pinned profiler packages, verified by the runner.
    extra = document(folder / "profiler.json").get("packages", {})
    same_identity(provenance["origin"], provenance["before"], extra)
    same_identity(provenance["origin"], provenance["after"], extra)
    rewritten = document(folder / "job.json")
    if (
        rewritten["origin"]["job"] != job
        or rewritten["origin"]["job_sha256"] != provenance["job_sha256"]
    ):
        raise ValueError("profile executed a different origin job")
    receipt = document(folder / "receipt.json")
    if (
        receipt.get("diagnostic_profile") is not True
        or receipt.get("valid_for_comparison") is not False
    ):
        raise ValueError("diagnostic profile receipt required")
    if (
        receipt["output"] != operation["output"]
        or identity(receipt, "development") != provenance["after"]
    ):
        raise ValueError("profile output or executed identity differs")
    validate_path(receipt, rewritten, candidate=True)
    threads = row["coverage"]["search_threads"]
    if threads["requested_workers"] != job["workers"] or len(threads["observed"]) < job["workers"]:
        raise ValueError("profile does not cover the native search workers")

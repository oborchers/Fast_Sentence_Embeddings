"""Source identity of a resumed run (DR-248): the controller and the measuring child.

Class descriptive, candidate build only, one worker. The timed call is
``fse._encoder._resume.identify(source, None)``: the fingerprint, the row count and the
line marks of an opened source. Opening the source (``FileSource`` or ``StringSource``),
reading the file into the page cache before the call, the probe and the checks are outside
the timer; encoding and writing are not part of the stage."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from _development_budget import check, check_args, stopped
from _development_identity import CHUNK, STAGE, admission, cut
from _development_locks import coordinated
from _development_pairs import wait_for_arm
from _development_process import checked, launch
from _development_statistics import summary
from _development_system import LABEL, observations
from _measurement import append, atomic


def run(args, cell, prepared, ledger):
    """One cell of the stage, as a descriptive result (no comparison, no decision gate)."""
    if args.local_phase != "descriptive":
        raise ValueError(f"{STAGE} is class descriptive: use --local-phase descriptive")
    corpus = prepared["corpora"][cell["corpus"]]
    plan = admission(cell, corpus, args.work, args.memory_limit_bytes)
    append(args.results / "memory.jsonl", {"cell_id": cell["id"], "step": "descriptive", **plan})
    if not plan["admitted"]:
        raise MemoryError("source-identity workload exceeds RAM or disk admission")

    def notify(row):
        if row["event"] == "waiting_for_lock":
            ledger.stage = "waiting_for_lock_or_load"
            ledger.progress()
        append(args.results / "coordination.jsonl", row)

    def quiet():
        return observations(args.work)["load"][0] < 5

    tag = cell["id"] + "-descriptive"
    with coordinated(
        "timing", args.worker_name, cell["id"], notify, lambda: stopped(ledger), quiet
    ) as owner:
        wait_for_arm(args, ledger, cell["id"])
        ledger.current, ledger.stage = cell["id"], "descriptive"
        job = {
            **cell,
            "action": "identity",
            "corpus": corpus,
            "tag": tag,
            "bench_owner": owner,
            "repetitions": args.descriptive_repetitions,
            "cpu_affinity": args.cpu_affinity,
            "functional_only": args.diagnostic_rehearsal,
            "admission": plan,
        }
        record = launch(args, job, args.candidate_python, tag, timed=True)
    append(args.results / "operations.jsonl", {"cell_id": cell["id"], **record})
    check_args(args)
    costs = summary(record["seconds"])
    costs.pop("max_deviation")  # a distance from one means nothing for absolute seconds
    result = dict(
        status="valid",
        decision="functional_only" if args.diagnostic_rehearsal else "descriptive_only",
        arm="candidate",
        seconds=costs,
        seconds_per_gib=costs["median"] * 1024**3 / record["identity_bytes"],
        repetitions=len(record["seconds"]),
        warmup_calls=len(record["warmup_seconds"]),
        aa="not_run",
        confirmation="not_run",
        valid_for_comparison=False,
        **{
            name: record[name]
            for name in ("probe", "output", "input", "warm_seconds", "peak_rss_bytes", "identity")
        },
    )
    atomic(args.results / (cell["id"] + "-descriptive.json"), result)
    return result


@contextmanager
def _file_source(corpus):
    from fse._text_file import FileSource

    started = time.perf_counter()
    with Path(corpus["path"]).open("rb", buffering=0) as stream:  # the file is in the page cache
        block = bytearray(CHUNK)
        while stream.readinto(block):
            pass
    warm = time.perf_counter() - started
    with FileSource(corpus["path"]) as source:
        yield source, warm


def _lines(corpus):
    """The first ``rows`` lines of the file, as the list of the strings case."""
    rows, texts = corpus["rows"], []
    with Path(corpus["path"]).open("rb") as stream:
        for line in stream:
            texts.append(line.removesuffix(b"\n").decode("utf-8"))
            if len(texts) == rows:
                return texts
    raise ValueError("the file has fewer lines than the strings case")


def _proof(identity):
    return dict(
        fingerprint=identity.fingerprint,
        form=identity.form,
        rows=identity.rows,
        nbytes=identity.nbytes,
        marks=len(identity.marks),
    )


def _probe(job, corpus):
    """Outside every timer: the fingerprint of a head of the file, from ``identify``, equals
    an independent computation from its bytes."""
    import numpy as np

    from fse._container import checksum, fingerprint
    from fse._encoder._resume import identify
    from fse._text_file import FileSource

    with Path(corpus["path"]).open("rb") as stream:
        raw = stream.read(job["probe_bytes"])
    raw = raw[: cut(raw, len(raw))]
    path = Path(job["work"]) / (job["tag"] + "-probe.txt")
    path.write_bytes(raw)
    try:
        with FileSource(path) as source:
            got = identify(source, None)
    finally:
        path.unlink()
    rows = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
    entries = {"text": ("U8", (len(raw),), checksum(np.frombuffer(raw, np.uint8)))}
    metadata = {
        "fse.kind": "text_source",
        "fse.source_form": "lines",
        "fse.rows": str(rows),
        "fse.doc_ids": "none",
    }
    if got.fingerprint != fingerprint(entries, metadata) or got.rows != rows:
        raise ValueError("the source identity differs from its independent computation")
    return dict(bytes=len(raw), rows=rows, fingerprint=got.fingerprint)


def measure(job):
    """The child: probe, one warm-up call and ``repetitions`` timed calls, in one process."""
    from _development_measure import _policy, _quiet

    from fse._encoder._resume import identify

    policy = _policy(job)
    waiting = [0.0]
    corpus = job["corpus"]
    checked(corpus["path"], corpus["sha256"])
    check(job.get("wall_deadline"))
    probe = _probe(job, corpus)
    texts = _lines(corpus) if corpus["form"] == "strings" else None
    calls = Path(job["receipt"]).with_suffix(".calls.jsonl")
    seconds, warmups, warms, expected = [], [], [], None
    before = _quiet(job, waiting)
    for number in range(job["warmup_calls"] + job["repetitions"]):
        check(job.get("wall_deadline"))
        phase = "warmup" if number < job["warmup_calls"] else "measured"
        _quiet(job, waiting)
        context = _file_source(corpus) if texts is None else _strings_source(texts)
        with context as (source, warm):
            started = time.perf_counter()
            identity = identify(source, None)
            elapsed = time.perf_counter() - started
        proof = _proof(identity)
        if expected is not None and proof != expected:
            raise ValueError("the source identity changed between calls")
        expected = proof
        (warmups if phase == "warmup" else seconds).append(elapsed)
        warms.append(warm)
        append(calls, {**LABEL, "phase": phase, "number": number, "seconds": elapsed, "warm": warm})
    if texts is not None and expected["rows"] != len(texts):
        raise ValueError("the identity counts a different number of rows than the list")
    return {
        "wait_seconds": waiting[0],
        "seconds": seconds,
        "warmup_seconds": warmups,
        "warm_seconds": warms,
        "output": expected,
        "identity_bytes": expected["nbytes"],
        "probe": probe,
        "input": {k: v for k, v in corpus.items() if k != "source"},
        "policy": policy,
        "telemetry_before": before,
        "telemetry_after": observations(job["work"], excluded=()),
        "peak_rss_bytes": _peak(),
    }


@contextmanager
def _strings_source(texts):
    """The list as a source, built before the timer (the timer covers the identity only)."""
    from fse._text import StringSource

    yield StringSource(texts), 0.0


def _peak():
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * (1 if sys.platform == "darwin" else 1024)

"""Untimed native join spans from an explicitly patched diagnostic build only."""

from types import SimpleNamespace

from _development_encode import output_identity
from _development_join import _arrays
from _harness.budget import check


def observe(texts, job, expected):
    from fse import _native

    read = getattr(_native, "_diagnostic_join_record", None)
    if read is None:
        raise ValueError("join_spans requires short-call-join-spans-diagnostic.patch")
    records = []
    for number in range(50):
        check(job.get("wall_deadline"))
        joined = _native.join_strings(texts, job["workers"])
        record = read()
        if joined is None or record is None:
            raise ValueError("native diagnostic join did not complete")
        data, offsets = joined
        proof = output_identity(_arrays(SimpleNamespace(_data=data, _offsets=offsets)))
        if proof != expected:
            raise ValueError("diagnostic native join differs from Python UTF-8 reference")
        if record["parts"] != job["workers"]:
            raise ValueError("diagnostic join did not execute the requested part count")
        records.append({"call": number, **record})
        del joined, data, offsets
    return {
        "format": "join-span-series@1",
        "valid": True,
        "functional_only": True,
        "valid_for_comparison": False,
        "output": expected,
        "requested_parts": job["workers"],
        "calls": records,
        "scope": "direct native join with explicit part budget, outside all timers",
        "limits": [
            "parallel start intervals overlap worker execution",
            "join intervals include waiting for running workers",
            "span sums are not additive savings or measured GIL waits",
            "diagnostic build and clock overhead cannot enter speed decisions",
        ],
    }

"""Many small encode calls: one pass is every call of a fixed pool, in a fixed order."""

from __future__ import annotations

from _harness.budget import check


class Calls(list):
    """The results of the calls of the last pass, one array per call, in call order."""


def groups(texts, call_texts):
    """The pool cut into the lists of the calls, made before any timer starts; the last
    call holds the remainder."""
    return [texts[i : i + call_texts] for i in range(0, len(texts), call_texts)]


def operation(model, texts, job):
    from _development_encode import output_identity

    check(job.get("wall_deadline"))
    calls = groups(texts, job["call_texts"])
    dtype = job["output_dtype"]
    workers = job["workers"]
    reference = model.encode(texts, workers=workers, dtype=dtype)
    expected = output_identity(reference)
    del reference
    encode = model.encode

    def one_pass():
        results = Calls()
        append = results.append
        for call in calls:
            append(encode(call, workers=workers, dtype=dtype))
        return results

    check(job.get("wall_deadline"))
    if output_identity(one_pass()) != expected:  # stacked calls equal one bulk encode
        raise ValueError("small calls differ from one bulk encode")
    return one_pass

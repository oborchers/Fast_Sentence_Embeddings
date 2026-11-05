"""The meta-tests of the group conformance: a deliberately wrong group must fail the
comparisons of tests/test_conformance_heads.py, so that these cases catch the usual
mistakes (a shifted column, a neighbour's bias, normalization twice or not at all, a head
run on the wrong path, flushed subnormals, an executor case that ran inline)."""

import threading

import numpy as np
import pytest

from fse._backends import get_backend
from fse._runtime import HeadsPipeline, Member, _heads, _heads_errors
from fse._text import StringSource
from support.head_cases import case_spec, head_of, sentences
from support.heads import assert_same, check_group, expected, group
from support.heads_cases import members_of, outputs_for, witness_head, witness_spec

F32 = np.float32
TEXTS = sentences(40)
CPU = type(get_backend("cpu"))


def _wrong(spec, members, vectors=False, texts=TEXTS):
    with pytest.raises(AssertionError):
        check_group(spec, members, "cpu", None, vectors, texts)


def test_the_comparison_passes_for_the_unchanged_group():
    spec = case_spec("float32", False)
    check_group(spec, members_of(spec, "M"), "cpu", None, False, TEXTS)


def test_a_column_shifted_by_one_is_caught(monkeypatch):
    original = HeadsPipeline._source

    def shifted(self, local, i, kind, read, n):
        source, weight = original(self, local, i, kind, read, n)
        return (
            np.ascontiguousarray(np.roll(source, 1, axis=1)) if weight is None else source
        ), weight

    monkeypatch.setattr(HeadsPipeline, "_source", shifted)
    spec = case_spec("float32", False)
    _wrong(spec, members_of(spec, "F2"))


def test_the_bias_of_the_neighbour_is_caught(monkeypatch):
    original = CPU.logits

    def other_bias(self, rows, weights, bias, out, first_row=0):
        original(self, rows, weights, np.roll(bias, 1).copy(), out, first_row)

    monkeypatch.setattr(CPU, "logits", other_bias)
    spec = case_spec("float32", False)
    _wrong(spec, members_of(spec, "F3"))


def test_normalization_twice_or_not_at_all_is_caught(monkeypatch):
    spec = case_spec("float32", False, normalize=False)
    members = members_of(spec, "N2")
    original = CPU.postprocess_pooled
    monkeypatch.setattr(
        CPU,
        "postprocess_pooled",
        lambda self, rows, steps, first: (
            original(self, rows, steps, first),
            original(self, rows, steps, first),
        )[0],
    )
    _wrong(spec, members, vectors=True)
    monkeypatch.setattr(CPU, "postprocess_pooled", lambda *_: None)
    _wrong(spec, members, vectors=True)


def test_the_thresholds_of_another_head_are_caught(monkeypatch):
    from fse._runtime import HeadSpec

    spec = case_spec("float32", False)
    members = members_of(spec, "F3")
    original = _heads.write_outputs

    def other(backend, local, head, logits, target, output, first, tag=""):
        if head.thresholds is not None:  # the cutoffs of a neighbour
            limits = np.clip(head.thresholds + F32(0.2), 0, 1).astype(F32)
            head = HeadSpec(head.weight, head.bias, head.normalized, head.link, limits, head.rule)
        original(backend, local, head, logits, target, output, first, tag)

    monkeypatch.setattr(_heads, "write_outputs", other)
    _wrong(spec, members)


def test_a_raw_head_run_unfolded_instead_of_folded_is_not_the_folded_bits():
    spec = case_spec("float32", False)
    members = (Member("a", head_of(spec, 17, False)),)
    folded, _ = group(spec, members).infer(StringSource(TEXTS), outputs=("logits",))
    unfolded = expected(spec, members[0], TEXTS, "logits", vectors=True)
    with pytest.raises(AssertionError):
        assert_same(folded[0], unfolded, "logits")


def test_flushed_subnormals_are_caught(monkeypatch):
    spec = witness_spec("float32")
    members = tuple(Member(f"h{i}", witness_head(spec, k)) for i, k in enumerate((33, 1, 31)))
    original = CPU.rows

    def flushed(self, batch, prepared, out):
        original(self, batch, prepared, out)
        out[np.abs(out) < np.finfo(F32).tiny] = 0.0

    monkeypatch.setattr(CPU, "rows", flushed)
    _wrong(spec, members, texts=["w1", "w2", "w0", "", "w3 w1"])


def test_an_error_with_the_wrong_head_is_caught(monkeypatch):
    from support.heads_cases import column_head, overflow_spec

    spec = overflow_spec()
    big = column_head(spec, 0, F32(3e38))
    members = (Member("small", column_head(spec, 0, 1.0)), Member("big", big))
    monkeypatch.setattr(_heads_errors, "_folded_heads", lambda *_: "head 'small'")
    with pytest.raises(ValueError, match="head 'small'") as caught:
        group(spec, members).infer(StringSource(["w1", "w5 w6"]), outputs=("logits", "logits"))
    assert "head 'big'" not in str(caught.value)  # the comparison of the error test would fail


def test_an_executor_case_that_ran_inline_is_caught(monkeypatch):
    from fse._runtime import _tasks

    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 1 << 40)
    tasks = []
    original = HeadsPipeline._work
    monkeypatch.setattr(
        HeadsPipeline,
        "_work",
        lambda self, *a, **k: (
            tasks.append(threading.current_thread().name),
            original(self, *a, **k),
        )[1],
    )
    spec = case_spec("float32", False)
    members = members_of(spec, "F3")
    pipeline = group(spec, members)
    pipeline.infer(
        StringSource(sentences(120) * 10),
        outputs=outputs_for(members, "proba"),
        workers=3,
        chunk_bytes=1 << 12,
    )
    assert len(tasks) > 1
    assert not any(name.startswith("fse-worker") for name in tasks)  # all in the caller

"""A group of heads in the runtime: the work of a task, the executor, the vectors and the
window."""

import threading

import numpy as np
import pytest

from fse._runtime import HeadsPipeline, Member, MemorySink
from fse._text import StringSource
from support.head_cases import BACKENDS, case_spec, head_of, sentences
from support.heads import alone, assert_same, bits, check_group, group
from support.heads_cases import members_of, outputs_for

F32 = np.float32


def _tasks(monkeypatch):
    """The tasks (and the threads) that ran a group."""
    seen = {"tasks": 0, "threads": set()}  # the names of the threads
    original = HeadsPipeline._work

    def counting(self, *args, **kwargs):
        seen["tasks"] += 1
        seen["threads"].add(threading.current_thread().name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HeadsPipeline, "_work", counting)
    return seen


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("name", ["F3", "M"])
def test_tasks_on_threads_give_the_bits_of_the_heads_alone(monkeypatch, backend, name):
    from fse._runtime._tasks import INLINE_MAX_BYTES

    name_, isa = backend
    spec = case_spec("float16", fasttext=True, normalize=False)
    members = members_of(spec, name)
    texts = sentences(120) * 10
    source = StringSource(texts)
    assert source.nbytes > INLINE_MAX_BYTES  # not run inline
    seen = _tasks(monkeypatch)
    pipeline = group(spec, members, name_, isa)
    outputs = outputs_for(members, "proba")
    results, _ = pipeline.infer(source, outputs=outputs, workers=3, chunk_bytes=1 << 12)
    assert seen["tasks"] > 1
    assert any(name.startswith("fse-worker") for name in seen["threads"])  # not inline
    for member, output, result in zip(members, outputs, results, strict=True):
        one, _ = alone(spec, member, name_, isa, False, texts, output)
        assert_same(result, one, output)


def test_a_stream_gives_the_bits_of_one_array():
    spec = case_spec("float32", fasttext=False)
    members = members_of(spec, "M")
    pipeline = group(spec, members, vectors=True)
    texts = sentences(60)
    outputs = outputs_for(members, "proba")
    whole, vectors = pipeline.infer(StringSource(texts), outputs=outputs, workers=2)
    from fse._stream import StreamSource

    parts, vparts = pipeline.infer(
        StreamSource(iter(texts), chunk_rows=7), outputs=outputs, workers=2
    )
    for a, b, output in zip(whole, parts, outputs, strict=True):
        assert_same(a, b, output)
    np.testing.assert_array_equal(bits(vectors), bits(vparts))


def test_the_vectors_are_those_of_the_encoder_in_every_dtype():
    spec = case_spec("int8", fasttext=True, normalize=True)
    members = members_of(spec, "M")
    pipeline = group(spec, members, vectors=True)
    texts = sentences(50)
    outputs = outputs_for(members, "logits")
    own = pipeline.plan.own
    for dtype in ("float32", "float16", "int8"):
        _, vectors = pipeline.infer(
            StringSource(texts), outputs=outputs, workers=2, vector_dtype=dtype
        )
        want = own.infer(StringSource(texts), workers=1, dtype=dtype)
        if dtype == "int8":
            np.testing.assert_array_equal(vectors.codes, want.codes)
            np.testing.assert_array_equal(bits(vectors.scales), bits(want.scales))
        else:
            np.testing.assert_array_equal(bits(vectors), bits(want))


def test_a_sink_per_head_and_one_for_the_vectors():
    spec = case_spec("float32", fasttext=False)
    members = members_of(spec, "F2")
    pipeline = group(spec, members, vectors=True)
    texts = sentences(30)
    sinks = [MemorySink(m.head.k) for m in members]
    vsink = MemorySink(spec.output_dim)
    results, vectors = pipeline.infer(
        StringSource(texts), outputs=("logits", "logits"), sinks=sinks, vector_sink=vsink
    )
    assert results == [None, None]
    assert vectors is None
    alone, avectors = pipeline.infer(StringSource(texts), outputs=("logits", "logits"))
    for sink, want in zip(sinks, alone, strict=True):
        np.testing.assert_array_equal(bits(sink.result()), bits(want))
    np.testing.assert_array_equal(bits(vsink.result()), bits(avectors))


def test_many_wide_heads_shrink_the_tasks_to_keep_the_window():
    spec = case_spec("float32", fasttext=False)
    few = group(spec, members_of(spec, "F2")[:1])
    many = group(spec, tuple(Member(f"h{i}", head_of(spec, 17, False, seed=i)) for i in range(30)))
    source = StringSource(["w1"] * 60000)
    sizes = []
    for pipe, outs in ((few, ("proba",)), (many, ("proba",) * 30)):
        row_bytes = pipe.plan.row_bytes(outs, "float32", False)
        plan, window, _ = pipe.plan.primary.plan_tasks(source, 4, None, row_bytes)
        sizes.append((max(t.n_rows for t in plan.tasks), window, row_bytes))
    assert sizes[1][2] > 10 * sizes[0][2]
    assert sizes[1][0] < sizes[0][0]  # smaller tasks
    # The window holds every row at the bytes the heads keep: columns of each folded head,
    # the logits and the outputs.
    heads = [m.head for m in many.plan.members]
    held = 4 * sum(h.k for h in heads) * 2 + sum(4 * h.width("proba") for h in heads)
    assert sizes[1][2] >= held


def test_a_group_of_one_keeps_the_name_of_its_head():
    from support.heads_cases import column_head, overflow_spec

    spec = overflow_spec()
    pipe = group(spec, (Member("only", column_head(spec, 0, F32(3e38))),))
    with pytest.raises(ValueError, match=r"^head 'only': row 1 "):
        pipe.infer(StringSource(["w1", "w5 w6"]), outputs=("logits",))


def test_an_empty_source_gives_empty_arrays():
    spec = case_spec("float32", fasttext=False)
    members = members_of(spec, "M")
    results, vectors = group(spec, members, vectors=True).infer(
        StringSource([]), outputs=outputs_for(members, "proba")
    )
    assert [len(r) for r in results] == [0] * len(members)
    assert vectors.shape == (0, spec.output_dim)


@pytest.mark.parametrize("vectors", [False, True])
def test_the_group_equals_each_head_alone_with_several_workers_and_a_stream(vectors):
    spec = case_spec("float32", fasttext=True, normalize=True)
    check_group(spec, members_of(spec, "M"), "cpu", None, vectors, sentences(80))


class Collector:
    """A tee that joins the rows of the tasks of a run."""

    def __init__(self):
        self.parts = {}

    def target(self, first_row, n_rows):
        pass

    def commit(self, first_row, rows):
        self.parts[first_row] = np.array(rows)

    def joined(self):
        return np.concatenate([self.parts[k] for k in sorted(self.parts)])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("name", ["F3", "N2"])
def test_a_second_output_has_the_bits_of_that_output_alone(backend, name):
    name_, isa = backend
    spec = case_spec("float16", fasttext=True, normalize=False)
    members = members_of(spec, name)
    texts = sentences(120) * 4
    pipe = group(spec, members, name_, isa)
    # Each head returns its labels and gives its probabilities (or its logits) on the side.
    labels = outputs_for(members, "labels")
    extras = [("logits" if i % 2 else "proba", Collector()) for i in range(len(members))]
    results, _ = pipe.infer(
        StringSource(texts), outputs=labels, workers=3, extras=extras, tee_bytes=64
    )
    alone_labels = group(spec, members, name_, isa).infer(
        StringSource(texts), outputs=labels, workers=1
    )[0]
    for got, want, output in zip(results, alone_labels, labels, strict=True):
        assert_same(got, want, output)
    for member, (kind, tee) in zip(members, extras, strict=True):
        want, _ = alone(spec, member, name_, isa, False, texts, kind)
        assert_same(tee.joined(), want, kind)


def test_a_tee_on_the_main_output_and_a_second_output_do_not_change_the_window():
    spec = case_spec("float32", fasttext=False)
    pipe = group(spec, members_of(spec, "F3"))
    outs = outputs_for(pipe.plan.members, "labels")
    source = StringSource(["w1 w2"] * 40000)
    plain = pipe.plan.row_bytes(outs, "float32", False)
    extras = ("logits", None, "proba")
    heavy = pipe.plan.row_bytes(outs, "float32", False, extras, 8 * 4 + 8)
    assert heavy > plain
    small, window, _ = pipe.plan.primary.plan_tasks(source, 4, None, heavy)
    big, _, _ = pipe.plan.primary.plan_tasks(source, 4, None, plain)
    assert max(t.n_rows for t in small.tasks) <= max(t.n_rows for t in big.tasks)
    assert window >= 1

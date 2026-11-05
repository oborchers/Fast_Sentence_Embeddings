"""The plan of a group of heads: which rows each head reads, how many poolings a task
makes, and the bytes of a row (for one head those of the pipeline of 2.0)."""

import numpy as np
import pytest

from fse._backends import get_backend
from fse._contract import Link
from fse._runtime import HeadSpec, Member, plan_heads
from fse._runtime._heads_plan import FOLDED, NORMALIZED, RAW, STANDALONE, _head_bytes, kind_of
from fse._runtime._pipeline import STEP_ROW_BYTES
from fse._text import StringSource
from support.head_cases import case_spec, head_of
from support.head_legacy import legacy_make_head
from support.heads import group, own_of
from support.heads_cases import members_of

F32 = np.float32


def test_the_kind_of_a_head_follows_its_rows():
    spec = case_spec("float32", False)
    raw, normed = head_of(spec, 2, False), head_of(spec, 2, True)
    assert kind_of(raw, False) == FOLDED
    assert kind_of(raw, True) == RAW
    assert kind_of(normed, False) == kind_of(normed, True) == NORMALIZED
    assert kind_of(HeadSpec(None, raw.bias, False, raw.link), False) == STANDALONE


def test_the_columns_of_the_folded_heads_lie_side_by_side():
    spec = case_spec("float32", False)
    members = tuple(
        Member(n, head_of(spec, k, normalized))
        for n, k, normalized in [("a", 3, False), ("b", 4, True), ("c", 1, False), ("d", 2, False)]
    )
    plan = group(spec, members).plan
    assert plan.kinds == (FOLDED, NORMALIZED, FOLDED, FOLDED)
    assert plan.starts == (0, 0, 3, 4)
    assert plan.folded_columns == 6
    assert plan.table.output_dim == 6
    assert plan.normed.output_dim == spec.output_dim
    assert plan.primary.output_dim == spec.output_dim  # the widest reader plans the tasks


def test_a_group_of_standalone_and_complete_heads_is_refused():
    spec = case_spec("float32", False)
    full = head_of(spec, 2, False)
    alone = HeadSpec(None, full.bias, False, full.link)
    members = (Member("a", full), Member("b", alone))
    with pytest.raises(ValueError, match="standalone heads and complete heads"):
        plan_heads(spec, members, "cpu", own=own_of(spec))
    with pytest.raises(ValueError, match="no vectors"):
        plan_heads(spec, (Member("b", alone),), "cpu", own=own_of(spec), with_vectors=True)
    with pytest.raises(ValueError, match="at least one head"):
        plan_heads(spec, (), "cpu", own=own_of(spec))
    with pytest.raises(ValueError, match="pipeline of spec"):
        plan_heads(case_spec("float32", True), members[:1], "cpu", own=own_of(spec))


class _Counting:
    """The calls of the cpu backend of a plan that pool and normalize."""

    def __init__(self, monkeypatch):
        self.calls = []
        cls = type(get_backend("cpu"))
        for name in ("rows", "postprocess_pooled"):
            monkeypatch.setattr(cls, name, self._counted(name, getattr(cls, name)))

    def _counted(self, name, original):
        def wrapped(backend, *args):
            self.calls.append(name)
            return original(backend, *args)

        return wrapped


EXPECTED = [
    # composition, encoder normalizes, vectors, poolings per task, normalizations per task
    ("F2", False, False, 1, 0),
    ("F3", True, False, 1, 0),
    ("N2", False, False, 1, 0),
    ("N2", True, False, 1, 0),
    ("M", False, False, 2, 0),  # a pooling of the folded table and one of the normalized rows
    ("M", True, False, 2, 0),
    ("F2", False, True, 1, 0),
    ("F2", True, True, 1, 1),  # raw rows pooled, then normalized into the vectors
    ("N2", False, True, 1, 1),  # raw rows are the vectors, a copy is normalized
    ("N2", True, True, 1, 0),
    ("M", False, True, 1, 1),
    ("M", True, True, 1, 1),
]


@pytest.mark.parametrize(("name", "normalize", "vectors", "poolings", "normalizations"), EXPECTED)
def test_a_task_pools_the_full_table_at_most_once_per_kind_of_rows(
    monkeypatch, name, normalize, vectors, poolings, normalizations
):
    spec = case_spec("float32", False, normalize=normalize)
    pipeline = group(spec, members_of(spec, name), vectors=vectors)
    counting = _Counting(monkeypatch)
    outputs = ("logits",) * len(pipeline.plan.members)
    pipeline.infer(StringSource(["w1 w2", "w3"]), outputs=outputs, workers=1)
    # A pipeline with a step normalizes inside rows; a copy goes through postprocess_pooled.
    # (Calls of rows inside the cpu backend to its own postprocess_pooled are not counted.)
    assert counting.calls.count("rows") == poolings
    assert counting.calls.count("postprocess_pooled") == normalizations


def _lone(spec, head, vectors):
    return legacy_make_head(spec, head, "cpu", own=own_of(spec), with_vectors=vectors)


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    ("k", "link"),
    [(1, Link.SIGMOID), (4, Link.SOFTMAX), (3, Link.OVR), (3, Link.MULTILABEL)],
)
def test_the_bytes_of_a_row_of_a_lone_head_are_those_of_2_0(
    k, link, normalized, normalize, vectors
):
    spec = case_spec("float16", True, normalize=normalize)
    base = head_of(spec, k, normalized, link=link)
    thresholds = None
    if link is Link.MULTILABEL:
        thresholds = np.full(k, 0.4, F32)
    head = HeadSpec(base.weight, base.bias, normalized, link, thresholds)
    old = _lone(spec, head, vectors)
    new = group(spec, (Member(None, head),), vectors=vectors).plan
    for output in ("logits", "proba", "labels"):
        for vdtype, inverse in [("float32", False), ("int8", True)]:
            assert new.row_bytes((output,), vdtype, inverse) == old._row_bytes(
                output, vdtype, inverse
            ), (output, vdtype)


def test_a_lone_head_plans_the_tasks_of_2_0():
    spec = case_spec("float32", False)
    head = head_of(spec, 17, False)
    old = _lone(spec, head, False)
    new = group(spec, (Member(None, head),)).plan
    texts = ["w1 w2 w3"] * 5000
    for workers in (1, 4):
        a = old.reader.plan_tasks(
            StringSource(texts), workers, None, old._row_bytes("proba", "float32", False)
        )
        b = new.primary.plan_tasks(
            StringSource(texts), workers, None, new.row_bytes(("proba",), "float32", False)
        )
        assert a[1] == b[1]
        assert [t.chunk for t in a[0].tasks] == [t.chunk for t in b[0].tasks]


def test_more_heads_hold_more_per_row():
    spec = case_spec("float32", False)
    one = group(spec, members_of(spec, "F2")[:1]).plan.row_bytes(("proba",), "float32", False)
    two = group(spec, members_of(spec, "F2")).plan.row_bytes(("proba", "proba"), "float32", False)
    assert two > one


def _composition(spec, shape):
    """Members of the given (k, normalized) pairs."""
    return tuple(Member(f"h{i}", head_of(spec, k, norm)) for i, (k, norm) in enumerate(shape))


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        [(4, False)],
        [(4, True)],
        [(3, False), (2, False)],
        [(3, False), (2, True)],
        [(3, False), (2, True), (1, False), (2, False)],
    ],
)
def test_what_the_plan_precomputes_is_what_a_task_would_compute(shape, vectors):
    spec = case_spec("float32", False)
    plan = group(spec, _composition(spec, shape), vectors=vectors).plan
    folded = [i for i, kind in enumerate(plan.kinds) if kind == FOLDED]
    assert plan.folded == plan.indices(FOLDED) == tuple(folded)
    assert plan.has_normalized == (NORMALIZED in plan.kinds)
    for i, kind in enumerate(plan.kinds):
        # A folded head reads the table as it is when it is the only one in the table.
        assert plan.direct[i] == (kind == STANDALONE or (kind == FOLDED and len(folded) == 1))
        assert plan.logit_buffers[i] == f"head_logits{i}"
        assert plan.column_buffers[i] == f"head_columns{i}"
        assert plan.tags[i] == str(i)


def test_a_lone_folded_head_in_a_mixed_group_reads_the_table_as_it_is():
    spec = case_spec("float32", False)
    plan = group(spec, _composition(spec, [(3, False), (2, True)])).plan
    assert plan.kinds == (FOLDED, NORMALIZED)
    assert plan.direct == (True, False)  # the other head is not folded: no columns to cut
    alone = HeadSpec(None, head_of(spec, 2, False).bias, False, Link.SOFTMAX)
    plan = group(spec, (Member(None, alone),)).plan
    assert plan.direct == (True,)


def _bytes_by_the_rule(plan, outputs):
    """The bytes of a row as the plan counted them before it kept the fixed part."""
    total = 4 * (plan.table.output_dim if plan.table else 0)
    if len(plan.indices(FOLDED)) > 1:
        total += 4 * sum(plan.members[i].head.k for i in plan.indices(FOLDED))
    full = 4 * plan.spec.output_dim
    if plan.indices(NORMALIZED, RAW):
        total += full + (STEP_ROW_BYTES * plan.spec.output_dim if plan.indices(NORMALIZED) else 0)
        total += full if plan.indices(NORMALIZED) and plan.indices(RAW) else 0
    for member, output in zip(plan.members, outputs, strict=True):
        total += _head_bytes(member.head, output)
    return total


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize(
    "shape",
    [[(4, False)], [(3, False), (2, False)], [(3, False), (2, True)], [(2, True), (3, True)]],
)
def test_the_bytes_of_a_row_are_counted_by_the_rule_of_the_kinds(shape, vectors):
    spec = case_spec("float32", False)
    plan = group(spec, _composition(spec, shape), vectors=vectors).plan
    for outputs in [("logits",) * len(shape), ("proba",) * len(shape), ("labels",) * len(shape)]:
        expected = _bytes_by_the_rule(plan, outputs)
        if vectors:
            expected += plan.own._row_bytes("float32", False)
        assert plan.row_bytes(outputs, "float32", False) == expected
    with pytest.raises(ValueError, match="zip"):
        plan.row_bytes(("logits",) * (len(shape) + 1), "float32", False)

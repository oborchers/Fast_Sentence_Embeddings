"""The plan of a group of heads: which rows each head reads, how many poolings a task
makes, and the bytes of a row (for one head those of the pipeline of 2.0)."""

import numpy as np
import pytest

from fse._backends import get_backend
from fse._contract import Link
from fse._runtime import HeadSpec, Member, plan_heads
from fse._runtime._heads_plan import FOLDED, NORMALIZED, RAW, STANDALONE, kind_of
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

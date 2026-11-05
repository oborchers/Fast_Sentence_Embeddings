"""Where the rows of a group go: one target per head, committed in task order."""

import numpy as np
import pytest

from fse._runtime import MemorySink
from fse._runtime._heads_targets import Targets
from fse._runtime._tasks import Task
from fse._text import Chunk, StringSource
from support.head_cases import case_spec, head_of

F32 = np.float32


def _task(first, n):
    return Task(StringSource(["a"] * (first + n)), Chunk(0, 1, first, n))


def _targets(heads, outputs, rows, sinks=None, vectors=None, base=0):
    return Targets(heads, outputs, sinks or [None] * len(heads), rows, base=base, vectors=vectors)


def test_every_head_has_its_own_array_of_its_width():
    spec = case_spec("float32", False)
    heads = [head_of(spec, 1, False), head_of(spec, 4, False)]
    targets = _targets(heads, ("proba", "labels"), 5)
    rows, vectors = targets.reserve(_task(2, 3))
    assert [r.shape for r in rows] == [(3, 2), (3,)]
    assert rows[1].dtype == np.int64
    assert vectors is None
    rows[0][...] = 1
    rows[1][...] = 2


def test_rows_are_committed_in_order_and_the_arrays_hold_them():
    spec = case_spec("float32", False)
    heads = [head_of(spec, 3, False), head_of(spec, 2, False)]
    targets = _targets(heads, ("logits", "logits"), 5, vectors=(spec.output_dim, "float32", None))
    first, second = _task(0, 2), _task(2, 3)
    a, b = targets.reserve(first), targets.reserve(second)
    for part, value in ((a, 1), (b, 2)):
        for rows in part[0]:
            rows[...] = value
        part[1][...] = value
    with pytest.raises(ValueError, match="expected rows from 0"):
        targets.commit(second, b)
    targets.commit(first, a)
    targets.commit(second, b)
    results, vectors = targets.results()
    assert results[0][:, 0].tolist() == [1, 1, 2, 2, 2]
    assert vectors.shape == (5, spec.output_dim)


def test_a_sink_takes_the_rows_of_its_head_numbered_from_the_run():
    spec = case_spec("float32", False)
    heads = [head_of(spec, 3, False), head_of(spec, 2, False)]
    sink = MemorySink(3)
    targets = _targets(heads, ("logits", "logits"), None, sinks=[sink, None], base=0)
    task = _task(0, 2)
    rows, _ = done = targets.reserve(task)
    rows[0][...] = 7
    targets.commit(task, (rows, done[1]))
    results, _ = targets.results()
    assert results[0] is None
    assert sink.result().tolist() == [[7, 7, 7]] * 2
    assert results[1].shape == (2, 2)


class Recorder:
    """A tee: it takes the rows of every task at the commit."""

    def __init__(self):
        self.opened, self.committed = [], []

    def target(self, first_row, n_rows):
        self.opened.append((first_row, n_rows))

    def commit(self, first_row, rows):
        self.committed.append((first_row, np.array(rows)))


def test_a_tee_gets_the_rows_of_a_task_at_its_commit_numbered_from_the_run():
    spec = case_spec("float32", False)
    heads = [head_of(spec, 3, False)]
    tee = Recorder()
    targets = Targets(heads, ("logits",), [None], 5, base=100, vectors=None, tees=[tee])
    first, task = _task(0, 2), _task(2, 3)
    targets.commit(first, targets.reserve(first))
    done = targets.reserve(task)
    assert tee.opened == [(100, 2), (102, 3)]
    assert [row for row, _ in tee.committed] == [100]  # only the first task is finished
    done[0][0][...] = 7
    targets.commit(task, done)
    assert [(row, part.shape) for row, part in tee.committed] == [(100, (2, 3)), (102, (3, 3))]
    assert (tee.committed[1][1] == 7).all()
    results, _ = targets.results()
    assert results[0].shape == (5, 3)  # the tee takes no part in the arrays returned


def test_slots_add_one_scratch_slot_for_each_extra_output():
    from fse._runtime._heads_targets import Scratch, Slots

    spec = case_spec("float32", False)
    heads = [head_of(spec, 1, False), head_of(spec, 4, False)]
    tee_a, tee_b = Recorder(), Recorder()
    slots = Slots(
        heads, ("labels", "logits"), [None, None], [None, tee_a], [("proba", tee_b), None]
    )
    assert slots.outputs == ("labels", "logits", "proba")
    assert slots.of_member == [[0, 2], [1]]
    assert slots.extra_kinds == ("proba", None)
    assert slots.tees == [None, tee_a, tee_b]
    assert isinstance(slots.sinks[2], Scratch)
    assert slots.heads(heads) == [heads[0], heads[1], heads[0]]
    scratch = slots.sinks[2]
    assert scratch.target(0, 4).shape == (4, 2)  # the two probabilities of one logit
    scratch.commit(0, None)


def test_slots_refuse_lists_of_other_lengths():
    from fse._runtime._heads_targets import Slots

    spec = case_spec("float32", False)
    heads = [head_of(spec, 1, False)]
    with pytest.raises(ValueError, match="one sink, tee and extra output"):
        Slots(heads, ("logits",), [None], [None, None], None)

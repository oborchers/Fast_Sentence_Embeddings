"""Task-team failures keep the previous executor's exception type and message."""

import pytest

from fse import _native
from fse._runtime._sinks import ArrayRows
from fse._runtime._worker import WorkerLocal
from fse._text import StringSource, copied_strings
from support.native_team import faults, pipeline, tasks_for
from support.runtime import texts


def caught(call):
    with pytest.raises(Exception, match=r".") as error:
        call()
    try:
        return type(error.value), str(error.value)
    finally:
        del error


@pytest.mark.parametrize("source_factory", [StringSource, copied_strings])
@pytest.mark.parametrize("kind", range(1, 11))
@pytest.mark.parametrize("workers", range(2, 9))
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_batch_corruption_matches_old_task(kind, workers, position, source_factory):
    model = pipeline(fasttext=True)
    source = source_factory(texts(4000))
    tasks = tasks_for(model, source, workers)
    assert len(tasks) > len(tasks_for(model, source, workers, refined=False))
    index = {"first": 0, "middle": len(tasks) // 2, "last": len(tasks) - 1}[position]
    worker = model._task_work(source, True, WorkerLocal(), 0, None)
    task = tasks[index]
    out = ArrayRows(len(source), model.output_dim, "float32")
    previous = _native._test_corrupt(kind)
    try:
        before = _native._test_team_calls()
        with faults(corrupt_task=index):
            found = caught(lambda: model.infer(source, workers=workers))
        assert _native._test_team_calls() == before + 1
        wanted = caught(lambda: worker((task, out._view(task.first_row, task.n_rows))))
        assert found == wanted
        assert _native._test_team_resources() == 0
    finally:
        _native._test_corrupt(previous)

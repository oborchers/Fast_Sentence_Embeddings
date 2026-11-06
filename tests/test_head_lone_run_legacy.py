"""The calls of 2.0 with ``sink=`` on one head stay as they were (DR-243).

The test uses only the public API of fse 2.0.0 and records every call a sink and an input
get, so the same file runs against the tag (``uv run --isolated --with fse==2.0.0 pytest
tests/test_head_lone_run_legacy.py``) and against the branch. A text that differs only in
its hints is matched on its stable part.
"""

import numpy as np
import pytest

import fse

F32 = np.float32
DIM = 6
WORDS = [f"w{i}" for i in range(40)]


def _encoder():
    rng = np.random.default_rng(0)
    return fse.Encoder.from_arrays(
        rng.standard_normal((len(WORDS), DIM)).astype(F32), words=WORDS, normalize=False
    )


def _classifier(k=3):
    rng = np.random.default_rng(1)
    return fse.Classifier.from_arrays(
        _encoder(), rng.standard_normal((k, DIM)), labels=[f"c{i}" for i in range(k)]
    )


def _regressor():
    rng = np.random.default_rng(2)
    return fse.Regressor.from_arrays(_encoder(), rng.standard_normal(DIM))


def _texts(n, start=0):
    return [" ".join(WORDS[(start + i + j) % len(WORDS)] for j in range(3)) for i in range(n)]


class Recorder(fse.MemorySink):
    """A sink that records its calls, in order, as plain tuples."""

    def __init__(self, dim, log, **attributes):
        super().__init__(dim)
        self.log = log
        self.attributes = attributes

    def __getattr__(self, name):
        attributes = self.__dict__.get("attributes", {})
        if name in attributes:
            self.__dict__["log"].append(("get", name))
            return attributes[name]
        raise AttributeError(name)

    def target(self, first_row, n_rows):
        self.log.append(("target", first_row, n_rows))
        return super().target(first_row, n_rows)

    def commit(self, first_row, rows, **extra):
        self.log.append(("commit", first_row, len(rows), tuple(sorted(extra))))
        super().commit(first_row, rows)
        self.done = getattr(self, "done", 0) + len(rows)

    def publish(self):
        self.log.append(("publish",))

    def flush(self):
        self.log.append(("flush",))

    @property
    def durable_rows(self):
        self.log.append(("durable_rows",))
        return getattr(self, "done", 0)


def _recorded(dim, **attributes):
    log = []
    return Recorder(dim, log, **attributes), log


class Items:
    """An input that records how far it was read."""

    def __init__(self, batches, log, fail_at=None):
        self.batches, self.log, self.fail_at = batches, log, fail_at

    def __iter__(self):
        for k, batch in enumerate(self.batches):
            self.log.append(("next", k))
            if k == self.fail_at:
                raise RuntimeError("input failed")
            yield batch
        self.log.append(("exhausted",))


def _drain(iterator, got, then=None):
    for batch in iterator:
        got.append(batch)
        if then is not None:
            then()


def _widen(sink):
    def change():
        sink.dim = 4

    return change


def _summary(batches):
    out = []
    for b in batches:
        out.append(
            (
                b.first_row,
                b.n_rows,
                None if b.outputs is None else np.asarray(b.outputs).tobytes(),
                None if b.vectors is None else np.asarray(b.vectors).tobytes(),
                b.durable_rows,
            )
        )
    return out


def _models():
    return [
        pytest.param(_classifier(), "decision", id="classifier"),
        pytest.param(_regressor(), "values", id="regressor"),
    ]


def _width(model, vectors):
    return DIM if vectors else (3 if model.TASK == "classification" else 1)


def _run(model, output, batches, vectors, **options):
    width = _width(model, vectors)
    sink, log = _recorded(width)
    items = Items(batches, log)
    try:
        out = list(
            model.predict_batches(
                items, output=output, return_vectors=vectors, sink=sink, **options
            )
        )
    except Exception as error:
        return log, [], (type(error).__name__, str(error))
    return log, _summary(out), None


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize(("model", "output"), _models())
def test_three_batches_publish_and_flush_in_the_2_0_order(model, output, vectors):
    log, batches, error = _run(model, output, [_texts(4), _texts(2), _texts(3)], vectors)
    assert error is None
    assert [b[:2] for b in batches] == [(0, 4), (4, 2), (6, 3)]
    kinds = [entry[0] for entry in log]
    assert kinds[-1] == "exhausted" or kinds[-2:] == ["flush", "exhausted"] or "flush" in kinds
    assert kinds.count("flush") == 1
    assert kinds.index("flush") > max(i for i, k in enumerate(kinds) if k == "commit")
    assert kinds.count("publish") == 3
    for first in (0, 4, 6):
        assert ("target", first, {0: 4, 4: 2, 6: 3}[first]) in log


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize(("model", "output"), _models())
def test_an_empty_input_flushes_once_and_yields_nothing(model, output, vectors):
    log, batches, error = _run(model, output, [], vectors)
    assert error is None
    assert batches == []
    assert [entry[0] for entry in log].count("flush") == 1


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize(("model", "output"), _models())
def test_stopping_early_does_not_flush(model, output, vectors):
    sink, log = _recorded(_width(model, vectors))
    it = model.predict_batches(
        Items([_texts(3), _texts(3)], log), output=output, return_vectors=vectors, sink=sink
    )
    first = next(it)
    assert first.first_row == 0
    it.close()
    assert ("flush",) not in log
    assert ("next", 1) not in log


@pytest.mark.parametrize(("model", "output"), _models())
def test_an_error_of_the_second_batch_names_the_batch(model, output):
    sink, log = _recorded(_width(model, False))
    items = Items([_texts(2), ["w1 w2", 7]], log)
    got = []
    with pytest.raises((ValueError, TypeError), match=r"batch 1 \(rows 2 to 3 of the run\)"):
        _drain(model.predict_batches(items, output=output, sink=sink), got)
    assert len(got) == 1
    assert ("flush",) not in log


@pytest.mark.parametrize(("model", "output"), _models())
def test_an_input_that_fails_is_not_swallowed(model, output):
    sink, log = _recorded(_width(model, False))
    items = Items([_texts(2), _texts(2)], log, fail_at=1)
    with pytest.raises(RuntimeError, match="input failed"):
        list(model.predict_batches(items, output=output, sink=sink))
    assert ("flush",) not in log


def test_the_checks_run_in_the_2_0_order_before_a_batch_is_read():
    model = _classifier()
    sink, log = _recorded(3)
    items = Items([_texts(1)], log)
    with pytest.raises(ValueError, match="labels go to no sink"):
        model.predict_batches(items, output="labels", sink=sink)
    with pytest.raises(ValueError, match="must take rows of 3 values"):
        model.predict_batches(items, output="decision", sink=fse.MemorySink(5))
    with pytest.raises(TypeError, match="target and commit"):
        model.predict_batches(items, output="decision", sink=object())
    with pytest.raises(ValueError, match=r"no probabilities|link none"):
        fse.Classifier.from_arrays(
            _encoder(), np.ones((3, DIM)), labels=["a", "b", "c"], link="none"
        ).predict_batches(items, output="proba")
    assert ("next", 0) not in log


@pytest.mark.xfail(reason="DR-240 refuses it until DR-243 restores the 2.0 behavior", strict=False)
def test_a_foreign_sink_with_has_doc_ids_true_runs_as_a_plain_sink():
    """2.0 does not know the attribute: such a sink took the batches as any sink."""
    model = _classifier()
    sink, log = _recorded(3, has_doc_ids=True)
    out = list(model.predict_batches(Items([_texts(3)], log), output="decision", sink=sink))
    assert [b.n_rows for b in out] == [3]
    assert ("commit", 0, 3, ()) in log


def test_a_sink_that_changes_its_dim_between_batches_fails_in_that_batch():
    model = _classifier()
    sink, log = _recorded(3)
    items = Items([_texts(2), _texts(2)], log)
    got = []
    with pytest.raises((ValueError, TypeError), match="batch 1"):
        _drain(model.predict_batches(items, output="decision", sink=sink), got, then=_widen(sink))
    assert len(got) == 1


def test_a_store_still_gives_its_outputs_to_a_sink(tmp_path):
    model = _classifier()
    with fse.ShardSink(tmp_path / "v", model.encoder) as vectors:
        list(model.encoder.encode_batches([_texts(5)], sink=vectors))
    with fse.VectorStore.open(tmp_path / "v") as store:
        sink = fse.MemorySink(3)
        out = list(model.predict_batches(store, output="decision", sink=sink))
        assert sum(b.n_rows for b in out) == 5
        assert all(b.outputs is None for b in out)
        assert sink.result().shape == (5, 3)
        with pytest.raises(ValueError, match="holds the vectors already"):
            model.predict_batches(store, return_vectors=True)


def test_the_2_0_parameters_stay_a_prefix_of_the_signature():
    import inspect

    for model in (_classifier(), _regressor()):
        names = list(inspect.signature(model.predict_batches).parameters)
        assert names[:6] == ["batches", "output", "return_vectors", "workers", "backend", "sink"]

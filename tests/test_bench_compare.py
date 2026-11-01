"""Shared-machine timing must fail closed and leave other owners alone."""

import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="local POSIX comparison runner")


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    module = importlib.import_module("bench_compare")
    monkeypatch.setattr(module.os, "getloadavg", lambda: (1, 1, 1))
    return module


def test_reservation_preserves_existing_owner(bench, tmp_path):
    lock = tmp_path / "bench.lock"
    lock.mkdir()
    (lock / "owner").write_text("other worker")
    with pytest.raises(FileExistsError), bench.reservation(lock):
        pytest.fail("entered another worker's reservation")
    assert (lock / "owner").read_text() == "other worker"


def test_reservation_releases_after_failure(bench, tmp_path):
    lock = tmp_path / "bench.lock"
    with pytest.raises(ValueError, match="child failed"), bench.reservation(lock):
        raise ValueError("child failed")
    assert not lock.exists()


@pytest.mark.parametrize("reason", ["load", "pause"])
def test_reservation_defers_before_timing(bench, tmp_path, monkeypatch, reason):
    lock = tmp_path / "bench.lock"
    if reason == "load":
        monkeypatch.setattr(bench.os, "getloadavg", lambda: (5, 1, 1))
    else:
        (tmp_path / "pause-local.marker").touch()
    with pytest.raises(RuntimeError), bench.reservation(lock):
        pytest.fail("timed during pause or high load")
    assert not lock.exists()


def test_reservation_rechecks_load_after_acquisition(bench, tmp_path, monkeypatch):
    loads = iter([(1, 1, 1), (5, 1, 1)])
    monkeypatch.setattr(bench.os, "getloadavg", lambda: next(loads))
    lock = tmp_path / "bench.lock"
    with pytest.raises(RuntimeError), bench.reservation(lock):
        pytest.fail("load rose while reserving")
    assert not lock.exists()


def test_default_model2vec_keeps_tokenizer_parallelism(bench, monkeypatch, capsys):
    import json

    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        SimpleNamespace(threadpool_limits=lambda **_: nullcontext(), threadpool_info=lambda: []),
    )
    monkeypatch.setattr(bench, "texts", lambda _: ["a short corpus"])
    monkeypatch.setattr(bench, "identity", lambda: {})

    def contender(*_args, **_kwargs):
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

        def encode(_texts):
            assert bench.os.environ["TOKENIZERS_PARALLELISM"] == "true"
            return bench.np.zeros((1, 2))

        return encode

    monkeypatch.setattr(bench, "contender", contender)
    args = SimpleNamespace(
        corpus="wikitext",
        kind="model2vec",
        threads=10,
        model="minishlab/potion-base-8M",
        passes=2,
        vectors_out=None,
        table_dtype="float32",
        fse_models=None,
        warmup_texts=None,
    )
    bench.child(args)
    row = json.loads(capsys.readouterr().out)
    assert len(row["warm_seconds"]) == 2
    assert row["texts"] == 1


def test_interrupted_round_is_persisted_as_incomplete(bench, tmp_path, monkeypatch):
    import json

    def operation(_args, label, _threads):
        if label == "other":
            raise RuntimeError("busy")
        return {"valid": True, "input_digest": "same"}

    monkeypatch.setattr(bench, "operation", operation)
    args = SimpleNamespace(results=tmp_path / "runs.jsonl")
    with pytest.raises(RuntimeError, match="busy"):
        bench.one_round(args, [("fse-a", 1), ("other", 1)], "series", 0)
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    assert rows[0]["series_id"] == "series"
    assert rows[1]["event"] == "round_complete"
    assert rows[1]["complete"] is False
    assert rows[1]["valid"] is False


def test_repeated_invocations_have_distinct_pair_identity(bench, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(bench, "operation", lambda *_a: {"valid": True, "input_digest": "same"})
    args = SimpleNamespace(
        results=tmp_path / "runs.jsonl",
        lock=tmp_path / "bench.lock",
        against="model2vec",
        threads=1,
        budget_s=10,
        rounds=3,
    )
    bench.run(args)
    bench.run(args)
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    complete = [row for row in rows if row["event"] == "series_complete"]
    assert len({row["series_id"] for row in complete}) == 2
    assert [row["valid_rounds"] for row in complete] == [3, 3]


def test_short_warmup_still_times_the_complete_corpus(bench, monkeypatch):
    calls = []

    def contender(*_args, **_kwargs):
        def encode(texts):
            calls.append(list(texts))
            return bench.np.zeros((len(texts), 2), dtype=bench.np.float32)

        return encode

    monkeypatch.setattr(bench, "contender", contender)
    args = SimpleNamespace(
        kind="fse",
        threads=1,
        model="minishlab/potion-base-8M",
        table_dtype="float32",
        fse_models=None,
        passes=1,
        warmup_texts=2,
    )
    record, vectors = bench.python_measure(args, ["a", "b", "c", "d"])
    assert calls == [["a", "b"], ["a", "b", "c", "d"]]
    assert record["first_encode_texts"] == 2
    assert vectors.shape == (4, 2)


def test_quiet_budget_expiry_persists_an_incomplete_series(bench, monkeypatch, tmp_path):
    import json

    now = [0.0]
    monkeypatch.setattr(bench.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(bench.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(bench.os, "getloadavg", lambda: (5, 1, 1))
    monkeypatch.setattr(bench, "operation", lambda *_: pytest.fail("timed at high load"))
    args = SimpleNamespace(
        results=tmp_path / "runs.jsonl",
        lock=tmp_path / "bench.lock",
        against="model2vec",
        threads=1,
        budget_s=20,
        rounds=3,
    )
    with pytest.raises(RuntimeError, match="only 0 valid rounds"):
        bench.run(args)
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    assert rows[0]["event"] == "round_complete"
    assert rows[0]["complete"] is False
    assert rows[-1]["event"] == "series_complete"
    assert rows[-1]["valid_rounds"] == 0
    assert rows[-1]["required_rounds"] == 3

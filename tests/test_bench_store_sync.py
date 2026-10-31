"""The benchmark ablations must actually remove the syncs they claim to remove."""

import importlib
import sys
from pathlib import Path

import pytest

from support.runtime import word_vectors

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX benchmark"),
    pytest.mark.skipif(
        not (BENCHMARKS / "bench_store_sync.py").exists(), reason="benchmarks absent"
    ),
]


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("bench_store_sync")


@pytest.mark.parametrize(("dtype", "files"), [("float32", 3), ("float16", 3), ("int8", 4)])
@pytest.mark.parametrize(
    "side", ["full", "fsync", "none", "deferred", "flush_model", "aa", "pend1g", "pend1g_n"]
)
def test_sync_counts_and_restoration(bench, tmp_path, monkeypatch, dtype, files, side):
    import fcntl
    from types import SimpleNamespace

    import fse
    from fse._container import _common
    from fse._store import _shard, _writer

    # Fake only the OS operations; keep real files, manifests and exclusive renames.
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda *_args: 0)
    monkeypatch.setattr(_common, "_fcntl", fcntl)
    monkeypatch.setattr(bench.run.os, "fsync", lambda _fd: None)
    encoder = fse.Encoder.from_word_vectors(word_vectors(), tokenizer="external")
    args = SimpleNamespace(
        dtype=dtype, side=side, shard_rows=4, batch=0, tmp=str(tmp_path), workers=1
    )
    sync = _shard.sync_data
    limits = (_writer.MAX_PENDING, _writer.MAX_PENDING_BYTES)
    result = bench.measure(args, encoder, ["a b"] * 8)
    counts = {k: v["calls"] for k, v in result["syncs"].items()}
    if side == "none":
        assert counts == {}
    elif side == "fsync":
        assert counts == {"fsync_file": 2 * files, "fsync_dir": 6}
    elif side in {"deferred", "flush_model"}:
        # Store header plus each shard's files and one directory per shard + root.
        assert counts == {
            "fsync_file": 1 + 2 * files + (2 if side == "flush_model" else 0),
            "fsync_dir": 3,
            "full": 3 if side == "flush_model" else 1,
        }
    else:
        assert counts == {"fsync_file": 2 * files, "fsync_dir": 2, "full": 4}
    assert result["initialisation"]["syncs"]
    assert _shard.sync_data is sync
    assert limits == (_writer.MAX_PENDING, _writer.MAX_PENDING_BYTES)
    assert list(tmp_path.iterdir()) == []
    assert result["events"]["last_commit"] <= result["events"]["encoder_flush_end"]
    assert result["events"]["encoder_flush_end"] <= result["events"]["encode_return"]
    assert result["events"]["encode_return"] <= result["events"]["close_end"]


def pairs(ratio=0.8, aa=1.02, n=3):
    return [
        {
            s: {"seconds": t, "digest": "same"}
            for s, t in (("full", 1), ("aa", aa), ("flush_model", ratio), ("none", ratio))
        }
        for _ in range(n)
    ]


def test_noise_is_symmetric_and_needs_three_pairs(bench):
    assert bench.evidence(pairs(), "flush_model", 0.9)["passes"]
    assert not bench.evidence(pairs(0.89, 1.2), "flush_model", 0.9)["passes"]
    assert not bench.evidence(pairs(0.89, 0.88), "flush_model", 0.9)["passes"]
    assert bench.evidence([], "none", 0.5)["passes"] is None
    assert bench.evidence(pairs(n=2), "none", 0.5)["passes"] is None
    mismatched = pairs()
    mismatched[1]["aa"]["digest"] = "bad"
    assert bench.evidence(mismatched, "none", 0.9)["passes"] is None


@pytest.mark.parametrize(
    ("bulk", "stream", "outcome"),
    [(0.8, 0.8, "tiers in 2.0"), (1, 0.4, "full + none in 2.0, flush in 2.1"), (1, 1, "2.1")],
)
def test_decision(bench, bulk, stream, outcome):
    cells = [
        {"name": name, "kept": pairs(ratio)}
        for name, ratio in (("C2-default", bulk), ("C3", bulk), ("C1-10000", stream))
    ]
    assert bench.decision(cells)["outcome"] == outcome
    assert bench.decision([])["outcome"] == "not decided"


def test_every_run_load_is_filtered(bench):
    round_ = {s: {"before": 3, "after": 3, "digest": "same"} for s in bench.SIDES}
    assert bench.valid_round(round_, bench.SIDES, 5)
    round_["none"]["digest"] = "wrong"
    assert not bench.valid_round(round_, bench.SIDES, 5)
    round_["none"]["digest"] = "same"
    round_["none"]["after"] = 5
    assert not bench.valid_round(round_, bench.SIDES, 5)
    assert not bench.valid_round({}, bench.SIDES, 5)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_stream_seals_each_batch_and_digest_ignores_partition(bench, tmp_path, dtype):
    from types import SimpleNamespace

    import fse

    encoder = fse.Encoder.from_word_vectors(word_vectors(), tokenizer="external")
    args = SimpleNamespace(
        dtype=dtype, side="none", shard_rows=20, batch=4, tmp=str(tmp_path), workers=1
    )
    sentences = [f"w{i}" for i in range(8)]
    streaming = bench.measure(args, encoder, sentences)
    args.batch = 0
    bulk = bench.measure(args, encoder, sentences)
    changed = bench.measure(args, encoder, sentences[::-1])
    assert changed["digest"] != bulk["digest"]
    assert streaming["shards"] == 2
    assert bulk["shards"] == 1
    assert streaming["digest"] == bulk["digest"]
    assert streaming["syncs"] == bulk["syncs"] == {}
    assert streaming["events"]["encoder_flush_end"] <= streaming["events"]["encode_return"]


def test_failed_ablation_restores_all_patches(bench):
    from fse._container import _npy
    from fse._store import _shard

    original = _shard.sync_data, _npy._sync_directory
    with pytest.raises(RuntimeError, match="failure"), bench.run.ablation("none"):
        raise RuntimeError("failure")
    assert original == (_shard.sync_data, _npy._sync_directory)


def test_budget_timeout_preserves_completed_pairs(bench, tmp_path, monkeypatch):
    import subprocess
    import time

    calls = []

    def child_result(_cmd, **_kwargs):
        calls.append(1)
        if len(calls) == 25:
            raise SystemExit("timeout") from subprocess.TimeoutExpired("child", 1)
        return {"seconds": 1, "digest": "same"}

    monkeypatch.setattr(bench.harness, "run_child", child_result)
    monkeypatch.setattr(bench.harness, "load", lambda: 3)
    args = bench.parser().parse_args(["cells", "--tmp", str(tmp_path), "--pairs", "4"])
    cell = next(c for c in bench.plan() if c["name"] == "C2-default")
    result = bench.run_cell(args, cell, time.monotonic() + 60)
    assert len(result["kept"]) == 3
    assert result["discarded"] == [{}]
    assert result["incomplete_run"]["reason"] == "shared budget expired during child"


def test_larger_full_shards_are_reported_as_simpler_answer(bench):
    rounds = pairs()
    for round_ in rounds:
        round_["single"] = {"seconds": 0.8, "digest": "same"}
        round_["large4"] = {"seconds": 1, "digest": "same"}
    result = bench.decision([{"name": "C3", "kept": rounds}])
    assert result["larger_shards"]["C3/single"]["passes"]
    assert "full with larger shards" in result["simpler_answer"]
    result = bench.decision([{"name": "C3", "kept": rounds, "correctness_failures": ["bad"]}])
    assert result["outcome"] == "not decided"


def test_timeout_removes_the_killed_child_store(bench, tmp_path, monkeypatch):
    import subprocess
    import time

    def interrupted_child(cmd, **_options):
        root = Path(cmd[cmd.index("--tmp") + 1])
        store = root / "left-by-killed-child"
        store.mkdir()
        (store / "vectors.npy").write_bytes(b"partial store")
        raise SystemExit("timeout") from subprocess.TimeoutExpired("child", 1)

    monkeypatch.setattr(bench.harness, "run_child", interrupted_child)
    monkeypatch.setattr(bench.harness, "load", lambda: 3)
    args = bench.parser().parse_args(["cells", "--tmp", str(tmp_path)])
    result = bench.run_cell(args, bench.plan()[0], time.monotonic() + 10)
    assert result["incomplete_run"]["reason"] == "shared budget expired during child"
    assert list(tmp_path.iterdir()) == []

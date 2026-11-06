"""Two stores written by one stream: different lengths, a missing partner, a history that does
not match, and the cut of a batch at the records each store holds."""

import numpy as np
import pytest

import fse
from fse._store._layout import shard_name
from support import positions_heads as ph
from support.positions import stream
from support.resume_heads import HEADS, drop_tail, score_sink, scores_of, vector_sink

TEXTS = ph.TEXTS[:320]


def _resume(directory, size, **kwargs):
    return ph.resume(directory, size, texts=TEXTS, rows=20, **kwargs)


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    directory = tmp_path_factory.mktemp("whole")
    ph.run(directory, stream(TEXTS, 20), rows=20)
    return ph.stores(directory)


def _cut(directory, vector_shards, score_shards, rows=20):
    drop_tail(directory / "v", vector_shards, rows=rows)
    drop_tail(directory / "s", score_shards, rows=rows)


@pytest.mark.parametrize(("vectors", "scores"), [(3, 4), (4, 3), (5, 5), (0, 6), (7, 0)])
@pytest.mark.parametrize("size", [50, 17])
def test_two_stores_of_any_length_continue_with_the_rows_of_one_run(
    tmp_path, whole, vectors, scores, size
):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=16)
    _cut(tmp_path, vectors, scores)
    shorter = "v" if vectors <= scores else "s"
    _resume(tmp_path, size, by=shorter)
    got = ph.stores(tmp_path)
    assert all(np.array_equal(a, b) for a, b in zip(got[:2], whole[:2], strict=True))


def test_a_batch_is_cut_at_the_rows_each_store_holds(tmp_path, whole):
    # the stores hold 60 and 80 rows; the shorter ends its last shard in the batch of the
    # record 40, so the stream is replayed from o40 with the skips 20 and 40
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=16)
    _cut(tmp_path, 3, 4)
    with vector_sink(tmp_path / "v", rows=20) as short, score_sink(tmp_path / "s", rows=20) as long:
        assert short.source_position == ("o40", 20)
        assert long.source_position == ("o60", 20)
    results, found = _resume(tmp_path, 50, by="v")
    assert found == ("o40", 20)
    # replay rows: 0-19 stored by both, 20-39 only by the scores, 40-49 by neither
    first = results[0]
    assert (first.first_row, first.n_rows) == (20, 30)
    got = ph.stores(tmp_path)
    assert all(np.array_equal(a, b) for a, b in zip(got[:2], whole[:2], strict=True))


def test_the_outputs_of_a_cut_batch_are_those_of_the_records_computed(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=16)
    _cut(tmp_path, 3, 4)
    results, _ = _resume(tmp_path, 50, by="v")
    (reference,) = HEADS.predict_batches([TEXTS[20 + 40 :]], return_vectors=True)
    # the replay starts at record 40: its first 20 are stored by both, its record 20 on computed
    computed = np.concatenate([r.outputs["quality"] for r in results])
    assert np.array_equal(computed, reference.outputs["quality"][: len(computed)])
    assert all(r.vectors is None for r in results)  # the vector sink took them


def test_a_complete_pair_gives_empty_results(tmp_path, whole):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20)
    results, _ = _resume(tmp_path, 40, by="s")
    assert [r.n_rows for r in results] == [0] * len(results)
    assert results[0].outputs["quality"].shape == (0, 1)
    assert results[0].vectors is None
    got = ph.stores(tmp_path)
    assert all(np.array_equal(a, b) for a, b in zip(got[:2], whole[:2], strict=True))


def test_the_durable_rows_are_the_least_of_both_stores(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=16)
    _cut(tmp_path, 3, 4)
    results, _ = _resume(tmp_path, 20, by="v")
    counts = [r.durable_rows for r in results]
    assert counts == sorted(counts)
    assert counts[0] == 20  # 20 records delivered, both stores hold them
    assert counts[1] <= 40


def test_a_store_that_is_missing_is_made_from_the_start_of_the_other(tmp_path, whole):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=10, scores=False)
    with vector_sink(tmp_path / "v", rows=20) as sink:
        assert sink.first_row == 200
    results = ph.run(tmp_path, stream(TEXTS, 30), rows=20)
    assert results[0].first_row == 0  # the scores start from record 0, the vectors hold 200
    got = ph.stores(tmp_path)
    assert all(np.array_equal(a, b) for a, b in zip(got[:2], whole[:2], strict=True))


def test_a_missing_store_needs_the_stream_from_the_start_of_the_other(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=10, scores=False)
    with pytest.raises(ValueError, match="must start at the position the other store started"):
        ph.run(tmp_path, stream(TEXTS, 30, start=40), rows=20)
    assert not (tmp_path / "s").exists()


def test_stores_with_another_history_are_refused(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 20), vectors=False, rows=20, stop=8, return_vectors=True)
    ph.run(tmp_path, stream(TEXTS, 30), scores=False, rows=20, stop=8)
    with pytest.raises(ValueError, match="do not share their history"):
        ph.run(tmp_path, stream(TEXTS, 20), rows=20)


def test_stores_of_other_shard_sizes_are_refused(tmp_path):
    ph.run(tmp_path / "x", stream(TEXTS, 20), vectors=False, rows=20, stop=8, return_vectors=True)
    ph.run(tmp_path / "x", stream(TEXTS, 20), scores=False, rows=40, stop=8)
    with (
        vector_sink(tmp_path / "x" / "v", rows=40) as vectors,
        score_sink(tmp_path / "x" / "s", rows=20) as scores,
        pytest.raises(ValueError, match="do not share their history"),
    ):
        HEADS.predict_batches(
            [], vector_sink=vectors, score_sink=scores, return_vectors=True, positions=True
        )


def test_stores_that_started_elsewhere_are_refused(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 20), vectors=False, rows=20, stop=8, return_vectors=True)
    ph.run(tmp_path, stream(TEXTS, 20, start=20), scores=False, rows=20, stop=8)
    with pytest.raises(ValueError, match="do not share their history"):
        ph.run(tmp_path, stream(TEXTS, 20), rows=20)


def test_a_stream_that_ends_before_the_stores_records_is_refused(tmp_path):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=16)
    _cut(tmp_path, 3, 4)
    with pytest.raises(ValueError, match="fewer than the 40 records"):
        ph.run(tmp_path, stream(TEXTS, 20, start=40, end=70), rows=20)


def test_one_store_alone_resumes_its_own_stream(tmp_path):
    ph.run(tmp_path / "ref", stream(TEXTS, 20), vectors=False, rows=20)  # the folded path
    ph.run(tmp_path, stream(TEXTS, 20), vectors=False, rows=20, stop=10)
    _, found = _resume(tmp_path, 25, by="s", vectors=False)
    assert found == ("o180", 20)
    with fse.ScoreStore.open(tmp_path / "s") as store:
        assert store.source_position == ("o305", 15)
    assert np.array_equal(scores_of(tmp_path / "s")[0], scores_of(tmp_path / "ref" / "s")[0])


def test_a_store_without_shards_starts_again_at_its_root(tmp_path, whole):
    ph.run(tmp_path, stream(TEXTS, 20), rows=20, stop=5)
    for name in ("v", "s"):
        (tmp_path / name / shard_name(0) / "manifest.json").unlink()
        assert fse.repair_store(tmp_path / name) == 0
    _, found = _resume(tmp_path, 30, by="v")
    assert found == ("o0", 0)
    got = ph.stores(tmp_path)
    assert all(np.array_equal(a, b) for a, b in zip(got[:2], whole[:2], strict=True))

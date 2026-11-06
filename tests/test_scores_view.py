"""``ScoreStore.open_view``: score stores read as one."""

import numpy as np
import pytest

from fse import ScoreStore
from support import scores as sc
from support.logs import expect_log
from support.resume_heads import run
from support.views import SIZES, score_parts

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")


def bits(array):
    return np.asarray(array).view(np.uint8).tobytes()


def joined(store, **options):
    return list(store.iter_batches(**options))


@pytest.fixture
def pair(tmp_path):
    whole, parts, _ = score_parts(tmp_path)
    with ScoreStore.open(whole) as one, ScoreStore.open_view(parts) as view:
        yield one, view


def test_what_the_view_holds(pair):
    one, view = pair
    assert len(view) == len(one) == sum(SIZES)
    assert view.path is None
    assert list(view.heads) == list(one.heads)
    assert view.encoder_fingerprint == one.encoder_fingerprint
    assert view.heads["quality"] == one.heads["quality"]


def test_the_stored_values_are_those_of_one_store(pair):
    one, view = pair
    for head in one.heads.values():
        assert bits(view.get_scores(head.name, head.stored)) == bits(
            one.get_scores(head.name, head.stored)
        )
    assert bits(view.get_scores("quality", "values", start=7, stop=33)) == bits(
        one.get_scores("quality", "values", start=7, stop=33)
    )


def test_iter_batches_run_over_the_joined_rows(pair):
    one, view = pair
    got = joined(view, block_rows=5, start=2, stop=38)
    assert got[0].store_row == 2
    assert sum(b.n_rows for b in got) == 36
    for head in one.heads:
        assert bits(np.concatenate([b.outputs[head] for b in got])) == bits(
            np.concatenate([b.outputs[head] for b in joined(one, block_rows=5, start=2, stop=38)])
        )


def test_predict_batches_with_thresholds_and_overrides(pair):
    one, view = pair
    for options in (
        {},
        {"thresholds": {"toxic": {"a": 0.2, "c": 0.9}}},
        {"thresholds": {"toxic": 0.0}},
    ):
        got = list(view.predict_batches(block_rows=6, **options))
        want = list(one.predict_batches(block_rows=6, **options))
        assert sum(b.n_rows for b in got) == sum(b.n_rows for b in want)  # blocks end at shards
        for head in one.heads:
            assert bits(np.concatenate([np.asarray(b.outputs[head]) for b in got])) == bits(
                np.concatenate([np.asarray(b.outputs[head]) for b in want])
            )


def test_document_ids_follow_the_members(tmp_path):
    _, parts, _ = score_parts(tmp_path, doc_ids=True)
    with ScoreStore.open_view(parts) as view:
        assert view.has_doc_ids
        expected = []
        for path in parts:
            with ScoreStore.open(path) as member:
                expected.extend(member.get_doc_ids().tolist())
        assert view.get_doc_ids().tolist() == expected
        assert view.get_doc_ids(start=10, stop=30).tolist() == expected[10:30]
        got = np.concatenate([b.doc_ids for b in view.iter_batches(block_rows=4)])
        assert got.tolist() == expected


def test_paths_members_and_locate(tmp_path):
    _, parts, _ = score_parts(tmp_path)
    with ScoreStore.open_view(parts) as view:
        assert view.paths == tuple(parts)
        assert [m["first_row"] for m in view.members] == [0, 13, 22]
        assert [m["rows"] for m in view.members] == list(SIZES)
        assert view.source is None
        member, row = view.locate([0, 13, 21, 39])
        assert member.tolist() == [0, 1, 1, 2]
        assert row.tolist() == [0, 0, 8, 17]
        with pytest.raises(ValueError, match="40 rows"):
            view.locate([40])


def test_thresholds_must_agree_across_the_boundaries(tmp_path):
    _, parts, _ = score_parts(tmp_path, limits=(0.5, 0.5, 0.7))
    with pytest.raises(ValueError, match="thresholds differ"):
        ScoreStore.open_view(parts)
    with pytest.raises(ValueError, match="thresholds differ"):
        ScoreStore.open_view([parts[2], parts[0]])
    with ScoreStore.open_view(parts[:2]) as view:
        assert len(view) == 22


def test_what_the_stores_must_share(tmp_path):
    _, parts, _ = score_parts(tmp_path, sizes=(6, 7))
    for name, other in (
        ("heads", sc.info(sc.head("other"))),
        ("paths", sc.info(sc.head(path="normalized"), sc.toxic(start=1))),
        ("doc ids", sc.default_info(True)),
    ):
        path = tmp_path / name.replace(" ", "_")
        sc.make_scores(path, other, sizes=(3,))
        with pytest.raises(ValueError, match=r"heads|path|doc"):
            ScoreStore.open_view([parts[0], path])


def test_a_view_has_no_refresh_and_close_closes_the_stores(tmp_path):
    _, parts, _ = score_parts(tmp_path, sizes=(6, 7))
    view = ScoreStore.open_view(parts)
    members = view._view.stores
    with pytest.raises(ValueError, match="does not refresh"):
        view.refresh()
    view.close()
    assert all(member._closed for member in members)
    with pytest.raises(RuntimeError, match="closed"):
        view.refresh()
    with pytest.raises(RuntimeError, match="closed"):
        list(view.iter_batches())


def test_verify_and_the_event(tmp_path):
    _, parts, _ = score_parts(tmp_path, sizes=(6, 7))
    with expect_log("scores.view_opened", stores=2, rows=13, mode="list"):
        view = ScoreStore.open_view(parts)
    with view, expect_log("scores.verified", rows=6), expect_log("scores.verified", rows=7):
        view.verify()


def test_the_parts_of_a_resumable_score_run_are_read_in_the_order_of_the_source(tmp_path):
    source = [f"w{i % 200} w{(i * 7) % 200}" for i in range(49_252)]  # three parts
    run(tmp_path / "whole", source, vectors=False, rows=8192)
    for k in range(3):
        run(tmp_path / f"part{k}", source, vectors=False, rows=4096, shard=(k, 3))
    paths = [tmp_path / f"part{k}" / "s" for k in (2, 0, 1)]
    with ScoreStore.open(tmp_path / "whole" / "s") as one, ScoreStore.open_view(paths) as view:
        assert view.paths == tuple(tmp_path / f"part{k}" / "s" for k in range(3))
        assert dict(view.source) == dict(one.source)
        assert view.source["shard"] == [0, 1]
        for head in one.heads:
            assert bits(view.get_scores(head, one.heads[head].stored)) == bits(
                one.get_scores(head, one.heads[head].stored)
            )
    with ScoreStore.open_view(paths[1:]) as partial:
        assert partial.source is None
        assert [dict(m["source"])["shard"] for m in partial.members] == [[0, 3], [1, 3]]
        assert len(partial) == 2 * 16_384

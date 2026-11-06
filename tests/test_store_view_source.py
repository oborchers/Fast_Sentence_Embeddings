"""Views over the parts of one source (the mode ``source_rows@1``): the parts are read in the
order of the source, and a position is a row of the source."""

import shutil

import numpy as np
import pytest

from fse import ShardSink, VectorStore
from support.resume import ENCODER, drop_tail, write

ROWS = 49_252  # four blocks of 16,384 rows: three parts of 16,384, 16,384 and 16,484 rows
SOURCE = [f"w{i % 200} w{(i * 7) % 200}" for i in range(ROWS)]
FIRST = (0, 16_384, 32_768)
QUERIES = np.random.default_rng(2).normal(size=(2, ENCODER.output_dim)).astype(np.float32)


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("parts")
    write(root / "whole", SOURCE, rows=8192)
    for k in range(3):
        write(root / f"part{k}", SOURCE, shard=(k, 3), rows=4096)
    return root


def bits(array):
    return np.asarray(array).view(np.uint8).tobytes()


def test_the_parts_are_read_in_the_order_of_the_source(stores):
    paths = [stores / "part2", stores / "part0", stores / "part1"]
    with VectorStore.open(stores / "whole") as one, VectorStore.open_view(paths) as view:
        assert view.paths == (stores / "part0", stores / "part1", stores / "part2")
        assert len(view) == len(one) == ROWS
        got, want = view.search(QUERIES, 12, mode="exact"), one.search(QUERIES, 12, mode="exact")
        assert bits(got.ids) == bits(want.ids)
        assert bits(got.scores) == bits(want.scores)
        assert bits(view.get_vectors(start=16_000, stop=17_000)) == bits(
            one.get_vectors(start=16_000, stop=17_000)
        )


def test_the_source_of_a_view_of_all_parts(stores):
    with VectorStore.open(stores / "whole") as one:
        whole = dict(one.source)
    paths = [stores / f"part{k}" for k in range(3)]
    with VectorStore.open_view(paths) as view:
        assert dict(view.source) == {
            "mode": "source_rows@1",
            "fingerprint": whole["fingerprint"],
            "form": "strings",
            "first_row": 0,
            "rows": ROWS,
            "shard": [0, 1],
        }
        assert whole["shard"] == [0, 1]
        assert [dict(m["source"])["shard"] for m in view.members] == [[0, 3], [1, 3], [2, 3]]
        assert [dict(m["source"])["first_row"] for m in view.members] == list(FIRST)
        assert [m["first_row"] for m in view.members] == list(FIRST)
        member, row = view.locate([0, 16_383, 16_384, 32_768, ROWS - 1])
        assert member.tolist() == [0, 0, 1, 2, 2]
        assert row.tolist() == [0, 16_383, 0, 0, 16_483]


def test_a_view_of_one_whole_store_is_that_store(stores):
    with VectorStore.open_view([stores / "whole"]) as view:
        assert view.source["shard"] == [0, 1]
        assert len(view) == ROWS


def test_every_part_once_and_finished(stores, tmp_path):
    parts = [stores / f"part{k}" for k in range(3)]
    with pytest.raises(ValueError, match="part 2 is missing"):
        VectorStore.open_view(parts[:2])
    with pytest.raises(ValueError, match="1 of the 3 parts"):
        VectorStore.open_view(parts[:1])
    copy = tmp_path / "copy"
    shutil.copytree(parts[1], copy)
    with pytest.raises(ValueError, match="part 1 of 3 is given twice"):
        VectorStore.open_view([parts[0], parts[1], copy, parts[2]])
    unfinished = tmp_path / "unfinished"
    shutil.copytree(parts[2], unfinished)
    drop_tail(unfinished, 2, rows=4096)
    with pytest.raises(ValueError, match="finish it"):
        VectorStore.open_view([parts[0], parts[1], unfinished])


def test_parts_of_another_source_or_cut_are_refused(stores, tmp_path):
    parts = [stores / f"part{k}" for k in range(3)]
    other = tmp_path / "other"
    write(other, SOURCE[:-1], shard=(2, 3), rows=4096)
    with pytest.raises(ValueError, match="another"):
        VectorStore.open_view([parts[0], parts[1], other])
    halves = tmp_path / "halves"
    write(halves, SOURCE, shard=(1, 2), rows=4096)
    with pytest.raises(ValueError, match="part of 2 parts"):
        VectorStore.open_view([parts[0], halves])


def test_stores_without_a_source_keep_the_order_given(stores, tmp_path):
    plain = tmp_path / "plain"
    with ShardSink(plain, ENCODER, shard_rows=3) as sink:
        ENCODER.encode(["w1 w2"] * 5, sink=sink, workers=1)
    with VectorStore.open_view([plain, stores / "part1"]) as view:
        assert view.paths == (plain, stores / "part1")
        assert view.source is None
        assert [dict(m["source"] or {}).get("shard") for m in view.members] == [None, [1, 3]]
        assert len(view) == 5 + 16_384

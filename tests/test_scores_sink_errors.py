"""ScoreSink: what is refused before a text is read."""

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import scores as sc

PATHS = {"quality": "folded", "toxic": "folded", "topic": "folded"}


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"store": "labels"}, ValueError, "store must be one of"),
        ({"store": "proba"}, ValueError, "head 'quality' cannot store 'proba'"),
        ({"store": {"nope": "decision"}}, ValueError, "names the head 'nope'"),
        ({"durability": "sometimes"}, Exception, "durability"),
        ({"shard_rows": 0}, Exception, "shard_rows"),
        ({"doc_ids": "yes"}, Exception, "doc_ids"),
    ],
)
def test_wrong_arguments_raise_before_anything_is_created(tmp_path, kwargs, error, message):
    with pytest.raises(error, match=message):
        fse.ScoreSink(tmp_path / "s", sh.group(), **kwargs)
    assert not (tmp_path / "s").exists()


def test_the_heads_must_be_a_group(tmp_path):
    with pytest.raises(TypeError, match=r"fse\.Heads"):
        fse.ScoreSink(tmp_path / "s", sh.group()["quality"])
    with pytest.raises(TypeError, match=r"fse\.Heads"):
        fse.ScoreSink(tmp_path / "s", {"quality": 1})


def test_a_missing_parent_directory_is_refused_when_the_run_attaches(tmp_path):
    sink = fse.ScoreSink(tmp_path / "missing" / "s", sh.group())
    with pytest.raises(FileNotFoundError, match="does not exist"), sink:
        sink._attach(PATHS)


def test_a_vector_store_is_not_a_score_store(tmp_path):
    from support.store import make_store

    make_store(tmp_path / "v", sizes=(2,))
    with pytest.raises(
        fse.UnsupportedArtifactError, match="sentence_vectors, not prediction_scores"
    ):
        fse.ScoreSink(tmp_path / "v", sh.group())


def test_the_paths_must_name_every_head(tmp_path):
    with fse.ScoreSink(tmp_path / "s", sh.group()) as sink:
        with pytest.raises(ValueError, match="paths name the heads"):
            sink._attach({"quality": "folded"})
        assert not (tmp_path / "s").exists()


def test_a_commit_must_be_the_target_reserved(tmp_path):
    with fse.ScoreSink(tmp_path / "s", sh.group()) as sink:
        sink._attach(PATHS)
        sink.target(0, 4)
        with pytest.raises(ValueError, match="not the target reserved"):
            sink.commit(0, np.zeros((4, 7), np.float32))
        with pytest.raises(ValueError, match="reserved, expected rows from 4"):
            sink.target(9, 2)


def test_document_ids_must_match_the_mode_and_the_rows(tmp_path):
    with fse.ScoreSink(tmp_path / "s", sh.group(), doc_ids=True) as sink:
        sink._attach(PATHS)
        target = sink.target(0, 4)
        with pytest.raises(ValueError, match="mode of the sink"):
            sink.commit(0, target)
        with pytest.raises(ValueError, match="one ID per committed row"):
            sink.commit(0, target, doc_ids=np.arange(3))
        sink.commit(0, target, doc_ids=np.arange(4))
    with fse.ScoreSink(tmp_path / "t", sh.group()) as sink:
        sink._attach(PATHS)
        target = sink.target(0, 2)
        with pytest.raises(ValueError, match="mode of the sink"):
            sink.commit(0, target, doc_ids=np.arange(2))


def test_a_sink_with_a_store_of_other_ids_is_refused(tmp_path):
    with fse.ScoreSink(tmp_path / "s", sh.group(), doc_ids=True) as sink:
        sink._attach(PATHS)
        sc.fill(sink, 0, 2, ids=np.arange(2))
    with pytest.raises(ValueError, match="has_doc_ids"):
        fse.ScoreSink(tmp_path / "s", sh.group())

"""The ids of texts under a length rule: they are those ``Encoder.encode`` pools, the
lookups share one vocabulary, the cache is bounded, and threads asking at once get the
ids of a serial run."""

import threading

import numpy as np
import pytest

import fse
from fse._static_model._ids import Ids
from support import converted as cv
from support import static_model as sm

pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
TEXTS = sm.texts(60)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return fse.StaticModel.from_pretrained(sm.model_folder(tmp_path_factory.mktemp("m") / "m"))


def _ids(ids, rule, texts=TEXTS):
    return [row.tolist() for rows in ids.chunks(texts, rule) for row in rows]


def test_the_ids_are_those_the_encoder_pools(model):
    ids = Ids(model._rules, model._variants.vocabulary)
    chars = 7 * model.median_token_length
    want = [i.tolist() for i in cv.ids_of(model._variants.get(7, False), TEXTS)]
    assert _ids(ids, (7, chars, "drop")) == want
    assert ids.lists(TEXTS, (7, chars, "drop")) == want
    assert [len(i) for i in want] != [len(i) for i in ids.lists(TEXTS, (None, None, "drop"))]


def test_the_unknown_id_is_kept_or_dropped_by_the_rule(model):
    ids = Ids(model._rules, model._variants.vocabulary)
    unk = model.unk_token_id
    kept = ids.lists(["zzzq hello"], (None, None, "keep"))[0]
    dropped = ids.lists(["zzzq hello"], (None, None, "drop"))[0]
    assert unk in kept
    assert unk not in dropped
    assert [i for i in kept if i != unk] == dropped


def test_every_rule_is_a_lookup_of_its_own_on_the_shared_vocabulary(model):
    ids = Ids(model._rules, model._variants.vocabulary, limit=3)
    first = ids.lookup((1, 5, "drop"))
    assert ids.lookup((1, 5, "drop")) is first
    assert ids.lookup((None, None, "keep")) is not first
    for length in range(2, 12):
        ids.lookup((length, None, "drop"))
    assert len(ids) == 3
    assert ids.lookup((1, 5, "drop")) is not first  # evicted, made again
    assert model._variants.vocabulary() is model._variants.vocabulary()


def test_the_chunks_hold_one_array_per_text_the_empty_ones_included(model):
    ids = Ids(model._rules, model._variants.vocabulary)
    rows = [
        row
        for chunk in ids.chunks(["", "hello", "", "zzzq"], (None, None, "drop"))
        for row in chunk
    ]
    assert (len(rows[0]), len(rows[2])) == (0, 0)
    assert len(rows) == 4
    assert all(r.dtype == np.uint32 for r in rows)
    assert not list(ids.chunks([], (None, None, "drop")))


def test_threads_asking_at_once_get_the_ids_of_a_serial_run(model):
    ids = Ids(model._rules, model._variants.vocabulary, limit=2)
    rules = [(length, None, "drop") for length in (1, 3, 9, 27, None)]
    want = {rule: ids.lists(TEXTS, rule) for rule in rules}
    failures = []

    def work(rule):
        try:
            for _ in range(5):
                assert ids.lists(TEXTS, rule) == want[rule]
        except BaseException as err:
            failures.append(err)

    threads = [threading.Thread(target=work, args=(rule,)) for rule in rules * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures
    assert len(ids) <= 2


def test_a_text_that_is_not_one_raises(model):
    ids = Ids(model._rules, model._variants.vocabulary)
    with pytest.raises(TypeError):
        ids.lists(["a", None], (None, None, "drop"))
    with pytest.raises(TypeError, match="must be ordered"):
        ids.lists({"a"}, (None, None, "drop"))

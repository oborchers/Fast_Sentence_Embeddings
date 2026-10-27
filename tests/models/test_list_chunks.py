"""Workers read IndexedList and CIndexedList themselves.

With two or more workers, each worker copies the tokens of its own chunk of rows as UTF-8
bytes while holding the GIL, then looks them up and averages without it. Every result must
equal the JobEncoder of the producer thread, which stays for one worker and for infer.
"""

import contextlib
import random

import numpy as np
import pytest
from gensim.models import FastText, Word2Vec

import fse
from fse.inputs import CIndexedList, IndexedList
from fse.models import average
from fse.models.average import CY_ROUTINES, Average

pytestmark = pytest.mark.skipif(not CY_ROUTINES, reason="list chunks need the extension")

WORDS = ["the", "dog", "is", "good", "ä", "東京", "naïve", "\U0001f600"]


@pytest.fixture
def w2v():
    model = Word2Vec(min_count=1, vector_size=12, seed=1, workers=1)
    model.build_vocab([WORDS * 3, WORDS[:4]])
    model.wv.vectors[:] = np.random.default_rng(0).random(model.wv.vectors.shape, dtype=np.float32)
    return model


@pytest.fixture
def ft():
    model = FastText(min_count=1, vector_size=12, seed=1, workers=1, bucket=500)
    model.build_vocab([WORDS * 3, WORDS[:4]])
    rng = np.random.default_rng(1)
    model.wv.vectors_vocab[:] = rng.random(model.wv.vectors_vocab.shape, dtype=np.float32)
    model.wv.vectors_ngrams[:] = rng.random(model.wv.vectors_ngrams.shape, dtype=np.float32)
    return model


@pytest.fixture
def small_chunks(monkeypatch):
    """Chunks of 7 rows, so a corpus has many chunks and a short last one."""
    monkeypatch.setattr(average, "_LIST_CHUNK_MIN_ROWS", 7)
    monkeypatch.setattr(average, "_LIST_CHUNK_MAX_ROWS", 7)


def _sentences(n=400, seed=0):
    """Token lists with unknown words, non-ASCII words and every 25th sentence empty."""
    rng = random.Random(seed)
    return [
        []
        if i % 25 == 0
        else [rng.choice([*WORDS, "unknown", "wörter"]) for _ in range(rng.randrange(1, 9))]
        for i in range(n)
    ]


def _spy(monkeypatch):
    """Record whether each training ran list chunks."""
    calls = []
    original = average.run_list_chunk

    def spy(*args):
        calls.append(args[5] - args[4])
        return original(*args)

    monkeypatch.setattr(average, "run_list_chunk", spy)
    return calls


def _producer_only(model):
    model._list_plan_for = lambda sentences, statistics: None
    return model


MODELS = [
    (fse.Average, {}),
    (fse.SIF, {"components": 2}),
    (fse.uSIF, {"length": 5, "components": 2}),
]


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize(("cls", "kwargs"), MODELS, ids=["Average", "SIF", "uSIF"])
@pytest.mark.parametrize("vectors", ["w2v", "ft"])
@pytest.mark.parametrize("wv_dtype", ["float32", "int8"])
@pytest.mark.parametrize("sv_dtype", ["float32", "float16"])
@pytest.mark.parametrize("workers", [2, 4])
def test_list_chunks_equal_the_job_encoder(
    request, monkeypatch, cls, kwargs, vectors, wv_dtype, sv_dtype, workers
):
    wv = request.getfixturevalue(vectors)
    sentences = _sentences()
    calls = _spy(monkeypatch)
    model = cls(wv, workers=workers, wv_dtype=wv_dtype, sv_dtype=sv_dtype, **kwargs)
    counts = model.train(IndexedList(sentences))
    # Every row once, or twice when float16 windows first sum the Gram matrix.
    assert sum(calls) in (len(sentences), 2 * len(sentences))
    reference = _producer_only(
        cls(wv, workers=workers, wv_dtype=wv_dtype, sv_dtype=sv_dtype, **kwargs)
    )
    assert reference.train(IndexedList(sentences)) == counts
    assert np.array_equal(np.asarray(model.sv.vectors), np.asarray(reference.sv.vectors))
    if cls is not fse.Average:
        assert np.array_equal(model.svd_res[1], reference.svd_res[1])


@pytest.mark.usefixtures("small_chunks")
def test_list_chunks_append_with_update(monkeypatch, w2v):
    first, second = _sentences(100), _sentences(60, seed=1)
    calls = _spy(monkeypatch)
    model = fse.SIF(w2v, components=2, workers=3)
    model.train(IndexedList(first))
    model.train(IndexedList(second), update=True)
    assert sum(calls) == len(first) + len(second)
    reference = _producer_only(fse.SIF(w2v, components=2, workers=3))
    reference.train(IndexedList(first))
    reference.train(IndexedList(second), update=True)
    assert np.array_equal(model.sv.vectors, reference.sv.vectors)


@pytest.mark.usefixtures("small_chunks")
@pytest.mark.parametrize(
    "index",
    [
        lambda n: list(range(n)),
        lambda n: list(range(n - 1, -1, -1)),
        lambda n: np.arange(n) // 3,
    ],
    ids=["positions", "permuted", "n-to-1-ndarray"],
)
def test_c_indexed_list_reads_its_indices(monkeypatch, w2v, index):
    sentences = _sentences(200)
    custom = index(len(sentences))
    calls = _spy(monkeypatch)
    model = fse.Average(w2v, workers=3)
    counts = model.train(CIndexedList(sentences, custom_index=custom))
    assert sum(calls) == len(sentences)
    reference = _producer_only(fse.Average(w2v, workers=3))
    assert reference.train(CIndexedList(sentences, custom_index=custom)) == counts
    if len(set(np.asarray(custom).tolist())) == len(custom):
        assert np.array_equal(model.sv.vectors, reference.sv.vectors)
    else:
        # Several sentences on one row are added in any order: float32 rounding only.
        np.testing.assert_allclose(model.sv.vectors, reference.sv.vectors, rtol=1e-6, atol=1e-7)


class _Words(list):
    pass


class _Word(str):
    pass


class _Reversed(IndexedList):
    def __getitem__(self, i):
        return (self.items[i][::-1], i)


@pytest.mark.parametrize(
    ("make", "workers"),
    [
        (lambda s: IndexedList(s), 1),
        (lambda s: IndexedList([_Words(x) for x in s]), 2),
        (lambda s: IndexedList([[_Word(w) for w in x] for x in s]), 2),
        (lambda s: _Reversed(s), 2),
        (lambda s: IndexedList(np.array(s, dtype=object)), 2),
    ],
    ids=["one-worker", "list-subclass", "str-subclass", "own-getitem", "ndarray"],
)
def test_other_inputs_keep_the_job_encoder(monkeypatch, w2v, make, workers):
    sentences = [x for x in _sentences(60) if x]
    calls = _spy(monkeypatch)
    model = fse.Average(w2v, workers=workers)
    model.train(make(sentences))
    assert calls == []
    assert "_list_plan" not in model.__dict__


def test_infer_keeps_the_job_encoder(monkeypatch, w2v):
    calls = _spy(monkeypatch)
    model = fse.Average(w2v, workers=4)
    model.train(IndexedList(_sentences(50)))
    trained = len(calls)
    model.infer(IndexedList(_sentences(50)))
    assert len(calls) == trained


def _change_after_scan(model, change):
    scan = model.scan_sentences

    def scan_then_change(*args, **kwargs):
        statistics = scan(*args, **kwargs)
        change()
        return statistics

    model.scan_sentences = scan_then_change


@pytest.mark.parametrize(
    ("change", "error", "message", "positions"),
    [
        (lambda s, idx: s.__setitem__(3, ("the", "dog")), RuntimeError, "no longer a list", True),
        (lambda s, idx: s[3].append(_Word("dog")), RuntimeError, "no longer holds only str", True),
        (lambda s, idx: s[3].append(3), RuntimeError, "no longer holds only str", True),
        # Indices that are not positions: the rows are not prefilled per chunk.
        (lambda s, idx: idx.__setitem__(3, 10**6), IndexError, "out of range", False),
        (lambda s, idx: idx.__setitem__(3, 4), RuntimeError, "but the scan found", True),
    ],
    ids=["tuple", "str-subclass", "int-token", "index-range", "index-moved"],
)
def test_a_list_changed_after_the_scan_raises(w2v, change, error, message, positions):
    sentences = [["the", "dog"] for _ in range(20)]
    custom = list(range(20)) if positions else list(range(19, -1, -1))
    model = fse.Average(w2v, workers=2)
    _change_after_scan(model, lambda: change(sentences, custom))
    with pytest.raises(error, match=message):
        model.train(CIndexedList(sentences, custom_index=custom))
    assert "_list_plan" not in model.__dict__


@pytest.mark.usefixtures("small_chunks")
def test_a_list_that_shrinks_after_the_scan_raises(w2v):
    sentences = [["the", "dog"] for _ in range(20)]
    model = fse.Average(w2v, workers=2)
    _change_after_scan(model, lambda: sentences.__delitem__(slice(10, None)))
    with pytest.raises(RuntimeError):
        model.train(IndexedList(sentences))


def test_run_list_chunk_validates_its_arguments(w2v):
    from fse.models.average_inner import KernelContext, VocabTable, run_list_chunk

    model = Average(w2v)
    model.prep.prepare_vectors(sv=model.sv, total_sentences=3)
    context = KernelContext(model, model.sv.vectors)
    table = VocabTable(model.wv.key_to_index)
    scratch = model._get_thread_working_mem()[0]
    items = [["the"], ["dog"], []]
    with pytest.raises(ValueError, match="outside the 3 items"):
        run_list_chunk(context, table, items, None, 0, 4, 0, scratch)
    with pytest.raises(ValueError, match="outside the 3 items"):
        run_list_chunk(context, table, items, None, 2, 1, 0, scratch)
    with pytest.raises(ValueError, match="offset must be non-negative"):
        run_list_chunk(context, table, items, None, 0, 3, -1, scratch)
    with pytest.raises(IndexError, match="out of range"):
        run_list_chunk(context, table, items, None, 0, 3, 1, scratch)
    assert run_list_chunk(context, table, items, None, 0, 3, 0, scratch) == (2, 2)


@pytest.mark.usefixtures("small_chunks")
def test_empty_tokens_and_lone_surrogates_match_the_job_encoder(monkeypatch, w2v):
    """An empty token and a word without a UTF-8 form are unknown words, as for the
    JobEncoder; FastText hashes unknown words, so both paths raise for a surrogate."""
    sentences = [["", "the", "\ud800", "dog"], [""], ["\ud800"], ["is", "", "good"]] * 30
    calls = _spy(monkeypatch)
    model = fse.Average(w2v, workers=2)
    counts = model.train(IndexedList(sentences))
    assert calls
    reference = _producer_only(fse.Average(w2v, workers=2))
    assert reference.train(IndexedList(sentences)) == counts
    assert np.array_equal(model.sv.vectors, reference.sv.vectors)


def test_fasttext_raises_for_a_lone_surrogate_on_both_paths(ft):
    sentences = [["the", "\ud800"]] * 20
    for model in (fse.Average(ft, workers=2), _producer_only(fse.Average(ft, workers=2))):
        with pytest.raises(UnicodeEncodeError):
            model.train(IndexedList(sentences))


def test_a_vocabulary_without_utf8_keeps_the_job_encoder(monkeypatch):
    model = Word2Vec(min_count=1, vector_size=8, seed=1, workers=1)
    model.build_vocab([["the", "dog", "\ud800"]])
    calls = _spy(monkeypatch)
    fse.Average(model, workers=2).train(IndexedList([["the", "\ud800"]] * 10))
    assert calls == []


class _ConstantIndex(list):
    def __getitem__(self, i):
        return 0


@pytest.mark.parametrize(
    "custom",
    [_ConstantIndex([1, 0, 3, 2]), np.array([1.0, 0.0, 3.0, 2.0]), np.array([[1], [0], [3], [2]])],
    ids=["list-subclass", "float-array", "2d-array"],
)
def test_other_index_containers_keep_the_job_encoder(monkeypatch, w2v, custom):
    calls = _spy(monkeypatch)
    model = fse.Average(w2v, workers=2)
    # The scan may reject the container itself; it must not reach list chunks.
    with contextlib.suppress(TypeError, ValueError):
        model.train(CIndexedList([["the"], ["dog"], ["is"], ["good"]], custom_index=custom))
    assert calls == []


@pytest.mark.usefixtures("small_chunks")
def test_a_c_indexed_list_that_shrinks_after_the_scan_raises(w2v):
    sentences = [["the", "dog"] for _ in range(20)]
    custom = list(range(19, -1, -1))
    model = fse.Average(w2v, workers=2)

    def shrink():
        del sentences[10:]
        del custom[10:]

    _change_after_scan(model, shrink)
    with pytest.raises(RuntimeError, match="the scan found 20"):
        model.train(CIndexedList(sentences, custom_index=custom))


def test_a_model_saved_during_training_holds_no_list_plan(monkeypatch, w2v, tmp_path):
    model = fse.Average(w2v, workers=2)
    post = Average._post_train_calls

    def save_then_post(self, **kwargs):
        assert "_list_plan" in self.__dict__
        self.save(str(tmp_path / "model"))
        return post(self, **kwargs)

    monkeypatch.setattr(Average, "_post_train_calls", save_then_post)
    model.train(IndexedList([["the", "dog"]] * 10))
    assert "_list_plan" not in model.__dict__
    assert "_list_plan" not in Average.load(str(tmp_path / "model")).__dict__


def test_infer_in_a_training_hook_keeps_the_job_encoder(monkeypatch, w2v):
    """A hook that infers other sentences during training does not use the training's plan."""
    calls = _spy(monkeypatch)
    post = Average._post_train_calls
    inferred = []

    def infer_then_post(self, **kwargs):
        trained = len(calls)
        inferred.append(self.infer(IndexedList([["the"], ["dog"], ["is"]])))
        assert len(calls) == trained
        return post(self, **kwargs)

    monkeypatch.setattr(Average, "_post_train_calls", infer_then_post)
    model = fse.Average(w2v, workers=2)
    model.train(IndexedList([["the", "dog"]] * 20))
    assert calls
    assert inferred[0].shape == (3, 12)


def test_a_garbage_collection_while_encoding_fails_is_memory_safe(w2v):
    """The exception of a word without a UTF-8 form may run the garbage collector, and a
    callback may then change the sentence that is being read."""
    import gc

    sentences = [["the", "\ud800", "dog", "\ud800", "is"] for _ in range(50)]

    def shrink(phase, info):
        if phase == "start":
            for sentence in sentences:
                del sentence[1:]

    threshold = gc.get_threshold()
    gc.callbacks.append(shrink)
    gc.set_threshold(1, 1, 1)
    try:
        fse.Average(w2v, workers=2).train(IndexedList(sentences))
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(shrink)
    assert all(sentence == ["the"] for sentence in sentences)

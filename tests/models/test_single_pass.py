"""Training sets EPS and sums the Gram matrix while it writes the rows.

With index i for the i-th sentence, the kernel prefills the rows of every job and SIF and
uSIF sum the Gram matrix of the finished rows during training. Every result must be
bit-identical to the separate passes (EPS in advance, Gram matrix after training).
"""

import itertools
import random

import numpy as np
import pytest
from gensim.models import FastText, Word2Vec

import fse
from fse.inputs import CIndexedList, CorpusFile, IndexedList, TextList
from fse.models import sif, usif, utils
from fse.models._single_pass import OrderedGram
from fse.models.average import CY_ROUTINES
from fse.models.utils import gram_block_dtype, gram_matrix

pytestmark = pytest.mark.skipif(not CY_ROUTINES, reason="the single pass needs the extension")

WORDS = ["the", "dog", "is", "good", "a", "cat", "runs", "far"]


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


def _texts(lines=3000, seed=0):
    """Sentences with unknown words and every 50th sentence empty."""
    rng = random.Random(seed)
    out = []
    for i in range(lines):
        if i % 50 == 0:
            out.append("")
        else:
            out.append(
                " ".join(rng.choice([*WORDS, "unknown"]) for _ in range(rng.randrange(1, 9)))
            )
    return out


def _input(name, texts, tmp_path):
    if name == "IndexedList":
        return IndexedList([t.split() for t in texts])
    if name == "TextList":
        return TextList(texts)
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    return CorpusFile(path)


def _separate_passes(model):
    """Make train take the passes of 1.1.0 before the single pass: no identity indices."""
    scan = model.scan_sentences

    def without_identity(*args, **kwargs):
        statistics = scan(*args, **kwargs)
        statistics["identity_indices"] = False
        return statistics

    model.scan_sentences = without_identity
    return model


def _same(a, b):
    assert np.array_equal(a.sv.vectors, b.sv.vectors)
    for name in ("svd_res", "svd_weights"):
        left, right = getattr(a, name, None), getattr(b, name, None)
        if isinstance(left, tuple):
            for x, y in zip(left, right, strict=True):
                assert np.array_equal(x, y)
        elif isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        else:
            assert left == right


@pytest.mark.parametrize("model_name", ["Average", "SIF", "uSIF"])
@pytest.mark.parametrize("input_name", ["IndexedList", "TextList", "CorpusFile"])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("vectors", ["w2v", "ft"])
def test_single_pass_is_bit_identical(
    request, monkeypatch, tmp_path, model_name, input_name, workers, vectors
):
    # Small blocks, so blocks end inside jobs and jobs span several blocks.
    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 97)
    wv = request.getfixturevalue(vectors)
    texts = _texts()
    kwargs = {"components": 3} if model_name != "Average" else {}
    cls = getattr(fse, model_name)
    single = cls(wv, workers=workers, **kwargs)
    single.batch_words = 300  # many jobs
    separate = _separate_passes(cls(wv, workers=workers, **kwargs))
    separate.batch_words = 300
    assert single.train(_input(input_name, texts, tmp_path)) == separate.train(
        _input(input_name, texts, tmp_path)
    )
    _same(single, separate)


@pytest.mark.parametrize("cls", [fse.SIF, fse.uSIF])
def test_single_pass_on_a_memmap(w2v, tmp_path, cls):
    texts = _texts()
    single = cls(w2v, components=2, workers=4, sv_mapfile_path=tmp_path / "single")
    separate = _separate_passes(
        cls(w2v, components=2, workers=4, sv_mapfile_path=tmp_path / "separate")
    )
    single.train(TextList(texts))
    separate.train(TextList(texts))
    _same(single, separate)
    assert isinstance(single.sv.vectors, np.memmap)


def _spy_passes(monkeypatch, module):
    calls = []
    original = module.compute_principal_components

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "compute_principal_components", spy)
    return calls


@pytest.mark.parametrize(("module", "cls"), [(sif, fse.SIF), (usif, fse.uSIF)])
def test_training_sums_the_gram_matrix_itself(monkeypatch, w2v, module, cls):
    calls = _spy_passes(monkeypatch, module)
    cls(w2v, components=2).train(TextList(_texts()))
    assert calls == []
    _separate_passes(cls(w2v, components=2)).train(TextList(_texts()))
    assert calls == [1]


@pytest.mark.parametrize(
    "sentences",
    [
        # Unique indices in another order: rows are not contiguous per job.
        CIndexedList([w.split() for w in _texts(200)], custom_index=list(range(199, -1, -1))),
        # Several sentences on one row.
        CIndexedList([w.split() for w in _texts(200)], custom_index=[i // 2 for i in range(200)]),
    ],
    ids=["permuted", "n-to-1"],
)
def test_other_indices_take_the_separate_passes(monkeypatch, w2v, sentences):
    calls = _spy_passes(monkeypatch, sif)
    model = fse.SIF(w2v, components=2, workers=2)
    model.train(sentences)
    assert calls == [1]
    reference = _separate_passes(fse.SIF(w2v, components=2, workers=2))
    reference.train(sentences)
    np.testing.assert_allclose(model.sv.vectors, reference.sv.vectors, rtol=1e-6, atol=1e-7)


def test_update_without_learned_components_takes_the_separate_pass(monkeypatch, w2v):
    calls = _spy_passes(monkeypatch, sif)
    model = fse.SIF(w2v, components=0)
    model.train(TextList(_texts(100)))
    model.components = 2
    model.train(TextList(_texts(100, seed=1)), update=True)
    assert calls == [1]
    reference = _separate_passes(fse.SIF(w2v, components=0))
    reference.train(TextList(_texts(100)))
    reference.components = 2
    reference.train(TextList(_texts(100, seed=1)), update=True)
    _same(model, reference)


def test_update_with_learned_components_prefills_the_new_rows(w2v):
    model = fse.SIF(w2v, components=2)
    model.train(TextList(_texts(300)))
    model.train(TextList(_texts(100, seed=1)), update=True)
    reference = _separate_passes(fse.SIF(w2v, components=2))
    reference.train(TextList(_texts(300)))
    reference.train(TextList(_texts(100, seed=1)), update=True)
    _same(model, reference)


def test_without_a_kernel_context_the_rows_are_filled_in_advance(monkeypatch, w2v):
    monkeypatch.setattr(fse.SIF, "_kernel_context", lambda self, target: None)
    model = fse.SIF(w2v, components=2)
    model.train(TextList(_texts(300)))
    reference = _separate_passes(fse.SIF(w2v, components=2))
    reference.train(TextList(_texts(300)))
    np.testing.assert_allclose(model.sv.vectors, reference.sv.vectors, rtol=1e-5, atol=1e-6)


def test_empty_sentences_keep_eps(w2v):
    model = fse.Average(w2v)
    model.train(TextList(["the dog", "", "good"]))
    assert np.all(model.sv.vectors[1] == utils.EPS)


def test_nan_in_word_vectors_raises_from_the_summed_gram_matrix(w2v):
    w2v.wv.vectors[w2v.wv.key_to_index["the"], 0] = np.nan
    with pytest.raises(ValueError, match="Gram matrix of the sentence vectors is not finite"):
        fse.SIF(w2v, components=2).train(TextList(["the dog", "is good", "the cat runs"]))


def test_fewer_sentences_in_training_than_in_the_scan_raise(w2v):
    class Shrinking(IndexedList):
        """Yields one sentence fewer once shrink is set."""

        shrink = False

        def __iter__(self):
            items = super().__iter__()
            if not self.shrink:
                return items
            return (item for i, item in enumerate(items) if i < len(self) - 1)

    sentences = Shrinking([["the"], ["dog"], ["is"]])
    model = fse.Average(w2v)
    scan = model.scan_sentences

    def scan_then_shrink(*args, **kwargs):
        statistics = scan(*args, **kwargs)
        sentences.shrink = True
        return statistics

    model.scan_sentences = scan_then_shrink
    with pytest.raises(RuntimeError, match="training wrote 2 of 3 rows"):
        model.train(sentences)


def test_a_worker_error_is_not_hidden_by_the_gram_matrix(monkeypatch, w2v):
    from fse.models import average

    calls = []

    def failing(original):
        def fail_second(*args):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")
            return original(*args)

        return fail_second

    # The producer's jobs, and the list chunks that workers read themselves (small ones).
    for name in ("run_job", "run_list_chunk"):
        monkeypatch.setattr(average, name, failing(getattr(average, name)))
    monkeypatch.setattr(average, "_LIST_CHUNK_MIN_ROWS", 10)
    monkeypatch.setattr(average, "_LIST_CHUNK_MAX_ROWS", 10)
    model = fse.SIF(w2v, components=2, workers=2)
    model.batch_words = 50
    with pytest.raises(OSError, match="disk full"):
        model.train(IndexedList([t.split() for t in _texts(500)]))


def test_prefill_rejects_jobs_with_rows_outside_the_target(w2v):
    from fse.models.average_inner import KernelContext, VocabTable, run_chunk

    model = fse.Average(w2v)
    model.prep.prepare_vectors(sv=model.sv, total_sentences=2)
    context = KernelContext(model, model.sv.vectors)
    context.prefill_rows = True
    data = b"the dog\nis\nthe\n"
    scratch = model._get_thread_working_mem()[0]
    with pytest.raises(ValueError, match="outside the 2 target rows"):
        run_chunk(context, VocabTable(w2v.wv.key_to_index), data, None, 0, len(data), 0, scratch, 3)
    with pytest.raises(ValueError, match="outside the 2 target rows"):
        run_chunk(context, VocabTable(w2v.wv.key_to_index), data, None, 0, len(data), 0, scratch)


def test_prefill_rejects_jobs_with_non_contiguous_rows(w2v):
    from fse.models.average_inner import JobEncoder, KernelContext, run_job

    model = fse.Average(w2v)
    model.prep.prepare_vectors(sv=model.sv, total_sentences=3)
    context = KernelContext(model, model.sv.vectors)
    job = JobEncoder(context, [(["the"], 0), (["dog"], 2)], 100).next_job()
    assert (job.first_row, job.contiguous) == (0, False)
    context.prefill_rows = True
    with pytest.raises(ValueError, match="not contiguous"):
        run_job(context, job, model._get_thread_working_mem()[0])


def test_jobs_covering_rows_twice_raise_before_writing(w2v):
    """The scan saw indices 0 to 3, training yields 0, 1, 0, 1: same count, rows twice."""

    class Repeating(IndexedList):
        repeat = False

        def __iter__(self):
            if not self.repeat:
                return super().__iter__()
            return iter([(["the"], 0), (["dog"], 1), (["is"], 0), (["good"], 1)])

    sentences = Repeating([["the"], ["dog"], ["is"], ["good"]])
    model = fse.Average(w2v)
    model.batch_words = 2  # two sentences per job
    scan = model.scan_sentences

    def scan_then_repeat(*args, **kwargs):
        statistics = scan(*args, **kwargs)
        sentences.repeat = True
        return statistics

    model.scan_sentences = scan_then_repeat
    with pytest.raises(RuntimeError, match="starts at row 0 instead of 2"):
        model.train(sentences)
    assert np.all(model.sv.vectors == utils.EPS)


@pytest.mark.parametrize("cls", [fse.SIF, fse.uSIF])
def test_a_failed_training_leaves_eps_rows(monkeypatch, w2v, tmp_path, cls):
    from fse.models import average

    original = average.run_chunk
    calls = []

    def fail_third(*args):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("disk full")
        return original(*args)

    monkeypatch.setattr(average, "run_chunk", fail_third)
    monkeypatch.setattr("fse.inputs._CHUNK_MIN_BYTES", 256)
    monkeypatch.setattr("fse.inputs._CHUNK_MAX_BYTES", 256)
    model = cls(w2v, components=2, workers=2, sv_mapfile_path=tmp_path / "sv")
    with pytest.raises(OSError, match="disk full"):
        model.train(TextList(_texts(500)))
    assert np.all(model.sv.vectors == utils.EPS)


def test_update_on_a_memmap_with_an_unaligned_offset(monkeypatch, w2v, tmp_path):
    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 97)
    single = fse.SIF(w2v, components=2, workers=4, sv_mapfile_path=tmp_path / "single")
    separate = _separate_passes(
        fse.SIF(w2v, components=2, workers=4, sv_mapfile_path=tmp_path / "separate")
    )
    for model in (single, separate):
        model.train(TextList(_texts(301)))
        model.train(TextList(["", *_texts(150, seed=2)]), update=True)
    _same(single, separate)


def test_a_replaced_post_train_hook_gets_no_summed_gram_matrix(monkeypatch, w2v):
    calls = _spy_passes(monkeypatch, sif)

    class Scaled(fse.SIF):
        def _post_train_calls(self, **kwargs):
            self.sv.vectors *= 2
            super()._post_train_calls(**kwargs)

    Scaled(w2v, components=2).train(TextList(_texts(200)))
    assert calls == [1]


@pytest.mark.parametrize("use_offsets", [True, False])
def test_run_chunk_rejects_a_wrong_sentence_count(w2v, use_offsets):
    from fse.models.average_inner import KernelContext, VocabTable, run_chunk

    model = fse.Average(w2v)
    model.prep.prepare_vectors(sv=model.sv, total_sentences=5)
    context = KernelContext(model, model.sv.vectors)
    context.prefill_rows = True
    table = VocabTable(w2v.wv.key_to_index)
    scratch = model._get_thread_working_mem()[0]
    if use_offsets:
        data, offsets, last = b"thedogis", np.array([0, 3, 6, 8], dtype=np.int64), 3
    else:
        data, offsets, last = b"the\ndog\nis\n", None, 11
    with pytest.raises(ValueError, match="more than its 2 sentences"):
        run_chunk(context, table, data, offsets, 0, last, 0, scratch, 2)
    with pytest.raises(ValueError, match="holds 3 sentences instead of 4"):
        run_chunk(context, table, data, offsets, 0, last, 0, scratch, 4)


@pytest.mark.parametrize("path", ["separate", "ordered"])
def test_overflow_of_opposite_signs_across_blocks_raises(monkeypatch, path):
    """+inf and -inf from two blocks add up to NaN without a numpy warning."""
    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 1)
    vectors = np.array([[1e20, 1e20], [1e20, -1e20]], dtype=np.float32)
    if path == "separate":
        with pytest.raises(ValueError, match="overflows"):
            utils.compute_principal_components(vectors, components=1)
    else:
        gram = OrderedGram(vectors)
        gram.add_rows(0, 2)
        with pytest.raises(ValueError, match="not finite"):
            utils.components_from_gram(gram.result(), 2, 1, np.float32)


@pytest.mark.parametrize("memmap", [False, True])
@pytest.mark.parametrize("where", ["pre-train", "worker", "thread start"])
def test_a_failed_update_keeps_the_old_rows(monkeypatch, w2v, tmp_path, memmap, where):
    import threading

    from fse.models import average

    kwargs = {"sv_mapfile_path": tmp_path / "sv"} if memmap else {}
    model = fse.SIF(w2v, components=2, workers=2, **kwargs)
    model.train(TextList(_texts(200)))
    old = np.array(model.sv.vectors)
    if where == "pre-train":

        def fail(*args, **kwargs):
            raise OSError("pre-train failed")

        monkeypatch.setattr(fse.SIF, "_pre_train_calls", fail)
    elif where == "worker":

        def fail(*args):
            raise OSError("worker failed")

        monkeypatch.setattr(average, "run_chunk", fail)
    else:
        original = threading.Thread.start
        workers = []

        def fail(self):
            # Only threads of _train_manager: the native scan starts threads of its own.
            target = getattr(self, "_target", None)
            if target == model._job_producer:
                raise OSError("thread start failed")
            if target == model._worker_loop:
                workers.append(self)
            return original(self)

        monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(OSError, match="failed"):
        model.train(TextList(_texts(100, seed=1)), update=True)
    if where == "thread start":
        assert len(workers) == 2
        assert not any(worker.is_alive() for worker in workers)
    assert np.array_equal(model.sv.vectors[:200], old)
    assert np.all(model.sv.vectors[200:] == utils.EPS)
    assert "_single_pass" not in model.__dict__


class TestOrderedGram:
    def _vectors(self, rows=1000, dim=6):
        return np.random.default_rng(3).random((rows, dim), dtype=np.float32)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_any_order_of_jobs_gives_gram_matrix(self, monkeypatch, seed):
        monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 64)
        vectors = self._vectors()
        bounds = sorted({0, len(vectors), *random.Random(seed).sample(range(1, 1000), 40)})
        jobs = list(itertools.pairwise(bounds))
        random.Random(seed).shuffle(jobs)
        gram = OrderedGram(vectors)
        for first, last in jobs:
            gram.add_rows(first, last)
        assert np.array_equal(gram.result(), gram_matrix(vectors, gram_block_dtype(vectors.dtype)))

    def test_waiting_products_stay_within_the_budget(self, monkeypatch):
        """A late first block holds at most max_pending products; the rest is recomputed."""
        from fse.models import _single_pass

        monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 10)
        monkeypatch.setattr(_single_pass, "PENDING_BUDGET_BYTES", 3 * 6 * 6 * 4)
        vectors = self._vectors(100)
        gram = OrderedGram(vectors)
        assert gram.max_pending == 3
        for k in range(9, 0, -1):
            gram.add_rows(10 * k, 10 * k + 10)
            assert sum(p is not None for p in gram._pending.values()) <= 3
        gram.add_rows(0, 10)
        assert np.array_equal(gram.result(), gram_matrix(vectors, np.float32))

    def test_a_budget_below_one_product_holds_none(self, monkeypatch):
        from fse.models import _single_pass

        monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 10)
        monkeypatch.setattr(_single_pass, "PENDING_BUDGET_BYTES", 10)
        vectors = self._vectors(50)
        gram = OrderedGram(vectors)
        assert gram.max_pending == 0
        gram.add_rows(10, 50)
        assert all(p is None for p in gram._pending.values())
        gram.add_rows(0, 10)
        assert np.array_equal(gram.result(), gram_matrix(vectors, np.float32))

    def test_concurrent_workers_stay_within_the_budget(self, monkeypatch):
        """While two products are being computed, other workers defer their blocks."""
        import threading
        import time

        from fse.models import _single_pass

        monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 10)
        monkeypatch.setattr(_single_pass, "PENDING_BUDGET_BYTES", 2 * 6 * 6 * 4)
        vectors = self._vectors(200)
        gram = OrderedGram(vectors)
        assert gram.max_pending == 2
        release = threading.Event()
        lock = threading.Lock()
        live, most_live = [0], [0]
        original = utils.gram_block_product

        def held_product(block):
            # Block 0 is late, so each product here needs a reservation; hold it open.
            with lock:
                live[0] += 1
                most_live[0] = max(most_live[0], live[0])
            try:
                release.wait(timeout=5)
                return original(block)
            finally:
                with lock:
                    live[0] -= 1

        monkeypatch.setattr(utils, "gram_block_product", held_product)

        def report(blocks):
            for k in blocks:
                gram.add_rows(10 * k, 10 * k + 10)

        threads = [threading.Thread(target=report, args=(range(1 + i, 20, 8),)) for i in range(8)]
        for t in threads:
            t.start()
        # Wait until two products are held open; the other workers report meanwhile.
        deadline = time.monotonic() + 5
        while most_live[0] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
        assert most_live[0] == 2
        assert len(gram._reserved) == 2
        release.set()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()
        assert most_live[0] == 2
        assert sum(p is not None for p in gram._pending.values()) <= 2
        monkeypatch.setattr(utils, "gram_block_product", original)
        gram.add_rows(0, 10)
        assert np.array_equal(gram.result(), gram_matrix(vectors, np.float32))

    def test_only_waiting_blocks_are_held(self, monkeypatch):
        monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 10)
        gram = OrderedGram(self._vectors(100))
        gram.add_rows(20, 100)  # blocks 2 to 9 wait for blocks 0 and 1
        assert len(gram._pending) == 8
        gram.add_rows(0, 20)
        assert not gram._pending
        assert gram.complete

    def test_offset_rows(self):
        vectors = self._vectors(300)
        gram = OrderedGram(vectors, first_row=100)
        gram.add_rows(100, 300)
        assert np.array_equal(gram.result(), gram_matrix(vectors[100:], np.float32))

    def test_incomplete_raises(self):
        gram = OrderedGram(self._vectors())
        gram.add_rows(0, 10)
        with pytest.raises(RuntimeError, match="0 of 1 blocks"):
            gram.result()

    def test_more_rows_than_a_block_has_raise(self):
        gram = OrderedGram(self._vectors())
        gram.add_rows(0, 995)
        with pytest.raises(RuntimeError, match="more rows of block 0 were reported"):
            gram.add_rows(0, 10)

    @pytest.mark.parametrize(("first", "last"), [(-1, 3), (5, 3), (0, 1001)])
    def test_rows_outside_raise(self, first, last):
        with pytest.raises(ValueError, match="outside the 1000 rows"):
            OrderedGram(self._vectors()).add_rows(first, last)


@pytest.mark.parametrize("cls", [fse.SIF, fse.uSIF])
def test_components_do_not_depend_on_the_workers(monkeypatch, w2v, cls):
    """With several workers BLAS gets fewer threads (utils.blas_threads) and the components
    are removed in parallel; components and vectors stay bit-identical to one worker."""
    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 97)
    models = []
    for workers in (1, 4):
        model = cls(w2v, components=3, workers=workers)
        model.batch_words = 300
        model.train(TextList(_texts()))
        models.append(model)
    _same(*models)
    inferred = [m.infer(TextList(_texts(500, seed=3))) for m in models]
    np.testing.assert_array_equal(*inferred)

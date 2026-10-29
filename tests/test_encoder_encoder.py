import pickle
import threading

import numpy as np
import pydantic
import pytest

from fse import _reference, _text
from fse._contract import Normalize
from fse._encoder import Encoder, QuantizedRows
from fse._runtime import Pipeline
from fse._wordvectors import FastTextParams, WordVectors
from support.encoder import finished
from support.runtime import expected, texts, word_vectors
from support.wordvectors import _flip_last_byte

SENTENCES = texts(700)
F32 = np.float32


def _bytes(rows):
    return b"".join(a.tobytes() for a in rows) if isinstance(rows, tuple) else rows.tobytes()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("table", ["float32", "int8"])
def test_encode_gives_the_reference_rows(table, dtype):
    encoder = finished(table)
    want = expected(encoder._spec, SENTENCES)
    got = encoder.encode(SENTENCES, dtype=dtype, workers=3)
    if dtype == "int8":
        assert isinstance(got, QuantizedRows)
    want_codes, want_scales = _reference.quantize_rows(want, dtype)
    assert _bytes(got) == want_codes.tobytes() + (
        b"" if want_scales is None else want_scales.tobytes()
    )
    assert _bytes(encoder.encode(SENTENCES, dtype=dtype, backend="numpy")) == _bytes(got)


def test_every_word_weighs_one_and_normalize_is_optional():
    wv = word_vectors(dim=6)
    plain = Encoder.from_word_vectors(wv, normalize=False)
    assert plain.steps == ()
    assert plain._spec.weights.tobytes() == np.ones(len(wv), F32).tobytes()
    rows = wv.get_vectors(np.array([1, 2]))
    mean = (_reference.fma32(1, rows[1], rows[0]) * F32(0.5))[None, :]  # (1/n) * acc
    assert plain.encode(["w1 w2"], workers=1).tobytes() == mean.tobytes()
    unit = Encoder.from_word_vectors(wv).encode(["w1 w2"], workers=1)
    assert unit.tobytes() == _reference.postprocess(mean, [Normalize()]).tobytes()


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("make", [finished, lambda: finished("int8", fasttext=True)])
def test_save_and_load_keep_the_fingerprint_and_the_rows(tmp_path, make, mmap):
    encoder = make()
    path = tmp_path / "encoder.safetensors"
    encoder.save(path)
    loaded = Encoder.load(path, mmap=mmap)
    assert loaded.fingerprint == encoder.fingerprint
    assert (loaded.steps, loaded.tokenizer, loaded.output_dim) == (
        encoder.steps,
        encoder.tokenizer,
        encoder.output_dim,
    )
    assert loaded.word_vectors.fingerprint == encoder.word_vectors.fingerprint
    assert loaded.encode(SENTENCES).tobytes() == encoder.encode(SENTENCES).tobytes()
    loaded.verify()
    copy = tmp_path / "copy.safetensors"
    loaded.save(copy)
    assert Encoder.load(copy).fingerprint == encoder.fingerprint


def test_load_takes_only_the_path(tmp_path):
    encoder = finished()
    encoder.save(tmp_path / "e.safetensors")
    with pytest.raises(pydantic.ValidationError):
        Encoder.load(tmp_path / "e.safetensors", encoder.word_vectors)
    with pytest.raises(pydantic.ValidationError):
        Encoder.load(tmp_path / "e.safetensors", word_vectors=encoder.word_vectors)


def test_an_encoder_of_loaded_word_vectors_is_saved_with_their_checksums(tmp_path):
    word_vectors(dim=6).save(tmp_path / "wv.safetensors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    encoder = Encoder.from_word_vectors(wv)
    encoder.save(tmp_path / "e.safetensors")
    assert Encoder.load(tmp_path / "e.safetensors").fingerprint == encoder.fingerprint
    _flip_last_byte(tmp_path / "wv.safetensors", "vectors")  # a mapped file sees it
    with pytest.raises(ValueError, match="does not match its checksum"):
        encoder.save(tmp_path / "copy.safetensors")
    assert not (tmp_path / "copy.safetensors").exists()


def test_fasttext_and_the_external_tokenizer():
    encoder = finished("int8", fasttext=True)
    want = expected(encoder._spec, SENTENCES)
    assert encoder.encode(SENTENCES).tobytes() == want.tobytes()
    external = finished(tokenizer="external")
    tokens = [s.split() for s in SENTENCES]
    assert external.encode(tokens).tobytes() == external.encode(SENTENCES).tobytes()


def test_from_word_vectors_gives_fasttext_its_defaults():
    wv = word_vectors(fasttext=True)
    encoder = Encoder.from_word_vectors(wv)
    assert (encoder._spec.max_ngrams, encoder._spec.oov_weight) == (40, np.float32(1))
    assert encoder._content.oov_weight.dtype == F32
    given = Encoder.from_word_vectors(wv, max_ngrams=3, oov_weight=0.5)
    assert (given._spec.max_ngrams, given._spec.oov_weight) == (3, np.float32(0.5))
    assert given.fingerprint != encoder.fingerprint


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"max_ngrams": 4}, ValueError, "only to FastText"),
        ({"oov_weight": 0.5}, ValueError, "only to FastText"),
        ({"tokenizer": "bert@1"}, ValueError, "tokenizer 'bert@1' is not supported"),
        ({"normalize": 1}, pydantic.ValidationError, "bool"),
        ({"max_ngrams": 0}, pydantic.ValidationError, "greater than 0"),
        ({"weights": np.ones(200, F32)}, pydantic.ValidationError, "weights"),
    ],
)
def test_from_word_vectors_checks_its_arguments(kwargs, error, match):
    with pytest.raises(error, match=match):
        Encoder.from_word_vectors(word_vectors(), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_ngrams": 2**24 + 1}, "max_ngrams must be at most"),
        ({"oov_weight": float("inf")}, "finite in float32"),
        ({"oov_weight": 1e300}, "finite in float32"),
    ],
)
def test_from_word_vectors_checks_the_fasttext_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Encoder.from_word_vectors(word_vectors(fasttext=True), **kwargs)


@pytest.mark.parametrize("env", [{"flush": True}, {"round_up": True}], ids=["flush", "round up"])
def test_oov_weight_is_converted_in_the_kernel_environment(env):
    """A thread that flushes subnormals or rounds up gives the oov_weight of one that does
    not: 1e-40 stays subnormal, 0.1 rounds to nearest."""
    from fse import _native

    wv = word_vectors(fasttext=True)
    want = {
        w: Encoder.from_word_vectors(wv, oov_weight=w)._content.oov_weight for w in (1e-40, 0.1)
    }
    assert want[1e-40].view(np.uint32) != 0
    saved = _native._test_fpenv_set(**env)
    try:
        got = {w: Encoder.from_word_vectors(wv, oov_weight=w)._content.oov_weight for w in want}
    finally:
        _native._test_fpenv_restore(saved)
    for w in want:
        assert got[w].tobytes() == want[w].tobytes(), w
        assert got[w].shape == ()
        assert got[w].dtype == F32


def test_from_word_vectors_takes_word_vectors():
    with pytest.raises(TypeError, match="must be WordVectors, not ndarray"):
        Encoder.from_word_vectors(np.ones((2, 3), F32))


def test_from_arrays_is_from_word_vectors_of_the_same_arrays():
    rng = np.random.default_rng(4)
    rows, ngrams = rng.standard_normal((3, 4)).astype(F32), rng.standard_normal((5, 4)).astype(F32)
    params = FastTextParams(2, 3, 5)
    got = Encoder.from_arrays(
        rows, words=["a", "b", "c"], fasttext=params, ngrams=ngrams, normalize=False, oov_weight=2.0
    )
    wv = WordVectors.from_arrays(rows, words=["a", "b", "c"], fasttext=params, ngrams=ngrams)
    want = Encoder.from_word_vectors(wv, normalize=False, oov_weight=2.0)
    assert got.fingerprint == want.fingerprint
    with pytest.raises(ValueError, match="either words or"):
        Encoder.from_arrays(rows)


def test_encoders_are_frozen():
    encoder = finished()
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(encoder)
    with pytest.raises(TypeError, match=r"Encoder\.from_word_vectors, Encoder\.from_arrays"):
        Encoder()
    with pytest.raises(AttributeError):
        encoder.tokenizer = "external"
    assert repr(encoder) == "Encoder(200 words, 6 dimensions, float32, fse_words@1, normalize@1)"
    assert repr(finished("int8", True, False, "external")) == (
        "Encoder(200 words, 6 dimensions, int8, FastText, external)"
    )
    assert (encoder.output_dim, encoder.steps) == (6, ("normalize@1",))
    for removed in ("folded", "recipe", "input_dim"):
        assert not hasattr(encoder, removed)


def test_arguments_are_checked_before_reading():
    encoder = finished()
    reads = []

    def counted():
        reads.append(1)
        yield "w1"

    with pytest.raises(ValueError, match="dtype must be one of"):
        encoder.encode(counted(), dtype="float64")
    with pytest.raises(ValueError, match="backend must be one of"):
        encoder.encode(counted(), backend="gpu")
    with pytest.raises(ValueError, match="workers must be a positive int"):
        encoder.encode(counted(), workers=0)
    with pytest.raises(TypeError, match="not one str"):
        encoder.encode("w1 w2")
    assert reads == []


def test_the_first_use_from_many_threads_builds_one_table_and_one_pipeline_per_backend(monkeypatch):
    built, pipelines = [], []
    real_table, real_pipeline = _text._native.VocabTable, Pipeline.__init__

    def table(*args):
        built.append(1)
        return real_table(*args)

    def pipeline(self, *args, **kwargs):
        pipelines.append(args[1] if len(args) > 1 else kwargs.get("backend", "cpu"))
        real_pipeline(self, *args, **kwargs)

    monkeypatch.setattr(_text._native, "VocabTable", table)
    monkeypatch.setattr(Pipeline, "__init__", pipeline)
    encoder = finished()
    want = encoder._spec
    barrier = threading.Barrier(8)
    results, errors = [], []

    def run(i):
        barrier.wait()
        try:
            results.append(
                encoder.encode(SENTENCES[:50], workers=1, backend=("cpu", "numpy")[i % 2])
            )
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert built == [1]
    assert sorted(pipelines) == ["cpu", "numpy"]
    rows = expected(want, SENTENCES[:50]).tobytes()
    assert all(r.tobytes() == rows for r in results)


def test_a_failed_first_use_can_be_retried(monkeypatch):
    encoder = finished()
    real = Pipeline.__init__
    calls = []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("no memory for the table")
        real(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "__init__", failing)
    with pytest.raises(MemoryError):
        encoder.encode(SENTENCES[:5])
    assert encoder.encode(SENTENCES[:5]).shape == (5, 6)


def test_verify_and_save_check_the_oov_weight_the_encoder_uses(tmp_path):
    path = tmp_path / "ft.safetensors"
    finished(fasttext=True).save(path)
    loaded = Encoder.load(path, mmap=True)
    # As if the copy of the spec had been read from a damaged file that was repaired since
    # (replacing a mapped file is not possible on every system).
    object.__setattr__(loaded._spec, "oov_weight", np.float32(0.7))
    with pytest.raises(ValueError, match=r"oov_weight of .* does not match its checksum"):
        loaded.verify()
    with pytest.raises(ValueError, match="changed since it was checked; nothing was written"):
        loaded.save(tmp_path / "copy.safetensors")


def test_verify_checks_the_table_on_disk(tmp_path):
    path = tmp_path / "e.safetensors"
    finished().save(path)
    loaded = Encoder.load(path, mmap=True)
    loaded.verify()
    _flip_last_byte(path, "vectors")
    with pytest.raises(ValueError, match="does not match its checksum"):
        loaded.verify()

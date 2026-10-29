import json
from pathlib import Path

import numpy as np
import pytest

import fse
from fse import _gram, _reference
from fse._stream import StreamSource
from fse._wordvectors import WordVectors
from support.runtime import expected, texts, word_vectors

F32 = np.float32
FIXTURE = Path(__file__).parent / "test_data" / "v1_reference"
ARRAYS = np.load(FIXTURE / "outputs.npz")
SENTENCES = json.loads((FIXTURE / "sentences.json").read_text(encoding="utf-8"))
WORDS = json.loads((FIXTURE / "words.json").read_text(encoding="utf-8"))["kv"]
KV = WordVectors.from_arrays(ARRAYS["kv.vectors"], words=WORDS, counts=ARRAYS["kv.counts"])


def _angles(a, b):
    """The largest principal angle between the row spaces of a and b, in degrees."""
    qa = np.linalg.qr(a.astype(np.float64).T)[0]
    qb = np.linalg.qr(b.astype(np.float64).T)[0]
    cosines = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cosines.min(), -1, 1))))


@pytest.mark.parametrize(
    ("recipe", "name"),
    [
        (fse.SIF(components=2, fold=False, normalize=False, tokenizer="external"), "SIF"),
        (
            fse.USIF(components=2, length=11, fold=False, normalize=False, tokenizer="external"),
            "uSIF",
        ),
    ],
)
def test_weights_and_components_follow_1_1_0(recipe, name):
    encoder = recipe.fit(KV, SENTENCES, workers=1)
    weights = encoder._content.weights
    assert weights.tobytes() == ARRAYS[f"{name}.float32.weights"].tobytes()
    step = encoder._content.steps[0]
    assert encoder.steps == ("remove_components@1",)
    # 2.0 pools with explicit FMA and no EPS, so the components agree closely, not bitwise.
    assert _angles(step.components, ARRAYS[f"{name}.float32.components"]) < 0.05
    if name == "uSIF":
        np.testing.assert_allclose(
            step.weights, ARRAYS["uSIF.float32.component_weights"], rtol=1e-3
        )
    rows = encoder.encode(SENTENCES)
    want = ARRAYS[f"{name}.float32.train"]
    nonempty = np.array([bool(s) for s in SENTENCES])  # 1.x filled empty rows with EPS
    np.testing.assert_allclose(rows[nonempty], want[nonempty], atol=1e-4)


def test_the_learned_usif_length_counts_every_token():
    encoder = fse.USIF(components=0, tokenizer="external").fit(KV, SENTENCES, workers=2)
    tokens = sum(len(s) for s in SENTENCES)
    assert encoder.recipe["length"] == tokens // len(SENTENCES)
    assert encoder.recipe["length"] == 17


def test_the_gram_pass_is_the_gram_matrix_of_the_weighted_rows():
    sif = {"normalize": False, "tokenizer": "external"}
    encoder = fse.SIF(components=1, fold=False, **sif).fit(KV, SENTENCES)
    rows = fse.SIF(components=0, **sif).fit(KV).encode(SENTENCES)
    gram = _gram.gram_matrix(rows, F32)
    _, want = _gram.components_from_gram(gram, len(rows), 1, F32)
    assert encoder._content.steps[0].components.tobytes() == want.tobytes()


@pytest.mark.parametrize("make", [fse.Average, fse.SIF, fse.USIF])
def test_fit_is_the_same_for_every_worker_count_and_chunk(make, monkeypatch):
    from fse._runtime import _tasks

    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    sentences = texts(4000)
    wv = word_vectors()
    counts = np.arange(1, len(wv) + 1)
    kwargs = {} if make is fse.Average else {"counts": counts}
    recipe = make(dim=4) if make is fse.Average else make(components=2, dim=4)
    prints = {recipe.fit(wv, sentences, workers=w, **kwargs).fingerprint for w in (1, 3, 8)}
    assert len(prints) == 1


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_folded_and_unfolded_agree_before_normalization(dtype):
    wv = word_vectors(dtype)
    counts = np.arange(1, len(wv) + 1)
    sentences = texts(2000)
    folded = fse.SIF(components=2, dim=6, normalize=False).fit(wv, sentences, counts=counts)
    unfolded = fse.SIF(components=2, dim=6, normalize=False, fold=False).fit(
        wv, sentences, counts=counts
    )
    assert folded.folded
    assert not unfolded.folded
    assert folded.steps == ()
    assert unfolded.steps == ("remove_components@1", "project@1")
    a, b = folded.encode(sentences), unfolded.encode(sentences)
    scale = np.abs(b).max(axis=1, keepdims=True) + 1e-30
    # float16 and int8 rows are stored again after folding (ARCHITECTURE.md).
    tolerance = {"float32": 1e-5, "float16": 2e-3, "int8": 5e-2}[dtype]
    assert (np.abs(a - b) / scale).max() < tolerance
    # Each is bit for bit the reference pipeline on its own frozen arrays.
    assert a.tobytes() == expected(folded._spec, sentences).tobytes()
    assert b.tobytes() == expected(unfolded._spec, sentences).tobytes()


def test_a_folded_fit_round_trips_and_keeps_its_provenance(tmp_path):
    wv = word_vectors("int8", fasttext=True)
    encoder = fse.USIF(components=2, length=11).fit(wv, texts(500), counts=np.ones(len(wv), int))
    assert encoder.folded
    assert encoder.steps == ("normalize@1",)
    assert encoder._content.folded_from == wv.fingerprint
    encoder.save(tmp_path / "usif.safetensors")
    loaded = fse.Encoder.load(tmp_path / "usif.safetensors")
    assert loaded.fingerprint == encoder.fingerprint
    assert loaded.encode(texts(50)).tobytes() == encoder.encode(texts(50)).tobytes()


def test_fasttext_defaults():
    wv = word_vectors(fasttext=True)
    average = fse.Average().fit(wv)
    assert average._content.oov_weight.tobytes() == np.array(1, F32).tobytes()
    assert average._content.max_ngrams == 40
    sif = fse.SIF(components=0).fit(wv, counts=np.ones(len(wv), int))
    assert sif._content.oov_weight == sif._content.weights.max()
    with pytest.raises(ValueError, match="apply only to FastText"):
        fse.Average(max_ngrams=10).fit(word_vectors())
    custom = fse.Average(max_ngrams=10, oov_weight=0.5).fit(wv)
    assert (custom._content.max_ngrams, float(custom._content.oov_weight)) == (10, 0.5)


def test_average_learns_nothing_without_dim():
    wv = word_vectors()
    encoder = fse.Average().fit(wv)
    assert encoder.steps == ("normalize@1",)
    assert encoder._content.weights is None
    assert not encoder.folded
    assert encoder.word_vectors is wv
    sentences = texts(100)
    rows = encoder.encode(sentences)
    plain = _reference.postprocess(
        expected(fse.Average(normalize=False).fit(wv)._spec, sentences), encoder._content.steps
    )
    assert rows.tobytes() == plain.tobytes()
    with pytest.raises(ValueError, match="learns nothing from texts"):
        fse.Average().fit(wv, sentences)
    with pytest.raises(ValueError, match="takes no counts"):
        fse.Average().fit(wv, counts=np.ones(len(wv), int))


def test_the_corpus_is_needed_exactly_when_something_is_learned():
    wv = word_vectors()
    counts = np.ones(len(wv), int)
    with pytest.raises(ValueError, match="learns from a corpus"):
        fse.SIF().fit(wv, counts=counts)
    with pytest.raises(ValueError, match="learns from a corpus"):
        fse.USIF(components=0).fit(wv, counts=counts)
    assert fse.USIF(components=0, length=10).fit(wv, counts=counts).steps == ("normalize@1",)
    with pytest.raises(ValueError, match="no counts: pass counts="):
        fse.SIF(components=0).fit(wv)


def test_streams_are_fitted_when_one_pass_suffices():
    wv = word_vectors()
    counts = np.ones(len(wv), int)
    sentences = texts(800)
    from_stream = fse.SIF(components=1).fit(wv, StreamSource(iter(sentences)), counts=counts)
    from_list = fse.SIF(components=1).fit(wv, sentences, counts=counts)
    assert from_stream.fingerprint == from_list.fingerprint
    reads = []

    def counted():
        reads.append(1)
        yield from sentences

    with pytest.raises(ValueError, match="reads the corpus 2 times"):
        fse.USIF(components=1).fit(wv, counted(), counts=counts)
    assert reads == []
    # A stream that only gives the length: one pass.
    assert fse.USIF(components=0).fit(wv, StreamSource(iter(sentences)), counts=counts)


def test_edge_cases_raise_clearly():
    wv = word_vectors()
    counts = np.ones(len(wv), int)
    with pytest.raises(ValueError, match="needs at least as many sentences"):
        fse.SIF(components=3).fit(wv, ["w1 w2", "w3"], counts=counts)
    with pytest.raises(ValueError, match="0 sentences and no tokens"):
        fse.USIF(components=0).fit(wv, [], counts=counts)
    with pytest.raises(ValueError, match="2 sentences and no tokens"):
        fse.USIF(components=0).fit(wv, ["", "  "], counts=counts)
    with pytest.raises(ValueError, match="no sentence vector that is not zero"):
        fse.SIF(components=1).fit(wv, ["", "unknown words only"], counts=counts)
    with pytest.raises(ValueError, match="no sentence vector that is not zero"):
        fse.Average(dim=2).fit(wv, ["zz yy"])
    with pytest.raises(ValueError, match="dim must be between 1 and 11"):
        fse.SIF(components=1, dim=12).fit(wv, texts(500), counts=counts)


def test_recipes_have_no_fit_transform():
    # fit, then encode: the one path (the combined call was removed before 2.0).
    for recipe in (fse.Average(), fse.SIF(), fse.USIF()):
        assert not hasattr(recipe, "fit_transform")


def test_valid_counts_with_a_large_sum_bound_are_accepted():
    from fse._encoder._weights import check_counts

    assert check_counts(np.array([2**62, 1], np.int64), 2).sum() == 2**62 + 1
    with pytest.raises(ValueError, match="too large"):
        check_counts(np.array([2**62, 2**62], np.int64), 2)


def test_usif_component_weights_refuse_a_variance_beyond_float32():
    from fse._encoder._weights import usif_component_weights

    with pytest.raises(ValueError, match="exceeds float32"):
        usif_component_weights(np.array([1.5e19, 1.5e19], F32))


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_rows_the_removal_nearly_cancels(dtype):
    """Before normalize the error is small relative to the size of the input row; after
    it, residues of rounding or quantization decide the direction (ARCHITECTURE.md)."""
    rng = np.random.default_rng(7)
    u = rng.standard_normal(12)
    u /= np.linalg.norm(u)
    rows = (np.outer(rng.uniform(1, 2, 200), u) + 1e-4 * rng.standard_normal((200, 12))).astype(F32)
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(200)])
    wv = wv if dtype == "float32" else wv.quantize(dtype)
    sentences = [f"w{i} w{(i * 7) % 200}" for i in range(400)]
    counts = np.ones(200, int)
    kwargs = {"components": 1, "normalize": False}
    folded = fse.SIF(**kwargs).fit(wv, sentences, counts=counts)
    unfolded = fse.SIF(fold=False, **kwargs).fit(wv, sentences, counts=counts)
    a, b = folded.encode(sentences), unfolded.encode(sentences)
    plain = fse.Average(normalize=False).fit(wv).encode(sentences)
    size = np.linalg.norm(plain, axis=1)
    error = np.linalg.norm(a - b, axis=1) / size
    assert error.max() < {"float32": 1e-6, "float16": 1e-3, "int8": 1e-2}[dtype]
    # what remains is tiny against the input: normalizing it amplifies the residues
    assert (np.linalg.norm(b, axis=1) / size).max() < 1e-2

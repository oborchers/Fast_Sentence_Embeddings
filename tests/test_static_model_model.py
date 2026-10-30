"""fse.StaticModel against model2vec 0.10.0: the rows of ``encode`` within the bound of
tests/support/converted.py, the attributes, the setters, and the bit identity with the
fse Encoder it wraps. The models are tiny and written by model2vec or by hand in the
layouts of sentence-transformers, so nothing is downloaded."""

import numpy as np
import pytest

import fse
from fse._encoder import Encoder
from fse._encoder._model2vec_rules import UNSET
from support import converted as cv
from support import model2vec as mv
from support import static_model as sm
from support import unigram as uni

model2vec = pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
TEXTS = sm.texts()


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    """A folder of each layout; the one of model2vec normalizes, the others do not (their
    config_sentence_transformers.json says nothing), and ``plain`` is a model2vec one
    that does not."""
    root = tmp_path_factory.mktemp("models")
    folders = {layout: sm.model_folder(root / layout, layout) for layout in sm.LAYOUTS}
    folders["plain"] = sm.model_folder(root / "plain", normalize=False)
    return folders


def _both(folders, layout="model2vec", **kwargs):
    folder = folders[layout]
    return (
        fse.StaticModel.from_pretrained(folder, **kwargs),
        model2vec.StaticModel.from_pretrained(folder, **kwargs),
    )


@pytest.mark.parametrize("layout", [*sm.LAYOUTS, "plain"])
def test_encode_gives_the_rows_of_model2vec(folders, layout):
    ours, theirs = _both(folders, layout)
    assert ours.normalize is theirs.normalize is (layout == "model2vec")
    want = theirs.encode(TEXTS)
    assert sm.close(ours, TEXTS, want) <= 1


@pytest.mark.parametrize("layout", sm.LAYOUTS)
def test_the_settings_of_one_call_are_those_of_model2vec(folders, layout):
    ours, theirs = _both(folders, layout)
    for max_length in (None, 1, 7, 512):
        for normalize in (None, True, False):
            want = theirs.encode(TEXTS, max_length=max_length, normalize=normalize)
            assert sm.close(ours, TEXTS, want, max_length=max_length, normalize=normalize) <= 1
    # the settings of the model are what they were before the calls
    assert (ours.max_length, ours.normalize) == (theirs.max_length, theirs.normalize)
    want = theirs.encode(TEXTS)
    assert sm.close(ours, TEXTS, want) <= 1


@pytest.mark.parametrize("layout", sm.LAYOUTS)
def test_the_arguments_of_loading_are_those_of_model2vec(folders, layout):
    for kwargs in (
        {"normalize": False},
        {"max_length": 5},
        {"max_length": None},
        {"dimensionality": 7},
        {"dimensionality": 7, "normalize": True, "max_length": 3},
        {"quantize_to": "float32", "force_download": False},
    ):
        ours, theirs = _both(folders, layout, **kwargs)
        assert (ours.dim, ours.normalize, ours.max_length) == (
            theirs.dim,
            theirs.normalize,
            theirs.max_length,
        )
        want = theirs.encode(TEXTS)
        assert sm.close(ours, TEXTS, want) <= 1


def test_a_text_is_1d_and_a_list_2d_with_the_same_rows(folders):
    ours, theirs = _both(folders)
    row = ours.encode("hello world")
    assert row.shape == theirs.encode("hello world").shape == (ours.dim,)
    assert row.dtype == np.float32
    assert row.tobytes() == ours.encode(["hello world"])[0].tobytes()
    for container in (tuple, np.array):
        rows = ours.encode(container(TEXTS[:5]))
        assert rows.tobytes() == ours.encode(TEXTS[:5]).tobytes()
    empty = ours.encode([])
    assert empty.shape == (0, ours.dim)
    assert empty.dtype == np.float32


def test_the_rows_are_those_of_the_encoder_bit_for_bit(folders):
    """The wrapper adds no arithmetic: under the model's own rules its rows are those of
    ``Encoder.from_pretrained`` of the same folder, whatever the sizes of the tasks."""
    folder = folders["model2vec"]
    ours = fse.StaticModel.from_pretrained(folder)
    encoder = Encoder.from_pretrained(folder)
    want = encoder.encode(TEXTS).view(np.uint32)
    assert ours.encode(TEXTS).view(np.uint32).tobytes() == want.tobytes()
    assert ours.encoder.fingerprint == encoder.fingerprint
    for kwargs in (
        {"use_multiprocessing": False},
        {"batch_size": 1},
        {"batch_size": 7, "multiprocessing_threshold": 0},
        {"show_progress_bar": True},
        {"device": "cpu", "convert_to_numpy": True, "convert_to_tensor": False},
    ):
        assert ours.encode(TEXTS, **kwargs).view(np.uint32).tobytes() == want.tobytes()
    # another setting of one call is the Encoder of that setting
    other = Encoder.from_pretrained(folder, max_length=4)
    got = ours.encode(TEXTS, max_length=4)
    assert got.view(np.uint32).tobytes() == other.encode(TEXTS).view(np.uint32).tobytes()


def test_setters_change_the_defaults_of_encode(folders):
    ours, theirs = _both(folders)
    ours.max_length, theirs.max_length = 3, 3
    ours.normalize, theirs.normalize = False, False
    assert (ours.max_length, ours.normalize) == (3, False)
    assert sm.close(ours, TEXTS, theirs.encode(TEXTS)) <= 1
    assert ours.encoder.steps == ()
    assert ours.encoder._content.tokenizer.max_tokens == 3
    ours.max_length, ours.normalize = None, True
    theirs.max_length, theirs.normalize = None, True
    assert ours.encoder.steps == ("normalize@1",)
    assert sm.close(ours, TEXTS, theirs.encode(TEXTS)) <= 1
    config = ours.config
    assert (config["max_length"], config["normalize"]) == (None, True)
    config["normalize"] = "changed"
    assert ours.config["normalize"] is True


def test_the_attributes_are_those_of_model2vec(folders):
    for layout in sm.LAYOUTS:
        ours, theirs = _both(folders, layout)
        assert ours.dim == theirs.dim
        assert tuple(ours.tokens) == tuple(theirs.tokens)
        assert ours.unk_token_id == theirs.unk_token_id
        assert ours.median_token_length == theirs.median_token_length
        assert ours.embedding_dtype == theirs.embedding_dtype == "float32"
        assert np.array_equal(ours.embedding, theirs.embedding)
        assert not ours.embedding.flags.writeable
        assert [ours.weights, ours.token_mapping, ours.vocabulary_quantization] == [None] * 3
        assert [theirs.weights, theirs.token_mapping, theirs.vocabulary_quantization] == [None] * 3
        assert (ours.base_model_name, ours.language) == (theirs.base_model_name, theirs.language)
        assert ours.config == theirs.config
        with pytest.raises(ValueError, match="read-only"):
            ours.embedding[0, 0] = 1.0
    assert "StaticModel(" in repr(ours)
    assert f"{ours.dim} dimensions" in repr(ours)


def test_a_unigram_model_is_that_of_model2vec(tmp_path):
    folder = mv.save_unigram_model(tmp_path / "uni", normalize=True)
    ours = fse.StaticModel.from_pretrained(folder)
    theirs = model2vec.StaticModel.from_pretrained(folder)
    texts = [*uni.corpus(300, seed=2), "", "[UNK] " * 600 + "word", "x" * 4000]
    assert ours.unk_token_id == theirs.unk_token_id
    assert ours.encoder.tokenizer == "unigram@1"
    for max_length in (UNSET, 4, None):
        want = theirs.encode(texts, **({} if max_length is UNSET else {"max_length": max_length}))
        assert sm.close(ours, texts, want, max_length=max_length) <= 1


def test_rows_past_the_token_cut_drop_the_unknown_ids_after_it(tmp_path):
    """model2vec cuts at max_length tokens and drops the unknown id afterwards, so a text
    of many unknown words keeps fewer than max_length ids; fse does the same."""
    folder = mv.save_model(tmp_path / "m", max_length=6)
    ours = fse.StaticModel.from_pretrained(folder)
    theirs = model2vec.StaticModel.from_pretrained(folder)
    texts = ["zzzq " * 5 + "hello world", "hello " * 5 + "zzzq world"]
    want = theirs.encode(texts)
    assert sm.close(ours, texts, want) <= 1
    cut = [t[: 6 * theirs.median_token_length] for t in texts]
    ids = cv.ids_of(ours.encoder, texts)
    assert [i.tolist() for i in ids] == theirs.tokenize(cut)
    assert not ids[0].size
    assert not want[0].any()

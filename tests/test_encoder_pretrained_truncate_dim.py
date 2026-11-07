"""Encoder.from_pretrained(truncate_dim=...) for Model2Vec models: the first dimensions of
the table, cut before normalize@1, recorded in fse.source; fse encoders stay refused."""

import numpy as np
import pytest

from fse._encoder import Encoder
from fse._encoder._format import Source
from support import model2vec as mv

pytest.importorskip("model2vec")

TEXTS = ["hello world", "zzz", "", "the cat sat on the mat", "ab [MASK]"]


@pytest.fixture(scope="module")
def table():
    return mv.rows(len(mv.wp.vocabulary()))


@pytest.fixture(scope="module")
def plain(tmp_path_factory, table):
    return mv.save_model(tmp_path_factory.mktemp("p") / "model", table, normalize=False)


def _bits(encoder):
    return np.ascontiguousarray(encoder.encode(TEXTS)).tobytes()


@pytest.mark.parametrize("kept", [1, 4, mv.DIM - 1])
def test_the_rows_are_the_prefix_of_the_uncut_rows(plain, kept):
    full = Encoder.from_pretrained(plain)
    cut = Encoder.from_pretrained(plain, truncate_dim=kept)
    assert cut.output_dim == kept
    assert _bits(cut) == np.ascontiguousarray(full.encode(TEXTS)[:, :kept]).tobytes()
    assert cut._content.source == Source("model2vec", None, None, (mv.DIM, kept))


@pytest.mark.parametrize("kept", [4, mv.DIM - 1])
@pytest.mark.parametrize("own", [False, True])
def test_the_cut_comes_before_normalize(tmp_path, table, kept, own):
    """The rows are those of a model whose table is the prefix, normalized after the cut,
    by the model's own rule or by the argument."""
    folder = mv.save_model(tmp_path / "full", table, normalize=own)
    prefix = mv.save_model(tmp_path / "prefix", table[:, :kept], normalize=own)
    want = Encoder.from_pretrained(prefix)
    cut = Encoder.from_pretrained(folder, truncate_dim=kept)
    assert cut.steps == want.steps == (("normalize@1",) if own else ())
    assert _bits(cut) == _bits(want)
    asked = Encoder.from_pretrained(folder, truncate_dim=kept, normalize=True)
    assert asked.steps == ("normalize@1",)
    assert _bits(asked) == _bits(Encoder.from_pretrained(prefix, normalize=True))
    dropped = Encoder.from_pretrained(folder, truncate_dim=kept, normalize=False)
    assert dropped.steps == ()
    assert _bits(dropped) == _bits(Encoder.from_pretrained(prefix, normalize=False))


@pytest.mark.parametrize("kept", ["model", None, mv.DIM])
def test_no_cut_is_the_uncut_model(plain, kept):
    full = Encoder.from_pretrained(plain)
    same = Encoder.from_pretrained(plain, truncate_dim=kept)
    assert same.fingerprint == full.fingerprint
    assert same.output_dim == mv.DIM
    assert same._content.source == Source("model2vec", None, None)
    assert _bits(same) == _bits(full)


def test_a_larger_k_raises(plain):
    with pytest.raises(ValueError, match=rf"truncate_dim {mv.DIM + 1}: the argument is more"):
        Encoder.from_pretrained(plain, truncate_dim=mv.DIM + 1)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="validation error"):
            Encoder.from_pretrained(plain, truncate_dim=bad)


def test_a_cut_model_round_trips_through_save_and_load(plain, tmp_path):
    cut = Encoder.from_pretrained(plain, truncate_dim=4)
    cut.save(tmp_path / "e.safetensors")
    for mmap in (False, True):
        loaded = Encoder.load(tmp_path / "e.safetensors", mmap=mmap)
        assert loaded.fingerprint == cut.fingerprint
        assert loaded.output_dim == 4
        assert loaded._content.source == cut._content.source
        assert _bits(loaded) == _bits(cut)
    again = Encoder.from_pretrained(tmp_path / "e.safetensors")
    assert again._content.source == Source("model2vec", None, None, (mv.DIM, 4))
    folder = cut.save_pretrained(tmp_path / "fse", license="mit")
    assert _bits(Encoder.from_pretrained(folder)) == _bits(cut)
    assert Encoder.from_pretrained(folder)._content.source == cut._content.source


def test_the_cut_works_with_the_length_rule_and_a_unigram_model(tmp_path):
    folder = mv.save_unigram_model(tmp_path / "uni")
    full = Encoder.from_pretrained(folder, normalize=False)
    cut = Encoder.from_pretrained(folder, truncate_dim=5, normalize=False, max_length=3)
    short = Encoder.from_pretrained(folder, normalize=False, max_length=3)
    assert _bits(cut) == np.ascontiguousarray(short.encode(TEXTS)[:, :5]).tobytes()
    assert full.output_dim == mv.DIM


def test_an_fse_encoder_cannot_be_cut(plain, tmp_path):
    encoder = Encoder.from_pretrained(plain)
    folder = encoder.save_pretrained(tmp_path / "fse", license="mit")
    encoder.save(tmp_path / "e.safetensors")
    for where in (folder, folder / "model.safetensors", tmp_path / "e.safetensors"):
        with pytest.raises(ValueError, match=r"truncate_dim: this model holds its own rules"):
            Encoder.from_pretrained(where, truncate_dim=4)

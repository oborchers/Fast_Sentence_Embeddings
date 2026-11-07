"""sentence-transformers models whose table is float16 (bekko-static-embedding-v1 is one):
read as an fse float16 table, unchanged, with every sum in float32 (DR-274).

The float16 rows are widened exactly, so the encoder gives the bits of the encoder of the
widened float32 table. Against sentence-transformers itself the rows are compared with its
weights made float32, within the bound of the float32 tables; its own float16 mean is only
measured. These tests need the group ``st`` for that last comparison and skip without it,
unless FSE_REQUIRE_ST is set.
"""

import os

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from support import converted as cv
from support import sentence_transformers as st
from support import wordpiece as wp

pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")
TOKENS = wp.vocabulary()
TEXTS = [
    *wp.corpus(TOKENS, count=300),
    "",
    "zzzq qqq",
    "[CLS] hello world [SEP]",
    "HELLO World",
    "中国 한 é́",
]


def _half():
    """A float16 table and its exact float32 widening."""
    half = st.table(len(TOKENS)).astype(np.float16)
    return half, half.astype(np.float32)


@pytest.fixture
def pair(tmp_path):
    half, wide = _half()
    return (
        st.write_model(tmp_path / "half", tensors={"embedding.weight": half}),
        st.write_model(tmp_path / "wide", tensors={"embedding.weight": wide}),
    )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kept", [None, 8])
def test_a_float16_table_gives_the_bits_of_its_widening(pair, normalize, kept):
    half, wide = (Encoder.from_pretrained(p, truncate_dim=kept, normalize=normalize) for p in pair)
    assert half.word_vectors.dtype == "float16"
    assert wide.word_vectors.dtype == "float32"
    assert half.output_dim == (kept or st.DIM)
    out = half.encode(TEXTS)
    assert out.dtype == np.float32
    assert out.tobytes() == wide.encode(TEXTS).tobytes()


def test_the_float16_table_is_kept_through_save_and_load(pair, tmp_path):
    half = Encoder.from_pretrained(pair[0])
    half.save(tmp_path / "h.safetensors")
    for mmap in (False, True):
        loaded = Encoder.load(tmp_path / "h.safetensors", mmap=mmap)
        assert loaded.word_vectors.dtype == "float16"
        assert loaded.fingerprint == half.fingerprint
        assert loaded.encode(TEXTS).tobytes() == half.encode(TEXTS).tobytes()
    with pytest.raises(ValueError, match="float32"):
        half.quantize("float16")


def test_the_card_of_a_saved_float16_encoder_names_its_dtype(pair, tmp_path):
    folder = Encoder.from_pretrained(pair[0]).save_pretrained(tmp_path / "out")
    assert Encoder.from_pretrained(folder).word_vectors.dtype == "float16"


def test_other_dtypes_are_still_refused(tmp_path):
    rows = st.table(len(TOKENS))
    cases = {
        "float64": rows.astype(np.float64),
        "int8": np.clip(rows * 10, -127, 127).astype(np.int8),
    }
    for name, table in cases.items():
        folder = st.write_model(tmp_path / name, tensors={"embedding.weight": table})
        with pytest.raises(UnsupportedArtifactError, match=f"{name} table"):
            Encoder.from_pretrained(folder)


def test_a_non_finite_float16_value_is_refused(tmp_path):
    half, _ = _half()
    half[3, 5] = np.inf
    folder = st.write_model(tmp_path / "m", tensors={"embedding.weight": half})
    with pytest.raises(ValueError, match="NaN or infinite"):
        Encoder.from_pretrained(folder)
    half[3, 5] = 0
    half[4, 2] = np.nan  # in a column that a cut drops
    folder = st.write_model(tmp_path / "n", tensors={"embedding.weight": half})
    with pytest.raises(ValueError, match="NaN or infinite"):
        Encoder.from_pretrained(folder, truncate_dim=2)


def test_the_rows_of_a_float16_table_must_match_the_tokens(tmp_path):
    half, _ = _half()
    folder = st.write_model(tmp_path / "m", tensors={"embedding.weight": half[:-1]})
    with pytest.raises(CorruptArtifactError, match="rows, but the tokenizer has"):
        Encoder.from_pretrained(folder)


@pytest.fixture(scope="module")
def sentence_transformers():
    if os.environ.get("FSE_REQUIRE_ST"):
        import sentence_transformers as module
    else:
        module = pytest.importorskip("sentence_transformers")
    return module


@pytest.mark.parametrize("kept", [None, 8])
def test_the_rows_are_those_of_sentence_transformers_with_float32_weights(
    sentence_transformers, tmp_path, kept
):
    from safetensors.numpy import save_file

    half, _ = _half()
    folder = st.save_model(tmp_path / "m")
    save_file({"embedding.weight": half}, folder / "model.safetensors")
    model = sentence_transformers.SentenceTransformer(str(folder), device="cpu")
    native = model.encode(TEXTS, batch_size=64, truncate_dim=kept)
    model[0].embedding.float()
    want = model.encode(TEXTS, batch_size=64, truncate_dim=kept)
    encoder = Encoder.from_pretrained(folder, truncate_dim=kept)
    worst = cv.assert_close(encoder.encode(TEXTS), want, encoder, TEXTS, False)
    assert worst <= 1
    gap = np.abs(native.astype(np.float64) - want).max()
    print(
        f"kept {kept}: worst {worst:.2e} of the bound, float16 mean of sentence-transformers "
        f"differs by at most {gap:.2e}"
    )

"""sentence-transformers models against sentence-transformers 6.2.0 itself: the ids of
their tokenizer and the rows of ``SentenceTransformer.encode`` within float32 rounding.

The models are tiny, written by sentence-transformers (tests/support/sentence_transformers.py)
and rearranged into the layouts of other versions. These tests need the group ``st``
(sentence-transformers and torch) and skip without it, unless FSE_REQUIRE_ST is set
(``make test-st`` and the CI job test-st), when a missing package fails them.
"""

import json
import os

import numpy as np
import pytest

from fse._encoder import Encoder
from support import converted as cv
from support import sentence_transformers as st
from support import wordpiece as wp

if os.environ.get("FSE_REQUIRE_ST"):  # make test-st and the CI job test-st: no skip
    import sentence_transformers
else:
    sentence_transformers = pytest.importorskip("sentence_transformers")
TOKENS = wp.vocabulary()
WORDS = [t for t in TOKENS[5:] if t.isalpha() and t.isascii()]
EDGE = [
    "",
    "zzzq qqq",
    "x" * 101,
    "[CLS] hello world [SEP]",
    "[PAD]",
    "[MASK] [UNK] [MASK]]",
    "HELLO World",
    "中国 한 é́",
    "",
]
LONG = [
    " ".join(WORDS[i % 40] for i in range(600)) + " " + WORDS[45],
    "",
    " ".join(WORDS[i % 40] for i in range(5000)) + " " + WORDS[46],
]
TEXTS = [*wp.corpus(TOKENS, count=1500), *EDGE, *LONG]


def _st(folder, **options):
    return sentence_transformers.SentenceTransformer(str(folder), device="cpu", **options)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return st.save_model(tmp_path_factory.mktemp("st") / "model")


def test_the_ids_are_those_of_the_tokenizer(saved):
    encoder = Encoder.from_pretrained(saved)
    module = _st(saved)[0]
    want = [e.ids for e in module.tokenizer.encode_batch(TEXTS, add_special_tokens=False)]
    assert [i.tolist() for i in cv.ids_of(encoder, TEXTS)] == want
    assert len(want[-1]) > 4096  # the long texts are read whole


@pytest.mark.parametrize(
    ("spelling", "subfolder", "tensor"),
    [
        (None, None, None),  # as sentence-transformers 6.2.0 writes it
        (st.LEGACY, "0_StaticEmbedding", None),  # as the target repos (ST 3.3)
        (st.V5, "0_StaticEmbedding", None),
        (st.V5, "", "embeddings"),
    ],
)
def test_the_rows_are_those_of_sentence_transformers(tmp_path, spelling, subfolder, tensor):
    folder = st.save_model(tmp_path / "m")
    if spelling is not None:
        st.relayout(folder, spelling=spelling, subfolder=subfolder, tensor=tensor)
    encoder = Encoder.from_pretrained(folder)
    want = _st(folder).encode(TEXTS, batch_size=64)
    got = encoder.encode(TEXTS)
    worst = cv.assert_close(got, want, encoder, TEXTS, False)
    assert worst <= 1
    ulps = np.abs(got.view(np.int32).astype(np.int64) - want.view(np.int32).astype(np.int64))
    print(f"{spelling}: worst {worst:.2e} of the bound, at most {ulps.max()} ulp")


@pytest.mark.parametrize("kept", [4, 8])
def test_truncate_dim_is_that_of_sentence_transformers(saved, kept):
    encoder = Encoder.from_pretrained(saved, truncate_dim=kept)
    want = _st(saved, truncate_dim=kept).encode(TEXTS)
    assert want.shape[1] == kept
    cv.assert_close(encoder.encode(TEXTS), want, encoder, TEXTS, False)
    normal = Encoder.from_pretrained(saved, truncate_dim=kept, normalize=True)
    want = _st(saved).encode(TEXTS, truncate_dim=kept, normalize_embeddings=True)
    cv.assert_close(normal.encode(TEXTS), want, normal, TEXTS, True)


def test_truncate_dim_of_the_config(tmp_path):
    folder = st.save_model(tmp_path / "m", truncate_dim=4)
    assert (
        json.loads((folder / "config_sentence_transformers.json").read_text(encoding="utf-8"))[
            "truncate_dim"
        ]
        == 4
    )
    encoder = Encoder.from_pretrained(folder)
    want = _st(folder).encode(TEXTS)
    assert encoder.output_dim == want.shape[1] == 4
    cv.assert_close(encoder.encode(TEXTS), want, encoder, TEXTS, False)


def test_a_plain_tokenizer_truncation(tmp_path):
    folder = st.save_model(tmp_path / "m", truncation={"max_length": 5})
    encoder = Encoder.from_pretrained(folder)
    assert encoder._content.tokenizer.max_tokens == 5
    want = _st(folder).encode(TEXTS)
    cv.assert_close(encoder.encode(TEXTS), want, encoder, TEXTS, False)


def test_rows_that_cancel(tmp_path):
    """Opposite rows cancel exactly in both libraries, also before fse's normalize."""
    rows = st.table(len(TOKENS))
    a, b = TOKENS.index("ab"), TOKENS.index("x")
    rows[b] = -rows[a]
    for name, table in (("zero", np.zeros_like(rows)), ("opposite", rows)):
        folder = st.save_model(tmp_path / name, table)
        texts = ["ab x", "x ab", "", "hello world"]
        for normalize in (False, True):
            encoder = Encoder.from_pretrained(folder, normalize=normalize)
            want = _st(folder).encode(texts, normalize_embeddings=normalize)
            got = encoder.encode(texts)
            cv.assert_close(got, want, encoder, texts, normalize)
            assert not got[0].any()


def test_written_by_sentence_transformers_and_refused(tmp_path):
    """A Normalize and a Dense module, as sentence-transformers writes them, are refused."""
    from sentence_transformers.base.modules import Dense, Normalize
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
    from torch import nn

    from fse._contract import UnsupportedArtifactError

    def save(name, *after):
        module = StaticEmbedding(st.tokenizer(), embedding_weights=st.table(len(TOKENS)))
        model = sentence_transformers.SentenceTransformer(modules=[module, *after])
        model.save_pretrained(str(tmp_path / name))
        return tmp_path / name

    with pytest.raises(UnsupportedArtifactError, match="Normalize module"):
        Encoder.from_pretrained(save("normalize", Normalize()))
    linear = Dense(st.DIM, 4, activation_function=nn.Identity())
    with pytest.raises(UnsupportedArtifactError, match="Dense module"):
        Encoder.from_pretrained(save("dense", linear))
    with pytest.raises(UnsupportedArtifactError, match="no static embedding model"):
        Encoder.from_pretrained(save("tanh", Dense(st.DIM, 4, activation_function=nn.Tanh())))

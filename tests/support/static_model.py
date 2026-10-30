"""Shared by the tests of fse.StaticModel: models in the folder layouts model2vec reads,
texts that cross its two cuts, and the comparison with model2vec's own ``encode``."""

import json
import shutil

import numpy as np

from fse._encoder._model2vec_rules import UNSET
from support import converted as cv
from support import model2vec as mv
from support import sentence_transformers as stm
from support import wordpiece as wp

TOKENS = wp.vocabulary()
UNK = TOKENS.index("[UNK]")
LAYOUTS = ("model2vec", "st-root", "st-nested")


def texts(count: int = 400) -> list[str]:
    """Texts of every length, the empty one, one of unknown words only, and ones that
    cross the token cut (many unknown ids before the 512th token) and the character cut."""
    long = " ".join(TOKENS[5:400]) * 3
    unknown = "[UNK] " * 700 + "hello world"
    return [*wp.corpus(TOKENS, count=count), long, unknown, "", "zzzq", "x" * 5000]


def model_folder(root, layout: str = "model2vec", *, normalize: bool = True, config=None):
    """A model of ``TOKENS`` in one of ``LAYOUTS``. A model2vec one is written by model2vec;
    the others by hand in the layout of sentence-transformers, whose
    config_sentence_transformers.json holds ``config``."""
    if layout == "model2vec":
        return mv.save_model(root, normalize=normalize)
    folder = stm.write_model(root, config=config)
    if layout == "st-root":
        for name in ("model.safetensors", "tokenizer.json"):
            shutil.move(folder / "0_StaticEmbedding" / name, folder / name)
        (folder / "0_StaticEmbedding").rmdir()
        modules = json.loads((folder / "modules.json").read_text())
        modules[0]["path"] = "."
        (folder / "modules.json").write_text(json.dumps(modules))
    return folder


def close(model, texts, want, *, max_length=UNSET, normalize=None):
    """``model.encode`` agrees with model2vec's rows ``want`` within the bound of
    tests/support/converted.py, for the settings given to ``encode``; gives the largest
    ratio of error to bound."""
    cut = model.max_length if max_length is UNSET else max_length
    norm = model.normalize if normalize is None else normalize
    got = model.encode(texts, max_length=max_length, normalize=normalize)
    encoder = model._variants.get(cut, norm)
    assert got.dtype == np.float32
    return cv.assert_close(got, want, encoder, texts, norm)


def hub_texts(model, extra=()):
    """Texts for a model of the Hub: the sentences of the STS benchmark, words sampled from
    its vocabulary, edge cases, and long texts that cross both cuts."""
    import csv
    import random
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    with open(root / "evaluation" / "sts-test.csv", encoding="utf-8", newline="") as f:
        sts = [s for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) for s in row[5:7]]
    rng = random.Random(7)
    words = rng.sample([t for t in model.tokens if not t.startswith("[")], 2000)
    edge = ["hello [MASK] [PAD] [UNK] [CLS] world [SEP]", "x" * 101, "İstanbul", "", "zzqxj"]
    long = [" ".join(rng.choice(words) for _ in range(n)) for n in (600, 5000)]
    return sts + list(extra) + wp.corpus(words, count=2000, seed=3) + edge + long


def assert_the_model_of_model2vec(ours, theirs, texts):
    """Attributes equal, and ``encode`` within the bound under the model's settings and
    under other ones for one call; gives the largest ratio of error to bound."""
    assert ours.dim == theirs.dim
    assert tuple(ours.tokens) == tuple(theirs.tokens)
    assert ours.config == theirs.config
    assert ours.unk_token_id == theirs.unk_token_id
    assert ours.median_token_length == theirs.median_token_length
    assert (ours.normalize, ours.max_length) == (theirs.normalize, theirs.max_length)
    assert (ours.base_model_name, ours.language) == (theirs.base_model_name, theirs.language)
    assert np.array_equal(ours.embedding, theirs.embedding)
    worst = close(ours, texts, theirs.encode(texts))
    for kwargs in ({"max_length": 64}, {"normalize": not ours.normalize}, {"max_length": None}):
        worst = max(worst, close(ours, texts, theirs.encode(texts, **kwargs), **kwargs))
    return worst


def assert_the_sequences_and_ids_of_model2vec(ours, theirs, texts):
    """``encode_as_sequence`` equal to model2vec's bit for bit (float32; a text without
    tokens gives a (0, dim) array, float64 there), with and without ``max_length``;
    ``tokenize`` and the ids of the tokenizer equal, also after a new
    ``max_length`` of the model; the vocabulary of the tokenizer equal."""
    for kwargs in ({}, {"max_length": 7}, {"max_length": 64}):
        got = ours.encode_as_sequence(texts, **kwargs)
        want = theirs.encode_as_sequence(texts, **kwargs)
        assert len(got) == len(want) == len(texts)
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            assert g.dtype == np.float32
            assert g.shape == w.shape, (i, kwargs)
            assert g.tobytes() == w.astype(np.float32).tobytes(), (i, kwargs)
    theirs_vocab = theirs.tokenizer.get_vocab()
    assert ours.tokenizer.get_vocab() == theirs_vocab
    assert ours.tokenizer.get_vocab_size() == theirs.tokenizer.get_vocab_size()
    assert all(ours.tokenizer.token_to_id(t) == i for t, i in list(theirs_vocab.items())[:500])
    original = ours.max_length
    for max_length in (original, 9, None, original):
        ours.max_length = theirs.max_length = max_length
        assert ours.tokenize(texts) == theirs.tokenize(texts)
        got = ours.tokenizer.encode_batch(texts, add_special_tokens=False)
        want = theirs.tokenizer.encode_batch(texts, add_special_tokens=False)
        assert [e.ids for e in got] == [e.ids for e in want]

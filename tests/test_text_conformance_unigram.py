"""The native unigram@1 gives the ids of fse._reference_unigram and of tokenizers 0.23.3.

The tokenizers of tests/support/unigram.py (the normalizer chain of potion-multilingual-128M
with a charsmap of keys of every kind) run through the native lookup: the corpus in item
and line mode, with and without both cuts, unknown ids dropped and kept, against the
reference; every code point in three contexts and the golden models against tokenizers.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from fse import _native
from fse._text import StringSource, make_lookup
from fse._tokenizer_json_unigram import parse_unigram
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors
from support import unigram as uni

tokenizers = pytest.importorskip("tokenizers")
VOCAB = uni.pieces_and_scores()
CORPUS = uni.corpus(3000, seed=5)
HF = uni.hf_tokenizer(VOCAB)
PIECES, CONFIG = parse_unigram(json.loads(HF.to_str()), Path("tokenizer.json"))
WORD_VECTORS = WordVectors.from_arrays(np.zeros((len(PIECES), 2), np.float32), words=PIECES)
REFERENCE = uni.Reference(VOCAB)
RULES = [("drop", None, None), ("keep", None, None), ("drop", 8, None), ("keep", 1, None)]
RULES += [("drop", 16, 12), ("keep", 3, 1)]
CODE_POINTS = [cp for cp in range(sys.maxunicode + 1) if not 0xD800 <= cp <= 0xDFFF]


def _spec(unknown="drop", max_tokens=None, max_chars=None, config=CONFIG):
    return TokenizerSpec("unigram@1", config, unknown, max_tokens, max_chars)


def reference_ids(texts, spec):
    rules = {"max_tokens": spec.max_tokens, "max_chars": spec.max_chars}
    return [REFERENCE.ids(t, drop_unknown=spec.unknown == "drop", **rules) for t in texts]


def _split(batches):
    return [
        b.tokens[b.offsets[s] : b.offsets[s + 1]].tolist()
        for b in batches
        for s in range(b.n_sentences)
    ]


def native_ids(texts, spec, word_vectors=WORD_VECTORS, chunk_bytes=None):
    source = StringSource(texts)
    lookup = make_lookup(word_vectors, spec)
    ws = _native.Workspace()
    return _split(source.encode(c, lookup, ws) for c in source.scan(chunk_bytes=chunk_bytes).chunks)


def assert_ids_equal(got, want, texts):
    """The comparison of every test: the same ids for every sentence, in order."""
    if len(got) != len(want):
        raise AssertionError(f"{len(got)} sentences, not {len(want)}")
    for text, a, b in zip(texts, got, want, strict=True):
        if a != b:
            raise AssertionError(f"{text!a}: {a} != {b}")


def test_the_comparison_catches_wrong_ids():
    assert_ids_equal([[1, 2], []], [[1, 2], []], ["a", "b"])
    for wrong in ([[2, 1], []], [[1], []], [[1, 2], [0]], [[1, 2]]):
        with pytest.raises(AssertionError):
            assert_ids_equal(wrong, [[1, 2], []], ["a", "b"])


@pytest.mark.parametrize(("unknown", "max_tokens", "max_chars"), RULES)
def test_native_matches_the_reference_on_the_corpus(unknown, max_tokens, max_chars):
    spec = _spec(unknown, max_tokens, max_chars)
    assert_ids_equal(native_ids(CORPUS, spec), reference_ids(CORPUS, spec), CORPUS)


def test_native_matches_tokenizers_on_the_corpus():
    assert_ids_equal(native_ids(CORPUS, _spec("keep")), uni.hf_ids(HF, CORPUS), CORPUS)


@pytest.mark.parametrize(
    "context", ["{}", "a{}b", "\uff21{}"], ids=["alone", "between", "after a key"]
)
def test_every_code_point_gives_the_ids_of_tokenizers(context):
    texts = [context.format(chr(cp)) for cp in CODE_POINTS]
    assert_ids_equal(native_ids(texts, _spec("keep")), uni.hf_ids(HF, texts), texts)


def _golden_models():
    golden = Path(__file__).parent / "test_data" / "v2" / "unigram.json"
    return json.loads(golden.read_text(encoding="utf-8"))["models"]


@pytest.mark.parametrize("model", _golden_models(), ids=lambda m: m["name"])
def test_the_golden_ids(model):
    hf = uni.hf_tokenizer(
        [tuple(p) for p in model["pieces"]],
        steps=model["steps"],
        charsmap=model["charsmap"],
        unk_id=model["unk_id"],
        added=model["added"],
    )
    pieces, config = parse_unigram(json.loads(hf.to_str()), Path("tokenizer.json"))
    wv = WordVectors.from_arrays(np.zeros((len(pieces), 1), np.float32), words=pieces)
    texts = [text for text, _ in model["cases"]]
    got = native_ids(texts, _spec("keep", config=config), wv)
    assert_ids_equal(got, [ids for _, ids in model["cases"]], texts)


def test_lines_give_the_ids_of_their_strings():
    texts = [t.replace("\n", " ") for t in CORPUS]
    spec = _spec("keep", 8)
    lookup = make_lookup(WORD_VECTORS, spec)
    data = np.frombuffer(("\n".join(texts) + "\n").encode(), np.uint8)
    batch = lookup.encode(data, None, 0, len(data), 0, _native.Workspace())
    assert_ids_equal(_split([batch]), reference_ids(texts, spec), texts)


def test_chunks_and_threads_give_the_same_ids():
    spec = _spec()
    want = native_ids(CORPUS, spec)
    assert native_ids(CORPUS, spec, chunk_bytes=1 << 12) == want
    source = StringSource(CORPUS)
    lookup = make_lookup(WORD_VECTORS, spec)
    chunks = source.scan(chunk_bytes=1 << 12).chunks

    def run(chunk):
        return source.encode(chunk, lookup, _native.Workspace())

    with ThreadPoolExecutor(4) as pool:
        assert _split(pool.map(run, chunks)) == want


def test_invalid_utf8_raises_where_python_does():
    lookup = make_lookup(WORD_VECTORS, _spec())
    for spec in (_spec(), _spec(max_tokens=2, max_chars=3)):
        lookup = make_lookup(WORD_VECTORS, spec)
        for data in (b"ab\xffcd", b"[PAD]\xc3", b"abcdef\xed\xa0\x80"):
            array = np.frombuffer(data, np.uint8)
            with pytest.raises(UnicodeDecodeError) as caught:
                lookup.encode(
                    array, np.array([0, len(data)], np.int64), 0, 1, 0, _native.Workspace()
                )
            with pytest.raises(UnicodeDecodeError) as python:
                data.decode("utf-8")
            assert caught.value.start == python.value.start


def test_long_sentences_and_pieces():
    texts = ["a" * 5000, "\u00e9" * 3000 + " [UNK] " + "zq" * 800, "\u2581" * 700, " " * 1000]
    for spec in (_spec("keep"), _spec("drop", 100, 2000)):
        assert_ids_equal(native_ids(texts, spec), reference_ids(texts, spec), texts)

"""The native wordpiece@1 gives the ids of fse._reference_wordpiece.

The corpus of tests/support/wordpiece.py and random text over all of Unicode run through
the native lookup, in item and line mode, with and without both cuts, with unknown ids
dropped and kept, and are compared with the reference per sentence. Meta-tests at the end
check that the corpus catches typical mistakes of an implementation, so the comparison
cannot pass by accident.
"""

import random

import numpy as np
import pytest

from fse import _native
from fse import _reference_wordpiece as ref
from fse import _unicode_bert as ucd
from fse._text import StringSource, make_lookup
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import WordVectors
from support import wordpiece as wp

TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}
CORPUS = wp.corpus(TOKENS)
UNICODE = wp.all_unicode()
CONFIG = WordPieceConfig(tuple(sorted((i, t) for t, i in ADDED.items())), wp.UNK_ID)
WORD_VECTORS = WordVectors.from_arrays(np.zeros((len(TOKENS), 2), np.float32), words=TOKENS)
RULES = [("drop", None, None), ("keep", None, None), ("drop", 8, None), ("keep", 1, None)]
RULES += [("drop", 16, 12), ("keep", 3, 1)]


def _spec(unknown="drop", max_tokens=None, max_chars=None, config=CONFIG):
    return TokenizerSpec("wordpiece@1", config, unknown, max_tokens, max_chars)


def reference_ids(texts, spec):
    config = spec.config
    return [
        ref.encode_ids(
            text,
            VOCAB,
            unk_id=config.unk_id,
            added={content: i for i, content in config.added_tokens},
            prefix=config.continuing_subword_prefix,
            max_input_chars_per_word=config.max_input_chars_per_word,
            max_tokens=spec.max_tokens,
            max_chars=spec.max_chars,
            drop_unknown=spec.unknown == "drop",
        )
        for text in texts
    ]


def _split(batches):
    out = []
    for batch in batches:
        for s in range(batch.n_sentences):
            out.append(batch.tokens[batch.offsets[s] : batch.offsets[s + 1]].tolist())
    return out


def native_ids(texts, spec, chunk_bytes=None):
    source = StringSource(texts)
    lookup = make_lookup(WORD_VECTORS, spec)
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


@pytest.mark.parametrize(("unknown", "max_tokens", "max_chars"), RULES[:3])
def test_native_matches_the_reference_on_random_unicode(unknown, max_tokens, max_chars):
    spec = _spec(unknown, max_tokens, max_chars)
    assert_ids_equal(native_ids(UNICODE, spec), reference_ids(UNICODE, spec), UNICODE)


def test_the_golden_ids():
    import json
    from pathlib import Path

    golden = json.loads(
        (Path(__file__).parent / "test_data/v2/wordpiece.json").read_text(encoding="utf-8")
    )
    tokens = golden["tokens"]
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 2), np.float32), words=tokens)
    config = WordPieceConfig.from_json(golden["config"])
    texts = [text for text, _ in golden["cases"]]
    source = StringSource(texts)
    lookup = make_lookup(wv, TokenizerSpec("wordpiece@1", config, "keep"))
    batches = [source.encode(c, lookup, _native.Workspace()) for c in source.scan().chunks]
    assert_ids_equal(_split(batches), [ids for _, ids in golden["cases"]], texts)


def test_lines_give_the_ids_of_their_strings():
    texts = [t.replace("\n", " ") for t in CORPUS]
    spec = _spec("keep", 8)
    lookup = make_lookup(WORD_VECTORS, spec)
    data = np.frombuffer(("\n".join(texts) + "\n").encode(), np.uint8)
    batch = lookup.encode(data, None, 0, len(data), 0, _native.Workspace())
    assert_ids_equal(_split([batch]), reference_ids(texts, spec), texts)


def test_chunks_and_threads_give_the_same_ids():
    from concurrent.futures import ThreadPoolExecutor

    spec = _spec()
    want = native_ids(CORPUS, spec)
    assert native_ids(CORPUS, spec, chunk_bytes=1 << 12) == want
    source = StringSource(CORPUS)
    lookup = make_lookup(WORD_VECTORS, spec)
    chunks = source.scan(chunk_bytes=1 << 12).chunks
    assert len(chunks) > 8

    def work(part):
        ws = _native.Workspace()
        return [source.encode(c, lookup, ws) for c in part]

    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(work, [chunks[k::4] for k in range(4)]))
    batches = sorted((b for part in parts for b in part), key=lambda b: b.first_row)
    assert _split(batches) == want


def test_the_word_at_the_token_limit_is_split_whole():
    tokens = ["[UNK]", "a"]
    wv = WordVectors.from_arrays(np.zeros((2, 2), np.float32), words=tokens)
    config = WordPieceConfig(((0, "[UNK]"),), 0)
    for unknown, want in (("keep", [[0], [1], [1]]), ("drop", [[], [1], [1]])):
        lookup = make_lookup(wv, TokenizerSpec("wordpiece@1", config, unknown, 1))
        texts = ["ab", "a b", "a [UNK]"]
        source = StringSource(texts)
        batch = source.encode(source.scan().chunks[0], lookup, _native.Workspace())
        assert _split([batch]) == want


def test_invalid_utf8_raises_where_python_does():
    rng = random.Random(3)
    pieces = [b"a", b" ", b"[MASK]", b"\xc3\xa9", b"\xe4\xb8\xad", b"\xf0\x9f\x98\x80", b"<<"]
    bad = [b"\xff", b"\xc3", b"\xe4\xb8", b"\xed\xa0\x80", b"\xc0\xaf", b"\xf4\x90\x80\x80"]
    for spec in (_spec(), _spec("drop", 2), _spec("drop", 2, 3)):
        lookup = make_lookup(WORD_VECTORS, spec)
        for _ in range(600):
            data = b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 6)))
            data += rng.choice(bad) + b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 6)))
            with pytest.raises(UnicodeDecodeError) as python:
                data.decode("utf-8")
            array = np.frombuffer(data, np.uint8)
            with pytest.raises(UnicodeDecodeError) as native:
                lookup.encode(array, None, 0, len(array), 0, _native.Workspace())
            assert native.value.start == python.value.start, data


# --- Meta-tests: the corpus catches typical mistakes ------------------------------------------


def _first_match(text, added):
    """Added tokens in their order, not the longest first."""
    out, i, last = [], 0, 0
    while i < len(text):
        match = next((t for t in added if text.startswith(t, i)), None)
        if match is None:
            i += 1
            continue
        if last < i:
            out.append((text[last:i], None))
        out.append((match, added[match]))
        i = last = i + len(match)
    if last < len(text):
        out.append((text[last:], None))
    return out


def _nfd_without_barrier(text):
    marks = {ch for ch in text if ord(ch) in ucd.NONSPACING_MARKS}
    zero = {ch for ch in marks if not ucd.COMBINING_CLASS.get(ord(ch), 0)}
    return NFD("".join(ch for ch in text if ch not in zero))


def _wordpiece_counting_bytes(word, vocab, unk_id, prefix, max_chars):
    if len(word.encode()) > max_chars:
        return [unk_id]
    return WORDPIECE(word, vocab, unk_id, prefix, max_chars)


def _drop_before_cut(text, vocab, **rules):
    ids = ENCODE(text, vocab, **{**rules, "max_tokens": None, "drop_unknown": False})
    ids = [i for i in ids if i != rules["unk_id"]] if rules["drop_unknown"] else ids
    return ids if rules["max_tokens"] is None else ids[: rules["max_tokens"]]


NFD, WORDPIECE, ENCODE = ref.nfd, ref.wordpiece, ref.encode_ids
MN9 = ucd.CodePoints(
    " ".join(f"{a:x}-{b:x}" for a, b in ucd.NONSPACING_MARKS.ranges) + " 8d4-8e1 1e944-1e94a"
)


@pytest.mark.parametrize(
    ("mistake", "attribute", "value"),
    [
        ("Mn of Unicode 9.0.0", "ucd.NONSPACING_MARKS", MN9),
        ("no removed mark of class 0 ends a run", "nfd", _nfd_without_barrier),
        ("no canonical ordering", "nfd", lambda t: "".join(NFD(ch) for ch in t)),
        (
            "accents removed before the decomposition",
            "normalize",
            lambda t: ref.lowercase(NFD(ref.strip_marks(ref.space_cjk(ref.clean(t))))),
        ),
        ("ASCII punctuation only", "ucd.PUNCTUATION", ucd.CodePoints("21-2f 3a-40 5b-60 7b-7e")),
        ("no CJK spaces", "space_cjk", lambda t: t),
        ("added tokens ignored", "split_added", lambda t, _added: [(t, None)]),
        ("the first matching added token, not the longest", "split_added", _first_match),
        ("the word limit counts bytes", "wordpiece", _wordpiece_counting_bytes),
        ("unknown ids dropped before the token cut", "encode_ids", _drop_before_cut),
    ],
)
def test_the_corpus_catches(monkeypatch, mistake, attribute, value):
    spec = _spec("drop", 8)
    want = reference_ids(CORPUS, spec)
    module, name = (ucd, attribute[4:]) if attribute.startswith("ucd.") else (ref, attribute)
    monkeypatch.setattr(module, name, value)
    caught = sum(a != b for a, b in zip(reference_ids(CORPUS, spec), want, strict=True))
    assert caught, f"no text catches {mistake}"


def test_the_corpus_catches_another_word_limit():
    spec = _spec("keep")
    want = reference_ids(CORPUS, spec)
    wider = _spec(
        "keep", config=WordPieceConfig(CONFIG.added_tokens, wp.UNK_ID, max_input_chars_per_word=200)
    )
    assert reference_ids(CORPUS, wider) != want

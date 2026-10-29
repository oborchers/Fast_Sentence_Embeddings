"""The reference of wordpiece@1 gives the ids of Hugging Face tokenizers 0.23.3.

The golden ids pin it; every code point alone and between two letters checks the
normalizer and the pre-tokenizer exhaustively; pairs of combining marks check the
ordering; a random corpus over a weighted alphabet checks the ids, with and without the
cuts. The tokenizers are built in code (tests/support/wordpiece.py), so nothing is
downloaded.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fse import _reference_wordpiece as ref
from fse import _unicode_bert as ucd
from support import wordpiece as wp

tokenizers = pytest.importorskip("tokenizers")
GOLDEN = json.loads(
    (Path(__file__).parent / "test_data" / "v2" / "wordpiece.json").read_text(encoding="utf-8")
)
CODE_POINTS = [cp for cp in range(sys.maxunicode + 1) if not 0xD800 <= cp <= 0xDFFF]
TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}
CORPUS = wp.corpus(TOKENS)


def _normalizer():
    from tokenizers.normalizers import BertNormalizer

    return BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
    )


def _golden_rules():
    config = GOLDEN["config"]
    added = {content: i for i, content in config["added_tokens"]}
    return {"unk_id": config["unk_id"], "added": added, "drop_unknown": False}


def test_the_golden_file_is_of_the_pinned_tokenizers():
    assert GOLDEN["tokenizer"] == "wordpiece@1"
    assert GOLDEN["tokenizers"] == tokenizers.__version__ == "0.23.3"


def test_the_reference_gives_the_golden_ids():
    vocab = {t: i for i, t in enumerate(GOLDEN["tokens"])}
    for text, ids in GOLDEN["cases"]:
        assert ref.encode_ids(text, vocab, **_golden_rules()) == ids, text


def test_tokenizers_gives_the_golden_ids():
    config = GOLDEN["config"]
    added = {content: i for i, content in config["added_tokens"]}
    hf = wp.hf_tokenizer(GOLDEN["tokens"], added, unk_id=config["unk_id"])
    texts = [text for text, _ in GOLDEN["cases"]]
    assert wp.hf_ids(hf, texts, None) == [ids for _, ids in GOLDEN["cases"]]


def _chunks(items, size=4096):
    for k in range(0, len(items), size):
        yield items[k : k + size]


def test_every_code_point_normalizes_as_tokenizers_does():
    """Each code point alone; the code points of a chunk are joined with a letter, which
    no rule changes and which ends every run of marks, so a chunk compares them all."""
    normalizer = _normalizer()
    for chunk in _chunks(CODE_POINTS):
        text = "a".join(map(chr, chunk))
        if normalizer.normalize_str(text) != ref.normalize(text):
            for cp in chunk:
                assert ref.normalize(chr(cp)) == normalizer.normalize_str(chr(cp)), hex(cp)


def test_every_code_point_between_letters_pre_tokenizes_as_tokenizers_does():
    """On the normalized text of tokenizers, which the test above checks for every code
    point; groups separated by a space split on their own."""
    from tokenizers.pre_tokenizers import BertPreTokenizer

    normalizer, pre = _normalizer(), BertPreTokenizer()
    for chunk in _chunks(CODE_POINTS):
        texts = [normalizer.normalize_str(f"a{chr(cp)}b") for cp in chunk]
        text = " ".join(texts)
        if [w for w, _ in pre.pre_tokenize_str(text)] != ref.pre_tokenize(text):
            for cp, s in zip(chunk, texts, strict=True):
                assert ref.pre_tokenize(s) == [w for w, _ in pre.pre_tokenize_str(s)], hex(cp)


def _survivors():
    """The code points with a nonzero class in 9.0.0 that are not Mn in 8.0.0."""
    return sorted(cp for cp in ucd.COMBINING_CLASS if cp not in ucd.NONSPACING_MARKS)


def test_83_marks_survive_accent_stripping():
    assert len(_survivors()) == 83


def test_pairs_of_marks_are_ordered_as_tokenizers_orders_them():
    marks = [*_survivors(), 0x0300, 0x0301, 0x0316, 0x0327, 0x05B0, 0x0E48, 0x034F, 0x0345]
    contexts = ["a{}{}", "{}{}", "a{}" + chr(0x034F) + "{}"]
    texts = [c.format(chr(x), chr(y)) for x in marks for y in marks for c in contexts]
    assert len(texts) > 24_000
    normalizer = _normalizer()
    for chunk in _chunks(texts, 512):
        joined = " ".join(chunk)  # a space ends every run of marks
        if normalizer.normalize_str(joined) != ref.normalize(joined):
            for text in chunk:
                assert ref.normalize(text) == normalizer.normalize_str(text), ascii(text)


@pytest.mark.parametrize("max_tokens", [None, 8, 1])
@pytest.mark.parametrize("drop_unknown", [True, False])
def test_the_ids_of_a_random_corpus_are_those_of_tokenizers(max_tokens, drop_unknown):
    hf = wp.hf_tokenizer(TOKENS, ADDED, max_length=max_tokens)
    want = wp.hf_ids(hf, CORPUS, wp.UNK_ID if drop_unknown else None)
    for text, ids in zip(CORPUS, want, strict=True):
        got = ref.encode_ids(
            text,
            VOCAB,
            unk_id=wp.UNK_ID,
            added=ADDED,
            max_tokens=max_tokens,
            drop_unknown=drop_unknown,
        )
        assert got == ids, ascii(text)


def test_the_character_cut_is_python_slicing_before_tokenizing():
    hf = wp.hf_tokenizer(TOKENS, ADDED, max_length=16)
    cut = [text[:12] for text in CORPUS]
    for text, ids in zip(CORPUS, wp.hf_ids(hf, cut), strict=True):
        rules = {"unk_id": wp.UNK_ID, "added": ADDED, "max_tokens": 16, "max_chars": 12}
        assert ref.encode_ids(text, VOCAB, **rules) == ids, ascii(text)


def test_another_limit_of_characters_per_word():
    hf = wp.hf_tokenizer(TOKENS, ADDED, max_input_chars_per_word=5)
    want = wp.hf_ids(hf, CORPUS, None)
    for text, ids in zip(CORPUS, want, strict=True):
        got = ref.encode_ids(
            text,
            VOCAB,
            unk_id=wp.UNK_ID,
            added=ADDED,
            max_input_chars_per_word=5,
            drop_unknown=False,
        )
        assert got == ids, ascii(text)


def test_the_word_at_the_token_limit_is_split_whole():
    """Vocabulary [UNK], a; text ab; one token: [UNK] (finding 3 of the plan review)."""
    tokens = ["[UNK]", "a"]
    hf = wp.hf_tokenizer(tokens, {"[UNK]": 0}, max_length=1, unk_id=0)
    assert wp.hf_ids(hf, ["ab"], None) == [[0]]
    rules = {"unk_id": 0, "added": {"[UNK]": 0}, "max_tokens": 1, "drop_unknown": False}
    assert ref.encode_ids("ab", {"[UNK]": 0, "a": 1}, **rules) == [0]
    assert ref.encode_ids("ab", {"[UNK]": 0, "a": 1}, **{**rules, "drop_unknown": True}) == []


def test_added_tokens_are_matched_leftmost_longest():
    added = {"ab": 1, "abc": 2, "bcd": 3}
    assert ref.split_added("xabcd", added) == [("x", None), ("abc", 2), ("d", None)]
    assert ref.split_added("abab", added) == [("ab", 1), ("ab", 1)]
    assert ref.split_added("bcdab", added) == [("bcd", 3), ("ab", 1)]
    assert ref.split_added("", added) == []
    assert ref.split_added("xy", {}) == [("xy", None)]


def test_the_steps_of_normalization():
    a, cgj = chr(0x1E949), chr(0x034F)
    assert ref.clean("a\x00b" + chr(0xFFFD) + "\x0b\tc" + chr(0x2028)) == "ab c "
    assert ref.space_cjk("x" + chr(0x4E2D) + "y") == "x " + chr(0x4E2D) + " y"
    assert ref.nfd(chr(0xD55C)) == chr(0x1112) + chr(0x1161) + chr(0x11AB)
    assert ref.nfd("a" + a + chr(0x11446)) == "a" + chr(0x11446) + a  # class 7 before 230
    assert ref.nfd("a" + a + cgj + chr(0x11446)) == "a" + a + cgj + chr(0x11446)
    assert ref.strip_marks("e" + chr(0x0301) + cgj) == "e"
    assert ref.lowercase(chr(0x0130) + chr(0x03A3)) == "i" + chr(0x0307) + chr(0x03C3)
    assert ref.normalize(chr(0x0130)) == "i"  # decomposed before it is lowercased
    assert ref.pre_tokenize("a,b  c!" + chr(0x3001)) == ["a", ",", "b", "c", "!", chr(0x3001)]


def test_wordpiece_splits_greedily_from_the_left():
    vocab = {"un": 0, "##aff": 1, "##able": 2, "u": 3, "##n": 4, "[UNK]": 5}
    assert ref.wordpiece("unaffable", vocab, 5, "##", 100) == [0, 1, 2]
    assert ref.wordpiece("unx", vocab, 5, "##", 100) == [5]
    assert ref.wordpiece("unaffable", vocab, 5, "##", 8) == [5]


def test_a_batch_has_the_ids_as_rows():
    batch = ref.encode_texts(["ab", "", "x[MASK]"], VOCAB, unk_id=wp.UNK_ID, added=ADDED)
    assert batch.offsets.tolist() == [0, 1, 1, 3]
    assert batch.tokens.dtype == np.uint32
    assert batch.tokens.tolist() == [VOCAB["ab"], VOCAB["x"], VOCAB["[MASK]"]]


def test_a_lone_surrogate_raises():
    with pytest.raises(UnicodeEncodeError):
        ref.encode_ids("a" + chr(0xD800), VOCAB, unk_id=wp.UNK_ID, added=ADDED)

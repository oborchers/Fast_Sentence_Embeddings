"""Vocabularies, texts and Hugging Face tokenizers built in code, for the tests of wordpiece@1.

Nothing of a BERT vocabulary is checked in: the vocabulary is made from the test sentences
and fixed pieces, and the tokenizers of Hugging Face are built from it in code, so the
comparisons need no network.
"""

import random
from pathlib import Path

from fse import _reference_wordpiece as ref

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
UNK_ID = 1
# Added tokens that are no special tokens of BERT: one that starts with another (the
# longest wins), a pair where one is a prefix of the other, and one that is not ASCII
# (matched on the raw text, so its NFD spelling is no match).
EXTRA_ADDED = ["[MASK]]", "<<", "<<<", "\u00fc"]
# Pieces beyond the test sentences: long tokens (more than the 16 bytes a slot of the
# vocabulary holds), continuation pieces, characters of several scripts and planes.
_FIXED = [
    *("internationalization", "##ization", "##izationally", "un", "##believ", "##able"),
    *("istanbul", "i", "##\u0307", "\u01f0", "j", "##\u030c", "\u4e2d", "\u56fd", "\u8c48"),
    *("\U00020000", "\u1100", "##\u1161", "##\u11a8", "\u1112", "##\u1175", "\u03c3"),
    *("##\u03c3", "\u00e9", "e", "##e", "a", "##a", "##b", "\U0001f600", "##\U0001f600"),
    *("ab", "x", "##x", "##y", "##9", "9", "6", "##6", "\u1e95", "##\u1e95", "##ing", "##s"),
    *("#", "!", ".", ",", "'", "##\U0001e949", "##\U00011446", "##\U0001d165"),
]
SENTENCES = Path(__file__).parent.parent / "test_data" / "test_sentences.txt"


def vocabulary() -> list[str]:
    """The tokens in id order: the special tokens, every second word of the test sentences
    whole, the first two characters and the continuation pieces of one to three characters
    of some of the rest, the letters, the fixed pieces and the added tokens. Many words
    stay unknown."""
    words = sorted(
        {
            w
            for line in SENTENCES.read_text(encoding="utf-8").splitlines()
            for w in ref.pre_tokenize(ref.normalize(line))
        }
    )
    pieces = set(_FIXED) | set("abcdefghijklmnopqrstuvwxyz")
    pieces |= {"##" + ch for ch in "abcdefghijklmnopqrstuvw"}  # x, y, z only as words
    for k, word in enumerate(words):
        if k % 2 == 0:
            pieces.add(word)
        elif k % 3:
            pieces.add(word[:2])
            pieces.update("##" + word[i : i + 1 + i % 3] for i in range(2, len(word)))
    tokens = SPECIAL + sorted(pieces - set(SPECIAL) - set(EXTRA_ADDED))
    return tokens + EXTRA_ADDED


def added_tokens(tokens: list[str]) -> dict[str, int]:
    return {t: tokens.index(t) for t in SPECIAL + EXTRA_ADDED}


def hf_tokenizer(
    tokens: list[str],
    added: dict[str, int],
    *,
    max_input_chars_per_word: int = 100,
    max_length: int | None = None,
    unk_id: int = UNK_ID,
):
    """A tokenizers.Tokenizer with BertNormalizer, BertPreTokenizer and WordPiece, its
    added tokens matched on the raw text, as a BERT tokenizer.json gives it."""
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer

    vocab = {t: i for i, t in enumerate(tokens)}
    model = WordPiece(
        vocab,
        unk_token=tokens[unk_id],
        max_input_chars_per_word=max_input_chars_per_word,
        continuing_subword_prefix="##",
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()
    options = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
    special = [t for t in added if t.startswith("[")]
    tokenizer.add_special_tokens([AddedToken(t, special=True, **options) for t in special])
    others = [t for t in added if t not in special]
    tokenizer.add_tokens([AddedToken(t, special=False, **options) for t in others])
    if tokenizer.get_vocab_size() != len(tokens):
        raise AssertionError("an added token is not in the vocabulary")
    if max_length is not None:
        tokenizer.enable_truncation(max_length)
    return tokenizer


def hf_ids(tokenizer, texts: list[str], unk_id: int | None = UNK_ID) -> list[list[int]]:
    """The ids without special tokens, with ``unk_id`` dropped (None keeps it), as
    model2vec (or sentence-transformers) takes them."""
    out = [e.ids for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
    return out if unk_id is None else [[i for i in ids if i != unk_id] for ids in out]


# Characters of every rule, weighted: letters of the vocabulary, added tokens and their
# parts, controls Cc and Cf (kept and removed), private use, unassigned code points,
# White_Space, punctuation (ASCII and of 8.0.0), CJK (unified, compatibility, plane 2),
# Hangul syllables and jamo, marks (Mn of class 0 and above, marks that are Mn only after
# 8.0.0, survivors of a nonzero class), decompositions and the case pairs of Unicode 16
# and 17.
_ALPHABET = [
    *"abcdeijsxyz9 6 ABCXYZ  ",
    *["[MASK]", "[MASK]]", "[UNK]", "[CLS]", "<<", "<", "\u00fc", "u\u0308", "\u00dc", "##"],
    *"\x00\x01\x0b\x0c\x1c\x7f\x85\t\n\r\u00ad\u200b\u200d\ufeff\ufffd\U000e0001",
    *"\ue000\U000f0000\U0010fffd\u0378\U0001fffe\u0890\u08e2",
    *"\u00a0\u1680\u2000\u2028\u2029\u202f\u205f\u3000",
    *"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\u00a1\u00bf\u2010\u2014\u300c\u060c\u055d",
    *"\U00016e97\u4e2d\u56fd\uf900\ufa6d\U00020000\U0002f800\u3400\u3005\u3041",
    *"\uac00\uac01\ud7a3\u1100\u1161\u11a8\u3131",
    *"\u0300\u0301\u0307\u030c\u0327\u0345\u034f\u05b0\u0e48\u08d4\u08e1",
    *"\U0001d165\U0001d16d\u0f39\U0001d15e\u1dfa\U0001e949\U00011446",
    *"\u0130\u03a3\u03c2\u01f0\u1e94\u1e9e\u00df\u212a\u212b\u0149\u1f88\u0390",
    *"\ua7cb\ua7cc\ua7ce\ua7d2\ua7d4\U00016ea0\U00016eb8\U0001e900",
    *"\U0001f600\u00e9\u0435\u0410",
]
_PLANES = [(0x80, 0x7FF), (0x800, 0xD7FF), (0xE000, 0xFFFF), (0x10000, 0x10FFFF)]


def random_text(rng: random.Random, words: list[str]) -> str:
    """A text of characters of the alphabet, vocabulary words and random code points."""
    parts = []
    for _ in range(rng.randint(0, 24)):
        roll = rng.random()
        if roll < 0.55:
            parts.append(rng.choice(_ALPHABET))
        elif roll < 0.85:
            parts.append(rng.choice(words) + rng.choice(["", " ", "", "x", "\u0301"]))
        else:
            lo, hi = rng.choice(_PLANES)
            parts.append(chr(rng.randint(lo, hi)))
    return "".join(parts)


def corpus(tokens: list[str], count: int = 6000, seed: int = 0) -> list[str]:
    """Random texts, words of 98 to 102 code points, the test sentences and edge cases."""
    rng = random.Random(seed)
    words = [t.removeprefix("##") for t in tokens if t not in SPECIAL]
    texts = [random_text(rng, words) for _ in range(count)]
    for n in (98, 99, 100, 101, 102):
        texts += ["a" * n, "a" + "\u00e9" * (n - 1), "x" * (n - 1) + "\u0301", "\u4e2d" * n]
        texts.append("ab" * (n // 2) + " " + "\U0001f600" * n)
    texts += SENTENCES.read_text(encoding="utf-8").splitlines()
    texts += ["", " ", "\x00", "[MASK]", "x[MASK]y", "[MASK]]]", "<<<<", "\u00fc\u00fc", "ab"]
    # Marks of classes 230 and 7 with and without a removed mark of class 0 between them.
    texts += ["a\U0001e949\u034f\U00011446", "a\U0001e949\U00011446", "e\u0301\u0327x"]
    # Words of 51 to 100 code points but more than 100 bytes, which the vocabulary splits.
    texts += ["\u03c3" * 70, "\U0001f600" * 60]
    return texts + edge_cases()


_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def _mark_runs() -> list[str]:
    """ASCII right after a run of marks, and an ASCII control or DEL inside a run (removed,
    and the run goes on)."""
    out = []
    for ascii_ in ("A", "b", "9", " ", ".", "\t", "x[MASK]"):
        out += [f"e\u0301\u0327{ascii_}", f"\u0301{ascii_}\u0327", f"a\U0001e949{ascii_}"]
    for control in ("\x00", "\x01", "\x1f", "\x7f", "\x0b"):
        out += [f"e\u0301{control}\u0327x", f"a\u0327{control}\u0301{control}B"]
    return out


def _ascii_classes() -> list[str]:
    """Uppercase letters, tab, newline and return, and every ASCII punctuation character
    next to letters, digits and code points that are not ASCII."""
    out = ["ABC Def GHIJKLMNOPQRSTUVWXYZ abcXYZ", "a\tb\nc\rd \t\n\r\x0b\x0c e"]
    for c in _PUNCTUATION:
        out.append(f"a{c}b 9{c}9 \u00e9{c}\u4e2d {c}{c}X\t{c}\r\nAB{c}")
    return out


def _word_limits() -> list[str]:
    """Words of exactly 100 and 101 code points, in ASCII and not, words longer than any
    token, and words the greedy match splits after a miss of the whole word."""
    out = []
    for n in (100, 101):
        out += ["b" * n, "\u00e9" * n, "B" * (n - 1) + "\u00c9", "a" * (n - 1) + "\u0301"]
    out += ["internationalizationally", "internationalizationallyxyz", "INTERNATIONALIZATION"]
    out += ["unbelievable unbelievably unbeliev unable", "abab ##able ##"]
    # The first piece matches, a later one does not: the whole word is one unknown id.
    out += ["unbelievqq", "abzz ab", "unbeliev\u0378able"]
    return out


def _added_and_bytes() -> list[str]:
    """A [ that starts no added token, added tokens back to back at the start and the end,
    single-byte words, and pieces of 0 to 17 bytes between added tokens."""
    out = ["[x [MAS [MASK [ ] [[MASK]] [MASK[MASK]", "[MASK][CLS]a[SEP]", "a[MASK][MASK]"]
    out += ["[UNK][UNK]", "<<[MASK]<<<\u00fc[CLS]", "a b c ~ ^ z q 9 6 , . ! \u00e9"]
    for k in range(18):
        out += [f"[MASK]{'aB'[: k % 2 + 1] * (k // 2)}{'c' * (k % 2)}[MASK]", "Ab" * k + "[CLS]"]
        out += ["[SEP]" + "x" * k + "\u00e9" + "A" * k, "\u0301" * (k % 3) + "B" * k + "[PAD]"]
    return out


def edge_cases() -> list[str]:
    """Texts for the paths of a native implementation: runs of marks against ASCII, the ASCII
    classes, the word limit, the greedy match, added tokens and short pieces."""
    return _mark_runs() + _ascii_classes() + _word_limits() + _added_and_bytes()


def ascii_heavy(count: int, seed: int) -> list[str]:
    """Random ASCII text with about 1 % of marks, controls, CJK, other code points that
    are not ASCII and added tokens: the mixtures where an ASCII path meets the generic one."""
    rng = random.Random(seed)
    rare = ["\u0301", "\u0327", "\U0001e949", "\x01", "\x7f", "\u4e2d", "\u00e9", "\u00c9"]
    rare += ["\u3000", "\u00a0", "\u2014", "\uac01", "[MASK]", "<<", "\u00fc", "\U0001f600"]
    common = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" + _PUNCTUATION
    common += " " * 12 + "\t\n\r"
    texts = []
    for _ in range(count):
        n = rng.randint(0, 40)
        chars = [rng.choice(rare) if rng.random() < 0.01 else rng.choice(common) for _ in range(n)]
        texts.append("".join(chars))
    return texts


def all_unicode(count: int = 4000, seed: int = 1) -> list[str]:
    """Random texts over every code point but the surrogates."""
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        cps = [rng.randint(0, 0x10FFFF) for _ in range(rng.randint(0, 30))]
        texts.append("".join(chr(c) for c in cps if not 0xD800 <= c <= 0xDFFF))
    return texts

"""Unigram vocabularies, charsmaps, texts and Hugging Face tokenizers built in code, for
the tests of unigram@1.

Nothing of a SentencePiece model is checked in: the pieces come from the test sentences
and fixed lists with seeded scores, the charsmap from a dict with keys of every kind
(``support.charsmap`` builds its blob), and the tokenizers of Hugging Face are built from
them in code with the normalizer chain of potion-multilingual-128M, so the comparisons
need no network.
"""

from __future__ import annotations

import random
from pathlib import Path

from fse import _reference_unigram as ref
from fse._charsmap import Charsmap
from support.charsmap import build

REPLACEMENT = "\u2581"
PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
# The normalizer of potion-multilingual-128M, as the parameters of unigram@1 hold it.
STEPS: list[dict] = [
    {"type": "precompiled"},
    {"type": "runs", "class": "space", "min": 2, "content": " "},
    *({"type": "replace", "pattern": c, "content": f" {c} "} for c in PUNCTUATION),
    {"type": "runs", "class": "white_space", "min": 1, "content": " "},
    {"type": "strip", "left": True, "right": True},
]
ADDED = ["[PAD]", "[UNK]"]
UNK_ID = 1
_COMPOSED = {
    "a\u0301": "\u00e1", "e\u0301": "\u00e9", "i\u0301": "\u00ed", "o\u0301": "\u00f3",
    "u\u0301": "\u00fa", "n\u0303": "\u00f1", "c\u0327": "\u00e7", "e\u0300": "\u00e8",
    "a\u0308": "\u00e4", "o\u0308": "\u00f6", "u\u0308": "\u00fc", "A\u030a": "\u00c5",
}  # fmt: skip
# Keys of every kind: whitespace and controls, fullwidth forms, composed pairs below
# 6 bytes, compatibility forms that expand (to punctuation too), keys that can never match
# (6 bytes and more), an InCB consonant (GB9c) and an Extended_Pictographic of 3 bytes
# (GB11), so a following mark is dropped only when they start a grapheme.
CHARSMAP = {
    **{c: " " for c in "\t\n\r\x0c\u00a0\u3000\u2028\u200b"},
    **{chr(c): "" for c in [*range(1, 9), 0x0B, *range(0x0E, 0x20), 0x7F, 0xFEFF]},
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},
    **_COMPOSED,
    "\ufb01": "fi", "\ufb02": "fl", "\u2460": "1", "\u2026": "...", "\u2122": "TM",
    "\u00bd": "1/2", "\u01c6": "d\u017e", "\u3392": "MHz",
    "\u00e9\u0302\u0303": "LONG", "\U0001f468\u200d": "FAMILY",
    "\u0937": "SSA", "\u2764": "HEART", "\U0001f600": "SMILE",
}  # fmt: skip
SENTENCES = Path(__file__).parent.parent / "test_data" / "test_sentences.txt"
_ALPHABET = (
    "abcdefghijklmnoprstuvwxyABCDEFGHIJKLMNOPRSTUVWXY0123456789"
    + PUNCTUATION
    + "\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00e7\u00e8\u00e4\u00f6\u00fc\u00c5\u00df\u017e"
    + "\u03b1\u03b2\u03b3\u0430\u0431\u0432\u4e2d\u56fd\u6587\uac00\ud55c\u1100\u1161"
    + "\u0915\u0937\u094d\u093f\u0627\u0644\u05d0\U0001f600\u2764\U0001f44d\U0001f3fd"
    + "\U0001f1e9\U0001f1ea\u0301\u0302\u200d"
)  # q, z, Q and Z stay unknown


def pieces_and_scores(seed: int = 0) -> list[tuple[str, float]]:
    """The pieces in id order with their scores: the added tokens, the replacement, every
    character of the alphabet alone and after the replacement, the words of the test
    sentences after the replacement and some of their parts. Scores are seeded floats of
    -14 to -2 (most not float32 values); the pieces of two characters share one score, so
    ties occur."""
    rng = random.Random(seed)
    words = sorted(
        {w for line in SENTENCES.read_text(encoding="utf-8").split("\n") for w in line.split()}
    )
    pieces = {REPLACEMENT, *_ALPHABET, *(REPLACEMENT + c for c in _ALPHABET)}
    for k, word in enumerate(words):
        if k % 2 == 0:
            pieces.add(REPLACEMENT + word)
        if k % 3 == 0:
            pieces.update(word[i : i + 2 + i % 3] for i in range(len(word) - 1))
    out = [(token, -13.5) for token in ADDED]
    for piece in sorted(pieces - set(ADDED)):
        score = -6.5 if len(piece) == 2 else rng.uniform(-14.0, -2.0)
        out.append((piece, score))
    return out


def hf_tokenizer(
    vocab: list[tuple[str, float]],
    *,
    steps: list[dict] = STEPS,
    charsmap: dict[str, str] = CHARSMAP,
    unk_id: int = UNK_ID,
    added: list[str] = ADDED,
    max_length: int | None = None,
):
    """A tokenizers.Tokenizer with this Unigram model, normalizer, Metaspace and the added
    tokens matched on the raw text, as a tokenizer.json of SentencePiece gives it."""
    from tokenizers import AddedToken, Regex, Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import Unigram

    def hf_step(step: dict):
        kind = step["type"]
        if kind == "precompiled":
            return normalizers.Precompiled(build(charsmap, share=True))
        if kind == "replace":
            return normalizers.Replace(step["pattern"], step["content"])
        if kind == "runs":
            pattern = " {2,}" if step["class"] == "space" else r"\s+"
            return normalizers.Replace(Regex(pattern), step["content"])
        return normalizers.Strip(left=step["left"], right=step["right"])

    tokenizer = Tokenizer(Unigram(vocab, unk_id=unk_id, byte_fallback=False))
    tokenizer.normalizer = normalizers.Sequence([hf_step(s) for s in steps])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=REPLACEMENT, prepend_scheme="always", split=False
    )
    options = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
    tokenizer.add_special_tokens([AddedToken(t, special=True, **options) for t in added])
    if tokenizer.get_vocab_size() != len(vocab):
        raise AssertionError("an added token is not in the vocabulary")
    if max_length is not None:
        tokenizer.enable_truncation(max_length)
    return tokenizer


def hf_ids(tokenizer, texts: list[str]) -> list[list[int]]:
    """The ids without special tokens, the unknown id kept."""
    return [e.ids for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]


class Reference:
    """The reference of unigram@1 on the same model, normalizer and added tokens."""

    def __init__(
        self,
        vocab: list[tuple[str, float]],
        *,
        steps: list[dict] = STEPS,
        charsmap: dict[str, str] = CHARSMAP,
        unk_id: int = UNK_ID,
        added: list[str] = ADDED,
        model: type[ref.Model] = ref.Model,
    ):
        self.model = model([p for p, _ in vocab], [s for _, s in vocab], unk_id)
        self.steps = steps
        self.charsmap = Charsmap(build(charsmap, share=True))
        ids = {p: i for i, (p, _) in enumerate(vocab)}
        self.added = {t: ids[t] for t in added}

    def normalize(self, text: str) -> str:
        return ref.normalize(text, self.steps, self.charsmap)

    def ids(self, text: str, **rules) -> list[int]:
        return ref.encode_ids(
            text,
            self.model,
            added=self.added,
            normalizer=self.steps,
            charsmap=self.charsmap,
            drop_unknown=rules.pop("drop_unknown", False),
            **rules,
        )


# Text of every rule, weighted: the alphabet, unknown letters, keys of the charsmap and
# what follows them in one grapheme, added tokens and their parts, whitespace runs,
# controls, emoji sequences (ZWJ, skin tone, flags), Indic conjuncts, the replacement.
_PARTS = [
    *_ALPHABET, *"qzQZ", *CHARSMAP, *_COMPOSED.values(),
    "[PAD]", "[UNK]", "[PAD", "UNK]", "[pad]", " ", "  ", "   ", "\t \n", "\x00",
    "\uff21\u0301", "\u0937\u0301", "\u0915\u094d\u0937\u0301", "\U0001f468\u200d\u2764\u0301",
    "\u2764\u0301", "\U0001f44d\U0001f3fd", "\U0001f1e9\U0001f1ea\U0001f1e9", "e\u0301\u0302",
    "\u00e9\u0302\u0303", REPLACEMENT, REPLACEMENT * 2, "\u180e", "\u0085", "\ue000",
]  # fmt: skip
_PLANES = [(0x80, 0x7FF), (0x800, 0xD7FF), (0xE000, 0xFFFF), (0x10000, 0x10FFFF)]


def random_text(rng: random.Random, words: list[str]) -> str:
    """A text of the parts above, words and random code points."""
    parts = []
    for _ in range(rng.randint(0, 20)):
        roll = rng.random()
        if roll < 0.6:
            parts.append(rng.choice(_PARTS))
        elif roll < 0.9:
            parts.append(rng.choice(words) + rng.choice(["", " ", " ", "\u0301", ","]))
        else:
            lo, hi = rng.choice(_PLANES)
            parts.append(chr(rng.randint(lo, hi)))
    return "".join(parts)


def corpus(n: int, seed: int = 0) -> list[str]:
    """The test sentences and ``n`` seeded random texts."""
    rng = random.Random(seed)
    lines = SENTENCES.read_text(encoding="utf-8").split("\n")
    words = sorted({w for line in lines for w in line.split()})
    return lines + [random_text(rng, words) for _ in range(n)]

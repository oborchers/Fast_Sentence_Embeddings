"""The reference of wordpiece@1 (docs/architecture/tokenizers.md, "wordpiece@1").

Plain Python on code points that follows the specification step by step and defines the
ids the native tokenizer must give. It uses the Unicode data of ``fse._unicode_bert`` as the
UCD states it (categories of 8.0.0, decompositions and classes of 9.0.0, lowercase and
White_Space of 17.0.0), never ``unicodedata`` or ``str.lower``, whose results depend on the
Python version. It is slow and meant for tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fse import _unicode_bert as ucd
from fse._contract import Batch

# Step 3.2: the ranges of handle_chinese_chars.
CJK = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B920, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)
ASCII_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_KEPT_CONTROLS = frozenset("\t\n\r")
# Hangul syllables, by the algorithm of the standard (section 3.12).
_S_BASE, _L_BASE, _V_BASE, _T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
_V_COUNT, _T_COUNT, _S_COUNT = 21, 28, 11172


def _is_space(ch: str) -> bool:
    return ch in _KEPT_CONTROLS or ord(ch) in ucd.WHITE_SPACE


def clean(text: str) -> str:
    """Step 3.1: remove NUL, U+FFFD and Cc, Cf, Co (8.0.0) except tab, line feed and
    carriage return; then every whitespace becomes U+0020."""
    kept = [
        ch
        for ch in text
        if ch not in ("\x00", "\ufffd") and (ch in _KEPT_CONTROLS or ord(ch) not in ucd.OTHER)
    ]
    return "".join(" " if _is_space(ch) else ch for ch in kept)


def space_cjk(text: str) -> str:
    """Step 3.2: U+0020 before and after every code point of the CJK ranges."""
    return "".join(f" {ch} " if any(a <= ord(ch) <= b for a, b in CJK) else ch for ch in text)


def _decompose(cp: int, out: list[int]) -> None:
    if _S_BASE <= cp < _S_BASE + _S_COUNT:
        s = cp - _S_BASE
        out.append(_L_BASE + s // (_V_COUNT * _T_COUNT))
        out.append(_V_BASE + (s % (_V_COUNT * _T_COUNT)) // _T_COUNT)
        if s % _T_COUNT:
            out.append(_T_BASE + s % _T_COUNT)
        return
    parts = ucd.DECOMPOSITION.get(cp)
    if parts is None:
        out.append(cp)
        return
    for part in parts:
        _decompose(part, out)


def nfd(text: str) -> str:
    """Step 3.3: the full canonical decomposition of 9.0.0, then every maximal run of code
    points with a nonzero combining class sorted stably by class."""
    cps: list[int] = []
    for ch in text:
        _decompose(ord(ch), cps)
    i = 0
    while i < len(cps):
        if not ucd.COMBINING_CLASS.get(cps[i], 0):
            i += 1
            continue
        j = i
        while j < len(cps) and ucd.COMBINING_CLASS.get(cps[j], 0):
            j += 1
        cps[i:j] = sorted(cps[i:j], key=lambda cp: ucd.COMBINING_CLASS[cp])
        i = j
    return "".join(map(chr, cps))


def strip_marks(text: str) -> str:
    """Step 3.4: remove every code point of category Mn of 8.0.0."""
    return "".join(ch for ch in text if ord(ch) not in ucd.NONSPACING_MARKS)


def lowercase(text: str) -> str:
    """Step 3.5: the full lowercase mapping of 17.0.0, without context."""
    return "".join("".join(map(chr, ucd.LOWERCASE.get(ord(ch), (ord(ch),)))) for ch in text)


def normalize(text: str) -> str:
    """Step 3: the normalized text of one piece between added tokens."""
    return lowercase(strip_marks(nfd(space_cjk(clean(text)))))


def _is_punctuation(ch: str) -> bool:
    return ch in ASCII_PUNCTUATION or ord(ch) in ucd.PUNCTUATION


def pre_tokenize(text: str) -> list[str]:
    """Step 4: split normalized text at whitespace (removed) and around punctuation."""
    words: list[str] = []
    word: list[str] = []
    for ch in text:
        if _is_space(ch) or _is_punctuation(ch):
            if word:
                words.append("".join(word))
                word = []
            if not _is_space(ch):
                words.append(ch)
        else:
            word.append(ch)
    if word:
        words.append("".join(word))
    return words


def wordpiece(
    word: str, vocab: Mapping[str, int], unk_id: int, prefix: str, max_chars: int
) -> list[int]:
    """Step 5: the greedy longest match of one word from the left, or ``[unk_id]``."""
    if len(word) > max_chars:
        return [unk_id]
    ids: list[int] = []
    start = 0
    while start < len(word):
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else prefix + word[start:end]
            if piece in vocab:
                ids.append(vocab[piece])
                start = end
                break
        else:
            return [unk_id]
    return ids


def split_added(text: str, added: Mapping[str, int]) -> list[tuple[str, int | None]]:
    """Step 2: the pieces of the raw text and the added tokens in it, leftmost-longest:
    ``(piece, None)`` for text between them, ``(content, id)`` for a token."""
    by_length = sorted(added, key=len, reverse=True)
    out: list[tuple[str, int | None]] = []
    i = last = 0
    while i < len(text):
        match = next((token for token in by_length if text.startswith(token, i)), None)
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


def encode_ids(
    text: str,
    vocab: Mapping[str, int],
    *,
    unk_id: int,
    added: Mapping[str, int],
    prefix: str = "##",
    max_input_chars_per_word: int = 100,
    max_tokens: int | None = None,
    max_chars: int | None = None,
    drop_unknown: bool = True,
) -> list[int]:
    """The ids of one sentence (steps 1 to 6). ``vocab`` maps every token string to its
    id, ``added`` the added tokens to theirs."""
    text.encode("utf-8")  # a lone surrogate raises
    if max_chars is not None:
        text = text[:max_chars]
    ids: list[int] = []
    for piece, token in split_added(text, added):
        if token is not None:
            ids.append(token)
            continue
        for word in pre_tokenize(normalize(piece)):
            ids += wordpiece(word, vocab, unk_id, prefix, max_input_chars_per_word)
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return [i for i in ids if i != unk_id] if drop_unknown else ids


def encode_texts(
    texts: Sequence[str], vocab: Mapping[str, int], *, first_row: int = 0, **rules
) -> Batch:
    """Stages 1 and 2: the batch of the sentences ``texts``; token id i is row i. The
    keyword arguments are those of ``encode_ids``."""
    tokens: list[int] = []
    offsets = [0]
    for text in texts:
        tokens += encode_ids(text, vocab, **rules)
        offsets.append(len(tokens))
    return Batch(first_row, np.array(offsets, np.int64), np.array(tokens, np.uint32))

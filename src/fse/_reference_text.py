"""The reference of stages 1 and 2: tokenizers and the lookup of tokens (ARCHITECTURE.md).

Plain Python that follows the specification step by step and defines the tokens and
batches the native text sources must give byte for byte. It uses neither ``str.lower``
nor ``unicodedata``, whose results depend on the Python version, only the checked-in
table of ``fse._unicode``. It is slow and meant for tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from fse._contract import EXTERNAL_TOKENIZER, OOV_TOKEN, Batch, Tokenizer, check_tokenizer
from fse._unicode import SIMPLE_LOWERCASE

WHITESPACE = frozenset(
    chr(cp)
    for cp in [
        *range(0x09, 0x0E),
        *range(0x1C, 0x21),
        0x85,
        0xA0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    ]
)
SEPARATORS = frozenset('!"#$%&()*;<>?@[]`{}') | frozenset(
    chr(cp) for cp in [0xAB, 0xBB, *range(0x2012, 0x2016), 0x2018, 0x2019, 0x201C, 0x201D, 0x201E]
)
CONTRACTIONS = {
    "cannot": ("can", "not"),
    "gimme": ("gim", "me"),
    "gonna": ("gon", "na"),
    "gotta": ("got", "ta"),
    "lemme": ("lem", "me"),
    "wanna": ("wan", "na"),
    "more'n": ("more", "n"),
    "d'ye": ("d", "ye"),
    "'tis": ("t", "is"),
    "'twas": ("t", "was"),
}
CLITICS = ("'s", "'m", "'d", "'ll", "'re", "'ve")
_ALNUM = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_DIGITS = frozenset("0123456789")


def simple_lower(text: str) -> str:
    """Step 1: the simple lowercase mapping of every code point."""
    return "".join(chr(SIMPLE_LOWERCASE.get(ord(ch), ord(ch))) for ch in text)


def _is_word(ch: str) -> bool:
    if ch in _ALNUM or ch == "_":
        return True
    return ord(ch) >= 0x80 and ch not in WHITESPACE and ch not in SEPARATORS


def _token_chars(s: str) -> list[bool]:
    """Steps 2 and 3: whether every code point of the lowercased sentence is a token char."""
    n = len(s)
    t = [ch not in WHITESPACE and ch not in SEPARATORS for ch in s]
    for i, ch in enumerate(s):
        if not t[i]:
            continue
        before = s[i - 1] if i > 0 else ""
        after = s[i + 1] if i + 1 < n else ""
        if ch in ".-":
            t[i] = before != ch and after != ch
        elif ch in ",:":
            t[i] = after in _DIGITS
        elif ch == "'":
            t[i] = _quote_is_token_char(i, t, before, after)
    return t


def _quote_is_token_char(i: int, t: list[bool], before: str, after: str) -> bool:
    """Whether the ' at ``i`` is a token char, given the classes ``t`` before it."""
    doubled = before == "'" or after == "'"
    # Not a token char after a token char that is no word char, before a word char.
    opening = i > 0 and t[i - 1] and not _is_word(before) and after != "" and _is_word(after)
    return not (doubled or opening)


def _parts(u: str) -> Sequence[str]:
    """Step 5b: the parts of a candidate after stripping."""
    if u in CONTRACTIONS:
        return CONTRACTIONS[u]
    if u == "n't":
        return ("not",)
    if u.endswith("n't") and len(u) > 3 and u[-4] != "'":
        return (u[:-3], "not")
    for clitic in CLITICS:
        rest = u[: -len(clitic)]
        if u.endswith(clitic) and rest and rest[-1] != "'":
            return (rest, clitic[1:])
    return (u,)


def _check_text(text: str) -> None:
    """A sentence is a str that UTF-8 can encode: lone surrogates raise."""
    text.encode("utf-8")


def tokenize_fse_words(text: str) -> list[str]:
    """The tokens of one sentence by fse_words@1."""
    _check_text(text)
    s = simple_lower(text)
    out: list[str] = []
    for start, end in _runs(_token_chars(s)):
        for part in _parts(s[start:end].strip(".:")):
            part = part.strip(".:'")
            if any(ch in _ALNUM for ch in part):
                out.extend(piece for piece in part.split("-") if piece)
    return out


def _runs(t: list[bool]) -> Iterator[tuple[int, int]]:
    """Step 4: the maximal runs of token chars, as (start, end)."""
    start = -1
    for i in range(len(t) + 1):
        if i < len(t) and t[i]:
            if start < 0:
                start = i
        elif start >= 0:
            yield start, i
            start = -1


def tokenize_external(text: str) -> list[str]:
    """The tokens of one sentence by external: split at whitespace, unchanged."""
    _check_text(text)
    out: list[str] = []
    token: list[str] = []
    for ch in text:
        if ch in WHITESPACE:
            if token:
                out.append("".join(token))
                token = []
        else:
            token.append(ch)
    if token:
        out.append("".join(token))
    return out


def tokenize(text: str, tokenizer: str) -> list[str]:
    """The tokens of one sentence by the named tokenizer."""
    tokenizer = check_tokenizer(tokenizer)
    if tokenizer == EXTERNAL_TOKENIZER:
        return tokenize_external(text)
    if tokenizer == Tokenizer.FSE_WORDS.id:
        return tokenize_fse_words(text)
    raise ValueError(f"{tokenizer} gives ids of its vocabulary; see fse._reference_wordpiece")


def split_lines(data: bytes) -> list[str]:
    """The sentences of a UTF-8 file: one per line, decoded strictly."""
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("the file starts with a UTF-8 byte order mark; remove it")
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return [line.decode("utf-8") for line in lines]


def _hash(data: bytes) -> int:
    """FNV-1a of fastText, each byte sign-extended, in uint32."""
    h = 2166136261
    for byte in data:
        h ^= (byte - 256 if byte > 127 else byte) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def ft_ngram_hashes(token: str, min_n: int, max_n: int, bucket: int, max_ngrams: int) -> list[int]:
    """fasttext@1: the buckets of the first ``max_ngrams`` ngrams of ``<token>``."""
    if bucket == 0:
        return []
    word = f"<{token}>".encode()
    # The start byte of every code point.
    starts = [i for i, byte in enumerate(word) if byte & 0xC0 != 0x80]
    out: list[int] = []
    for i, j in _ngram_spans(starts, len(word), min_n, max_n):
        if len(out) == max_ngrams:
            return out
        out.append(_hash(word[i:j]) % bucket)
    return out


def _ngram_spans(
    starts: list[int], length: int, min_n: int, max_n: int
) -> Iterator[tuple[int, int]]:
    """The byte spans (i, j) of the ngrams of ``min_n`` to ``max_n`` code points, by start,
    then length; the single "<" and ">" are no ngrams."""
    for k, i in enumerate(starts):
        for n in range(1, max_n + 1):
            if k + n > len(starts):
                break
            j = starts[k + n] if k + n < len(starts) else length
            if n < min_n or (n == 1 and (i == 0 or j == length)):
                continue
            yield i, j


def encode_texts(
    texts: Sequence[str],
    words: Mapping[bytes, int],
    *,
    tokenizer: str,
    fasttext: tuple[int, int, int] | None = None,
    max_ngrams: int = 0,
    first_row: int = 0,
) -> Batch:
    """Stages 1 and 2: the batch of the sentences ``texts`` for a vocabulary.

    ``words`` maps the UTF-8 bytes of every vocabulary word to its id. With ``fasttext``
    (min_n, max_n, bucket), a token that is no word becomes ``OOV_TOKEN`` with its first
    ``max_ngrams`` buckets; without it, the token is dropped.
    """
    tokens: list[int] = []
    offsets = [0]
    ngram_offsets = [0]
    ngrams: list[int] = []
    for text in texts:
        for token in tokenize(text, tokenizer):
            word = words.get(token.encode())
            if word is not None:
                tokens.append(word)
            elif fasttext is not None:
                tokens.append(OOV_TOKEN)
                ngrams.extend(ft_ngram_hashes(token, *fasttext, max_ngrams))
                ngram_offsets.append(len(ngrams))
        offsets.append(len(tokens))
    if fasttext is None:
        return Batch(first_row, np.array(offsets, np.int64), np.array(tokens, np.uint32))
    return Batch(
        first_row,
        np.array(offsets, np.int64),
        np.array(tokens, np.uint32),
        np.array(ngram_offsets, np.int64),
        np.array(ngrams, np.uint32),
    )

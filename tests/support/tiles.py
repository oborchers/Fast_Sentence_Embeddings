"""The cases and the numpy model of the tiled weighted mean (src/fse/_native_src/kernel/_native_kernel_tiles.h).

The kernel cuts the terms of a sentence (one row and factor per contribution) into tiles
of at most K terms and keeps the accumulators of a block of the dimension in registers
over a tile: blocks of 32 values, then one block each of 16, 8 and 4, then single values.
The cases put witnesses at every block, sub-block and the tail, and at tile boundaries
for every K up to 64; ``model_pool`` computes the tiled kernel in numpy, with a mistake if
asked, so the meta-tests can show that the cases catch each mistake.
"""

from __future__ import annotations

import numpy as np

from fse import _reference
from fse._contract import OOV_TOKEN
from support.conformance import F32, TINY, Case, _rng

BLOCK = 32
# Block edges of 32, 16, 8 and 4 values, the tail, and dimensions at and above the tile
# budget (1024 float32 values give K = 4, 1025 and more take the row loop).
DIMS = (
    *range(1, 10),
    12,
    15,
    16,
    17,
    28,
    31,
    32,
    33,
    47,
    48,
    49,
    50,
    60,
    63,
    64,
    65,
    95,
    96,
    97,
    127,
    128,
    129,
    300,
    301,
    768,
    1024,
    1025,
    2048,
)
LONGEST = 130  # terms: K - 1, K and K + 1 for every K up to 64, twice over
WITNESS_DIM = 63  # 32 + 16 + 8 + 4 + 3: every block and the tail


def regions(dim: int) -> dict[str, slice]:
    """The columns of the main blocks, each sub-block and the tail that ``dim`` has."""
    out, i = {}, BLOCK * (dim // BLOCK)
    if i:
        out["main"] = slice(0, i)
    for width in (16, 8, 4):
        if i + width <= dim:
            out[f"sub{width}"] = slice(i, i + width)
            i += width
    if i < dim:
        out["tail"] = slice(i, dim)
    return out


def _tokens(rng, words: int, size: int) -> tuple:
    return tuple(int(t) for t in rng.integers(0, words, size))


def _block_edges(dtype: str) -> list[Case]:
    out = []
    for dim in DIMS:
        rng = _rng(dim)
        base = rng.standard_normal((20, dim)).astype(F32)
        weights = rng.random(20).astype(F32)
        sentences = tuple(_tokens(rng, 20, size) for size in (1, 3, 0, 7, 70))
        out.append(Case(f"block edges {dim}", dtype, base, weights, sentences))
    return out


def _tile_edges(dtype: str) -> list[Case]:
    out = []
    for dim in (33, 300):
        rng = _rng(1000 + dim)
        base = rng.standard_normal((50, dim)).astype(F32)
        weights = rng.random(50).astype(F32)
        sentences = tuple(_tokens(rng, 50, size) for size in range(1, LONGEST + 1))
        out.append(Case(f"tile edges {dim}", dtype, base, weights, sentences))
    return out


def _unknown_words(dtype: str) -> Case:
    """Unknown words that straddle a tile boundary for every K up to 64, one of more than
    2 * 64 buckets, words without buckets before, at and after a boundary, and several
    unknown words in a row and across sentences. The bucket rows have different
    magnitudes, so their int8 scales differ."""
    rng = _rng(7)
    dim, words, buckets = 37, 30, 300
    base = rng.standard_normal((words, dim)).astype(F32)
    weights = rng.random(words).astype(F32)
    magnitudes = (F32(0.01) + 10 * rng.random((buckets, 1))).astype(F32)
    ngram_base = (rng.standard_normal((buckets, dim)) * magnitudes).astype(F32)
    sentences, ngrams = [], []

    def unknown(m):
        ngrams.append(tuple(int(b) for b in rng.integers(0, buckets, m)))
        return (OOV_TOKEN,)

    for p in range(LONGEST // 2 + 2):
        sentences.append(_tokens(rng, words, p) + unknown(5) + _tokens(rng, words, 3))
        sentences.append(_tokens(rng, words, p) + unknown(0) + _tokens(rng, words, 2) + unknown(3))
    sentences.append(unknown(2 * LONGEST))
    sentences.append(unknown(1) + unknown(0) + unknown(7) + unknown(2))
    sentences.append(unknown(9))
    sentences.append(unknown(0))
    return Case(
        "unknown words across tiles",
        dtype,
        base,
        weights,
        tuple(sentences),
        tuple(ngrams),
        ngram_base,
        0.75,
    )


def _documents(dtype: str) -> list[Case]:
    out = []
    for dim, size in ((300, 20_000), (768, 3_000)):
        rng = _rng(dim + 5)
        base = rng.standard_normal((500, dim)).astype(F32)
        weights = rng.random(500).astype(F32)
        sentences = (_tokens(rng, 500, size), (1, 2), _tokens(rng, 500, 100))
        out.append(Case(f"document {dim}", dtype, base, weights, sentences))
    return out


def _signed_zeros(dtype: str) -> list[Case]:
    """-0.0 in every block, sub-block and the tail (float32 and float16; int8 sums are
    multiples of the smallest subnormal and never reach -0.0 from +0.0).

    start: products of -0.0 from +0.0 give +0.0, a start from the first product -0.0.
    across tiles: fma(-TINY, 0.5, +0.0) is -0.0, and the products of -0.0 that follow keep
    it; a tile that reloads it as +0.0 ends at +0.0."""
    dim = WITNESS_DIM
    rows = np.vstack([np.full(dim, -0.0, F32), np.full(dim, 0.5, F32)])
    weights = np.array([1.0, -TINY], F32)
    lengths = range(1, LONGEST + 1)
    start = tuple((0,) * n for n in lengths)
    across = tuple((1,) + (0,) * (n - 1) for n in lengths)
    return [
        Case("signed zero start", dtype, rows, weights, start),
        Case("signed zero across tiles", dtype, rows, weights, across),
    ]


def _fused() -> Case:
    """The rounded negative product, then the fused one, at every position of a tile:
    zero rows before them add nothing (fma(1, +0.0, a) = a)."""
    f, x = F32(1 + 2**-12), F32(1 + 2**-12)
    dim = WITNESS_DIM
    rows = np.vstack([np.full(dim, x, F32), np.full(dim, -(f * x), F32), np.zeros(dim, F32)])
    weights = np.array([f, 1, 1], F32)
    sentences = tuple((2,) * p + (1, 0) for p in range(LONGEST))
    return Case("fused across tiles", "float32", rows, weights, sentences)


def tile_cases() -> list[Case]:
    out = []
    for dtype in ("float32", "float16", "int8"):
        out += _block_edges(dtype)
        out += _tile_edges(dtype)
        out.append(_unknown_words(dtype))
        out += _documents(dtype)
        if dtype != "int8":
            out += _signed_zeros(dtype)
    out.append(_fused())
    return out


def dirty(shape) -> np.ndarray:
    """An out array of NaN payloads and -0.0, so a kernel that reads it fails."""
    bits = np.full(shape, 0x7FC01234, dtype=np.uint32)
    bits.reshape(-1)[1::2] = 0x80000000
    return bits.view(F32)


# --- The model ----------------------------------------------------------------------------

BUILDER_MISTAKES = (
    "lose the boundary bucket",
    "duplicate the boundary bucket",
    "g from the remaining buckets",
    "the number of terms as divisor",
)
ACC_MISTAKES = (
    "start from the first product",
    "reload -0.0 as +0.0",
    "restart every tile",
    "reverse the terms of a tile",
    "first term of a tile not fused",
    "load the out row in the first tile",
)


def _stored(case):
    a = case.arrays
    rows = _reference.dequantize(a["rows"], None)
    nrows = None if a.get("ngram_rows") is None else _reference.dequantize(a["ngram_rows"], None)
    return rows, a["scales"], nrows, a.get("ngram_scales")


def _factor(weight, scales, index):
    return F32(weight) if scales is None else F32(F32(weight) * scales[index])


class _Tiles:
    """The terms of one sentence in tiles of at most K, as fse_fill_tile cuts them."""

    def __init__(self, k_max: int):
        self.k_max, self.done, self.tile, self.terms = k_max, [], [], 0

    def full(self) -> bool:
        return len(self.tile) == self.k_max

    def flush(self) -> None:
        self.done.append(self.tile)
        self.tile = []

    def push(self, term) -> None:
        if self.full():
            self.flush()
        self.tile.append(term)
        self.terms += 1

    def all(self) -> list:
        return [*self.done, self.tile]


def _unknown_terms(tiles, buckets, table, oov_weight, mistake) -> None:
    nrows, nscales = table
    m = len(buckets)
    g = F32(F32(F32(1) / F32(m)) * F32(oov_weight))
    for i, bucket in enumerate(int(b) for b in buckets):
        boundary = i > 0 and tiles.full()
        if boundary and mistake == "lose the boundary bucket":
            tiles.flush()
            continue
        if boundary and mistake == "duplicate the boundary bucket":
            tiles.push(tiles.tile[-1])
        if boundary and mistake == "g from the remaining buckets":
            g = F32(F32(F32(1) / F32(m - i)) * F32(oov_weight))
        tiles.push((nrows[bucket], _factor(g, nscales, bucket)))


def _sentence_tiles(case, table, sentence, unknown, k_max, mistake):
    rows, scales, nrows, nscales = table
    tiles = _Tiles(k_max)
    for token in sentence:
        if token == OOV_TOKEN:
            buckets = case.ngrams[unknown]
            unknown += 1
            if buckets:
                _unknown_terms(tiles, buckets, (nrows, nscales), case.oov_weight, mistake)
        else:
            tiles.push((rows[token], _factor(case.weights[token], scales, token)))
    return tiles, unknown


def _step(acc, term, where, alternative):
    x, f = term
    fused = _reference.fma32(f, x, acc)
    if alternative is None:
        return fused
    with np.errstate(over="ignore", invalid="ignore"):
        other = alternative(f, x, acc).astype(F32)
    return np.where(where, other, fused).astype(F32)


def _alternative(mistake, t, q):
    """The arithmetic a mistake puts at term q of tile t instead of the fma, or None."""
    if mistake == "start from the first product" and t == 0 and q == 0:
        return lambda f, x, _acc: f * x
    if mistake == "first term of a tile not fused" and t > 0 and q == 0:
        return lambda f, x, acc: (f * x).astype(F32) + acc
    return None


def _run_tiles(tiles, start, where, mistake):
    acc = np.zeros_like(start)
    if mistake == "load the out row in the first tile":
        acc = np.where(where, start, acc).astype(F32)
    for t, tile in enumerate(tiles):
        if t and mistake == "restart every tile":
            acc = np.where(where, F32(0), acc).astype(F32)
        if t and mistake == "reload -0.0 as +0.0":
            acc = np.where(where & (acc == 0), F32(0), acc).astype(F32)
        order = tile[::-1] if mistake == "reverse the terms of a tile" else tile
        for q, term in enumerate(order):
            acc = _step(acc, term, where, _alternative(mistake, t, q))
    return acc


def model_rows(case, k_max: int, mistake: str | None = None, region: str | None = None):
    """The rows of the tiled kernel with tiles of k_max terms, one sentence at a time,
    with a mistake in the columns of ``region`` (all columns if None)."""
    dim = case.base.shape[1]
    where = np.zeros(dim, bool)
    where[regions(dim)[region] if region else slice(None)] = True
    start = dirty(dim)
    table, unknown = _stored(case), 0
    for sentence in case.sentences:
        with np.errstate(over="ignore", invalid="ignore"):
            tiles, unknown = _sentence_tiles(case, table, sentence, unknown, k_max, mistake)
            acc = _run_tiles(tiles.all(), start, where, mistake)
            n = tiles.terms if mistake == "the number of terms as divisor" else len(sentence)
            yield F32(F32(1) / F32(n)) * acc if n else acc


def model_pool(case, k_max: int, mistake: str | None = None, region: str | None = None):
    """The rows of ``model_rows`` as one array."""
    dim = case.base.shape[1]
    out = np.zeros((len(case.sentences), dim), F32)
    for s, row in enumerate(model_rows(case, k_max, mistake, region)):
        out[s] = row
    return out

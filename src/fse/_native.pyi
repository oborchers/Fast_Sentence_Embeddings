# Type stubs of the compiled extension fse._native.

from collections.abc import Callable
from typing import Literal

from numpy import ndarray

from fse._contract import Batch

MAX_NGRAMS_LIMIT: int
MAX_CHUNK_ROWS: int
MAX_DIM: int

def ft_ngram_hashes(
    word: str, min_n: int, max_n: int, bucket: int, max_ngrams: int
) -> list[int]: ...

class VocabTable:
    def __init__(self, vocab_bytes: ndarray, vocab_offsets: ndarray) -> None: ...
    @property
    def nbytes(self) -> int: ...
    def __len__(self) -> int: ...
    def get(self, word: str | bytes) -> int | None: ...

def scan_spans(
    data: ndarray,
    offsets: ndarray | None,
    first: int,
    last: int,
    chunk_bytes: int,
    chunk_rows: int = ...,
) -> tuple[int, int, int, list[tuple[int, int, int]], int, int]: ...
def join_strings(texts: object, threads: int = 1) -> tuple[bytes, ndarray] | None: ...
def join_token_lists(
    sentences: object, threads: int = 1
) -> tuple[bytes, ndarray] | Literal[False] | None: ...
def _test_join_faults(failed_starts: int = 0, growth_limit: int = ...) -> tuple[int, int]: ...
def _test_join_freed() -> int: ...
def _test_sentence_limit(limit: int) -> int: ...
def arrow_null_error(row: int) -> ValueError: ...
def available_isas() -> tuple[str, ...]: ...
def default_isa() -> str: ...

class WeightedMean:
    dim: int
    isa_name: str
    def __init__(
        self,
        vectors: ndarray,
        weights: ndarray,
        scales: ndarray | None = None,
        ngram_vectors: ndarray | None = None,
        ngram_scales: ndarray | None = None,
        oov_weight: float | None = None,
        *,
        isa: str | None = None,
    ) -> None: ...
    def pool(self, batch: object, out: ndarray | None = None) -> ndarray: ...
    def accumulate(self, batch: object) -> tuple[ndarray, ndarray]: ...

def pool_weighted_mean(
    batch: object,
    vectors: ndarray,
    weights: ndarray,
    scales: ndarray | None = None,
    ngram_vectors: ndarray | None = None,
    ngram_scales: ndarray | None = None,
    oov_weight: float | None = None,
    *,
    isa: str | None = None,
) -> ndarray: ...
def accumulate(
    batch: object,
    vectors: ndarray,
    weights: ndarray,
    scales: ndarray | None = None,
    ngram_vectors: ndarray | None = None,
    ngram_scales: ndarray | None = None,
    oov_weight: float | None = None,
    *,
    isa: str | None = None,
) -> tuple[ndarray, ndarray]: ...
def dot16(rows: ndarray, other: ndarray, *, isa: str | None = None) -> ndarray: ...
def postprocess(
    rows: ndarray, steps: object, first_row: int = 0, *, isa: str | None = None
) -> ndarray: ...
def postprocess_pooled(
    rows: ndarray, steps: object, first_row: int = 0, *, isa: str | None = None
) -> ndarray: ...
def store_rows(
    rows: ndarray,
    codes: ndarray,
    scales: ndarray | None = None,
    first_row: int = 0,
    *,
    isa: str | None = None,
) -> None: ...
def score(
    stored: ndarray,
    scales: ndarray | None,
    weights: ndarray,
    bias: float = 0.0,
    *,
    isa: str | None = None,
) -> ndarray: ...

TOKENIZER_EXTERNAL: int
TOKENIZER_FSE_WORDS: int
TOKENIZER_WORDPIECE: int
TOKENIZER_UNIGRAM: int
MAX_SENTENCE_BYTES: int

def tokenize(text: str | bytes, tokenizer: str = "fse_words@1") -> list[str]: ...

class Workspace:
    tokens_seen: int
    def __init__(self) -> None: ...

class WordPieceState:
    table: VocabTable
    unk_id: int
    max_input_chars: int
    max_tokens: int
    max_chars: int
    drop_unknown: bool
    def __init__(
        self,
        table: VocabTable,
        vocab_bytes: ndarray,
        vocab_offsets: ndarray,
        unk_id: int,
        prefix: str,
        max_input_chars_per_word: int,
        added_ids: tuple[int, ...],
        max_tokens: int = 0,
        max_chars: int = 0,
        drop_unknown: bool = True,
        *,
        _generic: bool = False,
        _trie_units: int = ...,
        _trie_fail_at: int | None = None,
    ) -> None: ...
    def _trie_find(self, key: bytes, continuation: bool = False) -> int | None: ...
    @property
    def _trie_bytes(self) -> int: ...
    @property
    def _trie_blocks(self) -> int: ...

class UnigramState:
    table: VocabTable
    unk_id: int
    unk_score: float
    max_tokens: int
    max_chars: int
    drop_unknown: bool
    def __init__(
        self,
        table: VocabTable,
        vocab_bytes: ndarray,
        vocab_offsets: ndarray,
        scores: ndarray,
        unk_id: int,
        added_ids: tuple[int, ...],
        steps: tuple[tuple[object, ...], ...],
        charsmap: tuple[ndarray, bytes, int] | None,
        replacement: bytes,
        max_tokens: int = 0,
        max_chars: int = 0,
        drop_unknown: bool = True,
    ) -> None: ...

def _grapheme_props(code_points: object) -> ndarray: ...
def _grapheme_boundaries(code_points: object) -> list[int]: ...

class TextLookup:
    tokenizer: int
    def __init__(
        self,
        table: VocabTable,
        tokenizer: str,
        fasttext: tuple[int, int, int] | None = None,
        max_ngrams: int = 0,
        wordpiece: WordPieceState | None = None,
        unigram: UnigramState | None = None,
    ) -> None: ...
    def encode(
        self,
        data: object,
        offsets: ndarray | None,
        first: int,
        last: int,
        first_row: int,
        ws: Workspace,
    ) -> Batch: ...

def float32(values: object) -> ndarray: ...
def inverse_norms(stored: ndarray, first_row: int = 0) -> ndarray: ...
def cosines(
    stored: ndarray, inverse: ndarray, queries: ndarray, *, isa: str | None = None
) -> ndarray: ...
def scores(
    stored: ndarray,
    scales: ndarray | None,
    weights: ndarray,
    bias: float | ndarray = ...,
    *,
    isa: str | None = None,
) -> ndarray: ...
def exp1(x: ndarray, *, isa: str | None = None) -> ndarray: ...
def head_logits(
    rows: ndarray,
    weights: ndarray | None,
    bias: ndarray,
    out: ndarray,
    first_row: int = 0,
    *,
    isa: str | None = None,
) -> None: ...
def head_outputs(
    logits: ndarray,
    kind: str,
    link: object,
    out: ndarray,
    first_row: int = 0,
    *,
    isa: str | None = None,
) -> None: ...

class ArrowColumn:
    rows: int
    null_row: int
    nbytes: int
    copied_bytes: int
    @staticmethod
    def from_stream(capsule: object) -> ArrowColumn: ...
    @staticmethod
    def from_array(schema_capsule: object, array_capsule: object) -> ArrowColumn: ...
    @property
    def chunk_starts(self) -> ndarray: ...
    @property
    def byte_starts(self) -> ndarray: ...
    @property
    def released(self) -> bool: ...
    def row_bytes(self, row: int) -> bytes: ...
    def row_address(self, row: int) -> int: ...
    def span_bytes(self, first: int, last: int) -> int: ...
    def scan(
        self, first: int, last: int, chunk_bytes: int, chunk_rows: int = ...
    ) -> tuple[int, int, int, list[tuple[int, int, int]], int, int]: ...
    def encode(
        self, lookup: TextLookup, first: int, last: int, first_row: int, ws: Workspace
    ) -> Batch: ...
    def release(self) -> None: ...

def pool_text(
    kernel: WeightedMean,
    lookup: TextLookup,
    ws: Workspace,
    text: object,
    first: int,
    last: int,
    rows: int,
    first_row: int,
    out: ndarray,
    normalize: bool,
    after_encode: Callable[[], None] | None = None,
    *,
    isa: str | None = None,
) -> int: ...
def _test_corrupt(kind: int) -> int: ...
def _test_nogil_sections() -> int: ...

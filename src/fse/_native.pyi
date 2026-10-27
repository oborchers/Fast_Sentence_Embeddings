# Type stubs of the compiled extension fse._native.

from numpy import ndarray

MAX_NGRAMS_LIMIT: int
MAX_CHUNK_ROWS: int

def ft_ngram_hashes(
    word: str, min_n: int, max_n: int, bucket: int, max_ngrams: int
) -> list[int]: ...

class VocabTable:
    def __init__(self, vocab_bytes: ndarray, vocab_offsets: ndarray) -> None: ...
    def __len__(self) -> int: ...
    def get(self, word: str | bytes) -> int | None: ...

def scan_spans(data: ndarray, offsets: ndarray, first: int, last: int, *args: object) -> object: ...
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
MAX_SENTENCE_BYTES: int

def tokenize(text: str | bytes, tokenizer: str = "fse_words@1") -> list[str]: ...

class Workspace:
    def __init__(self) -> None: ...

class TextLookup:
    tokenizer: int
    def __init__(
        self,
        table: VocabTable,
        tokenizer: str,
        fasttext: tuple[int, int, int] | None = None,
        max_ngrams: int = 0,
    ) -> None: ...
    def encode(
        self,
        data: object,
        offsets: ndarray | None,
        first: int,
        last: int,
        first_row: int,
        ws: Workspace,
    ) -> object: ...

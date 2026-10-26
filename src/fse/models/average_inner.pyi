"""Type stub for the compiled Cython extension fse.models.average_inner."""

from typing import Any

from numpy import ndarray

MAX_WORDS_IN_BATCH: int
MAX_NGRAMS_IN_BATCH: int
FAST_VERSION: int
MAX_NGRAMS_LIMIT: int

def train_average_cy(
    model: Any,
    indexed_sentences: Any,
    target: ndarray,
    memory: tuple[ndarray, ndarray],
) -> tuple[int, int]: ...
def init() -> int: ...

class VocabTable:
    """Immutable map from the UTF-8 bytes of a word to its vocabulary index."""

    def __init__(self, key_to_index: dict[str, int]) -> None: ...
    def get(self, word: str) -> int | None: ...

def scan_spans(
    data: Any, offsets: ndarray | None, first: int, last: int, chunk_bytes: int
) -> tuple[int, int, int, list[tuple[int, int, int]], int, int]: ...
def run_chunk(
    ctx: KernelContext,
    table: VocabTable,
    data: Any,
    offsets: ndarray | None,
    first: int,
    last: int,
    first_row: int,
    scratch: ndarray,
    n_items: int = -1,
) -> tuple[int, int]: ...
def scan_indexed_sentences(
    sentences: Any, progress_per: float, progress: Any
) -> tuple[int, int, int, int, bool, bool]: ...

class KernelContext:
    """Read-only state of one train or infer call, shared by all workers."""

    sources: list[Any]
    prefill_rows: bool
    gram: Any

    def __init__(self, model: Any, target: ndarray) -> None: ...

class EncodedJob:
    """A batch of sentences encoded for the kernel."""

    n_items: int
    first_row: int
    contiguous: bool
    n_sentences: int
    eff_words: int

def encode_job(ctx: KernelContext, batch: Any) -> EncodedJob: ...

class JobEncoder:
    """Read (tokens, index) tuples from an iterable and encode them into jobs."""

    def __init__(self, ctx: KernelContext, iterable: Any, batch_words: int) -> None: ...
    def next_job(self) -> EncodedJob | None: ...

def run_job(ctx: KernelContext, job: EncodedJob, scratch: ndarray) -> tuple[int, int]: ...
def ft_ngram_hashes(
    word: str, min_n: int, max_n: int, bucket: int, max_ngrams: int
) -> list[int]: ...

class RowLocks:
    """Striped locks that serialize adding sentence averages into shared target rows."""

    def __init__(self) -> None: ...

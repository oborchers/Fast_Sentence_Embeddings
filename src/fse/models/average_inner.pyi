"""Type stub for the compiled Cython extension fse.models.average_inner."""

from typing import Any

from numpy import ndarray

MAX_WORDS_IN_BATCH: int
MAX_NGRAMS_IN_BATCH: int
FAST_VERSION: int

def train_average_cy(
    model: Any,
    indexed_sentences: Any,
    target: ndarray,
    memory: tuple[ndarray, ndarray],
) -> tuple[int, int]: ...
def init() -> int: ...
def scan_indexed_sentences(
    sentences: Any, progress_per: float, progress: Any
) -> tuple[int, int, int, int, bool]: ...

class KernelContext:
    """Read-only state of one train or infer call, shared by all workers."""

    def __init__(self, model: Any, target: ndarray) -> None: ...

class EncodedJob:
    """A batch of sentences encoded for the kernel."""

    n_items: int
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

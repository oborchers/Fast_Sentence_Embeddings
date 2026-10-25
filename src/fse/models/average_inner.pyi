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

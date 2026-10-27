# Type stubs of the compiled extension fse._native.

from numpy import ndarray

MAX_NGRAMS_LIMIT: int
MAX_CHUNK_ROWS: int

def ft_ngram_hashes(
    word: str, min_n: int, max_n: int, bucket: int, max_ngrams: int
) -> list[int]: ...

class VocabTable:
    def __init__(self, key_to_index: dict[str, int]) -> None: ...

def scan_spans(data: ndarray, offsets: ndarray, first: int, last: int, *args: object) -> object: ...

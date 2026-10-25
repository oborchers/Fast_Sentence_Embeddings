#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

import builtins
import mmap
import os
from collections.abc import Callable, Iterator, MutableSequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from operator import methodcaller
from pathlib import Path
from typing import Any

import numpy
from gensim.utils import any2unicode
from numpy import concatenate, ndarray
from smart_open import open

from fse._validation import PathLike, validate_arguments

try:
    from fse.models.average_inner import scan_spans
except ImportError:
    scan_spans = None  # type: ignore[assignment]

# Calls .split() on each item like __getitem__ does, also for numpy str_ items.
_split = methodcaller("split")


class BaseIndexedList(MutableSequence):
    def __init__(self, *args: list | ndarray):
        """Base object to be used for feeding in-memory stored lists of sentences to the
        training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single container. Can be single or multiple
            list/ndarray objects. A single list or ndarray is used as is, without a copy,
            so later changes to it are visible here.
        """

        self.items: Any = list()

        if len(args) == 1:
            self._check_list_type(args[0])
            self.items = args[0]
        else:
            for arg in args:
                self.extend(arg)

        super().__init__()

    def _check_list_type(self, obj: object):
        """Checks input validity."""
        if isinstance(obj, set):
            # A set has no order, so the sentence indices would be arbitrary.
            raise TypeError("Arg must be a list or ndarray. A set has no order; pass sorted(...)")
        if isinstance(obj, (list, ndarray)):
            return 1
        else:
            raise TypeError(f"Arg must be a list or ndarray. Got {type(obj)}")

    def _check_str_type(self, obj: object):
        """Checks input validity."""
        if isinstance(obj, str):
            return 1
        else:
            raise TypeError(f"Arg must be str type. Got {type(obj)}")

    def __len__(self):
        """List length.

        Returns
        -------
        int
           Length of the IndexedList
        """
        return len(self.items)

    def __str__(self):
        """Human readable representation of the object's state, used for debugging.

        Returns
        -------
        str
           Human readable representation of the object's state (words and tags).
        """
        return str(self.items)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple ([str], int)
            Returns the core object, a tuple, for every sentence embedding model.
        """
        raise NotImplementedError()

    def __delitem__(self, i: int):  # type: ignore[override]
        """Delete an item."""
        del self.items[i]

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        """Sets an item."""
        self._check_str_type(item)
        self.items[i] = item

    def insert(self, i: int, item: str):
        """Inserts an item at a position."""
        self._check_str_type(item)
        self.items.insert(i, item)

    def append(self, item: str):
        """Appends item at last position."""
        self._check_str_type(item)
        self.insert(len(self.items), item)

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        """Extens list."""
        self._check_list_type(arg)

        if not isinstance(arg, ndarray):
            self.items += arg
        else:
            self.items = concatenate([self.items, arg], axis=0)


class IndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray):
        """Quasi-list to be used for feeding in-memory stored lists of sentences to the
        training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        """
        super().__init__(*args)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i), i)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not IndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(self.items, range(len(self.items)), strict=True)


class CIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray, custom_index: list | ndarray):
        """Quasi-list with custom indices to be used for feeding in-memory stored lists
        of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_index : list, ndarray
            Custom index to support many to one mappings.
        """
        self.custom_index = custom_index

        super().__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(
                f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}"
            )

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i), self.custom_index[i])

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not CIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(self.items, self.custom_index, strict=True)

    def __delitem__(self, i: int):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def insert(self, i: int, item: str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item: str):
        raise NotImplementedError("Method currently not supported")

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")


class SplitIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray):
        """Quasi-list with string splitting to be used for feeding in-memory stored
        lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        """
        super().__init__(*args)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i).split(), i)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not SplitIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(_split, self.items), range(len(self.items)), strict=True)


class SplitCIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray, custom_index: list | ndarray):
        """Quasi-list with custom indices and string splitting to be used for feeding
        in-memory stored lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_index : list, ndarray
            Custom index to support many to one mappings.
        """
        self.custom_index = custom_index

        super().__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(
                f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}"
            )

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i).split(), self.custom_index[i])

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not SplitCIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(_split, self.items), self.custom_index, strict=True)

    def __delitem__(self, i: int):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def insert(self, i: int, item: str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item: str):
        raise NotImplementedError("Method currently not supported")

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")


class CSplitIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray, custom_split: Callable[[str], list[str]]):
        """Quasi-list with custom string splitting to be used for feeding in-memory
        stored lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_split : callable
            Split function to be used to convert strings into list of str.
        """
        self.custom_split = custom_split
        super().__init__(*args)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.custom_split(self.items.__getitem__(i)), i)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not CSplitIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(self.custom_split, self.items), range(len(self.items)), strict=True)


class CSplitCIndexedList(BaseIndexedList):
    def __init__(
        self,
        *args: list | ndarray,
        custom_split: Callable[[str], list[str]],
        custom_index: list | ndarray,
    ):
        """Quasi-list with custom indices and ustom string splitting to be used for
        feeding in-memory stored lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_split : callable
            Split function to be used to convert strings into list of str.
        custom_index : list, ndarray
            Custom index to support many to one mappings.
        """
        self.custom_split = custom_split
        self.custom_index = custom_index

        super().__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(
                f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}"
            )

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.custom_split(self.items.__getitem__(i)), self.custom_index[i])

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not CSplitCIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(self.custom_split, self.items), self.custom_index, strict=True)

    def __delitem__(self, i: int):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def insert(self, i: int, item: str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item: str):
        raise NotImplementedError("Method currently not supported")

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")


class IndexedLineDocument:
    @validate_arguments
    def __init__(self, path: PathLike, get_able: bool = True):
        """Iterate over a file that contains sentences: one line = tuple([str], int).

        Words are expected to be already preprocessed and separated by whitespace. Sentence tags are constructed
        automatically from the sentence line number.

        Parameters
        ----------
        path : str
            The path of the file to read and return lines from
        get_able : bool, optional
            Use to determine if the IndexedLineDocument is indexable.
            This functionality is required if you want to pass an indexable to
            :meth:`~fse.models.sentencevectors.SentenceVectors.most_similar`.

        """
        self.path = Path(path)
        self.line_offset: list[int] = []
        self.get_able = get_able

        if self.get_able:
            self._build_offsets()

    def _build_offsets(self):
        """Builds an offset table to index the file."""
        with open(self.path, "rb") as f:
            offset = f.tell()
            for line in f:
                self.line_offset.append(offset)
                offset += len(line)

    def __getitem__(self, i):
        """Returns the line indexed by i. Primarily used for.

        :meth:`~fse.models.sentencevectors.SentenceVectors.most_similar`

        Parameters
        ----------
        i : int
            The line index used to index the file

        Returns
        -------
        str
            line at the current index
        """
        if not self.get_able:
            raise RuntimeError("To index the lines, you must contruct with get_able=True")

        with open(self.path, "rb") as f:
            f.seek(self.line_offset[i])
            output = f.readline()
            f.seek(0)
            return any2unicode(output).rstrip()

    def __iter__(self):
        """Iterate through the lines in the source.

        Yields
        ------
        tuple : (list[str], int)
            Tuple of list of string and index
        """
        with open(self.path, "rb") as f:
            for i, line in enumerate(f):
                yield (any2unicode(line).split(), i)


# ---------------------------------------------------------------------------------------
# Native inputs: read, split and look up without the GIL when the extension is built.
# ---------------------------------------------------------------------------------------

# Compression formats that smart_open decompresses by file extension; CorpusFile reads
# raw bytes, so such files belong to IndexedLineDocument.
_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".lzma", ".zst", ".zstd", ".z", ".zip"}

# Bytes per training chunk: small enough to balance the workers and report progress,
# large enough that the per-chunk overhead does not matter.
_CHUNK_MIN_BYTES = 1 << 16
_CHUNK_MAX_BYTES = 1 << 20


class _NativeSource:
    """A contiguous UTF-8 buffer: the lines of a file, or strings delimited by offsets."""

    def __init__(self, data: Any, offsets: ndarray | None, keep: Any = None):
        self.data = data
        self.offsets = offsets
        self._keep = keep  # the mmap or file object that owns data

    @property
    def size(self) -> int:
        """Number of positions: bytes in line mode, strings otherwise."""
        return len(self.data) if self.offsets is None else len(self.offsets) - 1

    def line_start_after(self, position: int) -> int:
        """The first line start at or after position (line mode)."""
        if position == 0 or self.data[position - 1] == 10:
            return position
        newline = self._keep.find(b"\n", position) if self._keep is not None else -1
        return len(self.data) if newline < 0 else newline + 1


def _scan_native(owner: Any, workers: int, progress: Callable[[int, int], None]) -> dict | None:
    """Validate and count a native source in parallel; None without the extension.

    Splits the source into ranges that start at sentence starts, scans them in threads
    without the GIL and stores the training chunks on owner. Raises UnicodeDecodeError
    for the first invalid sentence of the whole source.
    """
    if scan_spans is None:
        return None

    source = owner._open_source()
    size = source.size
    n_ranges = max(1, min(4 * workers, size))
    bounds = [0]
    for k in range(1, n_ranges):
        position = size * k // n_ranges
        if source.offsets is None:
            position = source.line_start_after(position)
        if position > bounds[-1]:
            bounds.append(position)
    if bounds[-1] != size:
        bounds.append(size)
    bounds = sorted(set(bounds))
    ranges = list(pairwise(bounds))

    total_bytes = len(source.data)
    chunk_bytes = max(_CHUNK_MIN_BYTES, min(_CHUNK_MAX_BYTES, total_bytes // max(8 * workers, 1)))

    def scan(span: tuple[int, int]):
        return scan_spans(source.data, source.offsets, span[0], span[1], chunk_bytes)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(scan, ranges))

    total_sentences = total_words = empty = 0
    chunks: list[tuple[int, int, int, int]] = []
    for sentences, tokens, empty_here, range_chunks, error_byte, error_sentence in results:
        if error_byte >= 0:
            line = total_sentences + error_sentence
            raise UnicodeDecodeError(
                "utf-8",
                bytes(source.data[error_byte : error_byte + 4]),
                0,
                1,
                f"invalid UTF-8 in sentence {line} at byte {error_byte}",
            )
        bounds_here = [first for _, _, first in range_chunks] + [sentences]
        for (start, end, first), next_first in zip(range_chunks, bounds_here[1:], strict=True):
            chunks.append((start, end, total_sentences + first, next_first - first))
        total_sentences += sentences
        total_words += tokens
        empty += empty_here
        progress(total_sentences, total_words)

    owner._scan_state = (owner._fingerprint(), chunks)
    return {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "empty_sentences": empty,
        "max_index": total_sentences,
        "unique_indices": True,
    }


class CorpusFile:
    """A local UTF-8 text file with one sentence per line, read natively.

    Every line is one sentence; its index is the line number, empty lines included. A line
    ends after "\\n"; the last line needs none. Lines are decoded strictly as UTF-8 and
    split like str.split(), so the sentences equal those of IndexedLineDocument. With the
    compiled extension, scan and training read, split and look up the words without the
    GIL in every worker; without it, CorpusFile is iterated like IndexedLineDocument.

    Compressed or remote files are not supported; use IndexedLineDocument for those.
    The file must not change between the scan and the end of training.
    """

    @validate_arguments
    def __init__(self, path: PathLike):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"{self.path} is not a file")
        if self.path.suffix.lower() in _COMPRESSED_SUFFIXES:
            raise ValueError(
                f"{self.path} looks compressed; CorpusFile reads plain UTF-8 text. "
                "Use IndexedLineDocument for compressed files."
            )
        self._scan_state: Any = None
        self._line_offsets: ndarray | None = None

    def __getstate__(self):
        # The scan state and line offsets belong to one call and are rebuilt on demand.
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self._scan_state = None
        self._line_offsets = None

    def __iter__(self) -> Iterator[tuple]:
        """Yield (tokens, line number), exactly like IndexedLineDocument."""
        with builtins.open(self.path, "rb") as f:
            for i, line in enumerate(f):
                yield (any2unicode(line).split(), i)

    def __len__(self) -> int:
        """Number of lines."""
        data = numpy.frombuffer(self._open_source().data, dtype=numpy.uint8)
        if not len(data):
            return 0
        # Count in blocks, so the comparison never needs a temporary as large as the file.
        block = 1 << 26
        count = sum(
            int(numpy.count_nonzero(data[i : i + block] == 10)) for i in range(0, len(data), block)
        )
        return count + (0 if data[-1] == 10 else 1)

    def __getitem__(self, i: int) -> str:
        """The line i without its line break, for the indexable of most_similar."""
        if self._line_offsets is None:
            data = numpy.frombuffer(self._open_source().data, dtype=numpy.uint8)
            starts = numpy.flatnonzero(data == 10) + 1
            if len(data) and data[-1] != 10:
                starts = numpy.append(starts, len(data) + 1)
            self._line_offsets = numpy.concatenate([[0], starts]).astype(numpy.int64)
        offsets = self._line_offsets
        if not -(len(offsets) - 1) <= i < len(offsets) - 1:
            raise IndexError(f"line {i} is out of range for {len(offsets) - 1} lines")
        i %= len(offsets) - 1
        with builtins.open(self.path, "rb") as f:
            f.seek(int(offsets[i]))
            # rstrip() as in IndexedLineDocument.__getitem__.
            return any2unicode(f.read(int(offsets[i + 1] - offsets[i]))).rstrip()

    def _fingerprint(self) -> tuple[int, int]:
        stat = os.stat(self.path)
        return (stat.st_size, stat.st_mtime_ns)

    def _open_source(self) -> _NativeSource:
        with builtins.open(self.path, "rb") as f:
            if os.fstat(f.fileno()).st_size == 0:
                return _NativeSource(b"", None)
            mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return _NativeSource(memoryview(mapped), None, keep=mapped)

    def _scan_native(self, workers: int, progress: Callable[[int, int], None]) -> dict | None:
        return _scan_native(self, workers, progress)

    def _native_chunks(self) -> tuple[_NativeSource, list[tuple[int, int, int, int]]]:
        """The source and the chunks of the last scan; raises if the file changed since."""
        if self._scan_state is None:
            raise RuntimeError("CorpusFile must be scanned before training")
        fingerprint, chunks = self._scan_state
        if self._fingerprint() != fingerprint:
            raise RuntimeError(f"{self.path} changed between the scan and training")
        return self._open_source(), chunks


class TextList:
    """Strings in memory, one sentence each, split natively like str.split().

    The index of a sentence is its position. The strings are copied once into a UTF-8
    buffer, so later changes to the passed list are not seen. The sentences equal those of
    SplitIndexedList; with the compiled extension they are split and looked up without the
    GIL in every worker.
    """

    def __init__(self, strings: list | tuple | ndarray):
        if isinstance(strings, (set, frozenset)):
            raise TypeError(
                "TextList takes a list or ndarray. A set has no order; pass sorted(...)"
            )
        if not isinstance(strings, (list, tuple, ndarray)):
            raise TypeError(f"TextList takes a list, tuple or ndarray of str, got {type(strings)}")
        items = []
        for i, s in enumerate(strings):
            if not isinstance(s, str):
                raise TypeError(f"At {i}: TextList items must be str, got {type(s)}")
            items.append(str(s))
        self.items = tuple(items)
        self._source: _NativeSource | None = None
        self._scan_state: Any = None

    def __getstate__(self):
        return {"items": self.items}

    def __setstate__(self, state):
        self.items = state["items"]
        self._source = None
        self._scan_state = None

    def __iter__(self) -> Iterator[tuple]:
        """Yield (tokens, position), exactly like SplitIndexedList."""
        return zip(map(str.split, self.items), range(len(self.items)), strict=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> str:
        """The string i, for the indexable of most_similar."""
        return self.items[i]

    def _fingerprint(self) -> None:
        return None

    def _open_source(self) -> _NativeSource:
        if self._source is None:
            encoded = [s.encode("utf-8") for s in self.items]
            offsets = numpy.zeros(len(encoded) + 1, dtype=numpy.int64)
            numpy.cumsum([len(e) for e in encoded], out=offsets[1:])
            self._source = _NativeSource(b"".join(encoded), offsets)
        return self._source

    def _scan_native(self, workers: int, progress: Callable[[int, int], None]) -> dict | None:
        return _scan_native(self, workers, progress)

    def _native_chunks(self) -> tuple[_NativeSource, list[tuple[int, int, int, int]]]:
        if self._scan_state is None:
            raise RuntimeError("TextList must be scanned before training")
        return self._open_source(), self._scan_state[1]

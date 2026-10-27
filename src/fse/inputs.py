#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

import builtins
import mmap
import operator
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

# Bytes read at a time while IndexedLineDocument builds its line offsets.
_OFFSET_BLOCK_BYTES = 1 << 26

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

    # With the compiled extension, a SplitIndexedList of str is read like a TextList: the
    # scan encodes the strings into one UTF-8 buffer, and every worker splits and looks
    # up its own chunk without the GIL. The sentences are those of __iter__. The buffer is
    # built anew for every train or infer call, so changes to the list are seen, and
    # released when the call ends (_release_native). Other items, an override of
    # __getitem__ or __iter__, or strings that cannot be encoded keep the Python path.

    def _clear_native_scan(self):
        self._scan_state = None
        self._native_source = None

    def _release_native(self):
        self._clear_native_scan()

    def _open_source(self) -> "_NativeSource":
        return self._native_source

    def _scan_native(self, workers: int, progress: Callable[[int, int], None]) -> dict | None:
        if (
            scan_spans is None
            or type(self).__getitem__ is not SplitIndexedList.__getitem__
            or type(self).__iter__ is not SplitIndexedList.__iter__
            or any(type(s) not in (str, numpy.str_) for s in self.items)
        ):
            return None
        try:
            self._native_source = _encode_strings(self.items)
        except UnicodeEncodeError:
            # The Python path splits such strings and treats their words as unknown.
            return None
        return _scan_native(self, workers, progress)

    def _native_chunks(self) -> "tuple[_NativeSource, list[tuple[int, int, int, int]]] | None":
        """Source and chunks of the last native scan, or None after a Python scan."""
        if getattr(self, "_scan_state", None) is None:
            return None
        return self._native_source, self._scan_state[1]


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
        self.line_offset: ndarray = numpy.empty(0, dtype=numpy.int64)
        self.get_able = get_able

        if self.get_able:
            self._build_offsets()

    def _build_offsets(self):
        """Builds an offset table to index the file: the start of every line, as int64.

        A line ends after "\\n" and the last one needs none, as for ``for line in f``. The
        file is read in blocks of ``_OFFSET_BLOCK_BYTES``; the finished offsets take 8 bytes
        per line, and while the offsets of the blocks are joined they exist twice. For a
        compressed file the offsets are positions in the decompressed stream, where
        ``seek`` goes.
        """
        starts = [numpy.zeros(1, dtype=numpy.int64)]
        position = 0
        with open(self.path, "rb") as f:
            while block := f.read(_OFFSET_BLOCK_BYTES):
                data = numpy.frombuffer(block, dtype=numpy.uint8)
                starts.append(numpy.flatnonzero(data == 10).astype(numpy.int64) + (position + 1))
                position += len(block)
        offsets = concatenate(starts)
        # The position after a final line break starts no line, nor does 0 in an empty file.
        self.line_offset = offsets[:-1] if offsets[-1] == position else offsets

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
            # A scalar index only, as for the list the offsets used to be.
            f.seek(int(self.line_offset[operator.index(i)]))
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
    """A contiguous UTF-8 buffer: the lines of a file, or strings delimited by offsets.

    For a file, the buffer is a read-only memory map. Its fingerprint is read from the
    descriptor that was mapped; later checks compare it with the file at path.
    """

    def __init__(
        self,
        data: Any,
        offsets: ndarray | None,
        path: Path | None = None,
        mapped: Any = None,
        fingerprint: Any = None,
    ):
        self.data = data
        self.offsets = offsets
        self._path = path
        self._mapped = mapped
        self.fingerprint = fingerprint

    @property
    def size(self) -> int:
        """Number of positions: bytes in line mode, strings otherwise."""
        return len(self.data) if self.offsets is None else len(self.offsets) - 1

    def line_start_after(self, position: int) -> int:
        """The first line start at or after position (line mode)."""
        if position == 0 or self.data[position - 1] == 10:
            return position
        newline = self._mapped.find(b"\n", position)
        return len(self.data) if newline < 0 else newline + 1

    def sentence_bounds(self, byte: int) -> tuple[int, int, int]:
        """Start and end byte of the sentence that contains byte, and its number."""
        if self.offsets is not None:
            k = int(numpy.searchsorted(self.offsets, byte, side="right")) - 1
            return int(self.offsets[k]), int(self.offsets[k + 1]), k
        start = self._mapped.rfind(b"\n", 0, byte) + 1
        end = self._mapped.find(b"\n", byte)
        # Including the line break, like the lines IndexedLineDocument decodes.
        return start, len(self.data) if end < 0 else end + 1, -1

    def close(self):
        """Release the map early; otherwise it is released with the last reference."""
        if self._mapped is not None:
            self.data.release()
            self._mapped.close()
            self._mapped = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def changed(self) -> bool:
        """Whether the file differs from the one this source mapped (always False for strings)."""
        if self.fingerprint is None or self._path is None:
            return False
        try:
            return _stat_fingerprint(os.stat(self._path)) != self.fingerprint
        except OSError:
            return True


def _encode_strings(items: Any) -> _NativeSource:
    """The strings as one UTF-8 buffer with their offsets, for the native routines."""
    encoded = []
    for i, s in enumerate(items):
        try:
            encoded.append(s.encode("utf-8"))
        except UnicodeEncodeError as err:
            raise UnicodeEncodeError(
                err.encoding, err.object, err.start, err.end, f"{err.reason} in string {i}"
            ) from None
    offsets = numpy.zeros(len(encoded) + 1, dtype=numpy.int64)
    numpy.cumsum([len(e) for e in encoded], out=offsets[1:])
    return _NativeSource(b"".join(encoded), offsets)


def _stat_fingerprint(stat: os.stat_result) -> tuple[int, int, int, int]:
    """Identity and version of a file: device, inode, size and modification time."""
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _decode_error(source: _NativeSource, byte: int, sentence: int) -> UnicodeDecodeError:
    """The error Python's decoder raises for the sentence, naming its number and offset."""
    start, end, _ = source.sentence_bounds(byte)
    raw = bytes(source.data[start:end])
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as err:
        return UnicodeDecodeError(
            err.encoding,
            err.object,
            err.start,
            err.end,
            f"{err.reason} in sentence {sentence} (byte {start + err.start} of the source)",
        )
    raise RuntimeError(f"no UTF-8 error in the sentence at byte {byte}")  # pragma: no cover


# About this many bytes per scan task, so progress is reported and a failed or interrupted
# scan stops after a bounded amount of work. A single longer sentence stays in one task.
_SCAN_RANGE_MAX_BYTES = 1 << 26


def _scan_native(owner: Any, workers: int, progress: Callable[[int, int], None]) -> dict | None:
    """Validate and count a native source in parallel; None without the extension.

    Splits the source into ranges that start at sentence starts, scans them in threads
    without the GIL and stores the training chunks on owner. Raises UnicodeDecodeError
    for the first invalid sentence of the whole source.
    """
    if scan_spans is None:
        return None

    with owner._open_source() as source:
        return _scan_source(owner, source, workers, progress)


def _scan_source(
    owner: Any, source: _NativeSource, workers: int, progress: Callable[[int, int], None]
) -> dict:
    """The body of _scan_native for an open source."""
    size = source.size
    total_bytes = len(source.data)
    n_ranges = max(1, min(max(4 * workers, -(-total_bytes // _SCAN_RANGE_MAX_BYTES)), size))
    bounds = [0]
    for k in range(1, n_ranges):
        # Ranges of similar byte size; for strings, the string that holds that byte.
        target = total_bytes * k // n_ranges
        if source.offsets is None:
            position = source.line_start_after(target)
        else:
            position = int(numpy.searchsorted(source.offsets, target, side="right")) - 1
        if position > bounds[-1]:
            bounds.append(position)
    if bounds[-1] != size:
        bounds.append(size)
    ranges = list(pairwise(bounds))
    chunk_bytes = max(_CHUNK_MIN_BYTES, min(_CHUNK_MAX_BYTES, total_bytes // max(8 * workers, 1)))

    def scan(span: tuple[int, int]):
        return scan_spans(source.data, source.offsets, span[0], span[1], chunk_bytes)

    total_sentences = total_words = empty = 0
    chunks: list[tuple[int, int, int, int]] = []
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        # map yields the results in order as they finish, so progress and errors are
        # handled during the scan; pending ranges are cancelled on an error.
        for sentences, tokens, empty_here, range_chunks, error_byte, error_sentence in pool.map(
            scan, ranges
        ):
            if error_byte >= 0:
                raise _decode_error(source, error_byte, total_sentences + error_sentence)
            firsts = [first for _, _, first in range_chunks] + [sentences]
            for (start, end, first), following in zip(range_chunks, firsts[1:], strict=True):
                chunks.append((start, end, total_sentences + first, following - first))
            total_sentences += sentences
            total_words += tokens
            empty += empty_here
            progress(total_sentences, total_words)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    owner._scan_state = (source.fingerprint, chunks)
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

    Compressed or remote files are not supported; use IndexedLineDocument for compressed
    files. The file must not change from the scan to the end of training. Changes that
    alter its device, inode, size or modification time raise; a change that keeps all four
    is not detected. The file is memory-mapped, so truncating it while it is read can end
    the process with SIGBUS on Linux, as for any memory map.
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
        self._python_fingerprint: Any = None
        self._line_offsets: tuple[Any, ndarray] | None = None

    def __getstate__(self):
        # The scan state and line offsets belong to one file version and are rebuilt.
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self._scan_state = None
        self._python_fingerprint = None
        self._line_offsets = None

    def __iter__(self) -> Iterator[tuple]:
        """Yield (tokens, line number), exactly like IndexedLineDocument.

        The Python scan and the Python training path both iterate; the file must be the one
        the first iteration after the last scan read, which is checked before and after.
        """
        with builtins.open(self.path, "rb") as f:
            fingerprint = _stat_fingerprint(os.fstat(f.fileno()))
            if self._python_fingerprint is None:
                self._python_fingerprint = fingerprint
            elif fingerprint != self._python_fingerprint:
                raise RuntimeError(f"{self.path} changed between the scan and training")
            for i, line in enumerate(f):
                yield (any2unicode(line).split(), i)
            if _stat_fingerprint(os.fstat(f.fileno())) != fingerprint:
                raise RuntimeError(f"{self.path} changed while it was read")

    def __len__(self) -> int:
        """Number of lines."""
        with self._open_source() as source:
            data = numpy.frombuffer(source.data, dtype=numpy.uint8)
            try:
                if not len(data):
                    return 0
                # Count in blocks, so no temporary is as large as the file.
                block = 1 << 26
                count = sum(
                    int(numpy.count_nonzero(data[i : i + block] == 10))
                    for i in range(0, len(data), block)
                )
                last_is_newline = bool(data[-1] == 10)
            finally:
                del data  # the map can only close without exported buffers
        return count + (0 if last_is_newline else 1)

    def __getitem__(self, i: int) -> str:
        """The line i without trailing whitespace, for the indexable of most_similar."""
        with self._open_source() as source:
            if self._line_offsets is None or self._line_offsets[0] != source.fingerprint:
                data = numpy.frombuffer(source.data, dtype=numpy.uint8)
                try:
                    starts = numpy.flatnonzero(data == 10) + 1
                    if len(data) and data[-1] != 10:
                        starts = numpy.append(starts, len(data))
                finally:
                    del data  # the map can only close without exported buffers
                offsets = numpy.concatenate([[0], starts]).astype(numpy.int64)
                self._line_offsets = (source.fingerprint, offsets)
            offsets = self._line_offsets[1]
            n = len(offsets) - 1
            if not -n <= i < n:
                raise IndexError(f"line {i} is out of range for {n} lines")
            i %= n
            # rstrip() as in IndexedLineDocument.__getitem__.
            return any2unicode(bytes(source.data[offsets[i] : offsets[i + 1]])).rstrip()

    def _clear_native_scan(self):
        self._scan_state = None
        self._python_fingerprint = None

    def _open_source(self) -> _NativeSource:
        with builtins.open(self.path, "rb") as file:
            # The fingerprint of exactly the file that is mapped, read from its descriptor.
            fingerprint = _stat_fingerprint(os.fstat(file.fileno()))
            if fingerprint[2] == 0:
                return _NativeSource(b"", None, self.path, fingerprint=fingerprint)
            # The map stays valid after the file is closed.
            mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        return _NativeSource(memoryview(mapped), None, self.path, mapped, fingerprint)

    def _scan_native(self, workers: int, progress: Callable[[int, int], None]) -> dict | None:
        return _scan_native(self, workers, progress)

    def _native_chunks(self) -> tuple[_NativeSource, list[tuple[int, int, int, int]]] | None:
        """Source and chunks of the last native scan, or None after a Python scan.

        Raises if the file now differs from the scanned one.
        """
        if self._scan_state is None:
            return None
        fingerprint, chunks = self._scan_state
        source = self._open_source()
        if source.fingerprint != fingerprint:
            source.close()
            raise RuntimeError(f"{self.path} changed between the scan and training")
        return source, chunks


class TextList:
    """Strings in memory, one sentence each, split natively like str.split().

    The index of a sentence is its position. The strings are encoded once into a UTF-8
    buffer when the TextList is created, so later changes to the passed list are not seen
    and strings that cannot be encoded, such as lone surrogates, raise at once. Items must
    be str or numpy str_; use SplitIndexedList for str subclasses with their own split().
    The sentences equal those of SplitIndexedList; with the compiled extension they are
    split and looked up without the GIL in every worker.
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
            if type(s) not in (str, numpy.str_):
                raise TypeError(
                    f"At {i}: TextList items must be str, got {type(s)}. "
                    "Use SplitIndexedList for other types."
                )
            items.append(str(s))
        self.items = tuple(items)
        self._source = _encode_strings(self.items)
        self._scan_state: Any = None

    def __getstate__(self):
        return {"items": self.items}

    def __setstate__(self, state):
        self.items = state["items"]
        self._source = _encode_strings(self.items)
        self._scan_state = None

    def __iter__(self) -> Iterator[tuple]:
        """Yield (tokens, position), exactly like SplitIndexedList."""
        return zip(map(str.split, self.items), range(len(self.items)), strict=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> str:
        """The string i, for the indexable of most_similar."""
        return self.items[i]

    def _clear_native_scan(self):
        self._scan_state = None

    def _open_source(self) -> _NativeSource:
        return self._source

    def _scan_native(self, workers: int, progress: Callable[[int, int], None]) -> dict | None:
        return _scan_native(self, workers, progress)

    def _native_chunks(self) -> tuple[_NativeSource, list[tuple[int, int, int, int]]] | None:
        """Source and chunks of the last native scan, or None after a Python scan."""
        if self._scan_state is None:
            return None
        return self._source, self._scan_state[1]

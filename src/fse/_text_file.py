"""File sources: the lines of a local text file as sentences (ARCHITECTURE.md, "Tokenizers").

A ``FileSource`` memory-maps the file and scans and encodes it like the sources of
``fse._text``; it checks that the file does not change while it is read.
"""

from __future__ import annotations

import mmap
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from fse._contract import Batch
from fse._text import Chunk, Scan, TextLookup, Workspace, _Buffered

# Compressed files hold no text lines; read them decompressed.
_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".lzma", ".zst", ".zstd", ".z", ".zip"}
_BOM = b"\xef\xbb\xbf"


def _fingerprint(stat: os.stat_result) -> tuple[int, int, int, int]:
    """Identity and version of a file: device, inode, size and modification time."""
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class FileSource(_Buffered):
    """The lines of a local UTF-8 text file, one sentence per line, memory-mapped.

    A line ends at "\\n", which is not part of it; the last line needs none. The file must
    not change while the source is open: a change of its device, inode, size or
    modification time raises at the next encode, and truncating a mapped file can end the
    process with SIGBUS, as for any memory map. Use it as a context manager, or close it.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"{self.path} is not a regular file")
        if self.path.suffix.lower() in _COMPRESSED_SUFFIXES:
            raise ValueError(f"{self.path} looks compressed; FileSource reads plain UTF-8 text")
        self._offsets = None
        self._closed = False
        self._mapped: mmap.mmap | None = None
        with open(self.path, "rb") as file:
            self._stat = _fingerprint(os.fstat(file.fileno()))
            if self._stat[2]:
                self._mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self._data = memoryview(self._mapped) if self._mapped is not None else b""
        if bytes(self._data[:3]) == _BOM:
            self.close()
            raise ValueError(f"{self.path} starts with a UTF-8 byte order mark; remove it")

    def _check_open(self) -> None:
        """Raise if the source is closed, or the file differs from the mapped one."""
        if self._closed:
            raise ValueError(f"the FileSource of {self.path} is closed")
        self._check_unchanged()

    def __len__(self) -> int:
        data = self._array()
        if not len(data):
            return 0
        count = sum(
            int(np.count_nonzero(data[i : i + (1 << 26)] == 10))
            for i in range(0, len(data), 1 << 26)
        )
        self._check_unchanged()
        return count + (0 if data[-1] == 10 else 1)

    def _check_unchanged(self) -> None:
        try:
            current = _fingerprint(os.stat(self.path))
        except OSError:
            current = None
        if current != self._stat:
            raise RuntimeError(f"{self.path} changed while it was read")

    def _after_encode(self) -> Callable[[], None]:
        return self._check_unchanged  # a change while the chunk was encoded

    def encode(self, chunk: Chunk, lookup: TextLookup, workspace: Workspace) -> Batch:
        self._check_open()
        batch = super().encode(chunk, lookup, workspace)
        self._check_unchanged()  # a change while the chunk was encoded
        return batch

    def scan_until_error(
        self,
        *,
        workers: int = 1,
        chunk_bytes: int | None = None,
        chunk_rows: int | None = None,
        first_row: int = 0,
    ) -> tuple[Scan, Exception | None]:
        result = super().scan_until_error(
            workers=workers, chunk_bytes=chunk_bytes, chunk_rows=chunk_rows, first_row=first_row
        )
        self._check_unchanged()  # a change during the scan
        return result

    def close(self) -> None:
        """Release the memory map."""
        self._closed = True
        if self._mapped is not None:
            try:
                if isinstance(self._data, memoryview):
                    self._data.release()
                self._mapped.close()
            except BufferError:
                pass  # arrays still view the map; it is released with the last of them
            self._mapped = None
            self._data = b""

    def __enter__(self) -> FileSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

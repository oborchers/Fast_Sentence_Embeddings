"""Memory between two pages without access, to catch a read or a write past an array."""

from __future__ import annotations

import ctypes
import mmap

import numpy as np


class Guarded:
    """Memory of nbytes between two pages without access, at the end or at the start."""

    def __init__(self, nbytes: int, at_end: bool):
        page = mmap.PAGESIZE
        pages = max(1, -(-nbytes // page))
        self.map = mmap.mmap(-1, (pages + 2) * page)
        char = ctypes.c_char.from_buffer(self.map)
        base = ctypes.addressof(char)
        del char
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        for index in (0, pages + 1):
            assert libc.mprotect(base + index * page, page, 0) == 0, ctypes.get_errno()
        self.offset = page + (pages * page - nbytes if at_end else 0)
        self.nbytes = nbytes

    def array(self, values: np.ndarray) -> np.ndarray:
        view = np.frombuffer(self.map, np.uint8, self.nbytes, self.offset)
        view = view.view(values.dtype).reshape(values.shape)
        view[...] = values
        return view

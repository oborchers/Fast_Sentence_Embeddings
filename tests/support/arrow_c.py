"""A producer of the Arrow C Data and C Stream interfaces in ctypes, without pyarrow.

It exports strings in the three layouts fse reads (``u``, ``U``, ``vu``) as PyCapsules,
from buffers it builds or from raw ones a test gives (wrong offsets, views outside their
buffer, NULL pointers, misaligned buffers), and records every release: which struct, at
which address, with which ``private_data``. The capsules have no destructor, so an array
fse did not take over is never released and a test sees it.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

import numpy as np

c_int64 = ctypes.c_int64
c_void_p = ctypes.c_void_p


class ArrowSchema(ctypes.Structure):
    pass


class ArrowArray(ctypes.Structure):
    pass


class ArrowArrayStream(ctypes.Structure):
    pass


SCHEMA_RELEASE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowSchema))
ARRAY_RELEASE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArray))
STREAM_RELEASE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArrayStream))
GET_SCHEMA = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ArrowArrayStream), ctypes.POINTER(ArrowSchema)
)
GET_NEXT = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ArrowArrayStream), ctypes.POINTER(ArrowArray)
)
GET_LAST_ERROR = ctypes.CFUNCTYPE(c_void_p, ctypes.POINTER(ArrowArrayStream))

ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", c_void_p),
    ("flags", c_int64),
    ("n_children", c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", SCHEMA_RELEASE),
    ("private_data", c_void_p),
]
ArrowArray._fields_ = [
    ("length", c_int64),
    ("null_count", c_int64),
    ("offset", c_int64),
    ("n_buffers", c_int64),
    ("n_children", c_int64),
    ("buffers", ctypes.POINTER(c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ARRAY_RELEASE),
    ("private_data", c_void_p),
]
ArrowArrayStream._fields_ = [
    ("get_schema", GET_SCHEMA),
    ("get_next", GET_NEXT),
    ("get_last_error", GET_LAST_ERROR),
    ("release", STREAM_RELEASE),
    ("private_data", c_void_p),
]

# An own prototype: ctypes.pythonapi.PyCapsule_New is shared, and other code sets its types.
_capsule_new = ctypes.PYFUNCTYPE(ctypes.py_object, c_void_p, c_void_p, c_void_p)(
    ("PyCapsule_New", ctypes.pythonapi)
)

_NAMES = {
    name: ctypes.create_string_buffer(name)
    for name in (b"arrow_schema", b"arrow_array", b"arrow_array_stream", b"other")
}


def capsule(struct: ctypes.Structure, name: bytes) -> Any:
    """A PyCapsule of ``name`` that points to ``struct`` (kept alive by the caller)."""
    return _capsule_new(ctypes.addressof(struct), ctypes.addressof(_NAMES[name]), None)


@dataclass
class Releases:
    """Every release a producer saw: (kind, address of the struct, private_data)."""

    calls: list = field(default_factory=list)

    def count(self, kind: str) -> int:
        return sum(1 for k, _, _ in self.calls if k == kind)


def _views(strings: list[bytes], buffer: bytearray) -> np.ndarray:
    """The 16-byte views of strings, the long ones appended to ``buffer``."""
    views = np.zeros((len(strings), 16), dtype=np.uint8)
    for i, s in enumerate(strings):
        header = np.array([len(s)], dtype=np.int32).view(np.uint8)
        views[i, :4] = header
        if len(s) <= 12:
            views[i, 4 : 4 + len(s)] = np.frombuffer(s, dtype=np.uint8)
        else:
            views[i, 4:8] = np.frombuffer(s[:4], dtype=np.uint8)
            views[i, 8:12] = np.array([0], dtype=np.int32).view(np.uint8)
            views[i, 12:16] = np.array([len(buffer)], dtype=np.int32).view(np.uint8)
            buffer.extend(s)
    return views


def layout(fmt: bytes, strings: list[bytes], *, nulls: list[int] = ()) -> list[np.ndarray | None]:
    """The buffers of ``strings`` in format ``u``, ``U`` or ``vu``: validity (or None if
    there are no nulls), then offsets and data, or views, data and sizes."""
    validity = None
    if nulls:
        bits = np.ones(len(strings), dtype=bool)
        bits[list(nulls)] = False
        validity = np.packbits(bits, bitorder="little")
    if fmt == b"vu":
        data = bytearray()
        views = _views(strings, data)
        data_array = np.frombuffer(bytes(data) or b"\0", dtype=np.uint8)
        return [validity, views, data_array, np.array([len(data)], dtype=np.int64)]
    dtype = np.int32 if fmt == b"u" else np.int64
    offsets = np.zeros(len(strings) + 1, dtype=dtype)
    np.cumsum([len(s) for s in strings], out=offsets[1:])
    return [validity, offsets, np.frombuffer(b"".join(strings) or b"\0", dtype=np.uint8)]


def _address(buffer: Any) -> int | None:
    if buffer is None:
        return None
    if isinstance(buffer, int):
        return buffer
    return buffer.ctypes.data


# Every producer lives until the end of the process: a column fse holds calls the release
# callbacks (ctypes thunks of its producer) when it is freed, which may be after the test.
_ALIVE: list[Producer] = []


class Producer:
    """Arrays and streams of strings, with every buffer kept alive and every release
    recorded."""

    def __init__(self):
        _ALIVE.append(self)
        self.releases = Releases()
        self._keep: list[Any] = []
        self._schema_release = SCHEMA_RELEASE(self._release_schema)
        self._array_release = ARRAY_RELEASE(self._release_array)
        self._stream_release = STREAM_RELEASE(self._release_stream)

    def _release_schema(self, schema):
        self.releases.calls.append(
            ("schema", ctypes.addressof(schema.contents), schema.contents.private_data)
        )
        schema.contents.release = SCHEMA_RELEASE()

    def _release_array(self, array):
        self.releases.calls.append(
            ("array", ctypes.addressof(array.contents), array.contents.private_data)
        )
        array.contents.release = ARRAY_RELEASE()

    def _release_stream(self, stream):
        self.releases.calls.append(
            ("stream", ctypes.addressof(stream.contents), stream.contents.private_data)
        )
        stream.contents.release = STREAM_RELEASE()

    def keep(self, *objects: Any) -> None:
        self._keep.extend(objects)

    def schema(
        self,
        fmt: bytes = b"u",
        *,
        metadata: bytes | None = None,
        children=(),
        dictionary: bool = False,
        name: bytes = b"",
    ) -> ArrowSchema:
        s = ArrowSchema()
        s.format = fmt
        s.name = name
        if metadata is not None:
            block = ctypes.create_string_buffer(metadata, len(metadata))
            self.keep(block)
            s.metadata = ctypes.addressof(block)
        if children:
            kids = (ctypes.POINTER(ArrowSchema) * len(children))(
                *[ctypes.pointer(c) for c in children]
            )
            self.keep(kids, *children)
            s.n_children = len(children)
            s.children = kids
        if dictionary:
            values = self.schema(b"u")
            s.dictionary = ctypes.pointer(values)
        s.release = self._schema_release
        s.private_data = 0x5C4E
        self.keep(s)
        return s

    def array(
        self,
        buffers: list[Any],
        length: int,
        *,
        offset: int = 0,
        null_count: int = 0,
        n_buffers: int | None = None,
        no_buffers: bool = False,
        children: int = 0,
        private_data: int = 0xA77A,
    ) -> ArrowArray:
        """An array of raw buffers (numpy arrays, addresses or None)."""
        a = ArrowArray()
        a.length = length
        a.null_count = null_count
        a.offset = offset
        a.n_buffers = len(buffers) if n_buffers is None else n_buffers
        pointers = (c_void_p * max(len(buffers), 1))(*[_address(b) for b in buffers])
        self.keep(pointers, *[b for b in buffers if b is not None])
        if not no_buffers:
            a.buffers = ctypes.cast(pointers, ctypes.POINTER(c_void_p))
        a.n_children = children
        a.release = self._array_release
        a.private_data = private_data
        self.keep(a)
        return a

    def strings(
        self, fmt: bytes, strings: list[bytes], *, nulls=(), slice_=None, **kw
    ) -> ArrowArray:
        """An array of ``strings``; ``slice_`` (offset, length) slices it as Arrow does."""
        buffers = layout(fmt, strings, nulls=nulls)
        offset, length = slice_ if slice_ is not None else (0, len(strings))
        null_count = sum(1 for i in nulls if offset <= i < offset + length)
        return self.array(
            buffers, length, offset=offset, null_count=kw.pop("null_count", null_count), **kw
        )

    def array_capsules(self, schema: ArrowSchema, array: ArrowArray) -> tuple[Any, Any]:
        return capsule(schema, b"arrow_schema"), capsule(array, b"arrow_array")

    def stream(
        self,
        schema: ArrowSchema,
        arrays: list[ArrowArray],
        *,
        fail_at: int | None = None,
        code: int = 5,
        message: bytes | None = b"the producer failed",
        schema_code: int = 0,
    ) -> Any:
        """A stream capsule of ``arrays``; ``fail_at`` makes get_next fail there with
        ``code`` and ``message`` (None: get_last_error gives NULL)."""
        state = {"next": 0}
        error = ctypes.create_string_buffer(message) if message is not None else None
        self.keep(error, state)

        def get_schema(_stream, out):
            if schema_code:
                return schema_code
            ctypes.memmove(out, ctypes.addressof(schema), ctypes.sizeof(ArrowSchema))
            return 0

        def get_next(_stream, out):
            k = state["next"]
            if fail_at is not None and k == fail_at:
                return code
            if k == len(arrays):
                out.contents.release = ARRAY_RELEASE()
                return 0
            state["next"] = k + 1
            ctypes.memmove(out, ctypes.addressof(arrays[k]), ctypes.sizeof(ArrowArray))
            return 0

        def get_last_error(_stream):
            return ctypes.addressof(error) if error is not None else None

        s = ArrowArrayStream()
        callbacks = (GET_SCHEMA(get_schema), GET_NEXT(get_next), GET_LAST_ERROR(get_last_error))
        s.get_schema, s.get_next, s.get_last_error = callbacks
        s.release = self._stream_release
        s.private_data = 0x57E4
        self.keep(s, callbacks)
        return capsule(s, b"arrow_array_stream")

"""Executable model of the resumable 64-byte WordPiece scanner.

The class callback returns 0 (word), 1 (space), or 2 (punctuation).
Offsets and saved states use bytes, including across UTF-8 block boundaries.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

FULL = (1 << 64) - 1
Range = tuple[int, int]


def masks(block: bytes) -> tuple[int, int, int]:
    """Independent printable-ASCII formula, with bit zero for the first byte."""
    slow = space = punct = 0
    for j, byte in enumerate(block):
        slow |= int(not 0x20 <= byte <= 0x7E) << j
        space |= int(byte == 0x20) << j
        punct |= (
            int(
                0x21 <= byte <= 0x2F
                or 0x3A <= byte <= 0x40
                or 0x5B <= byte <= 0x60
                or 0x7B <= byte <= 0x7E
            )
            << j
        )
    return slow, punct, FULL ^ (slow | space | punct)


def ctz(value: int) -> int:
    """Index of the lowest set bit; zero is never an admissible input."""
    assert value > 0
    return (value & -value).bit_length() - 1


@dataclass
class State:
    """All state retained when a batch returns."""

    b: int = 0
    j: int = 0
    open: int = -1
    pend: Range | None = None
    done: bool = False


class Scanner:
    """One-action state machine, shared by the batch and capacity-one iterator."""

    def __init__(self, q: bytes, classify: Callable[[str], int]):
        self.q = q
        self.padded = q + b" " * 64
        self.classify = classify
        self.state = State()
        self.block = masks(self.padded[:64])

    def _advance(self) -> Range | None:
        st = self.state
        while st.j >= 64:
            st.b += 64
            st.j -= 64
            if st.b < len(self.q):
                self.block = masks(self.padded[st.b : st.b + 64])
        if st.b >= len(self.q):
            result = (st.open, len(self.q)) if st.open >= 0 else None
            st.open = -1
            st.done = True
            return result
        slow, punct, word = self.block
        above = FULL ^ ((1 << st.j) - 1)
        remaining = slow & above
        cut = ctz(remaining) if remaining else 64
        segment = above & ((1 << cut) - 1)
        if st.j == cut:
            return self._slow()
        return self._ascii(cut, word & segment, punct & segment)

    def _ascii(self, cut: int, word: int, punct: int) -> Range | None:
        st = self.state
        if st.open >= 0 and not (word & (1 << st.j)):
            result = (st.open, st.b + st.j)
            st.open = -1
            return result
        occupied = word | punct
        if not occupied:
            st.j = cut
            return None
        start = st.j if st.open >= 0 else ctz(occupied)
        if punct & (1 << start):
            st.j = start + 1
            return st.b + start, st.b + start + 1
        nonword = (FULL ^ word) >> start
        end = start + ctz(nonword) if nonword else 64
        begin = st.open if st.open >= 0 else st.b + start
        st.j = end
        if end < cut:
            st.open = -1
            return begin, st.b + end
        st.open = begin
        return None

    def _slow(self) -> Range | None:
        st = self.state
        pos = st.b + st.j
        lead = self.q[pos]
        length = 1 if lead < 0x80 else (2 if lead < 0xE0 else (3 if lead < 0xF0 else 4))
        kind = self.classify(self.q[pos : pos + length].decode("utf-8"))
        result = (st.open, pos) if kind and st.open >= 0 else None
        if kind:
            st.open = -1
        elif st.open < 0:
            st.open = pos
        if kind == 2:
            st.pend = pos, pos + length
        st.j += length
        return result

    def batch(self, capacity: int) -> tuple[list[Range], State]:
        """Return ranges and an independent snapshot at every resumption point."""
        assert capacity > 0
        result = []
        while not self.state.done and len(result) < capacity:
            if self.state.pend is not None:
                item = self.state.pend
                self.state.pend = None
            else:
                item = self._advance()
            if item is not None:
                result.append(item)
        return result, replace(self.state)


def scan(
    q: bytes, classify: Callable[[str], int], capacity: int
) -> list[tuple[list[Range], State]]:
    """Resume to completion, retaining each batch and return state."""
    scanner = Scanner(q, classify)
    calls = []
    while not scanner.state.done:
        calls.append(scanner.batch(capacity))
    return calls

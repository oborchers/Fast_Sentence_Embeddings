#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The machinery every sink of a shard store shares (docs/architecture/store.md,
"Shared shard base").

A sink hands rows to workers as buffers of their tasks (``target``), takes them back in order
(``commit``) and publishes a shard whole once all its rows are committed. This base keeps
the part that does not depend on what a row holds:

- the reservation of rows over shards, with a buffer of its own for a task that crosses
  the end of a shard (split over the shards at its commit);
- the three threads (hashing in order, writing, publishing with syncs and a rename), with
  one shared first failure; each starts only with its first work;
- ``seal``, ``publish``, ``flush`` and ``close``, with the deferred interrupt (Ctrl+C) of a
  close and the clean-up of the shards that were not published.

A family of stores supplies the rest through hooks: ``_new_writer`` (the shard to write),
``_own`` and ``_flat`` (the buffer of a split task and the target as the sink protocol
hands it out), ``_closed_event`` and ``_published_event`` (its log events).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal, Self

from numpy import ndarray

from fse._container import PublishedNotDurableError, note_cleanup
from fse._encoder._resume import ResumableSinkMarker

from ._layout import failed_cleanup
from ._shard import ShardBase
from ._stream_state import StreamState
from ._writer import WriterFailure, WriterThread, deferred_interrupts


class ShardSinkBase(ResumableSinkMarker):
    """The reservation, threads, publication and closing of a sink of shards."""

    _NAME = "sink"  # how a family names its sink in an error

    def __init__(self, path: Any, *, synced: bool, shard_rows: int, first_row: int):
        self.path = path
        self._synced = synced
        self.first_row = first_row
        self.shard_rows = shard_rows
        self._reserved = self._committed = self.published_rows = self.durable_rows = 0
        self._writers: deque[ShardBase] = deque()  # shards still taking rows
        self._made: set[ShardBase] = set()  # shards of the run not yet published
        self._targets: dict[int, tuple[list, tuple | None]] = {}  # run row -> parts, own
        self._closed = self._interrupted = False
        self._ready = True  # False for a resumable sink until a run attached its source
        self._stream: StreamState | None = None  # the stream state of a source_position@1 store
        self._interrupts: list = []  # those noted while it closes
        self._unnoted: list = []  # failed cleanups for the note of a deferred interrupt
        # Hashing (in order), writing and publishing (syncs, a rename) in three threads, so
        # the flush of one shard does not stop the hashing and writing of the next. One
        # thread writes: more gained nothing measurable (docs/architecture/store.md,
        # "sentence_vectors 1").
        failure = WriterFailure()
        self._hash = WriterThread("fse-shard-hash", failure=failure)
        self._io = WriterThread("fse-shard-write", failure=failure)
        self._publisher = WriterThread("fse-shard-publish", failure=failure)
        self._published = 0  # shards
        self._extras_published = 0  # companion artifacts (sign bits) published after shards

    # --- what a family of stores supplies ----------------------------------------------

    def _new_writer(self, store_row: int) -> ShardBase:
        raise NotImplementedError

    def _own(self, n_rows: int) -> tuple:
        raise NotImplementedError

    def _flat(self, buffers: tuple) -> tuple[ndarray, ...]:
        raise NotImplementedError

    def _closed_event(self, outcome: str, dropped: int) -> None:
        raise NotImplementedError

    def _published_event(self, writer: ShardBase, rows: int) -> None:
        raise NotImplementedError

    # --- the sink protocol, in the calling thread --------------------------------------

    @property
    def source_position(self) -> tuple[str, int] | None:
        """``(P, skip)`` of a store of the mode ``source_position@1``: seek the stream to the
        position ``P`` and drop the next ``skip`` records, then the next record is the next
        row of the store. It is the anchor of the last published shard (else of the root), so
        it moves forward as shards are published; ``durable_rows`` says what is durable. None
        for a sink of another mode and for a new store, which has no position yet."""
        return None if self._stream is None else self._stream.position()

    @property
    def durability(self) -> Literal["full", "none"]:
        """The publication policy fixed when this sink was opened."""
        return "full" if self._synced else "none"

    def _check_ready(self) -> None:
        if not self._ready:
            raise ValueError(
                f"the {self._NAME} was made with resume=True: it takes rows from "
                "Encoder.encode(texts, sink=sink), which reads the source first"
            )

    def _check_next(self, first_row: int) -> None:
        if first_row == self._reserved:
            return
        if first_row == 0:
            raise ValueError(
                f"the {self._NAME} took {self._reserved} rows of a run already; a sink takes "
                f"exactly one run, so open a new {self._NAME} on the same store to append the "
                "next one"
            )
        raise ValueError(f"rows from {first_row} reserved, expected rows from {self._reserved}")

    def _check_fresh(self) -> None:
        """Raise unless this sink is open and has reserved no row, so that it can take a
        whole run (a group run checks it before it reads anything)."""
        self._check_open()
        self._check_ready()
        self._check_next(0)

    def target(self, first_row: int, n_rows: int) -> tuple[ndarray, ...]:
        self._check_open()
        self._check_ready()
        self._check_threads()  # a failed write first: no row after it is taken
        self._check_next(first_row)
        parts = []
        left = n_rows
        while left:
            writer = self._writable()
            take = min(left, writer.end - writer.reserved)
            parts.append((writer, writer.reserved, writer.reserve(take)))
            self._reserved += take
            left -= take
        if len(parts) == 1:
            self._targets[first_row] = (parts, None)
            return self._flat(parts[0][2])
        own = self._own(n_rows)
        self._targets[first_row] = (parts, own)
        return self._flat(own)

    def _claim(self, first_row: int, rows: Any) -> tuple[list, tuple | None]:
        """The parts of the target reserved at ``first_row``, which ``rows`` must be."""
        self._check_open()
        self._check_threads()
        if first_row != self._committed or first_row not in self._targets:
            raise ValueError(
                f"rows from {first_row} committed, expected rows from {self._committed}"
            )
        parts, own = self._targets[first_row]
        given = rows if isinstance(rows, tuple) else (rows,)
        expected = self._flat(own if own is not None else parts[0][2])
        if len(given) != len(expected) or not all(
            _same(g, e) for g, e in zip(given, expected, strict=True)
        ):
            raise ValueError(f"the rows from {first_row} are not the target reserved for them")
        return parts, own

    def _handed_over(self, writer: ShardBase, n_rows: int) -> None:
        """``n_rows`` of ``writer`` are handed to the threads: count them, go on with full
        shards at once, so few files stay open."""
        writer.queued += n_rows
        self._committed += n_rows
        self._publish_ready()

    # --- durability --------------------------------------------------------------------

    def seal(self) -> None:
        """End the current shard at the rows reserved so far; it is published once they
        are committed."""
        self._check_open()
        self._check_threads()
        if self._writers and not self._writers[-1].full:
            writer = self._writers[-1]
            if writer.reserved == 0:
                self._writers.pop().abort()  # a failure: removed again (and reported) at close
            else:
                writer.seal()

    def publish(self) -> int:
        """Publish every shard whose rows are all committed and wait for it; returns
        ``durable_rows``."""
        self._check_open()
        self._check_threads()
        self._publish_ready()
        self._drain()
        return self.durable_rows

    def flush(self) -> int:
        """Seal and publish everything committed; every reserved row must be committed.
        Returns ``durable_rows``."""
        self._check_open()
        self._check_threads()
        if self._reserved != self._committed:
            raise RuntimeError(
                f"{self._reserved - self._committed} reserved rows were not committed"
            )
        self.seal()
        self._publish_ready()
        self._drain()
        return self.durable_rows

    def close(self) -> None:
        """Publish the committed rows and drop the rest; the sink takes no more rows.

        Called in the main thread while SIGINT has Python's default handler, an
        interrupt (Ctrl+C) abandons the work of every thread at once: the piece each one
        runs is finished, which has no time bound while a piece blocks, and nothing after
        it. Further interrupts are only noted; once every thread has ended and the
        temporary directories of the shards not published are removed, one
        KeyboardInterrupt is raised. Otherwise no interrupt is deferred.
        """
        self._close(None)

    def _close(self, error: BaseException | None) -> None:
        """``close``; ``error`` is the error the block of ``with`` raised, if any."""
        if self._closed:
            return
        try:
            self._close_deferring(error)
        except KeyboardInterrupt as interrupt:  # the deferred one, raised after the block
            if interrupt is not error:
                for path, failure in self._unnoted:
                    note_cleanup(interrupt, path, failure)
            raise

    def _close_deferring(self, error: BaseException | None) -> None:
        with deferred_interrupts(self._abandon_all) as self._interrupts:
            try:
                self._settle()  # the rows handed over are written, unless a thread failed
                if self._failed():  # nothing more is published
                    if not self._interrupted:
                        self._raise_unreported()
                else:
                    self._abort_uncommitted()
                    self._drain()
            except BaseException as raised:
                self._shut_down(raised)
                raise
            self._shut_down(error)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _kind: object, error: BaseException | None, _tb: object) -> None:
        self._close(error)

    # --- internals ---------------------------------------------------------------------

    def _abandon_all(self) -> None:
        """Abandon the work of every thread at once (from the handler of an interrupt)."""
        self._interrupted = True
        for thread in (self._hash, self._io, self._publisher):
            thread.abandon()

    def _shut_down(self, error: BaseException | None) -> None:
        """End every thread, the hash thread first (it hands finished shards to the
        publisher), then remove the shards not published. A removal that fails is raised
        once every thread has ended, unless an error is raised (``error``, or that of a
        thread): it is then noted on ``error`` and logged."""
        for thread in (self._hash, self._io, self._publisher):
            thread.stop()
        failures = [(w.tmp, w.abort()) for w in list(self._made)]  # what was not published
        self._writers.clear()
        self._made.clear()
        self._closed = True
        failing = error is not None or self._failed() or self._interrupted
        interrupted = self._interrupted or isinstance(error, KeyboardInterrupt)
        self._closed_event(
            "interrupt" if interrupted else "error" if failing else "ok",
            self._committed - self.published_rows,
        )
        self._report_cleanup([(path, f) for path, f in failures if f is not None], error)

    def _report_cleanup(self, failed: list, error: BaseException | None) -> None:
        """Raise the first failed removal unless an error is raised now (``error``, or a
        deferred interrupt; an error of a thread that a call raised before is no longer in
        flight), with the others noted on it; else note them all on that error."""
        if self._interrupted:  # also on the KeyboardInterrupt raised after the block
            self._unnoted = failed
        if error is None and not self._interrupted and failed:
            error = failed.pop(0)[1]
            for path, failure in failed:
                failed_cleanup(path, failure, error)
            raise error
        for path, failure in failed:
            failed_cleanup(path, failure, error, failing=self._interrupted)

    def _check_threads(self) -> None:
        for thread in (self._hash, self._io, self._publisher):
            thread.check()

    def _drain(self) -> None:
        for thread in (self._hash, self._io, self._publisher):  # hash hands on to publisher
            thread.drain()

    def _settle(self) -> None:
        for thread in (self._hash, self._io, self._publisher):
            thread.settle()

    def _raise_unreported(self) -> None:
        """Raise the shared first failure unless a call already reported it."""
        threads = (self._io, self._hash, self._publisher)
        if not any(t.reported for t in threads):
            for thread in threads:
                if thread.failed:
                    thread.check()

    def _failed(self) -> bool:
        return any(t.failed for t in (self._hash, self._io, self._publisher))

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"the {self._NAME} is closed; make a new one for another run")

    def _writable(self) -> ShardBase:
        if not self._writers or self._writers[-1].full:
            writer = self._new_writer(self.first_row + self._reserved)
            self._writers.append(writer)
            self._made.add(writer)
        return self._writers[-1]

    def _publish_ready(self) -> None:
        """Hand every shard whose rows are all handed over to be published, in order."""
        while self._writers and self._writers[0].queued_all:
            # After its last hash, in order, the shard goes to the publisher.
            writer = self._writers.popleft()
            if self._stream is not None:  # the anchor of the batches it holds rows of
                writer.anchor = self._stream.anchors.take(
                    writer.first_row + writer.end, self._stream.shared
                )
            self._hash.submit(self._hand_on, writer)

    def _hand_on(self, writer: ShardBase) -> None:
        """Hand ``writer`` to the publisher (in the hash thread): an error of the publisher
        met here is the publisher's, not yet reported to the caller."""
        self._publisher.submit(self._finish, writer, report=False)

    def _finish(self, writer: ShardBase) -> None:
        """Publish ``writer`` (in the writer thread, after its commits)."""
        try:
            rows = writer.finish(lambda: self._io.failed)
        except PublishedNotDurableError:
            self._count_published(writer, writer.committed, durable=False)
            writer.abort()
            raise
        except FileExistsError:
            exists = FileExistsError(
                f"another writer appended to {self.path} at row {writer.first_row} first; "
                f"this run published its first {self.published_rows} rows from store row "
                f"{self.first_row}, of which {self.durable_rows} are durable"
            )
            writer.abort()  # a failure: removed again and reported at close, in its thread
            raise exists from None
        except BaseException:
            writer.abort()  # likewise
            raise
        self._count_published(writer, rows, durable=self._synced)
        self._published_event(writer, rows)
        # What belongs beside the shard comes last: its rows are counted already, so a
        # failure here is an error of the sink and never changes what was published.
        self._extras_published += writer.publish_extras()

    def _count_published(self, writer: ShardBase, rows: int, *, durable: bool) -> None:
        self._made.discard(writer)
        if self._stream is not None and writer.anchor is not None:
            self._stream.published(writer.anchor, writer.first_row + rows)
        self.published_rows += rows
        if durable:
            self.durable_rows += rows
        self._published += 1

    def _abort_uncommitted(self) -> None:
        """Publish the committed rows of every shard, end it there, drop the rest."""
        self._targets.clear()  # uncommitted targets are never committed
        kept: deque[ShardBase] = deque()
        for writer in self._writers:
            writer.drop_uncommitted()
            if writer.committed:
                kept.append(writer)
            else:
                writer.abort()  # a failure: removed again (and reported) at close
        self._writers = kept
        self._reserved = self._committed
        self._publish_ready()


def _same(given: Any, reserved: ndarray) -> bool:
    return (
        isinstance(given, ndarray)
        and given.shape == reserved.shape
        and given.__array_interface__["data"][0] == reserved.__array_interface__["data"][0]
    )

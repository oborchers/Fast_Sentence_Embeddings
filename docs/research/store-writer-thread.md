# Writer thread behind a bounded queue: research notes

Scope: about 20 min, primary sources. "UNVERIFIED" marks claims I could not confirm in a source. "LOCAL" marks my own measurements (macOS, Python 3.15.0, load average about 4 to 6, so only orders of magnitude count).

## 1. GIL handoff costs

Facts
- `sys.setswitchinterval` sets the "ideal duration of the timeslices" (default 5 ms); the OS decides which thread runs next. https://docs.python.org/3/library/sys.html#sys.setswitchinterval
- Convoy effect: an I/O thread releases the GIL on every blocking call; a CPU-bound thread takes it; when the call returns at once, the I/O thread must wait out the switch interval to get the GIL back. Beazley, bpo-7946: https://www.python.org/pipermail/new-bugs-announce/2010-February/006827.html ; talk: https://dabeaz.com/GIL . Never fixed in 3.2+: deferred to "bake in" (https://mail.python.org/pipermail/python-bugs-list/2010-May/101247.html). The mechanism (waiter sets gil_drop_request after `interval`; FORCE_SWITCHING makes the dropper wait for the switch) is in `Python/ceval_gil.c`: https://sources.debian.org/src/python3.14/3.14.3-1/Python/ceval_gil.c
- Release points relevant here: `os.pwrite` always wraps the syscall in `Py_BEGIN_ALLOW_THREADS`, whatever the size (https://github.com/python/cpython/blob/main/Modules/posixmodule.c, `os_pwrite_impl`); `os.fsync` goes through `posix_fildes_fd`, which also releases it (same file). hashlib releases the GIL only when one `update()` gets more than 2047 bytes (docs: https://docs.python.org/3/library/hashlib.html ; `HASHLIB_GIL_MINSIZE 2048` in Modules/hashlib.h).
- Every release is followed by a re-acquire. With about 2 GIL-released calls per 4 MiB task that is a few hundred to a thousand re-acquisitions per second. Each one is free if the GIL is idle, and costs up to one switch interval (5 ms) if another thread is running Python bytecode.
- 3.13/3.14: free-threaded build is optional; the GIL stays default; single-thread penalty about 5 to 10 % in 3.14. Extension modules not marked safe re-enable the GIL. https://docs.python.org/3/howto/free-threading-python.html . Per-interpreter GIL (PEP 684) does not matter for one interpreter. I found no 3.12 to 3.14 change to the convoy behaviour (UNVERIFIED that none exists; I did not read the changelogs).

LOCAL measurement (writer thread: 100 x (os.pwrite + sha256.update) of 4 MiB; main thread idle vs spinning `sum(range(1000))` in Python):
- idle main: 1.47 GB/s. Busy Python main, 5 ms interval: 0.24 GB/s (6x slower, about 5 ms lost per release, as predicted). Busy main, interval 0.5 ms: about 1.0 GB/s.
- So it matters if and only if the calling thread runs Python bytecode between its GIL-free sections. If the calling thread is blocked in GIL-free C (waiting for workers) or in `queue.get`/`Condition.wait` (lock waits release the GIL), the writer re-acquires at once.

Recommendation
- Keep the calling thread out of long Python loops while the writer is active (do per-task Python work in small steps, or block on a lock/Event). Do not rely on `setswitchinterval`; it is process-global and hurts other libraries. If pure-Python work in the caller is unavoidable, measure first, then consider a lower interval only as a documented option.
- Hash in chunks of at least 64 KiB, never in small slices (below 2048 bytes the GIL is held).
- Test on a free-threaded build too (3.13t/3.14t): it removes the convoy but has no effect on correctness of this design.

## 2. Queue cost and primitives

- `queue.Queue` is pure Python: one `threading.Lock` plus three `Condition`s on it (`not_empty`, `not_full`, `all_tasks_done`). https://github.com/python/cpython/blob/main/Lib/queue.py
- `queue.SimpleQueue` is C, unbounded, `put` never blocks, no `task_done`/`join`. https://docs.python.org/3/library/queue.html . `concurrent.futures.ThreadPoolExecutor` uses SimpleQueue for its work queue: https://github.com/python/cpython/blob/main/Lib/concurrent/futures/thread.py
- LOCAL (2 threads, ping-pong of 50k small items): `Queue(16)` about 1.2 us/item, `SimpleQueue` about 0.06 us/item. At a few thousand items/s that is below 0.5 % of one core. External benchmarks are old or single-threaded (UNVERIFIED for 3.12+): https://stackoverflow.com/a/717261 (Queue about 20x slower than deque, Python 2/3 single thread).
- Contended case: when the producer is blocked on `not_full`, wake-up latency after `get` is the cost, and it also suffers from the convoy in section 1.

Recommendation
- `queue.Queue(maxsize)` is fast enough; the cost is invisible next to 1 to 16 MiB of IO. Prefer it for the plain bounded case because `put(block, timeout)` and, in 3.13+, `shutdown()` exist. Use SimpleQueue + your own byte-credit semaphore only if you need byte bounds (section 3). A `ThreadPoolExecutor(max_workers=1)` adds a Future per task and gives ordering and exception capture for free, but you lose the bound (unbounded SimpleQueue) and it joins at interpreter exit; fine if you add a semaphore.

## 3. Backpressure sizing

- Kafka Java producer bounds bytes: `buffer.memory` (default 32 MiB) "total bytes of memory the producer can use to buffer records", `send()` blocks up to `max.block.ms` then throws; `batch.size` 16 KiB; `linger.ms` default 5. https://kafka.apache.org/41/configuration/producer-configs/
- librdkafka (under confluent-kafka-python) bounds both: `queue.buffering.max.messages` (100000) and `queue.buffering.max.kbytes` (1 GiB), "kbytes has higher priority". https://raw.githubusercontent.com/confluentinc/librdkafka/master/CONFIGURATION.md
- gRPC Python bounds concurrency by count (`maximum_concurrent_rpcs` -> RESOURCE_EXHAUSTED, thread pool queue otherwise); secondhand sources only: https://groups.google.com/g/grpc-io/c/1oQtEpheHU4/m/juXZvmpGAgAJ (UNVERIFIED against grpc.io docs). Arrow, zarr, tensorstore, Lance: not researched in time (UNVERIFIED; no claim made).

Recommendation
- Bound by bytes (what costs memory), with an item cap as a secondary guard: e.g. `maxsize=16` items AND a byte budget of about 64 to 128 MiB (4 to 8 max-size tasks), enforced by a `threading.Condition` counter or `BoundedSemaphore` acquired in `put` with `timeout`, released by the writer after `pwrite`+hash. 16 x 16 MiB = 256 MiB worst case is the number to reduce. Always allow one oversized item through when the budget is empty, to avoid deadlock. Blocking `put` is the right backpressure (Kafka semantics); consider a timeout that raises, as `max.block.ms` does.

## 4. Errors and shutdown

- Daemon threads "are abruptly stopped at shutdown"; the docs advise non-daemon threads plus an Event for graceful stop. https://docs.python.org/3/library/threading.html
- Interpreter exit: `threading._register_atexit` hooks run before non-daemon threads are joined; ThreadPoolExecutor uses it and later `submit` raises `RuntimeError('cannot schedule new futures after interpreter shutdown')`. https://github.com/python/cpython/blob/main/Lib/concurrent/futures/thread.py . Joining a running daemon thread in late finalization raises `PythonFinalizationError` since 3.14 (threading docs above). Starting a new thread during finalization fails with RuntimeError (3.12 change; I did not find the exact doc text, UNVERIFIED).
- `Queue.shutdown(immediate=False|True)` (3.13+): blocked `put`s raise `ShutDown`, `get` drains then raises; `immediate=True` drops items and can unblock `join()` without work done. https://docs.python.org/3/library/queue.html (not available on 3.11/3.12, which fse supports).
- KeyboardInterrupt is delivered only to the main thread; a blocking `Queue.put` is interruptible (lock wait with signals handled in the main thread). The writer keeps running unless you stop it. (Behaviour of lock waits and signals: general CPython knowledge, no source fetched.)

Recommendation
- Writer is a non-daemon thread. Loop: `item = q.get()`; sentinel -> exit; if `self._error` is set, skip the work but still `task_done`/release the byte budget (drop, do not drain to disk) so producers never block forever. Store only the first exception (`if self._error is None`), re-raise in the caller at the next `put`/`flush`/`close` (`raise ... from err`), clear nothing afterwards (sticky). Wrap the whole writer body in try/except BaseException.
- `close()`: put the sentinel, `join()`, then re-raise the error. Use `try/finally` in the context manager `__exit__`; on `KeyboardInterrupt` or an exception in the caller, set an abort flag, drop queued items, join with a timeout, delete temp files, no rename. Also register `threading._register_atexit` is private: avoid it; rely on non-daemon join plus `weakref.finalize` only for a warning.
- Never fsync/rename after an error; the rename is the commit point.
- On macOS plain `fsync` is not durable, `fcntl(F_FULLFSYNC)` is needed (documented in os.fsync docs; whether `fcntl.fcntl` releases the GIL: UNVERIFIED, check `Modules/fcntlmodule.c`).

## 5. pwrite in the workers

- Positional writes at disjoint offsets are safe concurrently on one fd (POSIX pwrite does not use the shared file offset); `os.pwrite` always releases the GIL (source above). A C-level `pwrite` inside the GIL-free kernel needs no GIL at all, so no re-acquire is paid.
- Pros: no queue, no extra thread, no copy, memory bound = number of workers; the convoy disappears if the write happens inside the existing GIL-free section; write parallelism on NVMe.
- Cons: tasks finish out of order, so the file layout needs offsets known up front (rows x dim x itemsize makes this likely); the sha256 over the shard must run in row order (needs a reorder buffer or re-reading the file, which doubles the read and defeats the point; or hash per block and combine with a Merkle-style manifest, which changes the format); error handling has to cross threads anyway; a failed write leaves holes; `fsync` still needs a barrier at shard end; partial writes need a loop (pwrite may write fewer bytes); concurrent write ordering gives no durability guarantee before fsync.
- Verdict for the stated design: hashing in order is the deciding factor. Hash must see buffers in sequence, so either the calling thread hashes (ties up a core at about 2 GB/s SHA-256 without SHA extensions; UNVERIFIED speed) or one writer does. Keep one writer for hash + ordered pwrite; consider pwrite in workers only if the hash becomes per-block.

## 6. Recommendation and measuring

Design
- One non-daemon writer thread, `Queue(maxsize=16)` plus a byte budget of 64 to 128 MiB, first-error-sticky, sentinel close, sequence numbers asserted in the writer so a reordering bug fails loudly.
- Make the calling thread block (GIL-free) instead of spin; avoid Python work between commits.
- Chunk hashing at 1 to 4 MiB (always above 2048 bytes) and pwrite in a loop until all bytes are written.

Pitfalls
1. Convoy: a Python-busy caller turns 1.5 GB/s into 0.25 GB/s (LOCAL above).
2. Buffer lifetime: the caller must not reuse or free the memory (numpy array, mmap) before the writer is done; keep a reference in the queue item.
3. Error after the caller's last call: must surface at `close()`.
4. Deadlock if the writer dies without releasing the budget.
5. macOS F_FULLFSYNC cost (often 10s of ms); do it once per shard.

Measurements
- Time in `put` (`perf_counter` around it, sum + count), time in writer `get` (starvation), queue depth and bytes in flight at each put, writer time split pwrite / update / fsync. If `put` wait is above about 5 % of wall time the writer is the bottleneck; if `get` wait dominates, producers are.
- GIL wait: wrap writer calls: `t0; os.pwrite(...)` returns; the excess over a GIL-free baseline (same call with the main thread idle) is the re-acquire delay. Compare idle vs busy caller as in the LOCAL test; also `py-spy record --native --threads` (already in the repo dev group) and `sys.setswitchinterval` sweep (0.5 ms / 5 ms) to see sensitivity. In 3.12+ there is no public GIL-wait counter (UNVERIFIED).
- Benchmark script used (not committed): (a local scratch file, not kept)

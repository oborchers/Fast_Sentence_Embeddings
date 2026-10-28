# Fast shard writes (macOS APFS / Linux ext4, xfs): short research

Scope: about 20 min of web research. Tags: [V] = read in a primary source this session; [S] = secondary/indirect; [U] = unverified, my inference or from memory, measure before relying. No benchmark of our exact workload was found; the web has little hard data on multi-threaded buffered writes to one file, so the main recommendation rests on mechanism plus our own measurements.

## 1. Writing from several threads to one file

Findings
- Linux ext4 buffered writes take the inode lock exclusively: `ext4_buffered_write_iter()` calls `inode_lock(inode)` (exclusive i_rwsem), then `inode_dio_wait`. [V] https://raw.githubusercontent.com/torvalds/linux/master/fs/ext4/file.c
- ext4 direct I/O uses the shared lock only for writes within i_size (non-extending); it falls back to exclusive for extending writes, unaligned/zeroing cases. [V] same file. Background: https://lists.openwall.net/linux-ext4/2019/12/05/6 (shared DIO lock patches), https://lists.openwall.net/linux-ext4/2014/06/02/19 ("ext4 file write is already serialized with inode mutex"; parallel writes matter for DIO).
- XFS buffered write: `iolock = XFS_IOLOCK_EXCL` in `xfs_file_buffered_write`. [V] https://raw.githubusercontent.com/torvalds/linux/master/fs/xfs/xfs_file.c . Dave Chinner's 2025 proposals for shared buffered-write locking were RFC only. [S] https://lists.openwall.net/linux-kernel/2025/01/13/45
- So on Linux, N threads doing buffered pwrite to one file serialize on the inode lock, but the lock only covers the copy into page cache (memcpy at several GB/s), not the disk write. At 1 to 2 GB/s the lock is probably not the bound; [U] it becomes one when copies approach single-core memcpy speed (about 5 to 10 GB/s). Avoiding it: one file per worker/shard (we already have ~256 MiB shards, so several shards in flight in parallel is free), or O_DIRECT.
- APFS: no primary source on per-inode write serialization found. Apple engineer (Kevin Elliott) says APFS lock contention grows with core count, largely addressed in macOS 10.14, "something to watch" at high core counts; the cited slowdown there was F_FULLFSYNC, not writes. [V] https://developer.apple.com/forums/thread/800906 . Whether APFS serializes pwrite on one vnode: [U]. Our own measurement (pwrite from calling thread 1.96 M rows/s vs mmap 1.04 M) is the best evidence.
- O_DIRECT: alignment restrictions on length, address, offset; man page advises not mixing with buffered I/O or mmap on the same file. [V] https://man7.org/linux/man-pages/man2/open.2.html ; `statx(STATX_DIOALIGN)` reports alignment since Linux 6.1. At 1 to 2 GB/s O_DIRECT on NVMe is feasible but needs 4 KiB-aligned buffers/offsets and a manifest-hash read order; the .npy header (128 or 64 byte aligned) is not 4 KiB aligned, so the header block must be written separately. [U for the layout detail]
- macOS has no O_DIRECT; the analogue is `fcntl(F_NOCACHE, 1)` ("Turns data caching off/on"). [V] https://leancrew.com/all-this/man/man2/fcntl.html . It does not purge pages already cached and other processes can repopulate the cache. [S] https://github.com/ronomon/direct-io . No benchmark of F_NOCACHE on APFS for throughput found. [U] expect it to cost throughput for small writes (no write-behind coalescing) and help only for page-cache pressure.
- posix_fadvise(DONTNEED) only drops clean pages; must come after the writeback (`sync_file_range` or fsync). [U from memory; man page not fetched]. macOS has no posix_fadvise.
- io_uring: no source shows a gain for buffered writes of large blocks; one deck reports nearly identical write bandwidth across async backends on NVMe. [S] https://www.ixpug.org/images/docs/ISC26/Arnab_Paul_iouring_ixpug_ISC_2026.pdf . Buffered io_uring writes still copy into page cache (possibly in kernel worker threads). Not worth the complexity for us. dispatch_io on macOS: no data found; it is a convenience layer over the same write path. [U]
- writev/pwritev: only saves syscalls; with writes of tens of KiB to MiB per task, syscall cost is negligible. [U]
- Page-cache copy vs page faults: pwrite = one copy (memcpy into page cache, plus allocation of pages); mmap = one fault per page plus the same copy (see section 3).

Recommendation Q1: keep pwrite at disjoint offsets, but issue it from the worker threads in large chunks (>= 256 KiB, ideally a whole task buffer) rather than from the calling thread, on macOS first measure whether this scales (our 1.96 M rows/s from one thread suggests the calling thread's copy is already near the bound). On Linux the exclusive inode lock only serializes the copy; to scale further write different shards from different workers. Do not use io_uring, O_DIRECT or F_NOCACHE as the first step.

## 2. Preallocation

- macOS `F_PREALLOCATE` with `fstore_t`: layout `u_int32_t fst_flags; int fst_posmode; off_t fst_offset; off_t fst_length; off_t fst_bytesalloc (OUT)`. Flags: `F_ALLOCATECONTIG` ("the file system may ignore this request if fst_length is very large"), `F_ALLOCATEALL` ("all requested space or no space at all"); `F_PEOFPOSMODE` = allocate from the physical end of file, fst_length is the number of newly allocated bytes. Space allocated can be larger than requested. [V] https://leancrew.com/all-this/man/man2/fcntl.html
- Constants (from a Rust crate, check against your SDK `sys/fcntl.h`): F_PREALLOCATE=42, F_PEOFPOSMODE=3, F_ALLOCATECONTIG=0x2, F_ALLOCATEALL=0x4. [S] https://docs.rs/crate/libfreemkv/0.25.5/source/src/io/platform_macos.rs . Common pattern: try CONTIG|ALL, retry with ALL only. [S] same.
- Python: CPython main gains `fcntl.fstore`, `F_PREALLOCATE`, `F_ALLOCATECONTIG/ALLOCATEALL/ALLOCATEPERSIST`, `F_PEOFPOSMODE/F_VOLPOSMODE` (unreleased, 3.15 or later). [V] https://github.com/python/cpython/pull/139864/files . Before that: ctypes.
  ctypes sketch [U, untested here]:
  ```python
  class FStore(ctypes.Structure):
      _fields_ = [("fst_flags", ctypes.c_uint32), ("fst_posmode", ctypes.c_int32),
                  ("fst_offset", ctypes.c_int64), ("fst_length", ctypes.c_int64),
                  ("fst_bytesalloc", ctypes.c_int64)]   # 32 bytes, natural alignment, no padding
  fcntl.fcntl(fd, 42, bytes(FStore(0x4|0x2, 3, 0, n)))   # or libc.fcntl(fd, 42, ctypes.byref(fs))
  ```
  Python's fcntl.fcntl accepts a bytes argument and returns the modified buffer; with ctypes use `libc.fcntl` and `byref`. After F_PREALLOCATE you still need `ftruncate` to set the logical size (preallocation does not change st_size). [U]
- Linux: `fallocate(mode 0)` allocates the range and guarantees later writes do not fail with ENOSPC; filesystems prefer unwritten extents so nothing is physically zeroed. [V] https://man7.org/linux/man-pages/man2/fallocate.2.html . `posix_fallocate` falls back to writing zeros on filesystems without support (glibc). [U]
- Does it speed writes? Only on extent filesystems with delayed allocation by avoiding fragmentation and extent-tree updates at writeback; with delayed allocation (ext4, xfs, APFS?) the page cache write path does not allocate at write time anyway. No source quantifies a speed-up; our own measurement says a zero-fill costs 15 to 20 %. Treat preallocation as a space guarantee, not a speed-up. [U]
- Important for us: the zero-fill is the expensive variant; F_PREALLOCATE/fallocate avoid writing data. But `ftruncate` alone creates a sparse file and is the cheapest way to get a correct size for mmap or pwrite at arbitrary offsets.

Recommendation Q2: replace the zero-fill with `ftruncate` (size) plus, only if the ENOSPC-early guarantee matters, F_PREALLOCATE (macOS, CONTIG|ALL then ALL) / `os.posix_fallocate` (Linux). Expect the 15 to 20 % cost to drop to about 0 to 2 % [U]; benchmark both.

## 3. mmap writes

- Mechanism: each first touch of a page of a MAP_SHARED file mapping faults (page fault to allocate/lock the page cache page and mark it dirty); after writeback starts, the next write takes a write-protect fault again to re-mark dirty. Peter Zijlstra: mmap needs one fault per page, "more faults than write() syscalls". [S] https://lkml.iu.edu/hypermail/linux/kernel/1311.0/03290.html (thread on ext2 XIP, same family); write-protect re-fault after writeback described in https://besthub.dev/articles/why-mappedbytebuffer-beats-filechannel-and-when-it-doesn-t-deep-linux-kernel-insights-74dde9ce1c09 [S].
- munmap/page-cache eviction trigger TLB shootdowns; one-page munmap on 16 cores about 8 us, about 70 % of it the shootdown. [S] https://gts3.org/assets/papers/2018/kumar:latr-slides.pdf ; https://arxiv.org/html/2409.10946v1 (mmap/munmap cycles as a main shootdown source). On Apple Silicon TLB invalidation is broadcast in hardware (TLBI inner-shareable), so the IPI cost is different. [U]
- On macOS with 4 KiB... Apple Silicon uses 16 KiB pages, so one fault per 16 KiB; a 256 MiB shard means 16384 faults, each with VM+APFS work. [U for APFS cost; the 16 KiB page size is standard Apple Silicon knowledge]. Our measured 1.04 M rows/s for mmap vs 1.96 M for pwrite agrees with fault-dominated cost.
- MAP_POPULATE: Linux only, and for write mappings it pre-faults (reads in) pages; it does not exist on macOS. No benchmark found. [U] madvise(MADV_WILLNEED/SEQUENTIAL) are read-ahead hints and do not remove write faults; macOS madvise accepts them but I found no evidence they help writes. [U]. Large pages: file-backed THP for ext4/xfs is limited (large folios in xfs, ext4 since 6.x [U]); macOS has no superpages for file maps. [U]
- One mapping per shard instead of per task avoids repeated map/unmap and shootdowns, but still faults once per page. Only worthwhile if faults are not the bound. [U]

Recommendation Q3: abandon mmap for writing; use pwrite. If mmap is kept, map the whole shard once (ftruncate first), never munmap until the shard is finished, and expect only a partial gain. Remove it from the hot path.

## 4. sha256

- Apple M1 OpenSSL 3.3 SHA256 about 5.5 GB/s per core on a Mac mini (Phoronix OpenSSL test, single run, OpenSSL's `speed`-style loop). [S] https://openbenchmarking.org/test/system/openssl (result 2405144-NE-OPENSSLMA41). Other claims in search results (wolfSSL 205 MB/s stock, 2.3 GB/s tuned on M1) concern wolfSSL builds, not OpenSSL; ignore. [V] https://www.wolfssl.com/cryptographic-benchmarks-new-apple-m1/
- Python's hashlib uses the ARMv8 SHA2 instructions only if the OpenSSL/libcrypto it is linked to has them (OpenSSL does on arm64). Python's own HACL* fallback (built-in sha2) is slower. Which is used by uv's python-build-standalone on macOS: [U]; test `hashlib.sha256(buf)` throughput directly, our numbers say 15 to 20 % of runtime at about 1 to 2 GB/s output, i.e. consistent with one core at roughly 5 to 6 GB/s [U].
- x86 with SHA-NI: roughly 1.5 to 2 GB/s per core on Intel Ice Lake/Zen [U, from memory; no source fetched]; without SHA-NI about 0.4 to 0.6 GB/s.
- Parallelism: SHA-256 is Merkle-Damgard, a strict sequential chain over 64-byte blocks. A single digest cannot be parallelised; only a different construction (tree hash such as BLAKE3, or per-chunk hashes combined) can, which is a format change. [U for citation; standard property, see https://www.rfc-editor.org/rfc/rfc6234 for the definition]
- GIL: `HASHLIB_GIL_MINSIZE` = 2048; the GIL is released when `size > 2048` (strict) in the update path. [V] https://raw.githubusercontent.com/python/cpython/main/Modules/hashlib.h . Since 3.12 (gh-99108) built-in implementations release it too. [S] https://www.python.org/pipermail/python-checkins/2023-May/182967.html . Note the *LOCKED* variant takes the hash object's mutex while released, so one hash object must be fed by one thread at a time (which a pipeline does).
- Pipeline pattern: one dedicated hasher thread consumes a queue of buffers in row order; the hasher runs in parallel with the workers because the GIL is released. The sequential constraint means task k's bytes must be hashed after task k-1: reorder buffer (a dict by task index) feeding a single thread. Hash time is overlapped; cap = single-core sha256 speed (about 5 GB/s on M1 [S]) which is above 1 to 2 GB/s, so the hasher does not bound throughput if it is off the calling thread. Memory: hold task buffers until hashed (already needed if you pwrite from the same buffer, or hash while the data is hot in cache right after the worker produced it, in order).

Recommendation Q4: hasher thread fed in order from a reorder queue; hash from the same buffer that is written (cache-warm). Keep sha256; do not try to parallelise a single digest. For one digest per tensor and two tensors (int8 and float32 scale) use two hashers in parallel (one per tensor), which already doubles headroom.

## 5. fsync cost

- fsync on macOS pushes data to the drive but the drive may not persist it nor keep order; `F_FULLFSYNC` "asks the drive to flush all buffered data to permanent storage". [V] https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html
- `F_BARRIERFSYNC`: "Does the same thing as fsync(2) then issues a barrier command to the drive ... guaranteed to be persisted before any other I/O that would follow the barrier". [V] https://leancrew.com/all-this/man/man2/fcntl.html . It orders but does not make durable now: after a power loss you may lose the data written before the barrier as a whole, but not see later writes without earlier ones. For publication by rename this gives "if the renamed file exists, its contents exist", but not "the file exists by the time the call returns". [V for text, U for implications]. SQLite uses it on Darwin (secondary comment) [S] https://mjtsai.com/blog/2022/02/17/apple-ssd-benchmarks-and-f_fullsync
- Latency: no authoritative per-call number. Anecdotes: about 4 ms per F_FULLFSYNC on an M3 Max, about 30 us for plain fsync [S] https://coderapp.vercel.app/answer/78628333 ; internal Apple NVMe 58 fsync IOPS under F_FULLFSYNC in an HN thread (about 17 ms, small-write context) [S] https://hn-svelte.vercel.app/item/30370551 ; Hector Martin: Apple SSD flush much slower than other drives [S] https://mjtsai.com/blog/2022/02/17/apple-ssd-benchmarks-and-f_fullsync . Apple: F_FULLFSYNC "is expensive by design: flushes the volume and forces the drive to commit its cache" and plain fsync is not equivalent. [V] https://developer.apple.com/forums/thread/800906
- Does it scale with dirty data? F_FULLFSYNC = fsync (writes the file's dirty pages, which scales with the amount dirty, i.e. up to 256 MiB per shard) plus a device cache flush (roughly constant, ms). So most of its cost for big shards is the data write-out, which is unavoidable if the data must be durable; only its overlap can be improved. [U inference]
- Overlap: start write-back early and in the background: Linux `sync_file_range(WRITE)` starts write-out asynchronously but "does not flush disk write caches" and is "not suitable for data integrity operations"; still, it spreads the I/O so the final fsync has little left. [V] https://man7.org/linux/man-pages/man2/sync_file_range.2.html . On macOS there is no equivalent; `fsync` on a finished shard in a background thread (GIL released, `os.fsync` releases it) is the equivalent. Several shards can be synced concurrently from separate threads; whether the device flush is coalesced across files: [U].
- Apple also states APFS contention rises with core count for many small F_FULLSYNC calls from parallel processes. [V] https://developer.apple.com/forums/thread/800906

Recommendation Q5: do the sync of shard N in a background thread while shard N+1 is filled (one in flight at most, or two); F_FULLFSYNC per shard (not per task), then directory fsync after the rename (F_FULLFSYNC on the directory fd on macOS; `os.fsync(dirfd)` on Linux). Keep F_FULLFSYNC for the publication guarantee the format promises; F_BARRIERFSYNC is only acceptable if the spec of "durable" is relaxed to "ordered", which is a product decision for the maintainer. On Linux add `sync_file_range(SYNC_FILE_RANGE_WRITE)` per ~16 MiB written to start write-back early [U gain].

## 6. Recommended design for our case

Threads
1. Calling thread: only schedules tasks and collects completions; never copies or hashes.
2. Workers (GIL released): compute rows into a task-private buffer (this is where the 3 M rows/s is made), then `pwrite` that buffer at its disjoint offset into the shard file (pwrite releases the GIL; use os.pwrite or native). File opened once per shard; file size set once with `ftruncate`, no zero fill.
3. Hasher thread per tensor: consumes completed task buffers strictly in order (reorder buffer), `hashlib.sha256().update(memoryview)` (GIL released for > 2048 bytes). Buffers are returned to a pool after both pwrite and hash are done, so memory is bounded.
4. Syncer thread: when a shard is complete and hashed, `F_FULLFSYNC` (macOS) / `fsync` (Linux) on the file in the background, then publish by exclusive rename (link()+unlink or `renamex_np(RENAME_EXCL)` on macOS, `renameat2(RENAME_NOREPLACE)` on Linux [U]) and fsync the directory. Shard N syncs while shard N+1 is filled.

Expected gain [U, estimates]: pwrite from workers instead of the calling thread removes the serial copy, aiming at 2.4 to 2.8 M rows/s; replacing zero-fill by ftruncate recovers about 15 %; moving sha256 to a pipeline thread recovers another 15 % if a spare core exists (10 cores, so yes); a background F_FULLFSYNC recovers most of the last 15 to 20 % when the disk keeps up. The in-memory bound of 3.1 M is probably not reached because the page-cache copy still costs memory bandwidth (about 1 to 2 GB/s extra traffic, small vs M1 Pro ~200 GB/s) and the kernel's dirty-page throttling; expect 80 to 90 % of it.

Risks
- APFS may serialize writers on one vnode (not verified); mitigation: write 2 to 4 shards concurrently, each with its own file, so workers spread across files.
- Dirty-page throttling: Linux `vm.dirty_ratio` (default 20 % of RAM) blocks writers when the disk is slower than 1 to 2 GB/s; sustained rate is then the disk's. On a laptop SSD (APFS M1 Pro about 5 GB/s write burst, lower sustained) fine.
- Hasher order dependency: a slow task stalls the hash queue and buffers pile up; bound the reorder buffer and apply backpressure.
- F_FULLFSYNC can take hundreds of ms for a 256 MiB shard; hidden only if a spare shard is being filled.
- Atomicity: pwrite from several threads at disjoint offsets with no overlap needs no locking, but short writes must be looped (os.pwrite can return fewer bytes).
- Test each step with the existing benchmarks/bench_store.py; change one thing at a time since gains here are estimates.

## Source list (all fetched or surfaced this session)
- https://leancrew.com/all-this/man/man2/fcntl.html (macOS fcntl man page, F_FULLFSYNC, F_BARRIERFSYNC, F_NOCACHE, F_PREALLOCATE, fstore_t)
- https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html
- https://developer.apple.com/forums/thread/800906 (Apple engineer on APFS contention and F_FULLFSYNC)
- https://mjtsai.com/blog/2022/02/17/apple-ssd-benchmarks-and-f_fullsync
- https://github.com/python/cpython/pull/139864/files ; https://docs.rs/crate/libfreemkv/0.25.5/source/src/io/platform_macos.rs
- https://raw.githubusercontent.com/torvalds/linux/master/fs/ext4/file.c ; https://raw.githubusercontent.com/torvalds/linux/master/fs/xfs/xfs_file.c
- https://lists.openwall.net/linux-ext4/2019/12/05/6 ; https://lists.openwall.net/linux-ext4/2014/06/02/19 ; https://lists.openwall.net/linux-kernel/2025/01/13/45
- https://man7.org/linux/man-pages/man2/fallocate.2.html ; sync_file_range.2 ; open.2
- https://raw.githubusercontent.com/python/cpython/main/Modules/hashlib.h ; https://www.python.org/pipermail/python-checkins/2023-May/182967.html
- https://openbenchmarking.org/test/system/openssl ; https://www.wolfssl.com/cryptographic-benchmarks-new-apple-m1/
- https://gts3.org/assets/papers/2018/kumar:latr-slides.pdf ; https://arxiv.org/html/2409.10946v1 ; https://lkml.iu.edu/hypermail/linux/kernel/1311.0/03290.html
- https://www.ixpug.org/images/docs/ISC26/Arnab_Paul_iouring_ixpug_ISC_2026.pdf ; https://github.com/ronomon/direct-io

## Not researched (time)
Source code of numpy/zarr/Lance/safetensors/LMDB/RocksDB/ClickHouse was not read; no data on dispatch_io, madvise on macOS, APFS per-inode write locking, or F_BARRIERFSYNC latency. These are marked [U] above.

# Vector-DB storage, durability and cold-start: primary-source findings for fse

Date: 2026-10-28. Fills the gap of `vector-db-tricks.md` (DuckDB VSS, Lance, sqlite-vec, durability pages, warming) with source code and official docs read this session. Nothing was benchmarked; no heavy computation ran.

Tags: **[src]** = read in source code (file, lines of `main`/`master` as downloaded 2026-10-28; line numbers move, so grep the function names). **[doc]** = official documentation page fetched. **[man]** = man page on this Mac (`man fcntl`, `man madvise`, `man fsync`, `man mmap`) or man7.org. **[derived]** = my reasoning, not a published result. **[unverified]** = secondary source, search snippet, or not checked. Where a page was summarised by the fetch tool I name that; quotes were taken from fetched text.

## 1. Summary (most useful first)

1. **macOS has an intermediate sync level that fse can use, and `F_FULLFSYNC` is documented to cover earlier `fsync`s.** `man fcntl` [man]: `F_FULLFSYNC` "drains the entire queue of the device and acts as a barrier, data that had been fsync'd on the same device before is guaranteed to be persisted when this call returns"; `F_BARRIERFSYNC` orders I/O ("useful to guarantee valid state on disk when ordering is a concern but durability is not"). fse today issues `F_FULLFSYNC` separately for the vectors file, each small tensor, the manifest and the store directory per shard (docs/architecture/store.md, "sentence_vectors 1", `_shard.py:153,169,193`, `_npy_inplace.py:105-110`, `_container/_common.py:320-330`). Plain `fsync` on each file followed by ONE `F_FULLFSYNC` per publish (or per batch of shards) keeps full durability by the man page's wording and removes most of the cost [derived from [man]; power-loss behaviour not testable here]. `F_BARRIERFSYNC` is the "ordered, not durable" middle level.
2. **Every database in the survey separates "atomic/consistent" from "durable", and the cheap levels keep the first.** SQLite WAL+`synchronous=NORMAL`: "always consistent ... but lose durability" [doc]. PostgreSQL `synchronous_commit=off`: loses recent commits but "no risk of database inconsistency", unlike `fsync=off` which can corrupt [doc]. RocksDB `sync=false`: process crash loses nothing, machine crash loses the last writes [src]. LMDB `MDB_NOMETASYNC`: integrity kept, last commit may be undone; `MDB_NOSYNC`: may corrupt unless the filesystem preserves write order [src]. For fse the corruption risk of `durable=False` is not the manifest (a missing shard is just a shorter prefix) but "renamed into place but data never reached the disk": a published shard with zero/short data. That needs a recovery rule, not just a flag (section 5).
3. **Lance already ships `durable=False` as its local default**, with a documented knob: storage option `local_fsync` defaults to `false`; when true it fsyncs the temp file, then renames, then fsyncs the parent directory [src]. Its commit is "write temp manifest, `rename_if_not_exists`", the same shape as fse's exclusive rename [src].
4. **Cold first search: only populating the pages helps, measured.** A Qdrant user benchmark (Linux 6.2, 10M x 512d float32, 20.9 GB, exact search): cold first search 44.35 s normal, 47.11 s with `MADV_WILLNEED`, 457 ms with `MAP_POPULATE` (startup 5 s to 14 s) [gist, third-party, single run]. macOS has no `MAP_POPULATE` [man mmap shows only `MAP_NOCACHE`], so a portable `warm()` should read the files (large sequential reads, a few threads) or touch pages. Never `MADV_RANDOM` on these files (Lucene/ES measurement: readahead drops to one page).
5. **sqlite-vec and Lance confirm the shape of fse's scan; their tricks are small.** sqlite-vec: fixed 1024-row chunks (max 4096), per-chunk top-k then merge, int8 NEON kernels, bit Hamming NEON/AVX2, optional `rescore` with default oversample 8 [src]. Lance: flat search is a binary heap of size k; cosine stores per-row norms (like fse's inverse norms); an RQ scan prunes with a lower bound while keeping the compare bit-identical to the scalar path [src], which is the nearest published relative of the exact-prefilter idea in `vector-db-tricks.md` 3.1.
6. **DuckDB's brute force is scalar C++.** `array_inner_product`/`array_cosine_similarity` copy each array into a temp buffer and run a plain loop; a source comment says specialising for fixed sizes "may allow the compiler to vectorize the loop better" [src]. DuckDB VSS adds only an HNSW index; its persistence is experimental and rewrites the whole index at each checkpoint [doc]. Nothing to adopt for speed.

## 2. DuckDB and duckdb-vss

### 2.1 Brute-force path (core DuckDB, not the VSS extension)
- Where: `extension/core_functions/scalar/array/array_functions.cpp` (`ArrayGenericFold`, ~lines 146-190; registration comment ~218-221) and `extension/core_functions/include/core_functions/array_kernels.hpp` (`InnerProductOp` 11-29, `CosineSimilarityOp` 40-104) [src, `duckdb/duckdb` main]. The path `src/core_functions/...` is a 404; the code moved to `extension/core_functions`.
- What it does: per row, `ReadArrayElements` copies the FLOAT[n] child values into a `vector<TYPE>` buffer (checking NULLs), then calls `OP::Operation(lhs, rhs, n)`, a scalar loop `result += x*y`. Cosine accumulates dot, norm_l, norm_r in one pass, falls back to a scaled computation if the result is non-finite or the norm product underflows, and clamps to [-1, 1] [src].
- Vectorisation: none explicit. The comment says "In the future we could add a wrapper with a non-type template parameter to specialize for specific array sizes ... which may allow the compiler to vectorize the loop better" [src]. A float sum loop is not auto-vectorised by default (no reassociation without fast-math) [derived].
- Data layout: arrays are stored contiguously in the child vector, addressed as `idx * array_size` [src, same function]. The announcement post says the ARRAY type avoids per-row length branching, no numbers [doc, duckdb.org/2024/05/03].
- Measured: the sqlite-vec author's M1 mini benchmark has DuckDB at 46 ms vs sqlite-vec 33-35 ms on sift1m (1M x 128d, k=20) and 307 ms vs 41 to 89 ms on a 500k x 960d set [blog, author's own numbers, older DuckDB; see 4.4].
- Exactness: yes (float, single accumulator). For fse: nothing to adopt; the native dot16 kernel is already ahead.

### 2.2 HNSW index persistence (duckdb-vss)
- `src/hnsw/hnsw_index.cpp`: `SerializeToDisk` (~608), `SerializeToWAL` (~626), `Vacuum` (~648), option `hnsw_enable_experimental_persistence` (~768) [src]. Source files also include usearch and simsimd headers (`src/include/usearch`, `src/include/simsimd`).
- Docs [doc, duckdb.org/docs/2.0/core_extensions/vss.html]: persistence only with the experimental flag; "there are no incremental updates to persistent index storage"; "every time DuckDB performs a checkpoint the entire index will be serialized to disk and overwrite itself"; "WAL recovery is not yet properly implemented for custom indexes", so an unexpected shutdown can mean "data loss or corruption of the index"; the index "is not buffer managed and must be able to fit into RAM" and does not count toward `memory_limit`. The README says the same, plus that the index is reloaded in full on first access after restart.
- Relevance: an anti-pattern for fse: a single mutable index file needs a WAL; fse's immutable shards with an atomic rename avoid the problem. If fse ever adds an ANN sidecar, build it per shard (immutable) and publish it with the shard.

## 3. Lance / LanceDB

### 3.1 File and table layout
- Files: pages per column, no row groups, "typically several megabytes", recommended 8 MB; "we always align buffers to 64 byte boundaries"; fixed footer (magic "LANC"); footer readable in 1-2 reads from the end; direct I/O may need 4096-byte alignment so writers may pad [doc, lance.org/format/file/].
- Vectors: a fixed-size list of primitives is "treated as a primitive", flat buffer if non-nullable; fixed-width values avoid the second I/O of full-zip random access; compression opt-in (`lance-encoding:compression`, default none) [doc, lance.org/format/file/encoding/, via fetch summary].
- Table: immutable manifest per version; fragments (uint32 id) hold one or more data files plus optional deletion file; directories `data/`, `_versions/`, `_transactions/`, `_deletions/`, `_indices/`; optional `_versions/latest_version_hint.json` "purely an optimization" [doc, lance.org/format/table/ and /layout/].
- Reads: the local reader uses `read_exact_at` (pread), not mmap (`rust/lance-io/src/local.rs:365`; no `mmap` match in `local.rs` or `object_store.rs`; I did not grep the whole repo) [src]. fse's mmap choice is therefore not the Lance choice; Lance's object-store scheduler is built for remote reads.

### 3.2 Flat (brute-force) search and top-k
- `rust/lance-index/src/vector/flat/index.rs`: "A Flat index is any index that stores no metadata, and during query, it simply scans over the storage and returns the top k results" [src]. Candidates go through `push_candidate_local` into a `BinaryHeap` of size k (replace the root only if strictly better; lines ~36-50). Unsorted heap output; a `SortExec` later merges partitions (comment ~230).
- Fused scan: if there is no prefilter, more than `4*k` rows, and the distance calculator "certifies that its top-k scan keeps every row the push below would", it calls `accumulate_topk_with_scratch` and "can skip scoring most rows" (lines ~170-185) [src]. The implementation I located is the RaBitQ one (next bullet); flat float storage falls to `distance_all_with_scratch` plus heap pushes.
- Norm cache: `FlatFloatStorage` precomputes per-vector L2 norms for cosine (`cosine_norms_cache`, `rust/lance-index/src/vector/flat/storage.rs:29-39`), the same idea as fse's stored inverse norms [src].
- Prune with bit-identical compare: `rust/lance-index/src/vector/bq/prune.rs` header: after a binary scan, each row's lower bound is checked against the query upper bound and the current heap threshold; "only the survivors (typically well under 1%) are reranked" (author comment, not a measured table); the kernels use "exactly the operation order of the scalar helper, multiplies and adds, never FMA" and ordered-quiet `>=` so "the masks reproduce the scalar `>=` decisions exactly"; a stale heap threshold "can only over-select survivors, never wrongly prune" [src]. This is the closest published design to the exact int8 prefilter (derived in `vector-db-tricks.md` 3.1) and supports its two rules: keep the survivor test monotone (stale thresholds only add survivors) and keep compare semantics identical to the scalar reference. Exact for the reranked set only if the bound is valid; Lance uses it on an approximate (IVF_RQ) index, so its overall result is approximate.
- Shared scan: batch queries share one scan unless a `refine_factor` is set (`rust/lance/src/dataset/scanner.rs` ~7290) [src]; same idea as fse's rows x queries blocks.

### 3.3 Refine
- `Scanner::refine(factor)` (`scanner.rs` ~2668): "reading extra elements and using the original vector values to re-rank the distances ... if factor is 2, then the search will read 2x more elements than the requested k before performing the re-ranking. Even if the factor is 1, the results will still be re-ranked" [src]. The ANN sub-index is asked for `k * refine_factor` (`knn.rs` ~1685, 3086) [src]. LanceDB docs: `refine_factor` "Reads and reranks additional candidates in memory to recover recall lost to quantization"; `bypass_vector_index()` forces "an exact (flat) scan"; "exact queries may be prohibitively slow on production scales" [doc, docs.lancedb.com/indexing/vector-index]. Approximate (user-chosen multiplier), same as Qdrant/Weaviate.

### 3.4 Compaction
- `rust/lance/src/dataset/optimize.rs` header: many small streaming appends leave many small files; `compact_files` merges fragments below `target_rows_per_fragment` (default 1,048,576; `max_rows_per_group` 1024; `materialize_deletions_threshold` 0.1) when neighbours are also candidates; row ids change so indexes are remapped, and indexed fragments are not combined with unindexed ones [src, lines ~235-400].
- Concurrency: a Rewrite changes row addresses, so it conflicts with Overwrite/Restore, is retryable against Delete/Update/Rewrite on overlapping fragments, and conflicts with CreateIndex on overlap [doc, lance.org/format/table/transaction/ via fetch summary].
- fse: a compaction that merges adjacent small shards fits the same rule (only neighbours). fse's "rows run without a gap" invariant and shard-named-by-first-row make it harder than Lance (stable row ids not needed in fse: row numbers are positions). Effort medium-high; still only for streaming use.

### 3.5 Commit protocol and write durability
- Manifest naming V2 reverses version numbers so the latest sorts first (version 1 is `18446744073709551614.manifest`) for lexicographic listing [doc, transaction page].
- Atomic primitive: "rename-if-not-exists and put-if-not-exists (conditional PUT) ... exactly one writer succeeds". `RenameCommitHandler` writes a staging manifest, calls `rename_if_not_exists(tmp, final)`; on `AlreadyExists` it deletes the temp and returns `CommitConflict`; on success writes the version hint (`rust/lance-table/src/io/commit.rs` ~1570-1633) [src]. The module header names `ConditionalPutCommitHandler` as the default for most object stores [src].
- Conflict handling: writer retries; transaction files allow rebase; outcomes rebasable / retryable / incompatible [doc].
- Local durability: `ObjectStore.local_fsync` (`rust/lance-io/src/object_store.rs` ~209, option parser ~1950): storage option `local_fsync` "must be `true` or `false`", **default false** [src]. When true, `LocalWriter::persist` (`rust/lance-io/src/object_writer.rs` ~829-880) does: `file.sync_all()` on the temp file, close, `temp_path.persist` (rename), then on Unix opens the parent directory and `sync_all()` [src]. `create_dir_all_durable` (`lance-io/src/local.rs:45-65`) syncs newly created directories up to the first existing ancestor [src]. Native copies are replaced by a streamed durable copy when `local_fsync` is on (`object_store.rs` ~1580) [src]. Rust's `File::sync_all` uses `fcntl(F_FULLFSYNC)` on Apple platforms (`library/std/src/sys/fs/unix.rs`, matches at 1273 and 1287 [src, rust-lang/rust master]; I did not read which of those two is `sync_all`).
- MemWAL (Lance's streaming WAL, `rust/lance/src/dataset/mem_wal.rs` ~27): "Durable writes: wait for WAL flush before returning; Non-durable writes: buffer in memory, accept potential loss on crash" [src]. A per-write durability flag, comparable to RocksDB `WriteOptions::sync`.
- fse takeaway: (a) Lance chose default off with an opt-in; fse's default on is the safer choice for sentence stores that confirm offsets via `durable_rows`. (b) Lance's order (file sync, rename, directory sync) equals fse's. (c) Lance has no recovery logic for the non-durable case beyond "manifest is the commit point".

## 4. sqlite-vec

### 4.1 Storage (vec0)
- Rows live in fixed-size **chunks**: shadow tables hold per chunk a `size`, a validity bitmap (`chunk_size/8` bytes), a rowids blob (`chunk_size * 8` bytes) and per vector column a `vectors` blob of `chunk_size * dims * element_size` (`sqlite-vec.c` ~4675-4730) [src]. Default `chunk_size` 1024 (line 5140), must be divisible by 8, max 4096 (`SQLITE_VEC_CHUNK_SIZE_MAX`, line 5118) [src]. Deleted rows clear a validity bit; an empty chunk is deleted.
- Scan: `vec0Filter_knn_chunks_iter` (~7240-7600) iterates chunks, validates blob sizes, `sqlite3_blob_read` copies the whole chunk blob into a malloc'd buffer, computes a distance for each valid slot, applies distance constraints, selects the chunk's top `min(k, chunk_size)` with `min_idx` (repeated minimum search, O(n*k), line 6511) and `merge_sorted_lists` into the running top-k [src]. Pre-filters (rowid IN, metadata) are bitmaps ANDed into the validity bitmap.
- Why chunks: inserts/updates/deletes touch a small blob, the scan reads big contiguous blobs; the author says vec0 stores vectors "in chunked shadow tables, which makes inserts, updates, and deletes efficient but slows the initial build", and a "static" mode reading one contiguous block is faster [blog, alexgarcia.xyz/blog/2024/sqlite-vec-stable-release, via fetch summary].

### 4.2 Kernels
- float32: NEON when dims > 16, AVX when dims % 16 == 0, else scalar (`distance_l2_sqr_float` line 413); four independent accumulators in NEON (line ~135-150) [src].
- int8: `l2_sqr_int8_neon` loads 8 lanes, widens to i16/i32 ("widening multiply: i16*i16 -> i32 to avoid i16 overflow", line 245+); `cosine_int8_neon` (line 526) computes dot and norms inside one call; no separate int8 dot kernel [src]. No SDOT/I8MM use found in what I read [src, partial read].
- bit: `distance_hamming` (line 808): NEON `vcntq_u8` XOR-popcount with four accumulators for dims >= 128, AVX2 PSHUFB popcount for >= 32 bytes, u64 `popcount` if dims % 64 == 0, else a 256-entry table [src].
- Rescore: `SQLITE_VEC_ENABLE_RESCORE` index type; `oversample` default 8, allowed 1..128 (lines 2746, 2801) [src]. Approximate (fixed multiplier).

### 4.3 Published speed
- Author's M1 mini 8 GB, query times in ms: sift1m (1M x 128d, k=20): faiss 10, sqlite-vec static 17, vec0 33-35, DuckDB 46, usearch 56, numpy 136; GIST 500k x 960d: static 41, usearch 46, faiss 50, vec0 87-89, DuckDB 307, numpy 581 [blog, via fetch summary; author warns static may be an outlier]. On-disk 100k vectors: 1536d float 105 ms, 3072d bit 11 ms; 1M vectors on disk: binary quantisation about 124 ms, no float config under 100 ms [blog]. Author estimates 5-10% quality loss for bit vectors at about 10x speed [blog; his estimate, not measured here].
- For fse: its int8 scan (10-13 ms for 859k x 300d on 10 workers, per `vector-db-tricks.md`) is already in the range of faiss/static numbers on comparable hardware; nothing to copy except the "chunk = unit of IO, bitmap pre-filter" idea.

## 5. Durability levels: what each system offers

| System | Fast end | Middle | Strict end | Source |
|---|---|---|---|---|
| SQLite (WAL) | `synchronous=OFF`: no sync; process crash safe, OS crash/power loss may corrupt | `NORMAL`: WAL synced before each checkpoint, db synced after; no sync on most commits; consistent, may lose recent commits | `FULL`: WAL synced at each commit, ACID in WAL; `EXTRA` adds directory sync in rollback mode | [doc] pragma.html, wal.html |
| PostgreSQL | `fsync=off`: "unrecoverable data corruption" possible | `synchronous_commit=off`: lose up to 3x `wal_writer_delay` (default 200 ms, so about 600 ms) of commits, no corruption | `on` (default): wait for local WAL flush | [doc] runtime-config-wal, wal-async-commit |
| RocksDB | `disableWAL`; `sync=false`: like `write()` | `sync=false` plus manual `FlushWAL`/`SyncWAL`; `wal_bytes_per_sync` background sync; `manual_wal_flush` | `WriteOptions::sync=true`: like `write()` + `fdatasync()` | [src] options.h |
| LMDB | `MDB_NOSYNC`: may corrupt unless FS preserves write order | `MDB_NOMETASYNC`: one flush per txn, last txn may be undone, integrity kept; `mdb_env_sync(force)` to checkpoint | default: full sync | [src] lmdb.h |
| Lance | `local_fsync=false` (default) | none | `local_fsync=true`: file, rename, directory | [src] |
| macOS primitives | plain `fsync`: reaches drive cache only | `F_BARRIERFSYNC`: ordering | `F_FULLFSYNC`: drive flush, covers earlier fsyncs on the device | [man] |

Exact guarantees (quotes):
- SQLite NORMAL: "WAL mode is always consistent with synchronous=NORMAL, but WAL mode does lose durability. A transaction committed in WAL mode with synchronous=NORMAL might roll back following a power loss or system crash. Transactions are durable across application crashes regardless of the synchronous setting." OFF: "the database might become corrupted if the operating system crashes or the computer loses power"; recommended "when creating a new database from scratch, in a scenario where the process of creating the database can be repeated" [doc, pragma.html]. This is exactly fse's bulk-encode case. `PRAGMA fullfsync` and `checkpoint_fullfsync` default **off**; "Only Mac OS X supports F_FULLFSYNC" [doc]. So SQLite on macOS is not power-loss durable by default.
- PostgreSQL: "Unlike fsync, setting [synchronous_commit] to off does not create any risk of database inconsistency: an operating system or database crash might result in some recent allegedly-committed transactions being lost, but the database state will be just the same as if those transactions had been aborted cleanly" [doc]. macOS: "On macOS, write caching can be prevented by setting `wal_sync_method` to `fsync_writethrough`" [doc, wal-reliability]; in source `pg_fsync_writethrough` is `fcntl(fd, F_FULLFSYNC)` (`src/backend/storage/file/fd.c:462-468`) and `HAVE_FSYNC_WRITETHROUGH` is defined for Darwin if `F_FULLFSYNC` exists (`port/darwin.h`) [src]. The default method is the first of `open_datasync, fdatasync, fsync, fsync_writethrough, open_sync` available, `fdatasync` on Linux/FreeBSD [doc]; `xlogdefs.h` picks `open_datasync` when `O_DSYNC` differs from `O_SYNC` [src]; so by that rule the macOS default does not use `F_FULLFSYNC` [derived; I did not check macOS's `O_DSYNC` definition].
- RocksDB: `sync` comment: "a DB write with sync==false has similar crash semantics as the write() system call. A DB write with sync==true has similar crash semantics to a write() system call followed by fdatasync()"; "fsync and fdatasync are equally safe for our purposes" (`use_fsync`); `PosixWritableFile::Sync` uses `F_FULLFSYNC` under `HAVE_FULLFSYNC`, else `fdatasync` (`env/io_posix.cc` ~1826-1838) [src]. Recovery modes: default `kPointInTimeRecovery` ("stop the WAL playback on discovering WAL inconsistency"), `kTolerateCorruptedTailRecords` [src]. The prefix-recovery idea maps to fse's "valid prefix of shards".
- LMDB: NOSYNC "a system crash can corrupt the database or lose the last transactions ... However, if the filesystem preserves write order and the MDB_WRITEMAP flag is not used, transactions exhibit ACI properties and only lose D"; NOMETASYNC "maintains database integrity, but a system crash may undo the last committed transaction"; `mdb_env_sync(force)` flushes on demand. On Apple LMDB defines `MDB_FDATASYNC(fd)` as `fcntl(fd, F_FULLFSYNC)` (`mdb.c:185`) [src].
- SQLite's `full_fsync` uses `F_FULLFSYNC` only when `fullfsync` is set (`os_unix.c:3778-3817`) [src].

### What this suggests for fse `durable=False`

fse invariants today: a crash leaves a valid prefix of whole shards; durability against power loss rests on the syncs; `durable_rows` is the confirmation point for stream consumers (docs/architecture/store.md, "sentence_vectors 1", `_sink.py:63`).

Recommendation (my design, [derived]):
1. Make it an enum, not a bool, with names that state the guarantee: `durability="full" | "ordered" | "none"` (default `"full"`), or two flags if the project prefers. Avoid "async"/"fast". Names echo PostgreSQL (`off`), SQLite (`OFF/NORMAL/FULL`) and Lance (`local_fsync`). A plain `durable=False` as planned maps to `"none"`.
   - `full`: today's semantics, minus redundant FULLFSYNCs (below).
   - `ordered`: nothing is guaranteed to be on disk, but a published shard is never visible on disk before its contents. macOS: plain `fsync` of each file, `F_BARRIERFSYNC` before the rename [man]. Linux/others: no cheap equivalent in the sources I read, so `"ordered"` would need `fdatasync` of the files before the rename (cost kept, directory sync dropped) or be unavailable [derived].
   - `none`: no syncs; only the exclusive rename. Process crash still leaves a valid prefix (the data are in the page cache); power loss may leave a published shard with missing data (LMDB NOSYNC/SQLite OFF analogue).
2. **Sync at flush (checkpoint model) is the best intermediate level, and it needs one recovery rule.** `flush()`/`close()`/`seal()` then performs the syncs for every shard published since the last sync (SQLite's checkpoint sync; LMDB `mdb_env_sync`; RocksDB `SyncWAL`). `durable_rows` must then count only synced shards (it already means "published and synced"), so stream consumers stay correct. Recovery rule for the window between publishes and the next sync: on `open`, if a store was written without syncs, verify the xxh3-128 of shards after the last synced one and truncate the tail at the first bad shard (RocksDB `kPointInTimeRecovery` analogue, valid-prefix invariant preserved). Needs a small marker: for example `store.json` (or a `synced_through` file) written and synced at each sync point. Without such a rule, `mmap` open does not verify checksums (`_container/_common.py:77` "checks checksums unless `mmap`"), so a torn shard would be read as garbage silently, which conflicts with the AGENTS.md rule "never fail silently".
3. **Cheaper `full` on macOS (works for the default too).** Per shard fse syncs up to five things with `F_FULLFSYNC`. By `man fcntl`, `F_FULLFSYNC` makes everything fsync'd earlier on the device durable. Use `os.fsync` for the data file, the small tensors and the manifest, then one `F_FULLFSYNC` on the shard directory before the rename and one on the store directory after (or only the last one at flush for batched publishes). Saves several full drive flushes per shard; measure `F_FULLFSYNC` latency on the target SSD before claiming a gain (not measured; Apple says it "may take quite a while"). Risk: relies on the documented "same device" guarantee on APFS; I could not test power loss.
4. Cost context [derived, from `vector-db-tricks.md`]: at 256 MiB shards the syncs are amortised over about 0.4 s of encoding at 2.4 M rows/s, so durable=False matters most for small shards, `seal`-heavy streams and slow drives.
5. Docs/validation: `durable=False` with `seal()` should state that `durable_rows` then equals published rows only in level `none`, or raise if combined (decision for the maintainer).

## 6. Cold first query and page-cache warming

| Technique | What it does (source) | Published effect | Notes for fse |
|---|---|---|---|
| `MAP_POPULATE` (Linux, mmap flag) | "Populate (prefault) page tables for a mapping. For a file mapping, this causes read-ahead on the file." [man7 mmap(2)] | Qdrant gist: 10M x 512d float32, 20.9 GB, Linux 6.2, 32 GB RAM, exact search limit 1000: cold first search 44.35 s (normal) vs 457 ms (populate); startup 5 s vs 14 s; steady state about 410 to 500 ms in all variants [gist, single run, third party] | Python 3.10+ has `mmap.MAP_POPULATE` [docs.python.org]; numpy.memmap does not expose flags, so a populating mmap would be created separately. Not on macOS [man mmap]. |
| `MADV_WILLNEED` on a mapping | "Expect access in the near future" [man7]. | Same gist: cold first search 47.11 s, no improvement; with first-byte read 46.88 s [gist] | Do not rely on it for mapped files. Third-party measurement, one machine. |
| `MADV_POPULATE_READ` (Linux 5.14) | "Populate (prefault) page tables readable, faulting in all pages in the range just as if manually reading from each" [man7 madvise(2)] | none found | Python's mmap docs do not list it [docs.python.org], so it needs `ctypes` or a constant by value. |
| `posix_fadvise(WILLNEED)` | "initiates a nonblocking read of the specified region into the page cache"; kernel may read less under pressure, "a few megabytes" usually honoured [man7 fadvise] | none found | Fits the file, not the mapping; fine for chunked prefetch ahead of the scan. Linux/POSIX; not on macOS. |
| `posix_fadvise(SEQUENTIAL)` | Linux doubles the readahead window for the whole file [man7] | none found | Cheap hint for `vectors.npy`. |
| `MADV_SEQUENTIAL` / `MADV_RANDOM` | exist on macOS and Linux [man madvise, man7] | Elasticsearch issue 27748 (Lucene mmap, RHEL 7.3, spinning disks): with a patched `MADV_RANDOM`, cold average query 4259 ms vs 3917 ms plain mmap and 3747 ms NIO; warm 2545 vs 3361 vs 3470 ms; cache used 1607 MiB vs 7659 MiB; author: madvise "always reduce the readahead to 0: only 1 page of 4KB is read at a time" [GitHub issue, fetched] | Never advise RANDOM on the vector files of a sequential scan. An older, different workload; use only as the warning it is. |
| macOS `F_RDADVISE` / `F_RDADVISEV`, `F_RDAHEAD` | "Issue an advisory read async with no copy to user"; "Turn read ahead off/on" [man fcntl] | none found | The macOS counterpart of `posix_fadvise(WILLNEED)`; reachable via `fcntl` with a packed `radvisory` struct (offset, count); behaviour on current macOS unverified. |
| Read-through ("pre-touch") | Lucene `MMapDirectory` preload touches every page; recommended for `.vec/.veq` because HNSW random paging is "crazy slow" [Lucene mailing list, search snippet, unverified] | not measured by me | Portable and the only method that works everywhere: read each file with large sequential `readinto` calls into a reused buffer on a small thread pool. |
| Qdrant storage tiers | vectors always mmapped; `cached` preloads into the page cache at startup, `cold` does not [doc, qdrant.tech storage, from the earlier survey] | none | Same idea as `VectorStore.warm()`. |

Notes:
- The gist's cold first search reads 20.9 GB in about 44 s, about 470 MB/s [derived]; the gist's startup populate took about 9 s longer, so populate and search-time faulting are not equally fast (populate was faster by a factor of about 5 per byte in that run [derived]). Drive and kernel differ from Apple SSDs.
- Results stay exact; this is I/O only.
- Earlier survey said switching Lucene's `.vec` from RANDOM to normal advice "doubled latency" (GitHub issue, snippet): I could not confirm that figure; treat as unverified. The measured ES issue above points the same direction (RANDOM removes readahead) with smaller, mixed numbers.
- "Are you sure you want to use MMAP in your DBMS?" (CIDR 2022, db.cs.cmu.edu/mmap-cidr2022/): the page summary lists no numbers; the paper and `github.com/viktorleis/mmapbench` hold the measurements; not read here. Relevant caveat for fse: page-cache eviction and TLB cost under memory pressure are the paper's main mmap complaints [paper abstract, partial].

### Suggested `VectorStore.warm()` for fse [derived]
- Default: sequential reads of `vectors.npy` (and the small tensors) in 4 to 16 MiB chunks into a reused buffer, on 2 to 4 threads, in shard order, optionally with `posix_fadvise(SEQUENTIAL)` (Linux) first; skip the data region of nothing: the header is small. Optional `blocking=False` returns a handle so the first search can overlap with the tail of the warm-up. Platform branches: Linux may use `MAP_POPULATE`-style mapping later; macOS reads only (no populate flag, `F_RDADVISE` optional).
- Effort low, exact. Files: `src/fse/_store/_vectorstore.py` (method), `_shard.py` (paths of a shard's files), tests with a temporary store; no change to persistence names. Measure with a controlled cold cache (macOS `purge` needs sudo; another benchmark is running, so do this later); report cold first search before/after as AGENTS.md requires.
- Do not set `MADV_RANDOM`; `numpy.memmap` default is normal advice, which is right.
- Check available RAM before reading more than a fraction of it (warming a store larger than RAM just evicts itself); [derived] suggestion: cap by `os.sysconf` physical pages or raise a clear error, per the "never fail silently" rule.

## 7. Applicability to fse, ranked

| # | Finding | Exact? | Effort | Where |
|---|---|---|---|---|
| 1 | One `F_FULLFSYNC` per publish after plain `fsync`s (documented device-wide drain) | yes, same guarantee per the man page | low | `src/fse/_container/_npy_inplace.py` (`sync_file`), `_container/_common.py` (dir sync), `_store/_shard.py:153,169,193` |
| 2 | `durability` enum (`full` / `ordered` / `none`) with `F_BARRIERFSYNC` for `ordered` on macOS | data yes; durability weaker | low to medium | `_store/_sink.py`, `_writer.py`, `_shard.py`, docs/architecture/store.md, "sentence_vectors 1" |
| 3 | Sync-at-flush with prefix recovery on open (verify tail shards, truncate at first bad one) | yes | medium | `_store/_layout.py` (open), `_vectorstore.py`, `store.json` marker; needs a persistence-name decision |
| 4 | `warm()` as chunked sequential read-through | yes | low | `_store/_vectorstore.py`, `_shard.py` |
| 5 | Keep scan-side tricks from `vector-db-tricks.md`; Lance's prune-with-identical-compare contract is the template for the exact int8 prefilter | yes | high | `_native_kernel_search.h` etc. |
| 6 | Shard compaction of adjacent small shards (Lance `compact_files` model) | yes | medium-high | `_layout.py`, `_shard.py`, `_sink.py` |

## 8. Gaps and cautions
- No timing was run; every "effect" above is a published number from the named source or marked derived.
- Line numbers refer to files downloaded from `main`/`master` on 2026-10-28 and will drift; function and option names are the stable handles. No permalinks pinned to SHAs.
- Not verified: Lance reading through mmap anywhere in the repo (only checked `lance-io` local/object_store); that DuckDB's compiler does not vectorise the fold loop (my inference); macOS PostgreSQL default sync method; `F_RDADVISE` behaviour on current macOS; whether Python's `os.fsync` on macOS ever uses `F_FULLFSYNC` (fse handles this itself); power-loss behaviour of any proposal here.
- Lance docs pages were summarised by the fetch tool; quotes from the code files (`.rs`, `.c`, `.h`) come from the downloaded files themselves.
- The Qdrant gist and the Elasticsearch issue are single-machine, third-party measurements.

## 9. Sources
DuckDB
- https://github.com/duckdb/duckdb/blob/main/extension/core_functions/scalar/array/array_functions.cpp
- https://github.com/duckdb/duckdb/blob/main/extension/core_functions/include/core_functions/array_kernels.hpp
- https://github.com/duckdb/duckdb-vss (README, `src/hnsw/hnsw_index.cpp`, `src/hnsw/hnsw_optimize_topk.cpp`)
- https://duckdb.org/docs/2.0/core_extensions/vss.html
- https://duckdb.org/2024/05/03/vector-similarity-search-vss.html

Lance / LanceDB
- https://lance.org/format/file/ , https://lance.org/format/file/encoding/ , https://lance.org/format/table/ , https://lance.org/format/table/layout/ , https://lance.org/format/table/transaction/
- https://github.com/lance-format/lance : `rust/lance-index/src/vector/flat/index.rs`, `flat/storage.rs`, `vector/bq/prune.rs`; `rust/lance-table/src/io/commit.rs`; `rust/lance-io/src/local.rs`, `object_store.rs`, `object_writer.rs`; `rust/lance/src/dataset/optimize.rs`, `scanner.rs`, `mem_wal.rs`, `io/exec/knn.rs`
- https://docs.lancedb.com/indexing/vector-index

sqlite-vec
- https://github.com/asg017/sqlite-vec/blob/main/sqlite-vec.c
- https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html
- https://alexgarcia.xyz/sqlite-vec/features/vec0.html

Durability
- https://www.sqlite.org/pragma.html , https://www.sqlite.org/wal.html , https://github.com/sqlite/sqlite/blob/master/src/os_unix.c
- https://www.postgresql.org/docs/current/runtime-config-wal.html , .../wal-async-commit.html , .../wal-reliability.html ; https://github.com/postgres/postgres : `src/backend/storage/file/fd.c`, `src/include/access/xlogdefs.h`, `src/include/port/darwin.h`
- https://github.com/facebook/rocksdb : `include/rocksdb/options.h`, `env/io_posix.cc`; https://github.com/facebook/rocksdb/wiki/Basic-Operations
- https://github.com/LMDB/lmdb : `libraries/liblmdb/lmdb.h`, `mdb.c`
- https://github.com/rust-lang/rust/blob/master/library/std/src/sys/fs/unix.rs
- macOS manual pages on this machine: `man fcntl` (F_FULLFSYNC, F_BARRIERFSYNC, F_RDADVISE, F_RDAHEAD, F_PREALLOCATE), `man madvise`, `man fsync`, `man mmap`

Page cache warming
- https://man7.org/linux/man-pages/man2/mmap.2.html , .../madvise.2.html , .../posix_fadvise.2.html
- https://docs.python.org/3/library/mmap.html
- https://gist.github.com/timvisee/a775f9a0016f5410b46ce7fdbd1b253b
- https://github.com/elastic/elasticsearch/issues/27748
- https://db.cs.cmu.edu/mmap-cidr2022/ (page only; numbers not read)
- Lucene mailing-list pre-touch remark: seen only in a search snippet (https://go.mail-archive.com/java-user@lucene.apache.org/msg52397.html), unverified.


## 10. fse 2.0 publication policy

Rebello et al., [Can Applications Recover from fsync Failures? (USENIX ATC 2020)](https://www.usenix.org/conference/atc20/presentation/rebello), find that tested Linux filesystems mark pages clean after failed flushes. Retrying fsync can then miss the lost writes. fse adopts a conservative policy: a shard originally tagged `fse.durability=none` is never promoted by a later sync or successful checksum verification. This is an application policy informed by that failure study, not a claim that its Linux experiments establish Windows behavior.

Windows permits "none". The existing [MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw) call retains MOVEFILE_WRITE_THROUGH and omits replacement. Microsoft documents write-through behavior for that move flag, specifically flushing copy-and-delete moves. It does not establish fse's entire multi-file protocol. "none" omits explicit file flushes while preserving that rename. [FlushFileBuffers](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers) separately flushes a file's buffered information. Store identity is immutable, so publication does not replace an open store.json. Real Windows CI checks behavior and mapping lifetimes; POSIX power-loss models are not a physical Windows power-loss proof.

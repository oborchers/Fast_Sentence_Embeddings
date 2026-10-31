# Tricks of vector databases and search libraries for fse's exact search, scoring and writing

Date: 2026-10-28. Scope: what fse (store of float32/float16/int8 rows, bit-exact "exact" kernel, BLAS mode, floor-based top-k, three-thread writer) could adopt without becoming an ANN database.

Method and honesty notes. I read docs/architecture/store.md, "sentence_vectors 1", `_store/_search.py`, `_shard.py` and `_native_kernel_search.h`, then fetched primary docs/source (list at the end). Statements tagged **[verified]** come from a page I fetched this session. **[memory]** is my background knowledge, not re-checked. **[derived]** is my own reasoning or arithmetic, not a published result. **[unverified]** is a claim from a secondary source or a search snippet. No benchmark was run. Where the fetch tool summarised a page, numbers are as reported by that summary and worth a second look before they go into a PR description.

## 1. Summary

The most important finding is a design point specific to fse, not a library feature:

- fse's "exact" contract fixes the float32 summation order (dot16), which forbids integer SIMD for the final score. But integer sums are associative, so an **int8 x int8 integer prefilter with a provable error bound, followed by exact rescoring of the survivors with the existing kernel, stays bit-identical to the contract**. For int8 stores the cosine is `dot(int8 codes, unit query) * inv_norm` (the kernel decodes codes with a plain `(float)int8`), so the only error source is quantizing the *query*. The bound is `eps = ||q_unit - s*qi||_2` (Cauchy-Schwarz with the stored unit-norm rows), computable per query in microseconds. [derived; no library publishes exactly this. The pattern "cheap quantized scan, then exact rescore" is universal (Qdrant, Weaviate, pgvector, Elastic, ScaNN), but all of them choose the oversampling factor heuristically and so are approximate. RaBitQ gives a probabilistic bound; PDX-BOND is exact but for L2 partial distances.]
- Everything else is smaller: fuse the floor test into the kernel (FAISS fused kernels, ScaNN SIMD mask push), a `warm()`/readahead call for the cold first query (Qdrant "cached" tier, Lucene pre-touch), shard compaction (Qdrant optimizer, Lance compaction), a non-durable write mode (standard in databases), and an opt-in approximate binary/int8 prefilter with oversampling for those who want speed over bits.
- GPU: documented tricks exist (FAISS fused L2 + k-selection, fp16 storage, k <= 2048) but fse's import contract forbids mlx/torch, so this is an optional-extra project, not a trick to adopt now.

Ranked top picks are in section 8.

## 2. Faster exact brute force

### 2.1 FAISS: SIMD below a threshold, BLAS above, heap per query

- What: `IndexFlat*` decodes/reads all vectors sequentially. `knn_inner_product` picks a sequential SIMD path (`exhaustive_inner_product_seq`, `fvec_inner_product<SL>`, a heap per query) or `sgemm` with tiles. Constants in `faiss/utils/distances.cpp`: `distance_compute_blas_threshold = 128000`, `distance_compute_blas_query_bs = 4096`, `distance_compute_blas_database_bs = 1024`, `distance_compute_min_k_reservoir = 100`. The switch is `nx * d < threshold` (nx = number of queries) or a selector present. [verified, current `main`]
- The reservoir constant was defined but I did not see it used in the part of the file returned; do not rely on "reservoir above k=100" without reading `ResultHandler.h`. [verified that the constant exists; usage unverified]
- Effect: no number published on the wiki page; the constants are tuning, not a measured claim. FAISS wiki says brute force is preferred when the number of queries is limited. [verified]
- fse relation: fse already has the same two-path structure, with a crossover of 8 queries measured on its own hardware. FAISS's value is 128000/d = about 427 queries at d=300, but its SIMD path is a per-pair `fvec_inner_product` that threads over queries, so the two crossovers are not comparable. Nothing to adopt; just evidence that a crossover constant is the standard design.
- Also in FAISS: `exhaustive_L2sqr_fused_cmax` fuses distance and 1-NN selection to avoid storing temporary dot products; only for small d (PQ training). [verified]
- Exact: yes (the BLAS path is "close", like fse's `blas` mode).

### 2.2 Fuse threshold test and candidate push into the kernel

- What: ScaNN's `FastTopNeighbors` keeps an unsorted buffer of about 2k (rounded to 32), appends candidates, and when full prunes with `ApproxNthElement`, then sets an atomic `epsilon_` to the boundary value; candidates are filtered by `dist < eps` with a Highway SIMD compare mask whose set bits are pushed. [verified, `scann/utils/fast_top_neighbors.h`] FAISS's `HeapArray` does the same with a binary heap per query and a root compare. [verified, `Heap.h`]
- Effect: no published number for the fusion itself in these sources. The FAISS GPU paper reports that its fused L2 + k-selection kernel is at least 25% faster than the unfused version. [unverified, from a search summary of arXiv 1702.08734]
- Exact: yes, if the kernel compares the same float32 values with the same `>=` as `best()` does now (the existing code already treats a float compare as a superset and decides ties on integer order keys).
- fse relation: your floor is the same idea, applied after the kernel has written a (rows x queries) float32 block (about 4 MiB per task) that numpy then re-reads (`scores >= from_key(floor)`, `count_nonzero`, `flatnonzero`). Moving the compare into `fse_cosines_body` (write only rows with `o[j] >= floor[j]` as (row, query, value) triples, bounded by a cap with overflow flag) removes the 4 MiB store and the numpy passes.
- Difficulty: medium. Touches `src/fse/_native_src/kernel/_native_kernel_search.h` (new `fse_cosines_floor`), `_native.pyx`, `_backends/_cpu.py`/`_base.py`, `_store/_search.py` (`best`), conformance meta-tests. Must keep -0.0/subnormal handling (flush modes) identical. Gain unknown: profile first (the 1-query path at 10-13 ms may already be kernel-bound; for 8 queries the numpy part may matter more). Per AGENTS.md, profile with py-spy before building.

### 2.3 Integer SIMD dot products (NEON SDOT/I8MM, AVX2/AVX-512 VNNI)

- What: int8 x int8 products accumulated into int32. One ARM `SDOT` does 4 int8 pairs per lane into int32; x86 `VPDPBUSD` (VNNI) the same. [verified for the ARM description via the OpenSearch/Elastic material below; NumKong (successor of SimSIMD) lists AVX-512 VNNI `DPBUSD` for Int8xInt8 and NEON DotProd, and does not mention I8MM in the fetched page]
- Who: Lucene/Elasticsearch calls native C intrinsics for int8 dot via the Foreign Function API because the JVM Panama path lacks SDOT/UDOT: micro-benchmark about 6x over the Panama code. [verified from search snippet of "ludicrous speed" post; Graviton 2 6.3x at 256 dims to 13.4x at 4096 dims, Graviton 3 1.7x to 5.8x, from OpenSearchCon EU 2026 slides, **[unverified]** secondary]
- NumKong (SimSIMD successor) int8: 14.8 giga-scalar-ops/s single-thread 2048-d dot on Sapphire Rapids vs 1.1 for NumPy+OpenBLAS (which overflows, so not accuracy-matched). int32 accumulation, saturating reductions. [verified from README]
- Exact: the integer dot itself is exact and order-free (associative), but a float query must be quantized first, so the *score* is not the contract score. See 3.1 for how to make the whole search exact anyway.
- Hardware: M1 supports dot-product (SDOT); I8MM (SMMLA) I believe arrives with M2/ARMv8.6 [memory, check `sysctl hw.optional.arm.FEAT_I8MM` before designing]. BLAS libraries do not offer int8 GEMM on Accelerate [memory], so SMMLA/VNNI matrix tiles would be hand-written.
- Difficulty: medium-high for the kernel (NEON + AVX2 + optionally VNNI, ISA dispatch exists: `isa` argument, `fse_dot16_neon/avx2`), plus tests. `_native_kernel_search.h`, `_native_kernel_steps.h`.

### 2.4 Dimension-pruned exact scans (PDX, ADSampling, BSA)

- What: PDX stores blocks of 64 vectors "vertically" (dimension-major inside the block), so a scan can add dimensions one at a time over 64 rows and drop rows whose partial distance already exceeds the current k-th best. PDX-BOND is **exact** ("no recall trade-off"), orders dimensions by the query (decreasing query value or distance-to-means). ADSampling is approximate (random projection plus hypothesis test). [verified, arXiv 2503.04422 HTML]
- Effect: PDX-BOND on exact search 2.5x faster than FAISS, 3.7x USearch (intro says 4.0x), 4.0x Milvus (intro 3.3x), 6.2x scikit-learn on Intel Sapphire Rapids; the paper itself gives conflicting numbers between intro and section 6.5. BOND alone is limited to about 1.6x over exact KNN because of bound latency; PDX distance kernels average 40% faster than SIMD horizontal kernels (abstract) or 2.0x (Table 4). Gains largest at D <= 32. Tested on x86 and Graviton4; not on Apple M-series; bounds are described for Euclidean distance, not stated for inner product/cosine; int8 not discussed. [verified]
- Exact: BOND yes. fse relation: poor fit. A new row layout would break `sentence_vectors 1` and the memory-mapped row-major scan; cosine on unit rows needs a different bound; our float order contract forbids reordering dimensions. Rank low. A cheaper cousin is 3.1.

### 2.5 Query batching and cache blocking

- fse already blocks rows x queries to about 4 MiB of scores and reads each row once per query block. FAISS blocks queries by 4096 and database by 1024 in the BLAS path (above). Nothing to adopt beyond tuning: [derived] with 10 workers and 258 MB of int8 data at 859,346 x 300, one query costs 10-13 ms, i.e. about 20-25 GB/s of row traffic. M1 Pro's memory bandwidth is advertised at 200 GB/s [memory], so a bandwidth-bound int8 scan would be several times faster; the current kernel is compute-bound by the decode (`int8 -> float`) plus the 16-lane fma chain. That gap is the opportunity of 3.1.

### 2.6 Early termination by block-level bounds

- What: for `score(weights, bias)` a block's maximum row norm (1/min inv_norm) bounds `|w.x| <= ||w|| * max||x||`; for cosine, unit rows give nothing. Cluster/ball bounds (IVF-style, exact if you keep the cell radius) need clustered data; the store is in insertion order. Not published as a trick for flat stores in the sources I read. Low value for fse; [derived].

## 3. Two-stage search that stays exact or nearly

### 3.1 Provably exact int8 prefilter with exact rescoring (proposal, derived)

- Setup (int8 stores). Query unit vector q (after normalize@1), quantize `qi = round(q / s)` with `s = max|q| / 127`, error vector `e = q - s*qi`, `eps = ||e||_2` (exact float64 per query). For a row with codes c, the contract cosine is `c.q * inv` where `inv = 1/||c||`. Estimate `est = s * (c . qi) * inv` with the integer dot in int32 (no overflow: 300 x 127 x 127). Then `|est - cos| <= eps` because `|c.e| * inv <= ||e||` (Cauchy-Schwarz and `inv = 1/||c||`). Add a small relative slack for float rounding of `est`.
- Exact top-k proof sketch: after the integer scan of a block, let E_k be the k-th largest `est` among rows scanned so far for that query (or the merged floor from exact scores, whichever is higher). At least k rows have true cosine >= E_k - eps, so the true k-th best is >= E_k - eps. Any row with `est < E_k - 2*eps` has true cosine < E_k - eps and cannot be in the top k. Rescore all rows with `est >= E_k - 2*eps (- slack)` with the existing exact kernel on a gathered row list, merge with the same `Candidates` order. Ties: rows with equal true score have est within 2*eps of each other, so equal-score rows at the boundary are all rescored and the existing key/ties logic decides. The result is bit-identical to `mode="exact"`.
- Size of the survivor set: governed by how many rows lie within 2*eps of the k-th best. For d=300 and a unit query, `eps` is of order `s * sqrt(d/12)`; with `max|q|` around 0.2 to 0.3 that is about 0.01 (**[derived estimate, measure it]**). Whether that leaves few rows depends on the score density near the k-th best, which is data dependent; the algorithm stays exact either way and can fall back to the plain exact path when the survivors exceed a fraction of the block.
- Expected effect: [derived, not measured] the scan becomes memory-bandwidth bound instead of decode/fma bound; plausibly 2 to 5x at 1 to 8 queries on int8 stores. Float32/float16 stores cannot use it (no integer rows); a float16 variant would need an fp16 dot (NEON has FMLA.8H but accumulating in fp16 loses precision, which NumKong avoids for general dot products [verified]).
- Prior art: the filter-and-refine pattern is what Qdrant ("oversampling, rescore"), Weaviate (`rescoreLimit`), pgvector (re-rank subquery), Elastic (`rerank`), ScaNN (`reordering_num_neighbors`) all do, but with a user-chosen candidate multiplier, so none guarantees the exact top-k. RaBitQ has a "sharp theoretical error bound" (abstract only; form not stated in what I read) and Elastic's BBQ rescoring does not use the estimated error. [verified]
- Difficulty: high but well bounded. New native function(s) in `src/fse/_native_src/kernel/_native_kernel_search.h` (integer scan with floor, and `cosines` over a gathered index list), Cython glue `_native.pyx`, backend method in `_backends/_cpu.py`/`_base.py`, a pure numpy reference in `fse._reference` (the repo rule: every kernel has a reference and passes `tests/test_conformance.py`), and the orchestration in `_store/_search.py`. New persistence names are not needed (no stored-format change); it is a search strategy. Needs a new mode name or an internal switch under `auto`; because the result is bit-identical, it can be silent.

### 3.2 Binary / int4 pre-scan with oversampled rescoring (approximate, opt-in)

- What: keep 1 bit per dimension (sign) in a sidecar, scan with XOR+popcount, take `oversample * k` candidates, rescore with stored vectors. Qdrant docs: binary quantization 32x memory reduction, "up to a 40x speedup" (their claim), recall@100 0.98 at 4x oversampling for ada-002 1536d, 0.9966 at 3x for text-embedding-3-large 3072d; for 512-d text-embedding-3-small only 53 to 55% without and 71 to 91% with rescoring; they recommend oversampling 3 to start. Higher-dimension models do far better. [verified, qdrant.tech quantization guide]
- Hugging Face (mxbai-embed-large-v1, 1024-d, MTEB retrieval): binary 24.76x average speedup (15.05x to 45.8x), 92.53% performance retained, 96.45% with rescoring (multiplier 4); int8 3.66x average speedup (2.99x to 4.8x), 97% retained, about 99% with rescore multiplier 4 to 5. Retention is NDCG@10, not recall. [verified, huggingface.co/blog/embedding-quantization]
- Elastic BBQ (Lucene): per-segment single centroid, 1 bit per dimension for documents plus 2 or 3 corrective floats, query in int4, scored with bit-plane AND + popcount; rescoring on float32 originals; >90% recall at 3x oversampling (CohereV3 1M, 1024-d, with HNSW); 138M-vector table: recall 70/45/90% at candidates 100/20/1000. [verified]
- pgvector: `binary_quantize(embedding)::bit(n)` Hamming search for 20, then re-rank by cosine for 5 (the example uses a 4x factor). `halfvec` stores 2*dims + 8 bytes. [verified]
- Weaviate: over-fetch then rescore for SQ/RQ/BQ; 8-bit RQ 98 to 99% recall at 4x compression, 4-bit with rescoring of top 50 reaches 99.8% (internal tests); flat index accepts 8-bit and 1-bit RQ. [verified]
- For fse the data are mostly 300-d word-vector averages. All published binary numbers above are for 512 to 4096 dims from large transformer embedders; for 300-d expect a much larger oversampling factor (the 512-d number at 53 to 55% without rescoring is the nearest data point). Mark: [derived caution].
- Exact: no (recall below 1 unless the oversample is huge). Opt-in only.
- Difficulty: high. Needs a sidecar tensor per shard (new optional tensor name in the shard manifest, with hash and `verify` coverage; docs/architecture/store.md, "sentence_vectors 1" already reserves `row_ids` for 2.1, so extensions are anticipated) or computing signs on the fly from int8 codes (no extra storage but no bandwidth saving). Touches `_layout.py`, `_shard.py`, `_writer.py`, `_sink.py`, `_search.py`, kernels. Given 3.1 gives exact results from the same data, I would not build this before 3.1.

### 3.3 Asymmetric quantization

- Qdrant (v1.15+): binary stored vectors with scalar-quantized (8-bit or 4-bit) queries, "requires less rescoring for the same quality"; no numbers given. [verified] Elastic BBQ does the same with int4 queries. [verified] fse's 3.1 is the same asymmetry (int8 stored, int8-quantized query) with a bound.

### 3.4 Matryoshka / prefix shortlists

- Vespa's phased ranking: shortened 256-d embeddings first, full 3072-d on the top 100 by default. [unverified, search snippet of pyvespa notebook] Requires embeddings trained for truncation; fse's encoders (SIF, uSIF, averages) are not, so only useful if hub models are Matryoshka. Not applicable for now.

## 4. Storage and I/O

### 4.1 Memory tiers and warm-up

- Qdrant: vectors are always in a memory-mapped file; `cached` tier preloads them into the page cache at startup, `cold` does not; `memmap_threshold` chooses the default. [verified, storage docs]
- Lucene: `MMapDirectory` pre-touch of `.vec/.veq` files was the recommended workaround because paging in through random access is "crazy slow" (mailing list), and Lucene 10 calls `posix_madvise` from Java; switching `.vec` from RANDOM to normal read advice doubled latency on a 10M x 768d dataset (GitHub issue; random vs sequential advice must match the access pattern). [unverified, search snippets]
- A community gist for Qdrant found `MADV_WILLNEED` showed no visible improvement on the first search, whereas `MAP_POPULATE` pre-populated memory (cold cache cost about 45 s extra for 20.9 GB, 10M x 512d). [unverified, community gist]
- fse relation: the first search is disk-bound. A flat scan is perfectly sequential per shard, the case where `MADV_SEQUENTIAL`/readahead helps and `MADV_RANDOM` hurts. Options: (a) `VectorStore.warm()` reading the files with a few threads ahead of the search (works on every OS, can be overlapped with the first query by a prefetch thread that stays N blocks ahead of the workers), (b) `mmap.madvise(MADV_SEQUENTIAL/WILLNEED)` where available (Python's `mmap` has `.madvise`; numpy memmap exposes `_mmap`; macOS honors fewer hints; `F_RDADVISE`/`F_RDAHEAD` via `fcntl` [memory]), (c) `MAP_POPULATE`/`MADV_POPULATE_READ` on Linux [memory].
- Exact: yes (I/O only). Effort: low. Touch `src/fse/_store/_vectorstore.py`, `_shard.py` or the container loader that creates the memmaps. Effect: unknown, must be measured on cold cache (`purge` on macOS is a heavy hammer; the repo notes another benchmark is running, so do it later).

### 4.2 io_uring and direct I/O

- Qdrant's storage abstraction: mmap (queue depth 1, "cold read about 100 us on an SSD"), io_uring (depth 16, Linux 5.1+; batches submissions into one syscall when cached), `O_DIRECT` bypassing the page cache at the price of 4 KiB-aligned reads. No throughput numbers. [verified, qdrant.tech/articles/universal-io]
- fse relation: a sequential scan of GB-sized files gains little from io_uring when readahead works, and `O_DIRECT` would defeat the page cache that makes repeat searches fast (10-13 ms). Skip, except possibly a Linux-only prefetcher. Rank low.

### 4.3 Layout: alignment, padding, columnar

- Lance's file format stores column data in pages rather than Parquet row groups for random access; fragments hold data files per column subset. [verified, lance.org/format overview] For a brute-force scan of one dense column this is irrelevant: fse's `vectors.npy` is already a single contiguous row-major column.
- Row alignment/padding to 64 B (SIMD loads and cache lines): a 300-byte int8 row straddles cache lines. Padding to 320 costs 6.7% more bytes and would change `sentence_vectors 1` (shape/dtype contract) so only a 2.x option. Benefit unmeasured; common in FAISS/ScaNN code paths that pad dims to multiples of the SIMD width [memory, unverified]. Skip until 3.1 exists and shows misaligned loads in the profile.
- Compression: not useful for the scan (decode cost, random access); only for transport.

## 5. Top-k selection structures

- FAISS binary heap per query (`HeapArray`, replace-top only if better); ScaNN amortized buffer of about 2k with batched `nth_element` and an atomic threshold; FAISS GPU paper: full sort per query "more than 10x slower" than their k-selection (secondary summary). [verified for FAISS/ScaNN code, unverified for the GPU number]
- fse's `Candidates`/`best` is structurally the ScaNN scheme (unsorted candidates above a floor, periodic merge to k, floor raised) done in numpy, with an exact tie rule that the others do not have. Nothing to adopt except moving the first filter into C (2.2) and, for k large, `np.partition` already.

## 6. Write path

What fse does (hash thread, `pwrite`, publish with fsync and exclusive rename; about 2.4 M int8 rows/s at 10 workers) already matches the "immutable segment, atomic publish" family. Relevant comparisons:

1. Segments plus WAL (Qdrant): writes go first to a WAL with sequence numbers, then to segments; appendable vs non-appendable (immutable) segments; per-point versions let it replay after an unclean shutdown; an optimizer converts and merges segments. [verified, storage concepts page] Lance: fragments with manifests and ACID commits. [verified overview; compaction/atomic-rename details not in the fetched page, **[memory]** that Lance exposes a compaction call]. fse has no WAL by design; a crash leaves a valid prefix of whole shards, which is the stronger simplicity.
2. Compaction of small shards: after many `seal`/`publish` calls (streaming ingestion) a store has many small shards; each costs a task split, a file mapping and a manifest read at `open`. Qdrant's optimizer and Lance's compaction exist for this reason. A `compact(store)` that writes merged larger shards beside the old ones and swaps by renaming in order is possible because shards are named by first row (contiguity must be preserved, so only adjacent runs can merge and the swap must keep the "rows run without a gap" invariant for concurrent readers: this is the hard part). Effort medium-high; touches `_layout.py`, `_shard.py`, `_sink.py`. Exact: yes. Value: only for streaming use.
3. Durability vs throughput switch (`durable=False`): the standard dial in databases (SQLite `synchronous`, `PRAGMA fullfsync` on macOS, PostgreSQL `synchronous_commit`, RocksDB `WriteOptions::sync`) [memory; docs not fetched this session]. For fse: skip `F_FULLFSYNC`/`fsync` of each shard and of the directory, keep the exclusive rename so readers still see whole-or-nothing at the *process* level; a power cut can then leave a published shard whose data never reached disk, so `verify` (xxh3-128 over the data) becomes the recovery tool. Cheaper variants that stay durable: (a) overlap the sync of shard N with the writing of shard N+1 (the publish thread already runs separately), (b) one directory sync per batch of shards instead of per shard. Impact: at 256 MiB shards the fsync is amortized over about 0.4 s of writing at 2.4 M rows/s with 300-d int8 rows [derived: 256 MiB / 300 B = 895k rows], so the gain is mostly for small shards or `seal`-heavy streams; measure `F_FULLFSYNC` cost on the target SSD before building. Effort low (`_writer.py`, `_shard.py`, `_container/_npy_inplace.py`, a `durable` argument on `ShardSink`, docs/architecture/store.md, "sentence_vectors 1"). The AGENTS.md rule "validate strictly" suggests refusing `durable=False` for `seal()`'s promise, or documenting that `durable_rows` then means "published", not "synced".
4. Batching: pgvector docs recommend `COPY` for bulk loads and building the index afterwards. [verified] fse's analog is the 16,384-row aligned shard capacity; fine.
5. Checksums: fse uses xxh3-128 in a dedicated thread; no competitor in the sources has a stronger story for flat stores. Optional: a lazy verify at first `open` of a shard in the background (not on the search path).
6. Preallocation/`posix_fallocate`/`F_PREALLOCATE` to avoid fragmentation and `ENOSPC` late; fse writes the header for the full shard and shrinks at the end (sparse file). [memory, no source consulted]. Low priority.

## 7. GPU

- FAISS `GpuIndexFlat` is the GPU `IndexFlat`; k and nprobe must be <= 2048 and performance drops above about 512; float16 storage is available on all supported GPUs ("recall@N mostly unaffected, distances affected"); single GPU 5x to 10x faster than the CPU implementation, 8 GPUs 6x to 7x more; batch queries (not size 1); about 1 GB of scratch advisable. [verified, wiki "Faiss on the GPU"]
- Billion-scale GPU paper (Johnson, Douze, Jegou): k-selection "up to 55% of theoretical peak performance"; a nearest-neighbor implementation "8.5x faster than prior GPU state of the art"; the fused L2 + k-selection kernel is at least 25% faster than the unfused one. [first two verified from the abstract; the 25% [unverified] from a search summary]
- cuVS brute-force doc page could not be fetched (the NVIDIA docs URL returned 404); no cuVS claims here.
- Apple GPU (MLX/Metal) [derived, speculative]: the 256-query BLAS case is about 1.3e11 FLOP (256 x 859,346 x 300 x 2) in 170 ms, i.e. about 0.8 TFLOP/s effective on Accelerate. A Metal fp16/fp32 matmul on an M1 Pro (Apple quotes about 5 TFLOP/s fp32 for the 16-core GPU [memory]) could be faster but must decode int8 on the GPU and select top-k there (MLX has `argpartition`/`topk`-style ops [memory]); the unified memory makes zero-copy plausible but I found nothing documenting wrapping an mmap'd npy without a copy. Note `[tool.importlinter]` forbids mlx and torch in fse; this would need a separate optional package. Not recommended now.

## 8. Ranked list

| # | Trick | Exact? | Expected impact | Effort | Where |
|---|---|---|---|---|---|
| 1 | Integer int8 prefilter with provable bound, exact rescoring of survivors (section 3.1) | yes, bit-identical | Large for 1 to 8 queries on int8 stores: could approach memory bandwidth; my estimate 2 to 5x, to be measured. No effect on float stores or on BLAS-regime queries | High | `_native_kernel_search.h`, `_native.pyx`, `_backends/_cpu.py`, `_base.py`, `_reference.py`, `_store/_search.py`, tests/test_conformance.py |
| 2 | Cold-start `warm()` plus sequential readahead/prefetch thread (4.1) | yes | Only the first search; potentially large on a cold disk, but sources disagree on which madvise flags work. Measure with a controlled cold cache | Low | `_store/_vectorstore.py`, container memmap loader |
| 3 | Fuse the floor test into the native cosines kernel (2.2) | yes | Unknown, probably modest at 1 query, larger at 8 to 16 queries; profile first | Medium | `_native_kernel_search.h`, `_native.pyx`, `_backends`, `_search.py:best` |
| 4 | Opt-in approximate fast mode: the int8 scan of #1 without rescoring, or with a fixed oversample (Qdrant/HF/Weaviate pattern) | no | Same speed as #1 stage one; HF reports about 97% NDCG retained for int8 without rescoring on 1024-d embeddings (not recall, not 300-d) | Low once #1 exists | `_search.py` (new mode value; keep `auto` exact) |
| 5 | Shard compaction (Qdrant optimizer, Lance compaction) | yes | Search and open cost for stores with many small shards; none for stores written in full-size shards | Medium-high | `_layout.py`, `_shard.py`, `_sink.py`, new public function |
| 6 | `durable=False` and/or overlapped, batched syncs (6.3) | yes (data), weaker durability | Small at 256 MiB shards; larger for small shards on macOS where `F_FULLFSYNC` is slow (cost not measured here) | Low | `_writer.py`, `_shard.py`, `_container/_npy_inplace.py`, `_sink.py` |
| 7 | Binary-sign sidecar with oversampled rescoring (3.2) | no | Published 15x to 45x scan speedups are for 1024-d+ embeddings; 300-d recall will be worse; larger sidecar and format work | High | shard format, writer, search, kernels |
| 8 | Dimension-pruned exact scan, PDX-BOND style (2.4) | yes | Published 1.6x to 2.5x vs FAISS on x86 L2; not shown for cosine/int8/Apple; needs a vertical layout | Very high | new layout; not recommended |

Order rationale. #1 is the only item that raises the speed ceiling of the common single-query path without giving up the bit contract, and it fits the repo's existing "kernel vs numpy reference" discipline. #2 is cheap and addresses the one case your numbers say is slow (cold). #3 is a safe refactor-level speedup, gated on a profile. #4 follows for free from #1. #5 and #6 are about operating at scale and should wait for a user need. #7 and #8 do not pay off at 300 dimensions and 859k rows.

Before any of these: AGENTS.md requires before/after numbers with paired runs; I did none of that here.

## 9. Sources

Fetched this session (content as reported by the fetch tool's summary unless noted):
- FAISS index overview: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- FAISS distances.cpp (constants, BLAS dispatch): https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/utils/distances.cpp
- FAISS Heap.h: https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/utils/Heap.h
- FAISS fused kernel header: https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/utils/distances_fused/distances_fused.h
- FAISS brute-force wiki: https://github.com/facebookresearch/faiss/wiki/Brute-force-search-without-an-index
- FAISS on the GPU: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
- Billion-scale similarity search with GPUs (abstract): https://arxiv.org/abs/1702.08734
- Qdrant quantization guide: https://qdrant.tech/documentation/guides/quantization/
- Qdrant storage concepts: https://qdrant.tech/documentation/concepts/storage/
- Qdrant universal I/O article: https://qdrant.tech/articles/universal-io/
- pgvector README: https://github.com/pgvector/pgvector
- Hugging Face embedding quantization: https://huggingface.co/blog/embedding-quantization
- Elastic BBQ blog: https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch
- Weaviate vector quantization: https://docs.weaviate.io/weaviate/concepts/vector-quantization
- RaBitQ abstract: https://arxiv.org/abs/2405.12497
- PDX paper: https://arxiv.org/html/2503.04422v1
- ScaNN algorithms doc: https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
- ScaNN fast_top_neighbors.h: https://raw.githubusercontent.com/google-research/google-research/master/scann/scann/utils/fast_top_neighbors.h
- NumKong/SimSIMD README: https://github.com/ashvardanian/SimSIMD
- Lance format overview: https://lance.org/format/

From search results only (snippets, not opened; treat as unverified):
- Elastic "ludicrous speed" int8 dot post: https://www.elastic.co/search-labs/blog/vector-similarity-computations-ludicrous-speed
- Elastic BBQ SIMD post: https://www.elastic.co/search-labs/blog/bbq-vector-comparison-simd-instructions
- OpenSearchCon EU 2026 native SIMD slides: https://hosted-files.sched.co/opensearchconeu2026/8d/NATIVE_SIMD.pdf
- Lucene madvise/ReadAdvice discussion: https://github.com/elastic/elasticsearch/issues/27748 and https://www.elastic.co/search-labs/de/blog/lucene-and-java-moving-forward-together
- Qdrant madvise gist: https://gist.github.com/timvisee/a775f9a0016f5410b46ce7fdbd1b253b
- Vespa Matryoshka: https://pyvespa.readthedocs.io/en/latest/examples/Matryoshka_embeddings_in_Vespa-cloud.html and https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/
- LanceDB vector index docs (refine_factor): https://docs.lancedb.com/indexing/vector-index

Not covered (no time or no page reachable): USearch and hnswlib source, Milvus/knowhere, DuckDB VSS, cuVS brute-force docs (404 at the NVIDIA URL), DiskANN direct I/O, LanceDB scan internals, SQLite/PostgreSQL/RocksDB durability pages (cited from memory only).

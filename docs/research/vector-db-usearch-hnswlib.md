# USearch (with SimSIMD / NumKong) and hnswlib: what their source teaches fse's exact int8/f16/f32 search

Date: 2026-10-28. Read-only research; no benchmark was run, no repository file was changed.

Tags: **[read]** = I read the cited source lines this session. **[derived]** = my own arithmetic or reasoning from what I read, not a published result. **[bg]** = background knowledge, not re-checked. **[unverified]** = a claim whose primary source I could not confirm.

Pinned revisions (fetched from `main` on 2026-10-28; line numbers refer to these files):
- USearch `a985f77553d2d58d5ef1516eafaae81480bb3c30`
- NumKong (SimSIMD was renamed to NumKong; USearch now vendors it as submodule `numkong`, branch `v7.8.6`, `.gitmodules`) `254198d3cc1e9fbbfc194484d20cc9767dd8a676`. The old URL `ashvardanian/SimSIMD` returns no tree via the API; the code lives under `ashvardanian/NumKong`, `include/numkong/{dot,dots,spatial,spatials}/`.
- hnswlib `ca426729609b3221563047ceb269cc1213e0d376`

## 1. Summary

1. **No library gives fse a bit-exact int8 shortcut for free, but the integer building blocks are small, exact and well documented.** NumKong's `nk_dot_i8_*` kernels return an exact `int32` sum (order-independent, no rounding). Every ISA variant is a handful of lines: NEON `vdotq_s32` (SDOT), AVX-512 VNNI / AVX-VNNI `VPDPBUSD` with an XOR-0x80 bias and a `VPSADBW` correction, AVX2 `cvtepi8_epi16` + `VPMADDWD`. [read]
2. **fse's contract multiplies int8 codes with a float32 unit query in a fixed float32 order (`dot16`)**, so an integer kernel cannot replace it bit for bit: the query is a float. The only exact use of integer SIMD is a **conservative prefilter with a proven error bound, then exact rescoring of survivors** with the existing kernel (details in section 6). Neither USearch nor hnswlib does this (their exact paths are exact only in the sense of "scan everything"; scores are ISA-dependent floats/ints). [derived]
3. **Overflow is a non-issue for int8 x int8 at d <= 4096**: |sum| <= 127*127*4096 = 66,064,384 (about 3% of 2^31) [derived]. The real traps are instruction choice, not the sum: `VPMADDUBSW` saturates in int16 (do not use it with a biased uint8 operand), `VPDPBUSDS` saturates in int32 (use the non-S form), and `VPDPBUSD` needs one unsigned operand (hence the bias trick). Details in section 4.
4. **Where USearch is weaker than fse:** its exact path (`exact_search_t`) materialises the full dataset x queries distance matrix, transposes it single-threaded, then `std::partial_sort`s each query (`index_plugins.hpp:3325-3415`). hnswlib's `BruteforceSearch` is single-threaded per query with a `priority_queue` and a `lastdist` floor (`bruteforce.h:113-133`), the same idea as fse's running floor. fse's block/floor/order-key design is ahead of both on top-k; there is nothing to copy there. [read]
5. **What is worth copying** (ranked in section 8): (a) int8 SDOT/VNNI prefilter with provable bound (large win on int8 stores, medium effort, stays exact); (b) NumKong's packed/tiled "one row load, several queries" structure for the <= 8-query kernel (register blocking: 1 row x 4 queries, the `nk_dot_i8x16_update/finalize` state API); (c) stored squared norms at pack time, which fse already does (`inv_norms`); (d) on Apple M4/M5-class and Sapphire Rapids hardware, matrix engines (SME `SMOPA`, AMX `TDPBSSD`) give exact int32 GEMM at 5x or more over SDOT/VNNI (published, section 3), but the M1 Pro in use has neither [bg].
6. **Published speedups are thin and mostly vendor-reported.** The only exact-search speed claim in USearch's README (20x vs `faiss.IndexFlatL2`, 55.3 ms vs 2.54 ms) comes from a user's comment on 500 vectors where FAISS includes index construction (GitHub issue 176); it says nothing about steady-state brute force. NumKong's own tables (Sapphire Rapids, Apple M5) are the best numbers and are quoted below with their units.

## 2. USearch exact search, read from source

### 2.1 Entry points

- Python `usearch.index.search(dataset, queries, count, metric, exact=True, threads=...)` calls `_exact_search` (`python/usearch/index.py:1571-1650`) = C++ `search_many_brute_force` (`python/lib.cpp:537-630`). Requires 2-D, row-contiguous buffers, same dtype for both, dimensions equal; `threads=0` means `hardware_concurrency`; the GIL is released around the search. [read]
- `Index.search(..., exact=True)` goes to `index_gt::search` which sets `config.exact` and calls `search_exact_` (`include/usearch/index.hpp:3498-3501, 4796-4812`): a plain loop over all slots, `context.measure(query, slot, metric)`, then `top.insert(candidate_t{distance, slot}, count)`. One query per call; queries are parallelised by the caller (`python/lib.cpp:313-345`, thread per query). [read]
- `usearch.eval` does not contain its own exact kernel; it only builds ground-truth with `search(..., exact=True)` for recall measurement (grep of `python/usearch/eval.py`; I did not read it end to end). [partly read]

### 2.2 The `exact_search_t` three-step procedure (`index_plugins.hpp:3325-3415`)

- Step 1: `executor.dynamic(dataset_count, ...)`: each task is one dataset row and loops over all queries, one `metric(dataset, query)` call per pair, writing `{offset, distance}` into a `dataset*queries` array. Good: the row is read once for all queries. Bad: the scalar metric call per pair (function-pointer call, horizontal reduction per pair) and one atomic increment per row.
- Step 2: a **single-threaded transpose** of the full `N x Q` array of 8-byte `{u32 offset, f32 distance}` (comment: "to avoid contention"). Memory is `2 * N * Q * 8` bytes: for 859,346 rows and 8 queries about 110 MB, for 64 queries about 880 MB. [derived]
- Step 3: `std::partial_sort` (or `min_element` for k=1) per query over all N entries: O(N log k) per query, no running floor.
- The header comment itself says "Perfect if you have <1M vectors and <100 queries per call" (`:3320`). [read]
- Ties: `smaller_distance` compares only `distance` (`:3327`), so ties are in unspecified order (`partial_sort` is not stable). Not exact in fse's sense.

fse relation: nothing to adopt; this is the baseline fse's block + floor design already beats structurally (no N x Q matrix, no serial phase). Useful as a "what not to do" and as the reason USearch's exact numbers do not transfer.

### 2.3 Top-k structure in the graph path (`index.hpp:1047-1146`)

`sorted_buffer_gt`: an array kept sorted, `insert` does `std::lower_bound` (binary search) then a `memmove`-style shift; if the new element lands at slot == limit it is rejected. Comment: "Ideal for small collections under 128 elements" (`:1040-1046`). The first reject test is a binary search, not a one-compare floor test. fse's per-query floor (one compare against `from_key(floor)` before any structure is touched) is cheaper. [read]

### 2.4 Quantization to int8 and how cosine is computed (`index_plugins.hpp`)

- `cast_to_i8_gt` (`:1963-1986`): computes the float64 L2 norm of the vector, then `int8 = clamp(x * 127.0 / norm, -127, 127)` (truncating cast, not rounding). So USearch int8 = unit-normalise then scale to [-127,127]; one global scale (127), no per-row scale. Zero or NaN norm gives all zeros. README (`:274-276`): "only valid for cosine-like metrics". Differences from fse: fse's per-row scale `s = absmax/127` with `rint` is more accurate per row and keeps direction; USearch's global factor loses precision for rows with outliers. [read]
- Scalar fallback `metric_cos_i8_t` (`:2555-2580`): computes `ab`, `a2`, `b2` as int32 sums of int16 products in one `omp simd`/auto-vectorised loop, then `1 - ab / (sqrt(a2) * sqrt(b2))` in float32, clamps negatives to 0; `a2==0&&b2==0` gives 0, `ab==0` gives 1. **Norms are recomputed per pair** (three dot products), not stored. [read]
- Fast path: `configure_with_numkong` (`:3083-3132`) maps `cos_k` to `nk_kernel_angular_k` and picks the best kernel at runtime through `nk_find_kernel_punned(kind, dtype, caps)`; the result type switch (`:3105-3120`) is f64/f32/i32/u32 per kernel. [read]
- NumKong `nk_angular_i8_neonsdot` (`spatial/neonsdot.h:81-200`): three `vdotq_s32` per 16 bytes (`ab`, `a2`, `b2`), horizontal `vaddvq_s32` each, scalar tail, then `nk_angular_normalize_f32_neon_`. The source comments list two alternatives that were tried and rejected: a 64-bit `vdot` fallback and an **i8mm `vmmlaq_s32` variant "elegant, but requires `+i8mm` and is currently slower, at least on AWS Graviton 3"** (`:117-183`). [read] Useful data point: SMMLA is not an automatic win for a single pair.
- NumKong's batch form (`dots`/`spatials`, `dots/README.md:91-99`, `dots/neonsdot.h:34-57`) **precomputes norms at pack time** (`nk_dots_reduce_sumsq_i8_`, `norm_value_type=u32`) and applies a 4-wide finalizer `1 - dot / sqrt(qnorm * tnorm)` (`dots/README.md:101-123`). This is the same idea as fse's stored `inv_norms`; fse stores the *inverse* norm in float32 and multiplies, which is cheaper (no sqrt, no divide in the scan).

## 3. NumKong (ex SimSIMD) kernels, read from source

### 3.1 int8 dot, per ISA

| ISA | Function | Mechanism | Lines |
|---|---|---|---|
| NEON dotprod (Apple M1+, Graviton 2+) | `nk_dot_i8_neonsdot` | one `int32x4` accumulator, `vdotq_s32(acc, a, b)` per 16 bytes, `vaddvq_s32` at the end, scalar tail | `dot/neonsdot.h:116-129` |
| NEON, register-blocked state API | `nk_dot_i8x16_state_neonsdot_t` + `update`/`finalize` for 4 targets | one query vector loaded once, 4 targets, 4 independent accumulators, `finalize` reduces 4 states at once into an `int32x4` | `dot/neonsdot.h:45-70, 145-170` |
| AVX-512 VNNI (Ice Lake+) | `nk_dot_i8_icelake` | `a' = a ^ 0x80` (unsigned a+128), `acc = _mm512_dpbusd_epi32(acc, a', b)`; separately `sum(b+128)` via `_mm512_sad_epu8` into 64-bit lanes; result = `acc - 128*sum(b)` with a count_rounded correction; masked loads for the tail (padding is zero for b and 0x80 for a'; the correction uses the rounded count) | `dot/icelake.h:99-150` |
| AVX-VNNI (Alder Lake, 256-bit) | `nk_dot_i8_alder` | same algebra with `_mm256_dpbusd_avx_epi32` and `_mm256_sad_epu8`, 32 elements per step | `dot/alder.h:108-165` |
| AVX2 (Haswell) | `nk_dot_i8_haswell` | two 128-bit loads (to avoid port-5 `vextracti128`), `_mm256_cvtepi8_epi16`, `_mm256_madd_epi16` into two int32 accumulators, then add | `dot/haswell.h:1298-1322` |
| SME (Apple M4/M5, Arm v9.2) | `dots/sme.h` | `svmopa_za32_s8_m` outer-product accumulate into ZA tiles, "i8 -> i32 peak ~2 TOPS per core" (header comment) | `dots/sme.h:10-46` |
| Sapphire Rapids AMX | `nk_dots_*_sapphireamx` | `TDPBSSD`, 16x16 outputs, 16,384 int8 MACs per instruction, quad-interleaved packed data, Morton-order tiles | `dots/README.md:64-76` |

The "algebraic domain shifting" paragraph (`dot/README.md:91-98`, `dots/README.md:91-99`) is the best short explanation of the VNNI sign problem: `VPDPBUSD` is u8 x s8, so one operand is shifted by +128 and the bias `128 * sum(other operand)` is subtracted; in the batched (`dots`) kernels **the column sums of the packed side are computed once at pack time and stored, so the inner loop only carries the `DPBUSD` accumulator** (`dots/README.md:97-98`). For fse the stored (row) side is the pre-packable one: the per-row `sum(codes)` could be stored beside `inv_norms` (4 bytes per row) and the query side shifted, so the scan needs no `VPSADBW` at all. [derived from the NumKong description]

### 3.2 f16 / bf16 kernels (accumulation and accuracy)

- f16: `nk_dot_f16_neon` widens to f32 and FMAs; `nk_dot_f16_neonfhm` uses `FMLAL/FMLSL` (fused f16 -> f32 multiply-accumulate); x86 `haswell` uses F16C `VCVTPH2PS` + FMA; Sapphire has native `VFMADD_PH` with "flushes to Float32 every 128 elements to avoid overflow" (`dot/README.md:88-89, 127`). bf16: `neonbfdot` (`BFDOT`), `genoa` (`VDPBF16PS`). Accumulators are float32. [read]
- Accuracy is **not** bit-exact: the README tables report mean ULP error against a higher-precision reference, e.g. on Apple M5 `nk_dot_f16_neon` 33.4 / 37.4 / 23.1 ULP and `nk_dot_f16_neonfhm` 14.9 / 26.7 / 39.9 ULP for 256 / 1024 / 4096 elements; `nk_dot_f32_neon` and `nk_dot_f64_neon` report 0 ULP because of compensated or widened accumulation (`dot/README.md:76-83` for the Dot2 / TwoProd technique; table rows at the M5 section from line 309). The results also differ across ISAs by construction. This is the opposite of fse's contract (fixed order, bit-identical on all ISAs), so these kernels are speed references only. [read]
- f16 relation to fse: fse already widens f16 through a 64K table (`fse_half_table`, `_native_kernel_search.h:22`); NumKong shows the hardware alternatives (`VCVTPH2PS`, `FCVTL`/`FMLAL`). Hardware convert would be exact too (f16 -> f32 is exact) and removes a 128 KiB table lookup per element; rough effort small-medium; gain unmeasured. [derived]

### 3.3 Published numbers (all from NumKong READMEs, `main` at the pinned commit)

Unit: `gb/s` = "number of bytes read per second amortized for a large batch of vector pairs" (`dot/README.md:141`); the three columns are vector lengths 256 / 1024 / 4096 elements. Vendor-reported, single machine each, not independently reproduced. [read]

- Apple M5, `nk_dot` (`dot/README.md`, M5 section): f32 `neon` 46.4 / 38.0 / 34.8 gb/s; i8 `neonsdot` 92.8 / 87.4 / 59.9 gb/s; i8 `serial` (compiler auto-vectorised) 115 / 102 / 92.3 gb/s. **The auto-vectorised serial i8 loop beats the hand-written SDOT loop on this machine in the single-pair test**; the likely reason is the single accumulator (latency-bound SDOT chain), but that is my guess [unverified]. Lesson for fse: use several independent accumulators or a multi-query tile, and measure against what the compiler already produces.
- Intel Sapphire Rapids, `nk_dot_i8` (`dot/README.md:222-227`): serial 16.9 / 16.8 / 15.6; haswell 43.2 / 35.8 / 29.1; skylake 52.9 / 36.5 / 28.5; icelake (VNNI) 64.0 / 46.2 / 26.8; alder 42.8 / 40.4 / 31.1 gb/s. At 4096 elements all of them fall to cache/memory limits and VNNI is no better than Skylake-X.
- Batched GEMM, `dots_packed_i8` (`dots/README.md`): Apple M5: serial 18.9 / 20.0 / 20.2 gso/s, **`neonsdot` 345 / 419 / 477 gso/s, `sme` 2,348 / 2,687 / 2,570 gso/s** (lines 392-398). Sapphire Rapids: haswell 87.7 / 104 / 108, icelake 191 / 326 / 410, alder 180 / 229 / 270, **AMX 547 / 1,610 / 1,300 gso/s**, MKL `i8u8` baseline 250 / 627 / 1,670 gso/s (lines 224-235). gso/s = giga scalar operations per second, `2*M*N*K`, for square M=N=K = 256 / 1024 / 4096 matrix products (`dots/README.md:125-135`), so these are GEMM-shaped (many queries x many rows), not the 1-8 query case.
- `nk_angulars_packed_i8_*` (`spatials/README.md:238-254`) has the same shape as `dots` with the finalizer; 0 ULP vs the reference on all listed ISAs for i8 (the integer part is exact; the final float normalisation is one sqrt/divide).
- USearch README exact-search claim (`README.md:188-192`): "up to 20x" vs `faiss.IndexFlatL2` in Google Colab, 55.3 ms vs 2.54 ms, linked to GitHub issue 176. I read the thread: the 20x is a user's own report of 500 vectors x 1024 d (`l2sq`, "index has to be recreated every time"), so FAISS's number includes building the index per call; USearch's maintainer answered on an M2 MacBook. **Do not cite this as a brute-force kernel speedup.** [read, conclusion mine]
- USearch BENCHMARKS.md contains HNSW build/search numbers and `i8` rows, but no exact-search figures I could tie to the kernels (grep of `BENCHMARKS.md`); `python/scripts/bench_exact.py` exists to compare `usearch`, `faiss.knn` and `nk.cdist + np.argpartition` (the latter is a dense distance matrix + argpartition, not a streaming top-k). [read]

## 4. Overflow bounds for int8 x int8 accumulation (derived)

All numbers are exact integer arithmetic on the cited formulas; the instruction semantics are from the NumKong comments and the Intel/Arm manuals [bg].

| Quantity | Bound | d = 4096 | Verdict |
|---|---|---|---|
| plain int32 sum, codes in [-127,127] | 127*127*d = 16,129 d | 66,064,384 | 3.1% of 2^31 |
| plain int32 sum, full [-128,127] | 128*128*d = 16,384 d | 67,108,864 (= 2^26) | fine |
| largest d that fits int32 | 2^31 / 16,384 | 131,072 (full range); 133,143 for +-127 | far above 4096 |
| biased `VPDPBUSD` accumulator, (a+128) in [0,255] times b in [-128,127] | 255*128*d = 32,640 d | 133,693,440 | fine; max d 65,794 |
| `VPSADBW` accumulator of (b+128), 64-bit lanes | 255 d | 1,044,480 | cannot overflow |
| correction `128 * sum(b)` | 128*127*d | 66,584,576 | fine |
| `VPMADDWD` pair sum (AVX2 path, int16 x int16 -> int32) | 2*128*128 = 32,768 per lane-step | | never overflows int32 |
| `VPMADDUBSW` (u8 x s8 pair sum, saturating to int16) | up to 255*127*2 = 64,770 > 32,767 | | **saturates; wrong for a biased operand**; safe only if the signed operand has |b| <= 64 (2*255*64 = 32,640) |
| `VPDPBUSDS`, `VPDPWSSDS` (saturating accumulate) | | | must not be used |
| SDOT `vdotq_s32` / `VPDPBUSD` (non-S) | wraps, never saturates | | exact as long as the true sum fits int32 |

Notes:
- NumKong's Haswell source (`dot/haswell.h:1298-1322`, `dots/haswell.h:20` "VPMADDWD for efficient i16 pair ...") is overflow-free by using `cvtepi8_epi16` + `VPMADDWD`. The `dots/README.md:100` sentence "Haswell fallback uses VPMADDUBSW ... quantization ranges must be tighter ([-79, 79] vs [-127, 127])" does not match the Haswell code I read; treat that README line as stale or about a different path [unverified].
- Wider queries: for a **12-bit query** against int8 codes the guaranteed (any lane split) bound is 127 * Qmax * d < 2^31, i.e. Qmax < 2^31 / (127*4096) = 4,128 at d=4096, so 12-bit magnitudes (<= 4095) are overflow-safe in plain int32 accumulation; a full int16 query (32,767) is not (worst case 127*32767*4096 = 1.7e10). With 4 lanes x 2-element `VPDPWSSD`-style steps the per-lane bound is lower but depends on the instruction; stay on the global bound. [derived]
- On Arm there is no `i8 x i16` dot instruction; a 12-bit query needs `SMLAL` widening (about 4x slower per element than SDOT) [bg]. An int8 query keeps SDOT/VNNI but has a loose error bound (section 6).

## 5. hnswlib, read from source

### 5.1 Distance kernels (`hnswlib/space_ip.h`, `space_l2.h`)

- Selection (`InnerProductSpace` ctor, `space_ip.h:342-380`; `L2Space` `space_l2.h:209-236`): `dim % 16 == 0` -> `...SIMD16Ext`; `dim % 4 == 0` -> `...SIMD4Ext`; `dim > 16` -> `...SIMD16ExtResiduals`; `dim > 4` -> `...SIMD4ExtResiduals`; else the scalar loop. Variants chosen at compile time by `USE_SSE/USE_AVX/USE_AVX512` (set from `__AVX__`/`__AVX512F__` in `hnswlib.h:20-23`) plus a runtime `AVX512Capable()`/`AVXCapable()` cpuid check (`hnswlib.h:53-80`). [read]
- **Residual handling** (`space_ip.h:313-338`, `space_l2.h:149-204`): round `dim` down to a multiple of 16 (`qty >> 4 << 4`), call the SIMD kernel on that prefix, run the scalar `InnerProduct`/`L2Sqr` on the tail, **add the two partial results**. Cheap, but it fixes a summation order that depends on `dim % 16` and on which kernel was selected. Contrast with NumKong, which uses masked loads (`_bzhi_u64` mask on AVX-512, `svwhilelt` on SVE) so the tail is one more vector step (`README.md` of USearch `:74`, `dot/icelake.h:122-130`). [read]
- AVX IP (`space_ip.h:24-69`): 2 loads x 2 per iteration but **one accumulator** `sum256 = add(sum256, mul(v1, v2))` (no FMA), a dependency chain of 2 adds per 16 floats, then a 128-bit tail and a horizontal add through memory. AVX-512 IP (`:144-200`): unrolled 4x with `_mm512_fmadd_ps` into **one** accumulator, `_mm512_reduce_add_ps`. AVX-512 L2 (`space_l2.h:24-60`) deliberately uses `add(mul)` with the FMA line commented out (`:51`). Single-accumulator chains are latency-bound; NumKong's tile design (4 independent accumulators) is the fix, and fse's `dot16` (16 lanes) is already a wide-accumulator design. [read]
- Cross-ISA reproducibility: SSE, AVX and AVX-512 variants sum in different orders, and some use FMA and some do not, so **hnswlib results differ in the last bits by machine**. fse's contract is the opposite and should keep it. [read, conclusion mine]
- Integer space: only `L2SpaceI` for `unsigned char`, with an unrolled-by-4 scalar loop (`space_l2.h:255-320`). There is no int8 inner product or cosine, no f16/bf16 space. [read]

### 5.2 `BruteforceSearch` (`hnswlib/bruteforce.h`)

- Layout: one malloc'd array of `[vector bytes][label]` records (`size_per_element_ = data_size_ + sizeof(labeltype)`, `:60`); labels interleaved with the vectors, an `unordered_map` from external to internal id, `removePoint` swaps with the last element. [read]
- Search (`:113-133`): sequential scan, `dist = fstdistfunc_(query, row)`, `if (dist <= lastdist || topResults.size() < k)` then (filter) `emplace` into a `std::priority_queue<pair<dist,label>>` (max-heap), pop when > k, and `lastdist = top().first`. This is exactly a floor: the compare against the current k-th best is one float compare; `<=` lets ties in. One query, one thread. fse's floor + exact order keys do the same job with an exact tie rule. [read]
- Python `BFIndex.knn_query` parallelises across **queries** with `ParallelFor` (`python_bindings/bindings.cpp:23-75, 868-900`): spawns `numThreads` `std::thread`s per call, a shared `std::atomic<size_t> current` hands out one query row at a time (`fetch_add(1)`), `join` at the end; exception from any thread is captured and rethrown, and `current = end` stops the others. With few queries the scan itself is not split. For fse (rows split across workers, one floor merge) this is the less scalable of the two schemes. [read]

### 5.3 HNSW level-0 layout and prefetch (`hnswlib/hnswalg.h`)

- Level 0 stores, per element, one contiguous record `[link count + maxM0 neighbour ids][vector][label]` (`size_links_level0_ = maxM0_*sizeof(tableint)+sizeof(linklistsizeint)`, `size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype)`, `:126-129`). Reading a node's links and its vector touches adjacent memory, and a candidate's links and data come from the same place. [read]
- Prefetch (`:287-301, 398-436`; switch `HNSWLIB_USE_PREFETCH`, default 1, `hnswlib.h:3-7`): before expanding a node it prefetches `visited_array` entries and the **first cache line** of the next neighbour's vector (`_mm_prefetch(..., _MM_HINT_T0)`) one iteration ahead (`*(data + j + 1)`), and the best candidate's data. Only the first line of the vector is prefetched; the rest is left to the hardware prefetcher. In the distance kernels the prefetch is commented out (`space_ip.h:40`, `space_l2.h:111`). The pattern exists because graph search hops randomly. [read]
- fse relation: **a sequential scan over mmapped shards does not need software prefetch** (the hardware stream prefetcher handles it); the one place it could matter is the *rescoring pass* of the prefilter idea in section 6, which gathers scattered survivor rows from the mmap. There a `__builtin_prefetch` of the next survivor row (or `madvise(WILLNEED)` on the block) is the hnswlib pattern. Published effect: none that I could find in the hnswlib repository; the PR/commit that added it is not in the files I read [unverified]. Effort: small.

## 6. How to make an int8 search fast and still bit-exact for fse

Facts from the repo [read]: the contract is `score = dot16(stored codes as float32, q) * inv_norm` (`docs/architecture/numerical-contract.md#numerical-contract-transform_version-2`; `_native_kernel_search.h:46-55`), the int8 scale cancels, the row is decoded into float scratch per row (`fse_stored_row`), and top-k ties go to the smaller row via exact integer order keys.

**Why integer SIMD cannot compute the contract score:** `q` is a float32 vector; the contract rounds every product and sum in a fixed float order. An integer sum is exact but corresponds to a quantised `q`, a different number. No order-independence trick fixes that, because the target value itself is the float32 rounding of the float sum. (If the query had only a few significant bits and a bounded exponent range, float and integer sums would coincide, but unit queries from an encoder do not.) [derived]

**What does stay exact: prefilter + exact rescore** (the pattern of every vector DB, here with a proof instead of an oversampling heuristic):

1. Per query (cost microseconds): `q ~ sq * qi` with `qi` int8 (or 12-bit), `sq` a float scale. Compute the residual `eps = ||q - sq*qi||_2` (float64).
2. Per row: `approx_dot = sq * (int32 SDOT/VNNI sum of codes x qi)`. Real-number error versus the exact dot of the stored codes with `q`: `|code . (q - sq*qi)| <= ||code||_2 * eps` (Cauchy-Schwarz). Multiply by the row's `inv_norm` (`||code||_2 * inv_norm ~ 1`, since `inv_norm = float32(1/sqrt(sum code^2))`), so the cosine error is at most about `eps` plus float rounding.
3. Add the contract's own float32 rounding error: the float dot16 differs from the real dot by at most about `gamma_n * sum|code_i q_i|` with `gamma_n ~ (n/16 + 16) * 2^-24`, plus about 1 ulp from the `* inv_norm` rounding; a safe margin of `d * 2^-23 * ||q||_1 * 127 * inv_norm` is generous. Both terms are computable per query and per row-class in float64.
4. Keep every row with `approx_cos + bound >= floor`; rescore only those with the existing exact kernel (gather the rows from the mmap, optionally prefetching); the exact floor and order keys then proceed unchanged. Because `bound` is a proven upper bound, no row that the exact path would keep is dropped: the result is identical bit for bit. If a query is nearly flat (large `eps`) the bound is loose and more rows survive, which costs speed but never correctness.
5. For an int8 query `eps` is of order `||q||_2 * sqrt(d)/(2*127)` in the worst case, i.e. about 0.04 for d=300 [derived]; because cosines of related sentences are 0.3 to 0.9 and the floor after the first few blocks is high, the filter may still reject most rows, but that depends on the data and is **unmeasured**. A 12-bit query makes `eps` about 32x smaller (section 4) at the price of non-dot-instruction widening on Arm. This is the one experiment worth running before committing.

Cost-benefit [derived, no measurement]: the current 1-query path decodes each int8 row into float32 (`fse_stored_row`) and runs a 16-lane float dot; an SDOT pass reads 1 byte per element and does 16 MACs per instruction, so the prefilter pass is bandwidth-bound while the exact pass runs on a small fraction of rows. NumKong's figure of roughly 8x more elements per second for i8 SDOT than for f32 NEON on M5 (92.8 vs 46.4 gb/s, bytes basis, `dot/README.md` M5 table) is the nearest published datum, but it is a single-pair cache-resident microbenchmark. Run the paired 1-query and 8-query measurement from AGENTS.md before and after; per the maintainer's rules a PR that touches the kernels also needs the end-to-end float32 throughput.

Not applicable or dead ends for exactness: USearch/NumKong i8 cosine (float result from `sqrt` of integer norms, different rounding than the contract); NumKong f16/bf16 dots (non-zero ULP, ISA-dependent order); `nk.cdist` + `np.argpartition` (dense matrix, unstable ties).

## 7. Apply-to-fse table

| # | Trick | Where in source | Published effect | Exact for fse? | fse files | Effort |
|---|---|---|---|---|---|---|
| 1 | int8 SDOT / VNNI prefilter with proven error bound, then exact rescore of survivors | NumKong `dot/neonsdot.h:116-129`, `dot/icelake.h:99-150`, `dot/alder.h:108-165` | i8 SDOT 92.8 gb/s vs f32 NEON 46.4 gb/s (M5, 256 el., single pair, vendor); none for the bound itself | yes, by construction (section 6) | new `fse_cosines_prefilter` in `src/fse/_native_kernel_search.h`, hook in `_native.pyx`, `_backends/_cpu.py`, `_store/_search.py` (`best`), conformance meta-tests | large; gain unmeasured; needs per-ISA code (NEON, AVX2/VNNI) |
| 2 | Register blocking: one row load, 4 queries, independent accumulators | `dot/neonsdot.h:45-70, 145-170`; `dots/README.md:57-62, 101-123` | GEMM-shape i8 SDOT 345 to 477 gso/s vs 18.9 to 20.2 serial (M5); not directly comparable to 1-8 queries | applies to the integer prefilter (exact int sums); for the float `dot16` kernel it would have to keep each query's own 16-lane order, which it can (independent accumulators per query) | `fse_cosines_body` loop `for j` over queries | medium; free for float if the per-query accumulation order is unchanged |
| 3 | Store `sum(codes)` per row next to `inv_norms` so the VNNI bias correction needs no `VPSADBW` | `dots/README.md:97-98` | none published as an isolated effect | yes (integer) | `_store/_shard.py`, store format: **new tensor name, new `fse.` metadata, version bump**; docs/architecture/persistence.md, "Persistence names replace class paths" | small code, but a format change; not needed on NEON (signed SDOT) |
| 4 | Floor test inside the kernel (compare then write triple), no N x Q float block | hnswlib `bruteforce.h:119-125`, ScaNN (earlier survey) | none published for hnswlib | yes | `_native_kernel_search.h` + `_search.py` | medium (already in the earlier survey) |
| 5 | Masked/partial vector for the tail instead of prefix SIMD + scalar tail | `dot/icelake.h:122-145` vs hnswlib `space_ip.h:313-338` | none published | n/a for float contract (order is fixed by `dot16`); yes for integer prefilter (order-free) | only in the new integer kernel | small, comes with #1 |
| 6 | Hardware f16 -> f32 convert instead of 64K table | NumKong `dot/README.md:127`, `dot/haswell.h` (F16C) | none isolated; f16 neon 35.7 gb/s vs f32 neon 46.4 gb/s (M5, 256 el.) | yes (f16 to f32 is exact) | `fse_stored_row` in `_native_kernel_search.h`, and the `fse_half_table` users | small-medium; table lookup may already be fine, profile first |
| 7 | SME `SMOPA` / AMX `TDPBSSD` exact int32 GEMM for many-query int8 prefilter | `dots/sme.h:10-46`, `dots/README.md:64-90` | M5 `sme` 2,348 to 2,687 gso/s vs `neonsdot` 345 to 477; SPR AMX up to 1,610 vs icelake 410 (both vendor) | yes (int32) | new backend feature; needs Apple M4+/Sapphire Rapids hardware | large, hardware-gated; not on M1 Pro [bg] |
| 8 | Prefetch next survivor row in the rescore pass | hnswlib `hnswalg.h:287-301, 398-436` | none published | yes | rescore loop in the new prefilter path | small |
| 9 | Per-pair `exact_search_t`, serial transpose, `partial_sort` | USearch `index_plugins.hpp:3325-3415` | USearch's own comment: <1M vectors, <100 queries | no (ties unspecified) | not to be adopted | none |
| 10 | `ParallelFor` (thread-per-call, atomic row counter, exception capture) | hnswlib `bindings.cpp:23-75` | none | n/a | fse's worker threads are already blocked | none |

## 8. Ranked picks

1. **Integer SDOT/VNNI prefilter with the Cauchy-Schwarz bound (row 1), starting with an experiment**: write the NEON SDOT variant only (M1 Pro has `dotprod`), measure how many rows survive on the real 859,346-row store with int8 and 12-bit queries, and time the 1-query and 8-query paths. Kills the idea cheaply if survivors exceed a few percent.
2. **Multi-query register blocking (row 2)**: the float kernel already reads each decoded row once for all queries; an explicit 4-accumulator tile is the NumKong structure and is order-neutral per query.
3. **Fused floor test (row 4)**, already in the first survey; hnswlib's `lastdist` check confirms the idea, adds no new numbers.
4. Hardware f16 convert (row 6), after profiling `fse_half_table`.
5. Sum-of-codes tensor (row 3): only if an AVX-512 VNNI target matters; it needs a format change.
6. SME/AMX (row 7): note for the future; hardware-gated.

## 9. Sources (all fetched 2026-10-28)

USearch (`a985f77`): https://github.com/unum-cloud/usearch/blob/a985f77553d2d58d5ef1516eafaae81480bb3c30/
- `include/usearch/index_plugins.hpp` lines 1963-1986 (`cast_to_i8_gt`), 2555-2580 (`metric_cos_i8_t`), 3083-3132 (`configure_with_numkong`), 3310-3415 (`exact_search_t`)
- `include/usearch/index.hpp` lines 1040-1146 (`sorted_buffer_gt`), 3498-3501, 4796-4812 (`search_exact_`)
- `python/lib.cpp` lines 313-345, 537-630 (`search_many_brute_force`); `python/usearch/index.py` lines 1571-1650
- `README.md` lines 170-192 (exact search, "20x"), 270-276 (i8 quantization); `.gitmodules` (NumKong submodule); `python/scripts/bench_exact.py`
- Issue 176 and comments: https://github.com/unum-cloud/usearch/issues/176

NumKong (`254198d`): https://github.com/ashvardanian/NumKong/blob/254198d3cc1e9fbbfc194484d20cc9767dd8a676/include/numkong/
- `dot/neonsdot.h` 45-70, 116-129, 145-170; `dot/icelake.h` 99-150; `dot/alder.h` 108-165; `dot/haswell.h` 1298-1322
- `dot/README.md` 74-137 (optimizations, units), 222-227 (Sapphire Rapids i8), M5 section from 309 (i8 at 367-369)
- `dots/README.md` 47-123 (packing, tiles, AMX, SME, compensated integer GEMM, finalizer), 224-235 and 392-398 (tables)
- `dots/neonsdot.h`, `dots/haswell.h`, `dots/sme.h` 10-46
- `spatial/neonsdot.h` 81-200; `spatials/README.md` 238-254

hnswlib (`ca42672`): https://github.com/nmslib/hnswlib/blob/ca426729609b3221563047ceb269cc1213e0d376/
- `hnswlib/space_ip.h` 24-69, 144-200, 306-380; `hnswlib/space_l2.h` 22-60, 146-250, 255-320; `hnswlib/hnswlib.h` 3-80; `hnswlib/bruteforce.h` 60, 113-133; `hnswlib/hnswalg.h` 126-129, 287-301, 398-436; `python_bindings/bindings.cpp` 23-75, 868-900

Not read, so no claims made: USearch `include/usearch/index_dense.hpp` beyond the `exact` flag plumbing, `python/usearch/eval.py` beyond a grep, the NumKong C dispatch layer (`c/dispatch_*.c`), Cython/C of FAISS, and any Apple i8mm/SME hardware documentation (hardware availability statements are [bg]).

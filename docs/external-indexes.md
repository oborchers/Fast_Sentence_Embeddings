# External indexes on fse shards

fse builds no approximate nearest neighbor (ANN) index of its own. A vector store is a set of immutable shards, and this page shows how to put FAISS, USearch or hnswlib on them, how to turn their hits into exact fse scores, and how to keep the index current. The runnable version is [recipe 10](../recipes/10-external-ann-indexes.ipynb); its helper functions are in `recipes/_ann.py` and are not part of fse.

An external index finds likely rows quickly, with rounded scores. `VectorStore.rescore` then gives the rows it found the exact score and order of `search(mode="exact")`. You need no second database process, and the vectors stay on the disk of the store.

## What fse provides

- `VectorStore.iter_batches(start=, stop=, block_rows=)` yields `fse.VectorBatch` blocks. Their `vectors` are read-only views of the mapped shards in the stored dtype (int8 as `QuantizedRows`), C-contiguous, and they stay readable after `VectorStore.close`.
- `VectorBatch.inv_norms` holds the stored inverse norms of the rows. `VectorBatch.normalized()` returns a new float32 block with rows of approximately length one: `float32(stored) * inv_norm` per element, and for int8 the codes without their scale. The bits are fixed in the default IEEE environment only ([Reading rows](architecture/store.md)).
- `VectorStore.rescore(queries, candidates, /, k=10, *, workers=None, max_per_doc=None)` scores every query against its own candidate positions only and returns the best `min(k, |C|)` rows, bit for bit as `search(query, k, mode="exact", rows=sorted(C))`. `-1` marks a missing candidate. It takes the original float32 queries, never the index's rounded ones.
- `VectorStore.take_vectors` reads rows by position (for example a training sample), and `locate` names the member store of a position in a view.
- `VectorStore.refresh()` picks up shards a writer published, and `repair_store(path)` keeps the verified prefix of a damaged store.

## How each library takes the rows

| | FAISS (CPU) | USearch | hnswlib |
|---|---|---|---|
| Accepted dtypes | float32 only | f32, f16, i8 and others | float32 only |
| Copies the rows | Always, into its own memory | Not with `copy=False` and the same dtype | Always, into its own memory |
| Cosine | No such metric: normalized rows and `METRIC_INNER_PRODUCT` | `metric="cos"` on the raw rows | Normalized rows with `space="ip"`, or `space="cosine"` |
| What to pass | `batch.normalized()` | `batch.vectors` (float32, float16) or `batch.vectors.codes` (int8), with `dtype="f32"`, `"f16"` or `"i8"` | `batch.normalized()` |
| Keys | int64 (`add_with_ids`; HNSW and flat need `IndexIDMap`) | uint64 | `size_t` |
| Persistence | Own file; `IO_FLAG_MMAP_IFC` maps only FAISS files | `save` writes the vectors again; `view` maps a file | Loads the whole file into memory |
| Fit for 1e8 rows | `IVF…_HNSW32,SQ8` or `IVF…,PQ`, trained on a sample | `i8` index, graph in memory, vectors in the page cache | Up to the RAM limit only |

Sources: the FAISS [index guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) and [metric notes](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances), the sources of FAISS v1.15.1, USearch v2.26.6 and hnswlib v0.8.0 (`class_wrappers.py`, `index.py`, `index_dense.hpp`, `bindings.cpp`, `hnswalg.h`). fse reads no other index format and never reads shard files directly, so that the checks at `open` are not bypassed.

For an int8 store the cosine of the codes equals the fse cosine, because the scale cancels out of the inverse norm ([numerical contract](architecture/numerical-contract.md), "Reading stored rows"). That is why an int8 index needs no scales.

## Quantized queries in USearch

An `i8` USearch index also quantizes the query to int8 (`cast_to_i8_gt` in `index_dense.hpp`), and it computes distances in its own order. Its ranking therefore differs from the fse ranking by more than rounding. `rescore` always receives the original float32 query. The recall you measure after `rescore` includes this loss.

## Keys and the way back

In a single store the position is the key. It is stable, because shards are immutable and a store only grows. Document IDs are not unique, so they make no key; read them with `take_doc_ids` or from a `DocumentSearchResult`.

USearch and hnswlib return unsigned keys. USearch fills missing hits with a library value and reports the number of valid hits in `counts`; hnswlib returns exactly `k` hits or raises an exception. `rescore` refuses unsigned arrays, because a key such as `2**64 - 1` would turn into `-1` and vanish as a gap. Convert them yourself, as the recipe does (`candidates_from_unsigned`):

1. Find the valid hits (USearch: `counts`; hnswlib: all `k` of them, or the call raised).
2. Check that every valid key is below `2**63` and below `len(store)`; raise otherwise.
3. Convert to int64.
4. Set every other place to `-1` explicitly. Never rely on the library's filler.

Positions of a view (`open_view`) are not a stable key. They shift when a missing part of a `source_rows@1` source arrives, when an earlier member grows and the view is opened again, or when the order of the list changes. Keep **one index per member store**, keyed by the row inside the member, and convert with `view_candidates(view, path, keys, valid)`:

1. Find the member through `view.paths` (an unknown path is an error).
2. Mask the gaps.
3. Check `0 <= key < view.members[i]["rows"]` for the valid keys (raise otherwise).
4. Only then add `view.members[i]["first_row"]`.
5. Gaps stay `-1`.

Concatenate the candidates of all member indexes for each query and call `rescore` on the view. `locate` maps a position back to the member and its row.

## Replace FAISS `IndexRefine`

FAISS refines hits with `IndexRefine` and `k_factor`, whose usual form, `IndexRefineFlat`, needs a second flat index with the full vectors in memory. With fse, search the approximate index for `k_factor * k` hits, then call `store.rescore(queries, ids, k)`. The vectors stay in the store, on disk, and the final scores and order are the exact ones. `rescore` sorts and de-duplicates the candidates of each query and reads each row once, ordered by shard and position. Candidates scattered over a huge store can still touch many pages of the mapped shards, so bound the candidates per query.

## Add rows, continue, repair

Keep a small **sidecar file** next to every index. It holds the path of the store, a generation (a counter or UUID, new for every rebuild), `indexed_rows` (the last prefix that was indexed *and saved*), the dtype and dimension, and the bits of `take_vectors` at a few fixed positions (the first and last indexed row and the first row of every block). Write `indexed_rows` only after the index is saved.

- **Append.** `store.refresh()`, then `iter_batches(start=indexed_rows, block_rows=...)`. The key of a row is `batch.store_row + arange(batch.n_rows)`. hnswlib needs `resize_index` first. With USearch, each range may get an index of its own, searched together through `Indexes(paths=paths, view=True)`.
- **Continue after a stop.** Remove the keys from `indexed_rows` on, then continue from there: FAISS IVF and flat indexes (also behind `IndexIDMap`) with `remove_ids`, USearch with `remove`, hnswlib with `mark_deleted`. FAISS HNSW has no `remove_ids` (`Index::remove_ids` raises), so rebuild it, or drop the unsaved state and continue from the last saved index.
- **Repair.** After `repair_store(path)` returns `P`, the index is invalid as soon as `P < indexed_rows`. Remove the keys from `P` on (a rebuild for FAISS HNSW) and rebuild from `P`. Without that, a later append gives the same positions to other rows.
- **Binding.** fse promises no binding of an index to the content of a store. The generation and the sampled bits catch gross mix-ups, such as another store of the same length or a store that was refilled after a repair. They prove nothing about the other rows.

## Recipes for 100 million rows and more

These are plans, not executed code. The recipe notebook runs the same steps on a small store: 6,000 rows are indexed first, 1,000 are appended later, and 200 more articles serve as queries.

- **FAISS.** `IVF…_HNSW32,SQ8` or `IVF…,PQ` with a cell count `K` for your size (the FAISS guidelines list `IVF262144_HNSW32` for 10M to 100M and `IVF1048576_HNSW32` for 100M to 1B rows). Train on 30 to 256 rows per cell, read with `take_vectors` at sorted random positions and normalized. Add with `add_with_ids` from `batch.normalized()` in blocks of 16,384 rows, so no full normalized copy ever exists. Search for 50 to 200 candidates and call `rescore`.
- **USearch.** An `i8` index with `copy=False` on `batch.vectors.codes`: the graph lives in memory, the vectors in the page cache of the shards. Keep every batch alive as long as the index. A freed array leaves the index with a dangling pointer, and the program can crash instead of raising an error. On Windows the held mappings lock the files. `refresh` does not invalidate old mappings. USearch `save` writes the vectors again, so they are twice on disk (shard and index file).
- **hnswlib.** Only up to your RAM limit: every vector is copied, and loading reads the whole file.

Memory, from the cited sources and calculated, not measured: FAISS HNSW needs `d * 4 + M * 2 * 4` bytes per vector, hnswlib roughly `4 * d + M * 8` to `M * 10`. For 1e8 rows with `M = 32` that is about 128 GB at 256 dimensions and about 435 GB at 1,024 dimensions for HNSW over float32 vectors.

## Measure recall

Take a sample of queries, compute `search(queries, k, mode="exact")` and compare the ids of the index (and of `rescore`) with it: recall@k is the share of the exact top `k` found. The recipe does this with `recall_at_k`. Recall depends on the model, the data and the index settings. This page makes no speed claim; build time, index memory and latency against vector databases belong to the comparison ticket DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen).

## What fse does not do

fse has no ANN index. It gives FAISS no access without a copy (`IO_FLAG_MMAP_IFC` maps only FAISS files, and a header of its own would be a new format). It does not let you read shard files directly. `rescore` has no `rows=` or `doc_ids=` filter, and `locate` has no inverse method.

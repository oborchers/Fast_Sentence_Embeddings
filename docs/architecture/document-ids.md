# Document IDs and filtered store search

Implementation contract for DR-215 (Eigene Dokument-IDs im Store speichern (row_ids)).
Document IDs are signed int64 labels. They are not unique keys. Each matching row
remains a separate candidate, including rows sharing a document ID.

## API

`ShardSink(..., doc_ids=True)` fixes the mode when creating the store.
`Encoder.encode(texts, doc_ids=ids, sink=sink)` takes parallel finite IDs.
IDs are a list of integers, an int64 NumPy array, or a primitive Arrow int64
column without nulls. No coercion, table input, column-name selection or row pairs.
`encode_batches(..., doc_ids=True)` takes `(texts, doc_ids)` per batch in an ID-enabled sink; the argument is checked against the sink when called.
One sink spans all batches, allowing full shards across input batch boundaries.
The ordinary text-only batch API remains unchanged without document IDs.

Search accepts either `doc_ids` (document labels) or `rows` (snapshot positions).
Repeated filter values do not repeat rows. Missing document labels match nothing.
Invalid positions raise. A filtered result has `min(k, matching_rows)` entries.
`result.ids` and two-item unpacking retain their existing positional meaning.
An ID-enabled store additionally returns `result.doc_ids` with the same shape.
Tie-breaking remains by physical position, never by document label.

## Format 2

Vector-only stores without IDs or optional score tensors remain store version 1,
sentence_vectors format 1.
ID-enabled stores use `store_version=2` and `fse.format_version=2`, so 2.0.0
rejects them before applying its closed metadata schema. NpyContainer stays at 1.
Store metadata adds `fse.doc_ids=int64|none` and `fse.resume=none`.
Every shard repeats these keys and holds `doc_ids.npy`, I64, shape `(rows,)`,
exactly when enabled. Its checksum participates in the artifact fingerprint.
A store written by a pre-release of 2.1 with `fse.row_ids` is refused with
`UnsupportedArtifactError` (open, append, repair and verify) and must be written again.

The unpublished format 2 is shared with DR-199 (Lineare Köpfe zum vollwertigen
Bestandteil von fse machen). Document IDs or optional score tensors require this
same version, rather than separate format increments. Vectors, IDs and scores
share one physical shard writer and store path. Optional scores do not imply
document IDs: `doc_ids.npy` remains present exactly when the ID mode is enabled.
The score tensors and their metadata schema belong to that separate work. This
document-ID implementation neither writes score columns nor reserves their names.

ID shards also record `fse.doc_ids_min`, `fse.doc_ids_max` and
`fse.doc_ids_sorted=true|false`. Bounds are canonical signed int64 decimals.
Sorted means nondecreasing within this shard, including duplicates.
Verification recomputes all three statistics in bounded blocks.

The shared future resume mode for DR-165 (Shards und Resume für große Jobs
einführen) is `source_rows@1`, within version 2. Until implemented it is rejected
as unsupported, including during repair before any mutation. Its shard schema
adds source and execution fingerprints, source interval and partition identity:
`fse.source_fingerprint`, `fse.execution_fingerprint`, `fse.source_first_row`,
`fse.source_rows`, `fse.source_shard` (canonical JSON `[i,n]`). Source identity
includes ordered texts and document IDs. Execution identity binds the encoder,
numerical versions, output format and semantic input configuration. Scheduling
parameters do not change exact inference identity. No fictitious fingerprints
are written for ordinary non-resumable append jobs.

## Bounded search, writing and recovery

There is no secondary index, sorted shard copy or search sidecar. Min/max skips
irrelevant shards. Sorted ID columns are searched directly with `searchsorted`.
Unsorted columns are scanned in bounded blocks against the sorted filter list.
The filter list has a measured memory admission budget. It is never truncated.
Only eligible rows reach numerical scoring and top-k. Selected positions are
translated back before merging. Result document labels are read after top-k.

IDs and vectors share reservation splits, commit order and publication. The IO
completion counter advances only after both columns are written. Hashing uses
the same owned blocks as writing. Queue accounting includes ID bytes. Sealing
shrinks both columns and statistics describe only committed rows. The existing
sync and exclusive publication protocol remains in force for both columns.

Tagged shards verify IDs and their statistics on every reader open. Untagged
shards retain the existing lazy open contract. Explicit verify and offline repair
check IDs too. Repair quarantines whole shards and their tail, never just an ID
file. Unknown future formats or resume modes abort repair without mutation.

## Best hit per document (DR-218)

Implementation contract for DR-218 (Suche mit höchstens einem Treffer je Dokument anbieten).
`VectorStore.search(..., max_per_doc=1)` takes a keyword-only `PositiveInt | None`
(default `None`). Only 1 is allowed in 2.1; any other value raises a `ValueError` that says
larger values may come later. It requires a store with document IDs and raises before
reading data otherwise. The result
type stays `DocumentSearchResult`.

Semantics. For a query, take the total order of `fse._reference.top_k` over the selected
rows (the larger score first, equal scores, -0.0 equal to 0.0, by the smaller physical
position). The result is the first row of each document ID in that order, limited to the
first `k`. Documents tie by position, never by label. `fse._reference.top_k_unique` is
this definition; tests also compare against the first occurrences of a full ordinary
search. The result has `min(k, D)` entries for D distinct IDs in the selection.

Algorithm: an exact running maximum per document inside the existing block scan, as
Lucene's first-pass grouping collector keeps the best document of each group, rather
than oversampling and refetching (which has no fixed bound when one document holds
most rows). A task scores its block with the backend or BLAS, keeps the rows that reach
the floor of any query in its block, groups them (neighbouring IDs when the shard is
sorted, otherwise one stable argsort of the block's IDs), reduces to the best order key
per (group, query) with `maximum.reduceat`, and takes the smallest local row with that
key with `minimum.reduceat`. All of it is integer work on `order_key`. Groups under a
query's floor drop out, and each query keeps its first `k` groups by (key descending,
row ascending), so a block hands the merger at most `k` documents per query. The
returned score is read from the score matrix, which keeps -0.0 bits. `best_per_doc`
returns local rows; the caller translates them with `physical[local]`.

`DocumentCandidates` merges blocks and shards: per (query, document) the best row, then
the best `k` per query. The floor of a query is the key of its k-th best document, and
`LOWEST` before it has `k`. The maximum under a total order is associative, so the result
does not depend on block order or workers. Every query sees the same documents, so all
queries end with the same width; a difference is an internal error.

Memory per running task, for 2^20 scores per task and one group per row: the scores
(4 MiB), a mask, the grouped keys and indices (int32 matrices of the same size) and the
group maxima. The plan counts at most 38 MiB if nothing is freed early. With a very large
`k` (every group stays) the selected group and query indices add about 32 MiB more, so
about 44 MiB per task is the worst case. Intermediate arrays are freed in order, so the
usual peak is lower: a memray run of 256 queries (2^20 scores per task) on unordered IDs
with k=10 and two workers peaked at 41.7 MiB against 34.1 MiB for the ordinary search. The ordinary decode buffer (16 MiB) and
the gathered vectors of filtered blocks (16 MiB) are as before. The merger scales with
queries times `k`, adding one int64 column to the ordinary search.

Logging: `store.search` carries `max_per_doc=1` only on this path.

Result width rule for every search: at most `k` hits, `min(k, candidates)`, never an error
for a `k` above the number of rows (the unfiltered check `k > len(store)` was removed;
filtered searches already worked this way). An empty selection gives empty `ids` (int64),
`scores` (float32) and `doc_ids` (int64). The change needs no timing measurement (class K,
tests): it only removes a comparison before the search starts; the scan, its blocks and
its arithmetic are untouched.

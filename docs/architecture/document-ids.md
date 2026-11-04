# Document IDs and filtered store search

Implementation contract for DR-215 (Eigene Dokument-IDs im Store speichern (row_ids)).
Document IDs are signed int64 labels. They are not unique keys. Each matching row
remains a separate candidate, including rows sharing a document ID.

## API

`ShardSink(..., doc_ids=True)` fixes the mode when creating the store.
`Encoder.encode(texts, doc_ids=ids, sink=sink)` takes parallel finite IDs.
IDs are a list of integers, an int64 NumPy array, or a primitive Arrow int64
column without nulls. No coercion, table input, column-name selection or row pairs.
`encode_batches` takes `(texts, doc_ids)` per batch in an ID-enabled sink.
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
Store metadata adds `fse.row_ids=int64|none` and `fse.resume=none`.
Every shard repeats these keys and holds `row_ids.npy`, I64, shape `(rows,)`,
exactly when enabled. Its checksum participates in the artifact fingerprint.

The unpublished format 2 is shared with DR-199 (Lineare Köpfe zum vollwertigen
Bestandteil von fse machen). Document IDs or optional score tensors require this
same version, rather than separate format increments. Vectors, IDs and scores
share one physical shard writer and store path. Optional scores do not imply
document IDs: `row_ids.npy` remains present exactly when the ID mode is enabled.
The score tensors and their metadata schema belong to that separate work. This
document-ID implementation neither writes score columns nor reserves their names.

ID shards also record `fse.row_ids_min`, `fse.row_ids_max` and
`fse.row_ids_sorted=true|false`. Bounds are canonical signed int64 decimals.
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

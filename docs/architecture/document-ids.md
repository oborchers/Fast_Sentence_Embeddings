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
A sink is in ID mode only with `has_doc_ids` exactly `True`: every commit is then `commit(first_row, rows, doc_ids=ids)`, `ids` a read-only one-dimensional int64 array with one ID per committed row. Other values (0, 1, None) are ignored unless the call brings IDs, where a value that is no bool raises `TypeError`; this keeps sinks of 2.0 running. A sink of your own with `has_doc_ids` exactly `True` that gets a call without IDs also runs as in 2.0 (`commit(first_row, rows)`); only the shard sinks of fse in ID mode, which did not exist in 2.0, are refused then, before any batch is read. One validator (`fse._encoder._doc_ids`) serves `encode_batches`, `Heads.predict_batches` and the `predict_batches` of a single head: a `doc_ids` that is no bool, or a batch that is no `(texts, doc_ids)` pair, is a `TypeError`; a wrong number of IDs or a sink that does not fit the mode is a `ValueError`; every message names the method. `encode` takes the IDs themselves (`doc_ids=ids`), `encode_batches` the switch `doc_ids=True`.
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

Scores are not stored in the shards of vectors. They are a kind of their own,
`prediction_scores` ([scores.md](scores.md)), in a store of their own that shares the
physical shard writer, the document-ID column (`doc_ids.npy` exactly when `fse.doc_ids`
is `int64`) and the statistics below. Vectors and scores of one run are written to two
stores (DR-229, Vektoren und Scores in einem Lauf in zwei Stores schreiben). The two
stores share positions only while they were written together: a join by position needs
the same history, and a join by document ID needs IDs of your own that name each row
uniquely, because document IDs are not unique keys ([scores.md](scores.md), "Vectors and
scores in one run").

ID shards also record `fse.doc_ids_min`, `fse.doc_ids_max` and
`fse.doc_ids_sorted=true|false`. Bounds are canonical signed int64 decimals.
Sorted means nondecreasing within this shard, including duplicates.
Verification recomputes all three statistics in bounded blocks.

## Resume mode `source_rows@1`

`fse.resume=source_rows@1`, within version 2, marks a store written from a source that can
be read again (DR-165, Shards und Resume für große Jobs einführen; DR-248, Vektor-Stores
wiederaufnehmen und feste Partitionen schreiben). A store of the mode writes six keys into
`store.json` and into every shard, with the same values:

| Key | Value |
|---|---|
| `fse.source_fingerprint` | the fingerprint of the whole source: its bytes, its order, its document IDs |
| `fse.source_form` | `lines` (a text file), `strings` or `tokens` |
| `fse.execution_fingerprint` | the fingerprint of the root metadata without the keys above: encoder, dtype, dimension, ID mode and mode. Scheduling (workers, batches, shard size) is not part of it |
| `fse.source_first_row` | the source row where the part the store holds starts |
| `fse.source_rows` | the number of rows of the whole source |
| `fse.source_shard` | the part, canonical JSON `[i,n]` |

A shard may hold no row beyond the end of its part. The parser checks the six values, the
execution fingerprint (a mismatch is corruption) and that the first row is the start of the
part. An unknown mode such as `source_rows@2` is `UnsupportedArtifactError`, also in a later
shard or an orphan and in the preflight of repair, before anything is changed. The source
fingerprint is made of the bytes of the text (and, for strings and token lists, their
offsets) and the document IDs, so a changed byte, a moved row or another ID changes it.
No fictitious fingerprints are written for ordinary non-resumable append jobs.

## Resume mode `source_position@1`

`fse.resume=source_position@1`, within version 2 of sentence vectors and version 1 of score
stores, marks a store written from a stream that can be read again from a position, such as a
Kafka partition or a log with byte offsets (DR-165, Shards und Resume für große Jobs
einführen; DR-252, Unendliche Streams an einer wiederlesbaren Position fortsetzen). A stream
has no end and no identity to hash, so the store keeps *anchors* instead: an anchor `(P, R)`
says that the stream delivers the record that became store row `R` from the position `P`.
`P` is an opaque UTF-8 text of 1 to 65,536 bytes that the caller chooses (maintainer decision of
2026-11-06); fse stores it, compares it for equality between the roots of two stores and
hands it back, and never reads it. `R` is a canonical decimal number.

| Where | Keys |
|---|---|
| `store.json` | `fse.execution_fingerprint`, `fse.source_position` (the anchor `P_0` of the start, taken from the first batch), `fse.source_position_row` (`0`) |
| every shard | `fse.execution_fingerprint` and its own anchor, `fse.source_position` and `fse.source_position_row` |

For a shard that ends at the store row `E` (its last row is `E - 1`) the rule is `0 <= R < E`, and
the rows of neighbours do not decrease (`R_k <= R_{k+1}`). Snapshot, `verify`, `repair_store`
and `refresh` check the same chain; across the members of a view only the thresholds of score
stores are compared, because every member has a chain of its own. An unknown mode such as
`source_position@2` is `UnsupportedArtifactError`, also in a later shard or an orphan and in the
preflight of repair, before anything is changed. The execution fingerprint is the one of
`source_rows@1`: scheduling (workers, batches, shard size) is not part of it.

Which batch a shard belongs to: a sink keeps the anchors of the batch it writes and of the
batch before it. A shard that is handed over takes the anchor of the current batch if that
batch starts at or before the last row of the shard, else that of the batch before. Inside one
`encode_batches` every batch is committed before the next is read, so the last row of a shard
lies in the current batch or in the one before it. An example: batch A commits 5 rows, batch B
fails without a commit, and the sink closes with a shard of 5 rows; B starts at row 5, so the
shard takes A, with `R = 0 < 5`.

`sink.source_position` is `(P, skip)` of the last published shard (else of the root):
`skip` is the number of rows in the store after the row `R`. The caller seeks the stream to
`P` and gives its batches again; the first `skip` records are the rows the store holds, and
the run drops them without tokenizing or writing them. `VectorStore.source_position` and
`ScoreStore.source_position` give the same pair for a store that is read; `None` for stores of
other modes, for a store that is made but not yet written, and for a view, whose
`members[i]["position"]` holds the pair of each member.

The root of such a store may hold 1 MiB instead of 64 KiB (`json.dumps` writes a character
outside ASCII as up to 12 bytes). A new store is made with the first non-empty batch, after the
bytes of its root are counted with the real position; a position that does not fit is a
`ValueError` before anything is made. A `ScoreSink` made with `resume=True` checks in its
constructor that its heads and a position of 65,536 ASCII bytes fit.

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
`scores` (float32) and `doc_ids` (int64). The change needs no timing measurement (class correctness,
tests): it only removes a comparison before the search starts; the scan, its blocks and
its arithmetic are untouched.

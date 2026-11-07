# Store document IDs and filter search

Document IDs connect stored rows to your documents. They are signed int64 values.
Several chunks may share one ID. fse returns each matching row separately, unless you
ask for the best hit per document (see below).

```python
with fse.ShardSink(path, encoder, doc_ids=True) as sink:
    encoder.encode(texts, doc_ids=[42, 42, 81], sink=sink)

with fse.VectorStore.open(path) as store:
    found = store.search(query, 10, doc_ids=[42], mode="exact")
    positions, scores = found
    print(found.doc_ids)
```

`found.ids` still contains physical positions. `found.doc_ids` is an additional
array on ID-enabled stores. Equal scores still prefer the smaller position.
Repeated document IDs do not group or deduplicate the results unless you pass
`max_per_doc=1`.

ID inputs are Python integer lists, one-dimensional NumPy int64 arrays, or
primitive Arrow int64 columns without nulls. Other dtypes, nulls, bools, floats
and out-of-range integers raise. One ID is required per text. `encode(texts, doc_ids=ids, sink=sink)` takes the IDs themselves;
`doc_ids=True` is the switch of `encode_batches`, and `encode` refuses it with a `TypeError`. Keep input buffers
unchanged until processing finishes. IDs require an ID-enabled sink. Appending
with a different ID mode raises rather than changing an existing store.

## Best hit per document

With several chunks per document, the `k` best rows can all come from one document.
`max_per_doc=1` (keyword only, a store with document IDs) returns the `k` best documents
instead. Only the value 1 is allowed in this version; any other value raises a
`ValueError`, and larger values may be added later. The default `None` keeps the
ordinary search.

```python
found = store.search(query, 10, max_per_doc=1, mode="exact")
found.ids        # the position of the best chunk of each document
found.doc_ids    # ten different document IDs
```

Of every document ID only its best row counts: the largest score, and for equal scores
the smaller position. The documents follow in the order of those rows: the larger score
first, equal scores by the smaller position, never by the label. If all IDs are
different, the result equals the ordinary search bit for bit. `mode="exact"` returns the
scores and bits of the ordinary exact search for the rows it returns.

`k` counts documents. A result has `min(k, documents)` entries per query, as many as
the selection holds, without an error. This is the rule of every search: at most `k`
results, fewer if there are fewer candidates, and no error when `k` is larger than the
number of rows (earlier versions of 2.0 raised a `ValueError` for an unfiltered search).
An empty selection gives empty arrays of the usual types. `doc_ids=[...]` and `rows=[...]`
work as before and apply first: the rows are selected, then grouped by the IDs of the selected
rows. Several queries group independently and give a rectangular result.

`store.rescore(queries, candidates, k, max_per_doc=1)` groups the same way among the candidate
positions of each query (the hits of an external index) and gives, per query, the result
of `search(query, k, mode="exact", rows=sorted(candidates without -1), max_per_doc=1)` bit for bit. A
matrix result is filled up on the right with ids `-1`, scores `-inf` and document IDs `-1`,
so a gap is recognized by `ids == -1`, because `-1` can also be a real document ID.

fse does not fetch extra rows and cut them down. It keeps the best row per document in
every block of the scan, so the result is exact and the memory stays bounded. Sorted IDs
group by neighbouring values. Unordered IDs need one stable sort of the IDs of each
block, so writing IDs in nondecreasing order stays cheaper. The memory per running
task is a few times the 4 MiB of scores of a block; very many queries together with a
large `k` need memory in proportion to both, as in the ordinary search.

## Batches and Parquet

An ID-enabled `encode_batches` call takes `doc_ids=True` and `(texts, doc_ids)` for each
batch. fse checks when you call it, before it reads a batch, that the argument and the
sink agree: `doc_ids=True` needs a sink with `doc_ids=True`, and such a sink needs `doc_ids=True`.
Each tuple contains two whole columns, not one `(id, text)` row. Their lengths
must agree. One sink spans all batches, filling shards across batch boundaries.
Consume the iterator completely to flush the final partial shard.

```python
import pyarrow.parquet as pq

with pq.ParquetFile(input_path) as parquet:
    parts = (
        (batch["text"], batch["ticket_id"])
        for batch in parquet.iter_batches(columns=["text", "ticket_id"])
    )
    with fse.ShardSink(path, encoder, doc_ids=True) as sink:
        for batch in encoder.encode_batches(parts, sink=sink, doc_ids=True):
            confirmed_rows = batch.durable_rows
```

Select both columns from the same table or batch, after applying the same row
filter and ordering. fse checks lengths but cannot prove their common origin.
You may explicitly convert the ID column with `to_numpy()` instead. That producer
operation may copy a chunked column. fse does not import pyarrow, accept whole
tables, read Parquet files itself or resolve column names passed as `doc_ids`.
Without IDs, existing text-only `encode_batches` input remains unchanged.

## Reading document IDs

`store.has_doc_ids` tells whether every row has an ID. `store.get_doc_ids(start=0, stop=100)`
returns the IDs of those rows as an int64 copy, like `get_vectors`. A store without IDs
raises `ValueError`. `store.take_doc_ids(rows)` returns the IDs at a list of store positions,
in that order with duplicates kept: `store.take_doc_ids(result.ids)` has the shape of
`result.ids`, and in both search modes it equals `result.doc_ids`. `store.take_vectors(rows)`
does the same for the vectors, for example `store.take_vectors(result.ids)` of shape
`(queries, width, dim)`. `rows` is a list or an int64 array of one or two dimensions; negative
positions raise. Keep the input buffer unchanged until the call returns.
`fse.DocumentSearchResult` keeps `doc_ids` in `_replace`, `_asdict` and
`repr`; `_make` raises `TypeError`, because two values hold no IDs.

## Positions, counts and IDs

Four different things are easy to mix up. Positions are indexes that start at 0:

| Kind | Names |
| :--- | :--- |
| Store position | `SearchResult.ids`, `search(rows=)`, `VectorStore.get_vectors(start=, stop=)`, `get_doc_ids(start=, stop=)`, `take_vectors(rows)`, `take_doc_ids(rows)`, `iter_batches(start=, stop=)`, `VectorBatch.store_row`; `ScoreStore.get_scores(head, stored, start=, stop=)`, `ScoreStore.get_doc_ids(start=, stop=)`, `ScoreStore.iter_batches(start=, stop=)`, `ScoreBatch.store_row` |
| Start of a run in the store | `ShardSink.first_row`, `ScoreSink.first_row` (the store position of run position 0) |
| Run position | `first_row` of `EncodedBatch`, `PredictedBatch` and `HeadsBatch`; `first_row` of `Sink.target` and `Sink.commit` |

`store_row` is always a store position. `first_row` of a batch, a target or a commit is a run
position; only the `first_row` of a sink is the store position where its run begins.

A store position is the `first_row` of the sink (`ShardSink` or `ScoreSink`) plus the run
position. A new store has `first_row == 0`; an appended run starts after the rows already there.

Counts are lengths. `n_rows` of a batch is its length: the batch covers the run positions
`first_row` to `first_row + n_rows - 1`.

The durable prefix is a count too. `durable_rows` of `EncodedBatch`, `PredictedBatch` and
of the sink says how many rows at the start of the run are durable. `durable_rows=210`
means that the run positions 0 to 209 are durable. It is not a position.

Two stores written by one run, vectors in a `ShardSink` and scores in a `ScoreSink`
(`Heads.predict_batches(..., return_vectors=True, vector_sink=..., score_sink=..., doc_ids=True)`, or the same call of a single `Classifier` or `Regressor`),
have one store position for each run position: `vector_sink.first_row + r` and
`score_sink.first_row + r`. Both positions can differ when the stores held different numbers
of rows before the run, which a run accepts only with `doc_ids`. `HeadsBatch.durable_rows` (and `PredictedBatch.durable_rows` of a single head) is
the minimum of both sinks. A replay after a crash writes the rows beyond the confirmed ones
again, so a store can hold a row twice, and `max_per_doc=1` returns the best hit per document
without removing duplicates.

A document ID (`doc_ids`) is your own label for a document. It is not unique: several rows
may share one. It appears in the input (`doc_ids=`), in filters, in
`DocumentSearchResult.doc_ids` and in `get_doc_ids`.

## Filters and limits

Use `rows=[...]` to search specified snapshot positions, including stores without
document IDs. Use `doc_ids=[...]` for document values. Supply only one filter.
Filter lists and NumPy int64 arrays are validated before reading store data.
Repeated values act as a set. Missing document IDs and empty lists match no rows.
Invalid physical positions raise. Filtered results have `min(k, matching_rows)`
entries, including zero. Invalid queries still raise for an empty selection.

There is no secondary index or search sidecar. Shard bounds skip irrelevant data.
Nondecreasing IDs use binary search directly on the mapped column. Unordered IDs
need a blockwise ID scan. Overlapping shard ranges may require visiting every
shard. Writing document IDs in nondecreasing order is therefore recommended.

The filter admission budget is 64 MiB at 24 bytes per input entry. This admits
at most 2,796,202 entries before deduplication. Local allocation probes at that
boundary measured at most 47,536,182 additional filter-workspace bytes for ordered
and unordered ID columns, using the largest selection block of 65,536 rows.
The remaining budget provides a conservative margin.
Caller-owned inputs and numerical scoring buffers are separate. Larger filters
raise before normalization with the required bytes and budget. They are never
truncated. Split them into separate searches and merge hits by score and physical
position when necessary.

ID stores use store/format version 2 and are cleanly refused by fse 2.0.0.
Vector-only stores remain version 1. IDs, checksums and statistics are published with
their vectors under the same durability policy. Explicit verification and repair
check the ID column too. Repair quarantines whole shards, preserving the verified
prefix. A store written with `resume=True` continues after a stop; see
[resuming a run](../recipes/03-files-with-resume.ipynb).

A stream that never ends resumes at a **position**. `encode_batches(batches, sink=sink,
positions=True)` takes batches `(texts, position)`, and `predict_batches(..., positions=True)`
of `Heads`, `Classifier` and `Regressor` does the same into a vector store, a score store or
both. A position is a text of 1 to 65,536 UTF-8 bytes that you choose (a Kafka offset, a byte
offset); fse stores it with the shard in which the batch ends and never reads it. After a
crash `sink.source_position` is `(P, skip)`: seek the source to `P`, give the batches again,
and fse drops the first `skip` records, which the store holds. Three kinds of number appear
and must not be mixed up:

| Number | What it counts |
|---|---|
| store row | the rows in the store, from 0 |
| replay row | the records the source delivered since the seek, from 0 (`first_row` of each result) |
| record or offset | the position in the stream itself; offsets can have gaps |

`durable_rows` counts replay rows. Commit to the stream system the offset of the record of
replay row `durable_rows - 1`, plus one, and keep the map from replay rows to offsets
yourself. See [fse behind Kafka](../recipes/09-fse-behind-kafka.ipynb).

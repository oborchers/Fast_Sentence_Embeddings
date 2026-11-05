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
`unique_doc_ids=True`.

ID inputs are Python integer lists, one-dimensional NumPy int64 arrays, or
primitive Arrow int64 columns without nulls. Other dtypes, nulls, bools, floats
and out-of-range integers raise. One ID is required per text. Keep input buffers
unchanged until processing finishes. IDs require an ID-enabled sink. Appending
with a different ID mode raises rather than changing an existing store.

## Best hit per document

With several chunks per document, the `k` best rows can all come from one document.
`unique_doc_ids=True` (keyword only, a Python `bool`, a store with document IDs) returns
the `k` best documents instead.

```python
found = store.search(query, 10, unique_doc_ids=True, mode="exact")
found.ids        # the position of the best chunk of each document
found.doc_ids    # ten different document IDs
```

Of every document ID only its best row counts: the largest score, and for equal scores
the smaller position. The documents follow in the order of those rows: the larger score
first, equal scores by the smaller position, never by the label. If all IDs are
different, the result equals the ordinary search bit for bit. `mode="exact"` returns the
scores and bits of the ordinary exact search for the rows it returns.

`k` counts documents. A result has `min(k, documents)` entries per query, as many as
the selection holds, without an error. Without a filter, a `k` above the number of rows
still raises first, as in the ordinary search. `doc_ids=[...]` and `rows=[...]` work as
before and apply first: the rows are selected, then grouped by the IDs of the selected
rows. Several queries group independently and give a rectangular result.

fse does not fetch extra rows and cut them down. It keeps the best row per document in
every block of the scan, so the result is exact and the memory stays bounded. Sorted IDs
group by neighbouring values. Unordered IDs need one stable sort of the IDs of each
block, so writing IDs in nondecreasing order stays cheaper. The memory per running
task is a few times the 4 MiB of scores of a block; very many queries together with a
large `k` need memory in proportion to both, as in the ordinary search.

## Batches and Parquet

An ID-enabled `encode_batches` call takes `(texts, doc_ids)` for each batch.
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
        for batch in encoder.encode_batches(parts, sink=sink):
            confirmed_rows = batch.durable_rows
```

Select both columns from the same table or batch, after applying the same row
filter and ordering. fse checks lengths but cannot prove their common origin.
You may explicitly convert the ID column with `to_numpy()` instead. That producer
operation may copy a chunked column. fse does not import pyarrow, accept whole
tables, read Parquet files itself or resolve column names passed as `doc_ids`.
Without IDs, existing text-only `encode_batches` input remains unchanged.

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
prefix. This change does not yet implement automatic resume.

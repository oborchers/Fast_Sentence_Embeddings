## prediction_scores 1

Binding for score stores (`fse._store._scores_info`, `_scores_format`, `_score_shard`, `fse._scores`). DR-162 (Scores und Vorhersagen dauerhaft speichern).

A score store keeps what the heads of a group gave for every text, so that labels, new thresholds and filters need no new inference. It is a directory of shards like a store of sentence vectors ([store.md](store.md), "Shared shard base"), written by `fse.ScoreSink` and read by `fse.ScoreStore`.

### What is stored

- **One tensor `scores`**, F32 `(rows, columns)`, with the columns of all heads side by side in head order. `fse.score_heads` names the columns of each head.
- A classifier stores its **decision** (the float32 logits) or its **probabilities**; a regressor its **values**. The default is the decision, because probabilities and labels follow from float32 logits bit for bit (the links are functions of the finished logits, [numerical contract](numerical-contract.md)), and the way back does not exist. `ScoreSink(store=...)` takes `None`, one name for all heads or a mapping of head names to `decision`, `proba` or `values`.
- Optional `doc_ids` (I64, `(rows,)`) exactly when the store has document IDs, and `thresholds.<head>` (F32 `(k,)`) for each head with `multilabel@1`, the same in every shard, kept whole in a shard that ends early.
- Without document IDs the row `n` of a run is the text `n` of the run: `ScoreSink.first_row` plus the row in the run is the row of the store. No tensor and no metadata name the run. A resume across runs belongs to DR-165 (Shards und Resume für große Jobs einführen) and would add `fse.resume=source_rows@1` within this version.

### store.json and the metadata

`{"store": "fse_shards", "store_version": 2, "metadata": {...}}`, strict JSON, written once and never changed. `store_version` is 2, so `2.0.0` refuses a score store before it reads any metadata. The metadata of the root and of every shard:

| Key | Value |
|---|---|
| `fse.kind` | `prediction_scores` |
| `fse.format_version` | `1` |
| `fse.transform_version` | `2` |
| `fse.encoder` | fingerprint of the encoder of the heads |
| `fse.doc_ids` | `int64` or `none` |
| `fse.resume` | `none` |
| `fse.score_heads` | canonical JSON list of the heads in head order |

Each entry of `fse.score_heads` has exactly `name`, `model` (fingerprint of the complete head artifact), `task`, `rule`, `link`, `stored`, `path`, `labels`, `width`, `columns` (`[begin, end]` in `scores`, without a gap), `flat`. A shard adds `fse.created_by`, `fse.first_row`, `fse.rows`, for document IDs the three statistics of [document-ids.md](document-ids.md) and optionally `fse.durability=none`.

**Size.** A score store lists its heads and their labels, so its `store.json` may be up to 1 MiB; a store of sentence vectors keeps 64 KiB. `ScoreSink` serializes the document it will write and checks the limit before anything is created.

**Path.** A head on raw means runs `folded` without the vectors and `unfolded` with them, and the two differ in the last bits. The path of each head is therefore part of what is stored (`folded`, `normalized`, `unfolded`, `standalone`). It is known only when `Heads.predict_batches(return_vectors=...)` is called, so `ScoreSink(path, heads)` checks everything but the path and creates nothing; `predict_batches` attaches the run to the sink, before it reads a text, and passes the paths; a new store gets its `store.json` then, whole. Appending needs the same paths (`ValueError` naming the head and both paths, before the first text). Closing a sink that never attached leaves nothing behind.

### Compatibility table

`k` is the number of logits. Widths follow `HeadSpec.width`; the label rule follows the link.

| task | rule | link | k | `stored` | `width` | `labels` | `flat` |
|---|---|---|---|---|---|---|---|
| classification | `linear@1` | `sigmoid@1` | 1 | decision, proba | 1, 2 | 2 | null |
| classification | `linear@1` | `softmax@1` | 1 | decision, proba | 1, 2 | 2 | null |
| classification | `linear@1` | `softmax@1`, `ovr@1` | k >= 2 | decision, proba | k | k | null |
| classification | `linear@1` | `none` | k >= 1 | decision | k | 2 for k = 1, else k | null |
| classification | `linear@2` | `sigmoid@1`, `softmax@1`, `ovr@1` | as `linear@1` | decision, proba | as `linear@1` | as `linear@1` | null |
| classification | `linear@2` | `multilabel@1` | k >= 1 | decision, proba | k | k | null |
| regression | `linear@1` | `none` | k >= 1 | values | k | null | bool for k = 1, else null |

A known combination outside the table is a `CorruptArtifactError`; a well-formed unknown identifier for `rule`, `link`, `path`, `task` or `stored` is an `UnsupportedArtifactError`, also in the repair before any change.

### Values and the chain

`verify`, a tagged open and the repair check, in blocks of at most 1 MiB: the checksums; every score finite; the columns with `stored=proba` within [0, 1] (the ranges come from the root through `ScoreShard`, not from state of the kind, so two stores of different layout can be open at once); thresholds finite within [0, 1]; the statistics of the document IDs.

The only rule between shards is that two neighbours have equal thresholds (`summary` is the tuple of their checksums from the manifest). Snapshot, `verify` and `repair_store` use the same chain; the repair moves the first shard that breaks it, with its tail. `ScoreSink` compares the thresholds of the last shard of an existing store with those of its heads before it takes a text, so a run cannot break the chain itself.

### Writing

`ScoreSink` is a float32 sink of `dim` values per row (the sum of the widths) on the shared base: reservation over shards, three threads (hash, write, publish), publication by an exclusive rename, `durability="full"` or `"none"` with the tag and the refusal of a full append behind a tagged prefix. The default `shard_rows` is the largest multiple of 16,384 rows within 256 MiB of `scores` and `doc_ids` (at most 2**20).

`Heads.predict_batches(batches, score_sink=sink, doc_ids=False)`:

- It checks that the sink was made for these heads (names, order, model fingerprints, encoder) and the document-ID mode, then attaches.
- What a batch returns (`output=`, default the labels or values) and what the sink stores (`store=`) are resolved independently. The worker computes the logits of each head once. If the returned output is the stored one, that target gets a tee; else the worker derives the stored output into a scratch target of the task, which gets the tee. The tee copies the finished rows of each task into the columns of the sink matrix in the calling thread, in task order, and the last tee of a task commits the matrix. The budgets of the worker threads do not change.
- `HeadsPlan.row_bytes` counts the second outputs, the sink matrix and the IDs per row, so the window keeps its 512 MiB with a sink. The queues of the sink (16 pieces and 64 MiB per thread) and the shared buffer of a task over a shard boundary are outside the window. Without a sink the sum is the one of a group without one.
- `HeadsBatch.durable_rows` is, after each batch, the sink's `durable_rows` once `publish` ran: run-relative, monotonic, at most the rows given so far. `flush` runs after the last batch of a complete iteration, so the last batch can report an older value; the consumer reads `sink.durable_rows` for the final count. Stopping early does not flush: close the sink. With `durability="none"` the value is 0.
- `doc_ids=True` takes `(texts, doc_ids)` batches as `Encoder.encode_batches(doc_ids=True)` does (lists, int64 arrays, Arrow int64 columns without nulls); the mode must equal that of the sink.
- `Encoder`, `Classifier` and `Regressor` refuse a `ScoreSink` before they read anything; only a group writes it.

### Reading

`ScoreStore.open(path, heads=None, backend="cpu")` maps the shards published at that moment (a snapshot, with the chain), verifying tagged shards. `heads=` requires the same names, order and model fingerprints. `len`, `heads` (name to `ScoreHead`: stored output, path, rule, link, labels, columns), `encoder_fingerprint` and `has_doc_ids` describe the store; `get_scores(head, stored)` and `get_doc_ids` return copies (`stored` must be what the head stored); `verify` hashes everything.

- **`iter_batches(block_rows=65536, start=0, stop=None)`** gives `fse.ScoreBatch(store_row, n_rows, outputs, doc_ids)` with the stored values as **read-only views** of the mapping, at most `block_rows` rows, never over the end of a shard. A head of several columns is not contiguous in its view; copy it if a tool needs that. Views stay readable after `close`; a closed store gives nothing more. A flat head gives `(n,)`.
- **`predict_batches(output=None, thresholds=None, block_rows=65536, backend=None)`** gives labels or values as `Heads.predict_batches` gives them, derived from what is stored:

| stored | `decision` | `proba` | `labels` | `values` |
|---|---|---|---|---|
| decision | the logits | the link of the logits | `linear@1` from the logits; `linear@2` from derived probabilities and the cutoffs | error |
| proba | error | the probabilities | `linear@2` from them and the cutoffs; `linear@1`: error | error |
| values | error | error | error | the values |

  What cannot be derived raises before a row is read, naming the head and the `store=` to write. `thresholds` maps multilabel heads to new cutoffs (a partial mapping keeps the stored cutoffs of the labels it does not name; a scalar or vector replaces all; `None` resets to 0.5); the comparison is strictly above. No encoder, tokenizer or table is loaded. The backends take C-contiguous float32 matrices, so each block copies the columns of a head of a store of several heads into a reused buffer (at most `block_rows` times the width times 4 bytes); a store of a single head is passed as it is. The rule `resolve_outputs` that `Heads` uses takes an `OutputRule(task, link)` per head, so the store builds no model.

### Errors

Before the first text or block: a `heads` that is no `fse.Heads`; a `store=` that does not fit the table; an unknown head name in `store`, `output`, `thresholds` or `get_scores`; appending with other heads, models, paths, ID mode or thresholds; a full append behind a tagged prefix; a root over 1 MiB; a `ScoreSink` given to something else than a group, a `target` before the attach and a second attach; an output that cannot be derived. Damage (an unreadable shard, other thresholds between neighbours, a gap, an orphan) is a `CorruptArtifactError` naming the shard, its first row and `fse.repair_store(path)`. An unknown version, kind, rule, link or path is an `UnsupportedArtifactError`.

## Logging

Scope: outside the former "2.0 design" section.

fse logs through the standard library only, on the fixed loggers `fse.artifact` (load, save, convert and verify of artifacts, their cleanup), `fse.hub` (a repository resolved to a commit, the files taken), `fse.backend` (the choice of a backend), `fse.runtime` (the plan and summary of a run, the default workers, the fold of a head, the Arrow inputs) and `fse.store` (sinks, shards, stores, searches), below their parent `fse`, which logs nothing itself. `fse._log` holds the names and the registry of events; `fse._log_config` holds `fse.enable_logging(level="INFO")` and `fse.disable_logging()`, the only public surface (no `logger=` parameter, no environment variable, no public formatter). With nothing configured, a WARNING of fse reaches stderr through the standard library's `lastResort`; INFO and DEBUG stay silent. `enable_logging` prints fse's records to the stderr of the moment, one line each (time, level, logger, message; the thread for records of other threads than the main thread; at DEBUG the fields; colours only on a terminal and never with `NO_COLOR`), through one handler fse owns on "fse", and stops propagation from "fse" meanwhile so a root handler prints nothing twice; it raises if the application attached a handler of its own to "fse" or below. `disable_logging` detaches it and restores the level and propagation of "fse" where nobody changed them since; a record in flight in another thread at that instant may reach both fse's handler and a root handler, because the standard library reads `propagate` after it called the handlers of "fse", without a lock. Applications that configure logging themselves attach their handler to "fse" (structlog through `ExtraAdder`, loguru through an `InterceptHandler`); downloads of the Hub show huggingface_hub's own progress bars (`HF_HUB_DISABLE_PROGRESS_BARS=1`).

1. fse logs only through `fse._log.emit`, with an event of `fse._log.EVENTS` named as a string literal (a test checks every call and that every event is emitted); every event is in the table below. Logger, event and field names are fixed for 2.x: an event may gain fields, never lose or rename one, and renaming a logger is a breaking change recorded in CHANGELOG.md.
2. fse adds no handler, sets no level and never touches the root logger or the warnings filters, except in `enable_logging` and `disable_logging` as their docstrings say.
3. Logging never changes a result or an exception. `emit` returns at once for a level nobody listens to and catches every `Exception` of the logging machinery (a handler, filter, formatter or record factory of the application, a field that cannot be formatted): it reports the first failure of each event once to `sys.__stderr__` (if `logging.raiseExceptions`) and goes on; KeyboardInterrupt and SystemExit propagate. The handler of `enable_logging` reports a record it cannot format or write the same way, once per event. No state of an operation depends on a record.
4. Errors raise and are never also logged. A slower but correct fallback logs a WARNING once per cause and object. The failed removal of a temporary file or directory is raised when no other error is raised (a store raises it once every thread of its sink has ended); while another error is raised (also a Ctrl+C that `ShardSink.close` deferred), that error is raised unchanged with a note naming the temporary path, and the failure is logged as `*.cleanup_failed` with its class and errno. A temporary path that is already gone counts as removed.
5. INFO is for operations on artifacts and stores, never for each call of `encode`, `search` or `predict`; DEBUG for plans, summaries and automatic choices, at most two records per call.
6. No record per task, row or sentence: nothing in the native code (the extension imports neither `logging` nor `warnings`), nothing in the tasks of the runtime (stages 1 to 5 in the workers, the section without the GIL). Fields that would hash, read or stat a file are computed only behind `fse._log.enabled` at the call site, since the arguments of `emit` are evaluated before it is called; the artifact records take kind, fingerprint and size from what a header states, so logging hashes nothing.
7. The events of a call are emitted in the calling thread, where the application's context variables are. The exception is `store.shard_published` (the publisher thread).
8. Records contain no paths, OS error texts, repository ids or revision names, tokens of the Hub, input texts, tokens, words, labels, vectors or scores: counts, sizes, dtypes, dimensions, durations, commits, fingerprints and fse's own names only (a class name of a source, sink or producer). Errors appear as class and errno. Exceptions keep naming paths: they go to the caller, not to an aggregator.
9. Every WARNING message names fse and its component in its text ("fse runtime: ..."), so it reads on its own under `lastResort`.

| Event (`fse_event`) | Level | Logger | Fields (each `fse_<name>`) | Emitted in |
|---|---|---|---|---|
| `backend.selected` | DEBUG | `fse.backend` | backend, isa | `get_backend`, once per backend and kernel per process |
| `artifact.loaded` | INFO | `fse.artifact` | kind, fingerprint, bytes, mmap, seconds | the safetensors container: `open`, `open_bytes` |
| `artifact.saved` | INFO | `fse.artifact` | kind, fingerprint, bytes, seconds | the safetensors container: `write` |
| `artifact.verified` | INFO | `fse.artifact` | kind, bytes, seconds | the safetensors container: `verify` (a failure raises, no record) |
| `artifact.converted` | INFO | `fse.artifact` | kind, source (`model2vec`, `sentence_transformers`, `gensim`, `sklearn`), rows, dim, tokenizer, normalize; dtype, fasttext (gensim); estimator, outputs, dim (sklearn) | a converted encoder (`_encoder._persist.converted`), `fse.gensim.from_gensim`, `from_sklearn` |
| `artifact.ignored` | DEBUG | `fse.artifact` | source, keys (fse's own names of known keys, never values) | the Model2Vec and sentence-transformers readers |
| `artifact.cleanup_failed` | WARNING | `fse.artifact` | error, errno | a temporary file of a failed write that could not be removed (containers, Hub folders) |
| `hub.resolved` | INFO | `fse.hub` | commit, offline, files | `HubFiles`, once the commit is known |
| `hub.file` | DEBUG | `fse.hub` | commit, bytes, pin (`lfs_sha256`, `git_sha1`) | `HubFiles.read`, after the check; `pin` is the digest the file was checked with (an LFS file that is no safetensors file is `lfs_sha256` too) |
| `runtime.workers` | DEBUG | `fse.runtime` | workers, origin (`override`, `affinity`, `cpu_count`, `cgroup`) | `usable_cpus`, when the default or its origin changes |
| `runtime.plan` | DEBUG | `fse.runtime` | source (class), rows, workers, inline, backend, dtype, sink (class), window_tasks | `Pipeline.infer` after its checks |
| `runtime.done` | DEBUG | `fse.runtime` | rows, tasks, seconds, rows_per_second | `Pipeline.infer` after a run |
| `runtime.arrow_copied` | WARNING | `fse.runtime` | bytes | an Arrow column whose offsets or views were copied to align them |
| `runtime.arrow_stream` | WARNING | `fse.runtime` | producer (class), error (class) | a producer that cannot export Arrow data, read as a stream |
| `head.folded` | INFO | `fse.runtime` | rows, outputs, workers, backend, seconds | `fold` of a head on raw means |
| `store.sink_opened` | INFO | `fse.store` | mode (`new`, `append`), rows, dim, dtype, shard_rows, durability (`full`, `none`) | `ShardSink` |
| `store.shard_published` | DEBUG | `fse.store` | first_row, rows, published_rows, durable_rows | the publisher of `ShardSink` (thread `fse-shard-publish`) |
| `store.sink_closed` | INFO | `fse.store` | rows, shards, published_rows, durable_rows, dropped_rows, durability (`full`, `none`), outcome (`ok`, `error`, `interrupt`), seconds | `ShardSink.close` |
| `store.cleanup_failed` | WARNING | `fse.store` | error, errno | a temporary shard or store directory that could not be removed while an error is raised |
| `store.opened` | INFO | `fse.store` | rows, shards, dim, dtype, backend, unsynced_shards, verified_bytes, verify_seconds | `VectorStore.open` |
| `store.verified` | INFO | `fse.store` | rows, shards, seconds | `VectorStore.verify` |
| `store.repaired` | INFO | `fse.store` | rows, moved_shards, moved_rows, seconds | `repair_store`, in the calling thread |
| `store.search` | DEBUG | `fse.store` | queries, k, mode (`exact`, `blas`), rows, workers | `VectorStore.search`, once per call |

Tests fail on every WARNING or higher of fse that no `support.logs.expect_log` expected, in setup, call and teardown (`tests/conftest.py`); a suite that causes a warning by design marks itself `expected_log(event)`.

For `store.opened`, `unsynced_shards` counts shards tagged `fse.durability=none`. `verified_bytes` counts their mapped tensor data bytes, excluding headers and manifests. `verify_seconds` covers their checksum and value checks. All three fields are zero for an untagged snapshot.

For `store.repaired`, `rows` is the kept prefix length. `moved_rows` sums the row counts of moved entries whose structure was readable; it is `None` if any moved entry has an unknown row count. It is zero when nothing moved. A failed repair raises without also emitting this event.

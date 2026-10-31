# Research notes

Historical note (31.10.2026): MLX was removed from fse. These research findings describe the earlier investigation.

Prior art gathered before planning a pull request (AGENTS.md: research hard questions before planning them). The notes are inputs to plans, not decisions: what was decided is in ARCHITECTURE.md, the CHANGELOG and the pull requests. Each note marks its claims by source (read in the source code or docs, derived, background knowledge, unverified); numbers come from the cited sources or from measurements described in the note, and a cited number is only as good as its source.

| Note | Question | Used for |
|---|---|---|
| [store-sinks.md](store-sinks.md) | Sinks, stores and durable writes in other libraries | PR 6 (#104), the store |
| [store-fast-writes.md](store-fast-writes.md) | Fast writes of many rows: mmap, pwrite, hashing | PR 6, the write path |
| [store-writer-thread.md](store-writer-thread.md) | Writer threads, queues and back-pressure | PR 6, the writer threads |
| [vector-db-survey.md](vector-db-survey.md) | Tricks of vector databases for exact search, scoring and writing (FAISS, Qdrant, Weaviate, pgvector, Lance, ScaNN and others) | PR 6b, later search work |
| [vector-db-usearch-hnswlib.md](vector-db-usearch-hnswlib.md) | int8 SIMD kernels (USearch, NumKong, hnswlib), overflow bounds, exact brute force | an exact int8 prefilter |
| [vector-db-milvus-diskann.md](vector-db-milvus-diskann.md) | Milvus/knowhere and DiskANN: brute force, quantized sidecars, stores larger than memory | read-ahead, two-stage search |
| [vector-db-storage-durability.md](vector-db-storage-durability.md) | DuckDB VSS, Lance, sqlite-vec; durability levels of SQLite, PostgreSQL, RocksDB, LMDB; warming the page cache | PR 6b: syncs, durability levels, warm() |
| [throughput-and-gpu-codex.md](throughput-and-gpu-codex.md) | Fusion, Python overhead, memory layout, a tolerance tier and a GPU backend boundary (German) | PR 6b, the MLX decision |
| [mlx-metal-kernels.md](mlx-metal-kernels.md) | Bit-exact Metal kernels with MLX: FMA, contraction, sqrt, subnormals, threads | PR 9 (MLX) |
| [runtime-prior-art.md](runtime-prior-art.md) | Runtimes of tokenizers and embedding libraries | the runtime (PR 4) |
| [vocab-lookup.md](vocab-lookup.md) | Hash tables for vocabulary lookup | the vocabulary table (PR 5b) |
| [profiling-tools.md](profiling-tools.md) | Profilers for CPU and memory with native frames | the profiling tools (#105) |
| [static-embeddings-2018-2026.md](static-embeddings-2018-2026.md) | Static sentence embeddings since 2018 | PR 12, modern static models |

File paths and line numbers in the notes refer to the commit of fse or of the other project they name; they drift as the code changes.

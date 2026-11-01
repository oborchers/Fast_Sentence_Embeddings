# Comparison profile summary

Historical POTION-8M float32 profiles used all 5,352 WikiText inputs on the M1 Pro. Instrumented durations are not speed or cost evidence. Original metadata, load endpoints, CPU samples, memray HTML/traces and RSS logs are preserved in the [evidence manifest](../../benchmarks/baseline/dr151_evidence_manifest_20261101.json).

| Instrumented study | Hottest recorded functions or allocations | Memory peak |
| --- | --- | --- |
| fse CPU, source `0de9c3d` | `fse_loop_f32_neon` 18.0% self samples; `_wp_piece` 15.6%; `fse_normalize_body` 1.2% | Separate process-tree RSS profile: 231,424,000 bytes (220.7 MiB) |
| Historical model2vec CPU diagnostic | Unicode category lookup 9.5% self samples; memory moves 8.7%; BERT normalizer appears in 22.4% of stacks including callees | No comparable resident peak established by this CPU profile |
| fse native allocation profile | Imports, `StringSource`, `MemorySink._new`, `CpuBackend.text_rows` | Memray tracked allocation peak 1.169 GB, including a 1.074 GB import mapping |

The fse CPU profile has 6,612 samples; the historical model2vec diagnostic has 4,034. Self samples and stack shares measure different things. Tracked mappings are not resident RAM, and cumulative allocations are not simultaneous peaks. Profiles include imports, preparation, startup and repeated encoding. They do not isolate kernel buffers or cold-load costs.

Later window5 counterpart profiles carry the known recorder identity defect. They remain diagnostics, not independently attributed counterpart evidence. No paired speed headline is established by those profiles. The archive preserves the original records and warnings without changing their numbers.

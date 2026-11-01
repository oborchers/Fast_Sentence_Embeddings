# Comparison evidence and compact summaries

Raw measurements live on the orphan branch `benchmark-evidence-2.0`. It has no master ancestry and must never be merged. The initial archive is pinned at [04e5cbeb1af5d36a45384906d91457d423829680](https://github.com/oborchers/Fast_Sentence_Embeddings/commit/04e5cbeb1af5d36a45384906d91457d423829680). The first Pod 4 quiet probe is secured at b734a9b78829988b52788be3e3707ac6a68c5c10. It was rejected, and no later stage ran. Pod 4b passed its probe but failed before paired measurement at the conservative memory admission check. Its ten files are secured at e8b338363def0f43a9460ea94a66cd2b96e8bb52. Pod 4c uses the reviewed bounded calibration fix and has a separate frozen handoff.

The [complete manifest](../../benchmarks/baseline/dr151_evidence_manifest_20261101.json) lists every archived path, byte size and SHA256. Hashes describe the archived payloads. Compact master summaries may retain a historical filename with a smaller schema. The complete original rich summary remains at that path in the pinned archive. All retained measured values are copied without rounding or recalculation.

The initial archive contains 454 files and 67,496,939 bytes, including all 48 README transfer files at their original paths. The ae79c80 snapshot additionally preserves two original CRLF worktree representations alongside their canonical LF Git files, for 456 files and 67,499,492 bytes at ae79c80. The subsequent b734a9b snapshot adds all nine files from the rejected Pod 4 probe, bringing the manifest to 465 files and 67,727,161 bytes. The e8b3383 snapshot additionally preserves the cleanup review and all ten Pod 4b files, for 477 files and 68,097,415 bytes. No measured value changed. The README payload’s original source is `7164b12637370d8064346d18161d73eee27dcb84`. The [README transfer receipt](../../benchmarks/baseline/dr154_evidence_verification_20261101.json) preserves the complete list for independent verification. The eight historical 1.1 baseline result files also have an archival copy and remain on master, as required by the maintainer. Numerical conformance fixtures under `tests/test_data` remain in the code tree.

## Study summaries

| Study | Compact master source | Full evidence |
| --- | --- | --- |
| Initial local comparison | [Four-cell summary](../../benchmarks/baseline/dr151_summary_20261101.json) | Manifest paths `benchmarks/baseline/results/dr151-model2vec-*` |
| Local model/dtype matrix | [54-cell summary](../../benchmarks/baseline/dr151_matrix_summary_20261101.json) | Original rich matrix summary and every journal |
| Pod 2 | [Coverage and selected cells](../../benchmarks/baseline/dr151_pod2_audit_20261101.json) | Full 303-cell audit, quality records and journals |
| Pod 3 panels | [All 30 cells, including noisy cells](../../benchmarks/baseline/dr151_pod3_panels_20261101.json) | Full correctness, runtime and paired observations |
| Pod 3 10M write paths | [Original small summary](../../benchmarks/baseline/dr151_pod3_large_write_20261101.json) | Complete write-path journal and invocations |
| Pod 3 100M showcase | [Original small summary](../../benchmarks/baseline/dr151_pod3_showcase_20261101.json) | Input identity, both operations, bootstrap and invocations |
| Local quality | `benchmarks/baseline/dr151_quality_*_20261101.json` | Per-task raw JSONL, original rich summaries and fit records |
| Store sync | [Compact summary](../../benchmarks/baseline/store_sync_20261031.json) | [Unchanged original](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/e8b338363def0f43a9460ea94a66cd2b96e8bb52/benchmarks/baseline/store_sync_20261031.json) |
| Store durability none | [Compact summary](../../benchmarks/baseline/store_none_20261031.json) | [Unchanged original](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/e8b338363def0f43a9460ea94a66cd2b96e8bb52/benchmarks/baseline/store_none_20261031.json) |
| Thread pool | [Compact summary](../../benchmarks/baseline/thread_pool_20261031.json) | [Unchanged original](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/e8b338363def0f43a9460ea94a66cd2b96e8bb52/benchmarks/baseline/thread_pool_20261031.json) |
| Thread-pool profiles | [Compact summary](../../benchmarks/baseline/thread_pool_profiles_20261031.json) | [Unchanged original](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/e8b338363def0f43a9460ea94a66cd2b96e8bb52/benchmarks/baseline/thread_pool_profiles_20261031.json) |
| DR190 Linux observations | [Compact summary](../../benchmarks/baseline/dr190_runpod_20261031.json) | [Unchanged original](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/e8b338363def0f43a9460ea94a66cd2b96e8bb52/benchmarks/baseline/dr190_runpod_20261031.json) |
| Profiles | [Hot functions and memory peaks](dr151-profiles.md) | CPU samples, memray traces/HTML and RSS logs |

The matrix uses `cell_columns` to label each cell array, and `backend_columns` to label each backend array. A backend's configuration index points into `configurations`. Attribution status indices point into `attribution_labels`, preserving the original interpretation warnings. These are storage references, not new measurements. Every cell retains its A/A spread and noise label. fse-a remains the primary summary; fse-b stays separate. Archive-only metadata includes full package paths, diagnostic configuration histories and repeated environment records.

Noisy cells remain available and are not headline evidence. Pod 3 classification failed before measurement. Its scaling stage exported measurements already produced by the panel stage. The 100M showcase is an explicitly approved one-run-plus-A/A demonstration. It is a streamed file-to-ShardSink int8 write with durability none, not an encode-only library comparison. Its primary rate, A/A repeat and output checks remain separate from the three-pair comparison protocol.

## Restore an individual raw file

Fetch the archive explicitly, then restore only the required file. The helper checks the pinned commit, byte size and SHA256 before creating its destination. It refuses overwrite. It never fetches automatically.

```bash
git fetch origin benchmark-evidence-2.0
uv run python benchmarks/quality/evidence_files.py \
  --path benchmarks/baseline/results/dr151-pod3-20261101/showcase-100m/speed-results.jsonl \
  --output "$HOME/.cache/fse-quality/evidence/showcase-100m.jsonl"
```

Use a single-branch clone if you want the code checkout without downloading the evidence branch. Results and reports are preserved. Only regenerable intermediates may be deleted under the shared data protocol. Shared caches need coordinator and maintainer approval.

## MIND scope

MIND is not computed: MTEB 2.26.25 MindSmallReranking needs more memory than the 64 GB pod (two attempts, status 137 and earlier ENOSPC). No further MIND run is planned. English scores cover 40 of 41 tasks and must say so. Historical raw records that say deferred retain their original wording in the archive.

The source summaries span different model, fitting, runtime and machine cohorts. Archiving does not make those cohorts directly comparable. The original warnings, normalization settings and backend attribution limits remain applicable.

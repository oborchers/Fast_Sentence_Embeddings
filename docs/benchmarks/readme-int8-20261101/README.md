# POTION-8M int8 throughput, 1 November 2026

These two complete series come from the committed comparison snapshot
[ceff3acbad889cfbababa55a11c31db2f1bb1090](https://github.com/oborchers/Fast_Sentence_Embeddings/commit/ceff3acbad889cfbababa55a11c31db2f1bb1090).
The full append-only journals are byte-identical to that source. Earlier attempts remain intact.
Only each journal's last complete three-valid-round series contributes to [summary.json](summary.json).

## Configuration

Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4, tokenizers 0.23.3 and model2vec 0.10.0.
Measurements occurred on 1 November 2026 UTC, around 06:16.
Both libraries use minishlab/potion-base-8M at revision bf8b056651a2c21b8d2565580b8569da283cab23.
fse stores its table as int8; the native Model2Vec control remains float32. Both return float32 vectors with 256 values.
The fse file contains 8,105,274 bytes. Its provenance names the pinned source and float32 parent fingerprint.
Different table precisions do not establish equal task quality or an engine-only speedup.

Every call uses all 5,352 nonblank WikiText validation and test texts.
Dataset: Salesforce/wikitext, wikitext-103-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3.
Input digest: xxh3-128:15151abfbf2f482115b1180347b4295e.

Each of three valid paired rounds rotates fse-a, fse-b and Model2Vec in separate child processes.
Each child loads its model, performs an excluded full-corpus warmup and times one full-corpus call.
Rate is 5,352 divided by the timed call's duration. Primary rates use fse-a, with fse-b retained separately as A/A control.
All load endpoints are below 5: 4.388671875 to 4.4228515625 for one worker, and 4.2421875 to 4.54296875 for ten.

## Results and chart eligibility

| fse workers | Texts/s, median (full range) | Symmetric A/A band | Status |
|---|---:|---|---|
| 1 | 214,848 (210,828 to 226,852) | 0.9722 to 1.0286 | Within 10%, included in chart |
| 10 | 919,009 (918,772 to 952,730) **noisy, to be re-measured** | 0.8953 to 1.1169 | Excluded from headline chart |

For each paired round, divide fse-b time by fse-a time. Let U be the largest ratio or reciprocal.
The symmetric band is 1/U to U, an empirical variation diagnostic rather than a confidence interval.
The ten-worker upper band exceeds 1.10, so its rate is not used as a stable headline observation.
The chart compares fse configurations only. It makes no dtype speedup, classifier-speed or library-ratio claim.

The largest primary process peaks were 169,639,936 bytes (161.8 MiB) for one worker and
168,886,272 bytes (161.1 MiB) for ten. These include loading, warmup and outputs, not just encoding allocations.
The noisy timing label is about timing variation, not a separate measurement of memory variation.

## Executed identities

The executing source commit is ae805751447dfbf35ecd1e37ead22195ac3daf51.
Selected rows carry comparison_identity_version=3 and backend-specific package names, locations, versions and content fingerprints.
The fse fingerprint is xxh3-128:19dd73e91757cd9b93c8e0a83a86440b, identical to the earlier chart's fse runtime.
Model2Vec has its own package location and fingerprint xxh3-128:4852905a78404abddd67f88f333e9f5d.
The published compatibility guard accepts each selected series, including identical fse-a/fse-b signatures.
No runtime identity was retroactively assigned to an old operation.

Native Model2Vec tokenizer parallelism is off beside the one-worker series and on beside the ten-worker series.
Its exact tokenizer thread count was not measured. No batch-worker pool runs below 10,000 texts.
Those controls and their unrounded results remain in the journals/summary, separate from the float32-series controls elsewhere.
Tariff-derived cost fields are omitted. These local runs do not measure cloud hardware throughput or cost.

## Sources and reproduction

- [One worker, full journal](dr151-matrix-potion-base-8M-int8-w1.jsonl)
- [Ten workers, full journal](dr151-matrix-potion-base-8M-int8-w10.jsonl)
- [Unrounded summary and identities](summary.json)
- [Published strict summary](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ceff3acbad889cfbababa55a11c31db2f1bb1090/benchmarks/baseline/dr151_matrix_summary_20261101.json)
- [Executed runner](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ae805751447dfbf35ecd1e37ead22195ac3daf51/benchmarks/bench_compare.py)
- [Compatibility guard](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ceff3acbad889cfbababa55a11c31db2f1bb1090/benchmarks/compare_compatibility.py)
- [Backend identity recorder](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ae805751447dfbf35ecd1e37ead22195ac3daf51/benchmarks/compare_runtime.py)

From the executed checkout, with its recorded environment and pinned corpus/models prepared:

```sh
.venv-quality/bin/python benchmarks/bench_compare.py run \
  --model minishlab/potion-base-8M --against model2vec \
  --table-dtype int8 --threads 1 --passes 1 --rounds 3 --budget-s 540 \
  --fse-models /path/to/prepared/models \
  --lock /path/to/shared/bench.lock --results results.jsonl
```

Use --threads 10 for that series. Follow current shared machine and timing rules before any new run.
The source records checkout commits and installed distribution labels separately. A package version alone is not a content identity.

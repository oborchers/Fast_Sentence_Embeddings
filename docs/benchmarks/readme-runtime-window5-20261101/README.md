# Runtime-attributed WikiText measurements, 1 November 2026

These three complete cells come from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen), timing window 5.
The full append-only journals are byte-identical to publication
[436e36b5da5bee43713551ca21ede4ac8c05df24](https://github.com/oborchers/Fast_Sentence_Embeddings/commit/436e36b5da5bee43713551ca21ede4ac8c05df24).
Earlier attempts remain in each journal. Only its last completed three-valid-round series contributes to this summary.
[summary.json](summary.json) records those series identifiers, unrounded values and executed identities.
The fse rates have executed-runtime attribution. The Model2Vec rates require re-measurement for runtime attribution.
These paired cells cannot supply a final current-runtime comparison headline.

## Model, input and environment

Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4,
tokenizers 0.23.3 and model2vec 0.10.0. Measurements occurred on 1 November 2026 UTC.
Both libraries use minishlab/potion-base-8M at revision bf8b056651a2c21b8d2565580b8569da283cab23,
with 256 values per vector, default text limits and normalized float32 outputs.
fse uses the labeled table dtype. Model2Vec always uses float32.
The float16 comparison does not establish equal task quality or a dtype speedup.

Every call receives all 5,352 nonblank validation and test texts from
Salesforce/wikitext, wikitext-103-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3.
Input digest: xxh3-128:15151abfbf2f482115b1180347b4295e.
This is WikiText, not the earlier 151,674-text MASSIVE input.

The executed source commit is 2ef19caa4424c3aec3055d26816c772fa89ba01c.
All selected operations record comparison_identity_version=2, versions and thread settings.
However, package_location and runtime_fingerprint describe fse even in Model2Vec rows.
The fingerprint xxh3-128:19dd73e91757cd9b93c8e0a83a86440b hashes only fse Python/native contents.
Model2Vec code location and content identity were not captured. Its version string alone does not fill that gap.
The raw artifact provenance binds the loaded table to its prepared identity.
The source compatibility guard accepts the selected pairs, including threadpool settings.
It does not check that the recorded package identity belongs to the backend being timed.
The summary preserves the owner classification separately and marks counterpart/headline eligibility false.
Checkout and distribution versions are retained separately, rather than treating a version string as a content fingerprint.

## Method and results

Each series has three valid paired rounds. Each round rotates fse-a, fse-b and Model2Vec in separate processes.
Each process loads the model, performs an excluded full-corpus warmup, then times one full-corpus call.
A rate is 5,352 divided by that call's duration. Primary results use the three fse-a rates.
fse-b repeats the same configuration as an A/A control and is summarized separately.
Setup and corpus loading are outside encode timing.
All recorded load endpoints are below 5, spanning 4.30908203125 to 4.99658203125 across these cells.

| fse table | Workers | fse-a texts/s, median (range) | Model2Vec float32 texts/s, median (range) | Tokenizer parallelism | Symmetric A/A band |
|---|---:|---:|---:|---|---|
| float32 | 1 | 222,701 (209,179 to 229,351) | 8,926 (8,861 to 8,971) **to be re-measured for runtime attribution** | off | 0.9707 to 1.0302 |
| float32 | 10 | 918,299 (856,137 to 938,323) | 23,312 (20,426 to 27,554) **to be re-measured for runtime attribution** | on | 0.9378 to 1.0664 |
| float16 | 10 | 946,796 (928,663 to 955,245) | 26,315 (26,129 to 28,109) **to be re-measured for runtime attribution** | on | 0.9180 to 1.0893 |

Model2Vec's exact tokenizer thread count was not measured. No parallel batch-worker pool runs below 10,000 texts.
Tokenizer on does not mean ten Model2Vec encoding workers.
The float16 series has its own Model2Vec control. Do not pool it with the float32 series.
There is no separate Model2Vec A/A test. Its table classification inherits the paired fse control.

For each round, divide fse-b time by fse-a time. Let U be the largest ratio or reciprocal.
The symmetric A/A band is 1/U to U. All three upper bounds are at most 1.10.
These are empirical variation bands, not confidence intervals or bounds on future runs.

The largest primary-process memory peaks across three rounds are 183.0 MiB (float32, one worker),
184.3 MiB (float32, ten workers), and 169.6 MiB (float16, ten workers).
MiB means bytes divided by 1,048,576. Peaks include imports, model/corpus loading, warmup, encoding and outputs.
They are process peaks, not incremental call memory, saved size or universal memory bounds.
Saved files are 30,664,410 bytes for float32 and 15,546,202 bytes for float16.

## Sources and reproduction

- [Float32, one worker](dr151-matrix-potion-base-8M-float32-w1.jsonl)
- [Float32, ten workers](dr151-matrix-potion-base-8M-float32-w10.jsonl)
- [Float16, ten workers](dr151-matrix-potion-base-8M-float16-w10.jsonl)
- [Published strict summary](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/436e36b5da5bee43713551ca21ede4ac8c05df24/benchmarks/baseline/dr151_matrix_summary_20261101.json)
- [Executed runner](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/2ef19caa4424c3aec3055d26816c772fa89ba01c/benchmarks/bench_compare.py)
- [Executed fse-only runtime recorder](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/2ef19caa4424c3aec3055d26816c772fa89ba01c/benchmarks/compare_artifacts.py)
- [Environment recorder](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/2ef19caa4424c3aec3055d26816c772fa89ba01c/benchmarks/compare_inputs.py)
- [Executed compatibility guard](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/2ef19caa4424c3aec3055d26816c772fa89ba01c/benchmarks/compare_compatibility.py)
- [Backend calls](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/2ef19caa4424c3aec3055d26816c772fa89ba01c/benchmarks/compare_backends.py)

From the executed checkout, with its recorded environment and pinned corpus/models prepared:

```sh
.venv-quality/bin/python benchmarks/bench_compare.py run \
  --model minishlab/potion-base-8M --against model2vec \
  --table-dtype float32 --threads 1 --passes 1 --rounds 3 --budget-s 540 \
  --fse-models /path/to/prepared/models \
  --lock /path/to/shared/bench.lock --results results.jsonl
```

Set --threads 10 for the ten-worker series. Also set --table-dtype float16 for that cell.
Follow the shared heavy-job, timing-window, bench.lock and load rules before running.

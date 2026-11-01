# Additional WikiText matrix cells, 1 November 2026

These three completed series come from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen).
The journals are byte-identical snapshots from the comparison worktree after completion.
The producing source commit is c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22.
They cover three table-precision/worker configurations, not a complete model or task-quality matrix.
The raw files and strict owner summary were published at d20b9bcde4e74818c9ee0258cb03e7b7020583b5.

## Method

Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4,
tokenizers 0.23.3 and model2vec 0.10.0. Measurement date: 1 November 2026 UTC.
Each encode call receives all 5,352 nonblank WikiText validation and test texts.
Dataset: Salesforce/wikitext, wikitext-103-raw-v1, revision
b08601e04326c79dfdd32d625aee71d232d685c3.
Input digest: xxh3-128:15151abfbf2f482115b1180347b4295e.

Each series contains three completed valid paired rounds. Each round runs
fse-a, fse-b and model2vec in rotating order, in separate child processes.
fse-b repeats the same fse configuration as an A/A control. Each child loads
its model, runs an excluded full-corpus warmup and times one full-corpus call.
The primary fse-a rates are 5,352 divided by each call's time. The table shows
their three-round median and full range. Outputs are float32 in every case.
No setup time or corpus loading is included in encode timing.

All recorded load endpoints are below 5: 4.3681640625 to 4.65869140625 for
GloVe int8, 4.302734375 to 4.7587890625 for POTION float16, and
4.302734375 to 4.32958984375 for POTION float32 with ten workers.
These cells use one timed call per child, unlike the three warm timed calls
per child in the earlier POTION float32 sample. No dtype speed ratio is
inferred across these separate series. Task-quality scores remain pending.

| fse model and table | fse workers | Texts/s, median (range) | Process peak, MiB |
|---|---:|---:|---:|
| GloVe-300, int8 | 1 | 210,198 (197,759 to 212,257) **noisy, to be re-measured** (A/A band 0.8730 to 1.1455) | 319.1 **noisy, to be re-measured** (A/A band 0.8730 to 1.1455) |
| POTION-8M, float16 | 1 | 206,438 (202,631 to 214,714) **noisy, to be re-measured** (A/A band 0.8899 to 1.1237) | 166.6 **noisy, to be re-measured** (A/A band 0.8899 to 1.1237) |
| POTION-8M, float32 | 10 | 914,709 (827,901 to 928,079) **noisy, to be re-measured** (A/A band 0.5117 to 1.9543) | 182.7 **noisy, to be re-measured** (A/A band 0.5117 to 1.9543) |

A/A labels attached to memory peaks describe timing variation, not memory variation.

Process peak is the largest primary fse-a process ru_maxrss across three
rounds, divided by 1,048,576. On macOS ru_maxrss is in bytes. It includes
imports, corpus and model loading, warmup, encoding and outputs. It is not
saved-file size, incremental encoding memory or a bound for other inputs.
Separate fse-b and Model2Vec peaks remain in the raw records and summary.

GloVe uses all 400,000 Wiki/Gigaword rows, 300 values per vector, unweighted
means, fse_words@1 tokens and normalized outputs. Its int8 file is 127,757,570
bytes. POTION uses its original 256 output values and a 15,546,202-byte
float16 file. The POTION float32 file is 30,664,410 bytes.
All source revisions, saved fingerprints and file sizes match
the committed model-size observations.

All paired Model2Vec controls use float32 minishlab/potion-base-8M at revision
bf8b056651a2c21b8d2565580b8569da283cab23. Tokenizer parallelism is off for
one-worker fse series and on for the ten-worker series. No batch-worker pool runs. The exact tokenizer thread count is unmeasured.
For GloVe this is a different model, with 256 instead of 300 output values and
different text rules. Its control is not an equal-model library comparison.
POTION series use the same source model. The one-worker fse table is float16,
while the paired Model2Vec table remains float32. No task-quality equivalence follows.

A/A time ratios divide fse-b time by fse-a time in each round.
GloVe median (range): 0.9284 (0.8730 to 0.9633).
POTION float16 median (range): 0.9931 (0.8899 to 1.0134).
POTION float32, ten workers: 1.2597 (1.0567 to 1.9543).
In one round, the repeated fse call took almost twice as long as the primary call.
This wide variation limits conclusions despite all load endpoints being below 5.
This is variation between repeated configurations, not an implementation gain.

## Evidence and reproduction

- [GloVe int8, one worker](dr151-matrix-glove-wiki-gigaword-300-int8-w1.jsonl)
- [POTION float16, one worker](dr151-matrix-potion-base-8M-float16-w1.jsonl)
- [POTION float32, ten workers](dr151-matrix-potion-base-8M-float32-w10.jsonl)
- [Strict published owner summary](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/d20b9bcde4e74818c9ee0258cb03e7b7020583b5/benchmarks/baseline/dr151_matrix_summary_20261101.json)
- [Unrounded calculated summary](summary.json)
- [Producing runner](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/benchmarks/bench_compare.py)
- [Backend calls](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/benchmarks/compare_backends.py)
- [Classic preparation](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/benchmarks/quality/prepare_word_models.py)
- [Hub-model preparation](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/benchmarks/quality/prepare_models.py)

From that checkout, with its environment, pinned models and corpus prepared:

```sh
.venv-quality/bin/python benchmarks/bench_compare.py run \
  --model fse/glove-wiki-gigaword-300 \
  --other-model minishlab/potion-base-8M --against model2vec \
  --table-dtype int8 --threads 1 --passes 1 --rounds 3 --budget-s 540 \
  --fse-models /path/to/prepared/models \
  --lock /path/to/shared/bench.lock --results results.jsonl
```

For POTION float16, set --model minishlab/potion-base-8M and --table-dtype float16.
For POTION float32, set --model minishlab/potion-base-8M, --table-dtype float32
and --threads 10.
Default corpus and warmup are the full WikiText input. Follow the shared
load, timing-window and pause rules. The raw records distinguish the source
checkout commit from the editable installed distribution version.


## A/A repeatability classification

A/A repeats the same fse configuration to measure timing variation. For each
round, divide fse-b time by fse-a time. Let U be the largest ratio or reciprocal.
The symmetric band is 1/U to U. It is an empirical diagnostic, not a confidence
interval. A band above 1.10 is noisy and must be re-measured before headlining.
Incomplete attempts provide no accepted speed, regardless of diagnostic bands.
Reference controls inherit their paired series label, not a separate Model2Vec A/A test.

| Complete series | A/A band | Status |
|---|---|---|
| dr151-matrix-glove-wiki-gigaword-300-int8-w1 | 0.8730 to 1.1455 | **noisy, to be re-measured** |
| dr151-matrix-potion-base-8M-float16-w1 | 0.8899 to 1.1237 | **noisy, to be re-measured** |
| dr151-matrix-potion-base-8M-float32-w10 | 0.5117 to 1.9543 | **noisy, to be re-measured** |

[Committed owner audit](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/f7537d970e4d883b4bfcd61d991859a36cb6da19/docs/fse-comparison-aa.md).

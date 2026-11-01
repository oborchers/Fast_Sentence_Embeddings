# Additional WikiText matrix cells, 1 November 2026

These two completed series come from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen).
The journals are byte-identical snapshots from the comparison worktree after completion.
The producing source commit is c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22.
They add two table-precision measurements, not a complete model or task-quality matrix.

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
GloVe int8, and 4.302734375 to 4.7587890625 for POTION float16.
These cells use one timed call per child, unlike the three warm timed calls
per child in the earlier POTION float32 sample. No dtype speed ratio is
inferred across these separate series. Task-quality scores remain pending.

| fse model and table | fse workers | Texts/s, median (range) | Process peak, MiB |
|---|---:|---:|---:|
| GloVe-300, int8 | 1 | 210,198 (197,759 to 212,257) | 319.1 |
| POTION-8M, float16 | 1 | 206,438 (202,631 to 214,714) | 166.6 |

Process peak is the largest primary fse-a process ru_maxrss across three
rounds, divided by 1,048,576. On macOS ru_maxrss is in bytes. It includes
imports, corpus and model loading, warmup, encoding and outputs. It is not
saved-file size, incremental encoding memory or a bound for other inputs.
Separate fse-b and Model2Vec peaks remain in the raw records and summary.

GloVe uses all 400,000 Wiki/Gigaword rows, 300 values per vector, unweighted
means, fse_words@1 tokens and normalized outputs. Its int8 file is 127,757,570
bytes. POTION uses its original 256 output values and a 15,546,202-byte
float16 file. Both source revisions, saved fingerprints and file sizes match
the committed model-size observations.

Both paired Model2Vec controls use float32 minishlab/potion-base-8M at revision
bf8b056651a2c21b8d2565580b8569da283cab23. Tokenizer parallelism is off, and
no batch-worker pool runs. The exact tokenizer thread count is unmeasured.
For GloVe this is a different model, with 256 instead of 300 output values and
different text rules. Its control is not an equal-model library comparison.
The POTION series uses the same source model at different table precisions.
No task-quality equivalence follows from either pairing.

A/A time ratios divide fse-b time by fse-a time in each round.
GloVe median (range): 0.9284 (0.8730 to 0.9633).
POTION median (range): 0.9931 (0.8899 to 1.0134).
This is variation between repeated configurations, not an implementation gain.

## Evidence and reproduction

- [GloVe int8, one worker](dr151-matrix-glove-wiki-gigaword-300-int8-w1.jsonl)
- [POTION float16, one worker](dr151-matrix-potion-base-8M-float16-w1.jsonl)
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

For POTION, set --model minishlab/potion-base-8M and --table-dtype float16.
Default corpus and warmup are the full WikiText input. Follow the shared
load, timing-window and pause rules. The raw records distinguish the source
checkout commit from the editable installed distribution version.

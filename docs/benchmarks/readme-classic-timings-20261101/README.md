# Classic-model WikiText timing sample, 1 November 2026


**Historical observations, to be re-measured for runtime attribution.**
The speed records lack executed-package paths and content fingerprints.
A/A repeatability and model fingerprints do not identify the code that ran.
None of these rates is a final current-runtime throughput result.

These four completed series come from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen).
The raw journals were copied byte for byte from the comparison worktree after completion.
Their producing source commit is c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22.
Incomplete series and rounds rejected for load are not included.

## Workload and method

Machine: Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4,
tokenizers 0.23.3 and model2vec 0.10.0. Measurement date: 1 November 2026 UTC.
The corpus contains all 5,352 nonblank validation and test texts from
Salesforce/wikitext, configuration wikitext-103-raw-v1, revision
b08601e04326c79dfdd32d625aee71d232d685c3.
Its input digest is xxh3-128:15151abfbf2f482115b1180347b4295e.

Each series has three complete valid rounds, each with three child processes
in rotating order: fse-a, fse-b and model2vec. fse-b repeats the same fse
configuration as an A/A control. All operation load endpoints were below 5.
Each process loads its model, runs one excluded full-corpus warmup, then times
**one** full-corpus encode call. Setup, corpus reads and warmup are outside
encode timing. This differs from the three warm calls per child in the earlier
POTION-8M timing sample. Outputs are float32.

The primary fse rates are texts divided by the fse-a call time, summarized as
median and full range across the three rounds. fse-b remains a separate control.
The saved classic models use all 400,000 GloVe Wiki/Gigaword rows or all
3,000,000 Google News word2vec rows, with 300 values per vector, unweighted
means, fse_words@1 tokens and normalized outputs. Source revisions and saved
sizes are recorded in each operation and in the model-size observations.

The paired Model2Vec operation uses **POTION-8M**, not GloVe or word2vec.
It has 256 output values rather than 300 and different text processing.
It is a timing control on the same input, not an equal-model library baseline.
No speed ratio or task-quality comparison is inferred from these pairs.
Its tokenizer parallelism is off at runner setting 1 and on at setting 10.
The input is below its batch-pool threshold. Its exact tokenizer thread count
was not measured. fse itself uses one or ten workers as recorded.

| Classic model and table | fse workers | Texts/s, median (range) | Process peak, MiB |
|---|---:|---:|---:|
| GloVe-300, float32 | 1 | 212,783 (203,964 to 219,669) **to be re-measured for runtime attribution** | 661.3 **to be re-measured for runtime attribution** |
| GloVe-300, float32 | 10 | 828,045 (708,401 to 901,244) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.7850 to 1.2739) | 662.0 **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.7850 to 1.2739) |
| word2vec-300, int8 | 1 | 212,606 (211,846 to 213,981) **to be re-measured for runtime attribution** | 1,552.5 **to be re-measured for runtime attribution** |
| word2vec-300, int8 | 10 | 924,000 (886,643 to 932,838) **to be re-measured for runtime attribution** | 1,552.7 **to be re-measured for runtime attribution** |

A/A labels attached to memory peaks describe timing variation, not memory variation.

The paired Model2Vec POTION-8M float32 controls, on the same texts:

| Paired classic configuration | Tokenizer off, texts/s | Tokenizer on, texts/s |
|---|---:|---:|
| GloVe float32 | 8,915 (8,896 to 8,935) **to be re-measured for runtime attribution** | 24,694 (24,511 to 27,057) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.7850 to 1.2739) |
| word2vec int8 | 8,944 (8,927 to 8,951) **to be re-measured for runtime attribution** | 27,280 (23,108 to 27,714) **to be re-measured for runtime attribution** |

Each cell is the median and range of three control-process rates. Off and on
correspond to the one-worker and ten-worker fse series, respectively. These
are different-model comparisons with the limitations stated above.

Process peak is the largest resource.getrusage(RUSAGE_SELF).ru_maxrss among
the three primary fse-a processes, divided by 1,048,576. On macOS this source
reports bytes. It covers imports, corpus/model loading, warmup and measured
encoding, including outputs. It is not table size, incremental encode memory,
or a maximum across every possible workload. The independent fse-b processes
and Model2Vec controls retain their own peaks in the raw records and summary.

A/A ratios compare fse-b time with fse-a time, not an implementation change.
Median (range), in table order: 1.0404 (0.9471 to 1.0407),
1.0290 (0.7850 to 1.1500), 1.0458 (0.9527 to 1.0693),
and 0.9840 (0.9302 to 1.0281). This variation limits small speed conclusions.

## Evidence and reproduction

- [GloVe float32, one worker](dr151-matrix-glove-wiki-gigaword-300-float32-w1.jsonl)
- [GloVe float32, ten workers](dr151-matrix-glove-wiki-gigaword-300-float32-w10.jsonl)
- [word2vec int8, one worker](dr151-matrix-word2vec-google-news-300-int8-w1.jsonl)
- [word2vec int8, ten workers](dr151-matrix-word2vec-google-news-300-int8-w10.jsonl)
- [Unrounded calculated summary](summary.json)
- [Producing runner](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/benchmarks/bench_compare.py)
- [Backend calls](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/benchmarks/compare_backends.py)
- [Model preparation](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/benchmarks/quality/prepare_word_models.py)

From the producing checkout, with its environment and pinned corpus/models
already prepared, run on a quiet machine:

```sh
.venv-quality/bin/python benchmarks/bench_compare.py run \
  --model fse/glove-wiki-gigaword-300 \
  --other-model minishlab/potion-base-8M --against model2vec \
  --table-dtype float32 --threads 1 --passes 1 --rounds 3 --budget-s 540 \
  --fse-models /path/to/prepared/models \
  --lock /path/to/shared/bench.lock --results results.jsonl
```

Repeat with --threads 10. For word2vec, use --model fse/word2vec-google-news-300
and --table-dtype int8. The default corpus is wikitext. The default warmup
uses the full corpus. Honor the shared timing window, pause marker and load
rules. The source checkout is recorded separately from the editable installed
fse distribution version. All original records remain available above.


## A/A repeatability classification

A/A repeats the same fse configuration to measure timing variation. For each
round, divide fse-b time by fse-a time. Let U be the largest ratio or reciprocal.
The symmetric band is 1/U to U. It is an empirical diagnostic, not a confidence
interval. A band above 1.10 is noisy and must be re-measured before headlining.
A band within 10% is insufficient without executed-runtime identity. New final
rates must record package location/content identity and keep each series compatible.
Incomplete attempts provide no accepted speed, regardless of diagnostic bands.
Reference controls inherit their paired series label, not a separate Model2Vec A/A test.

| Complete series | A/A band | Status |
|---|---|---|
| dr151-matrix-glove-wiki-gigaword-300-float32-w1 | 0.9471 to 1.0559 | within 10% A/A band |
| dr151-matrix-glove-wiki-gigaword-300-float32-w10 | 0.7850 to 1.2739 | **noisy, to be re-measured** |
| dr151-matrix-word2vec-google-news-300-int8-w1 | 0.9352 to 1.0693 | within 10% A/A band |
| dr151-matrix-word2vec-google-news-300-int8-w10 | 0.9302 to 1.0751 | within 10% A/A band |

[Committed owner audit](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/f7537d970e4d883b4bfcd61d991859a36cb6da19/docs/fse-comparison-aa.md).

[Runtime identity policy and mixed-series rejection](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/4e888dc18aadc3b5bf6da5b7db7afaa5d0e4b944/benchmarks/compare_compatibility.py).

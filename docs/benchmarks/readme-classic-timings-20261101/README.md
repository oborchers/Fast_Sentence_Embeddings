# Classic-model WikiText timing sample, 1 November 2026

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
| GloVe-300, float32 | 1 | 212,783 (203,964 to 219,669) | 661.3 |
| GloVe-300, float32 | 10 | 828,045 (708,401 to 901,244) | 662.0 |
| word2vec-300, int8 | 1 | 212,606 (211,846 to 213,981) | 1,552.5 |
| word2vec-300, int8 | 10 | 924,000 (886,643 to 932,838) | 1,552.7 |

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
- [Model preparation](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/c0ee537cc0515a7ecbfd4b20eec3bfc227b57a22/quality/prepare_models.py)

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

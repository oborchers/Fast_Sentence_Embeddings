# Baseline: fse 1.1.0

The values fse 2.0 is compared with, measured with fse 1.1.0 from PyPI, never with the repository. The 2.0 numerics differ from 1.x by design (no EPS, fused multiply-add instead of `saxpy`, no BLAS in the removal of components); the release of 2.0 measures the same tables again.

Machine: Apple M1 Pro (8 performance and 2 efficiency cores, 16 GB), macOS 27, Python 3.12, numpy 2.5.4, scipy 1.18.1, gensim 4.4.0. Other sessions ran on the machine. The result files record the load average before each measurement (for the lifecycle benchmark before each timed phase, not before the warm searches and `infer` calls): 3.1 to 11.0 for the training throughput, 3.3 to 7.6 for the lifecycle, 3.8 for the numerics, 4.2 to 17.1 for STS and 3.6 to 30.3 for MTEB. Throughput and latency were measured under that load and are no measure of an idle machine; quality values do not depend on it.

## Setup

```bash
uv venv -p 3.12 .venv-baseline
uv pip install -p .venv-baseline/bin/python "fse==1.1.0" -r benchmarks/quality/requirements.txt psutil
```

The scripts that measure only fse 1.1.0 (`sts.py`, `numerics.py`, `bench_train.py`, `bench_lifecycle.py`) were removed from the repository with the 1.x API; they are at commit 4931f5e, for example `git show 4931f5e:benchmarks/bench_train.py > bench_train.py`. `make_v1_fixture.py` stays, frozen: it wrote the fixture `tests/test_data/v1_reference/` that the tests read. The results below never change.

| Result | Command (from the repository root; removed scripts from commit 4931f5e) |
|---|---|
| `tests/test_data/v1_reference/` | `.venv-baseline/bin/python benchmarks/baseline/make_v1_fixture.py` |
| `results/sts-1.1.0.csv` | `.venv-baseline/bin/python sts.py` (`benchmarks/baseline/sts.py` at 4931f5e) |
| `results/mteb-1.1.0.jsonl` | `.venv-baseline/bin/python benchmarks/quality/run_mteb.py run --results benchmarks/baseline/results/mteb-1.1.0.jsonl --tasks baseline --models fse/Average-paranmt-300-float32 fse/SIF-paranmt-300-float32 fse/uSIF-paranmt-300-float32 fse/uSIF-paranmt-300-int8 fse/uSIF-glove-wiki-gigaword-300-float32 minishlab/potion-base-8M` |
| `results/numerics-1.1.0.json`, `results/neighbors-1.1.0.npz` | `.venv-baseline/bin/python numerics.py` (`benchmarks/baseline/numerics.py` at 4931f5e) |
| `results/train-1.1.0-m1pro.json` | `.venv-baseline/bin/python bench_train.py --models Average SIF uSIF AverageFT --dims 100 300 --workers 1 2 4 8 10 --repeats 3 --json ...` |
| `results/train-dtypes-1.1.0-m1pro.json` | `... bench_train.py --inputs IndexedList TextList CorpusFile --models Average SIF --dims 300 --wv-dtypes float32 int8 --sv-dtypes float32 int8 --workers 1 4 10 --repeats 3 --json ...` |
| `results/lifecycle-1.1.0-m1pro-sif-<sv>.json` | `.venv-baseline/bin/python bench_lifecycle.py --vectors glove-wiki-gigaword-300 --corpus ~/.cache/fse-quality/wikitext-103-b08601e0.txt --model SIF --sv-dtype <sv> --json ...` |

The MTEB and numerics scripts use the WikiText-103 fit corpus of `benchmarks/quality/run_mteb.py` (859,346 tokenized paragraphs, written once to `~/.cache/fse-quality`).

## STS

The procedure of `notebooks/STS-Benchmarks.ipynb` (Pearson × 100 on the 1,379 test pairs) for the 15 hub vector sets with CBOW, SIF-10 and uSIF. The float32 values reproduce the README table. Differences to float32 over all 45 combinations:

| wv_dtype | sv_dtype | largest gain | largest loss | mean |
|---|---|---|---|---|
| float16 | float32 | +0.003 | -0.003 | -0.000 |
| int8 | float32 | +0.017 | -0.128 | -0.002 |
| float32 | float16 | +0.002 | -0.001 | +0.000 |
| float32 | int8 | +0.072 | -0.020 | +0.006 |

## MTEB

35 tasks: the 9 STS, 3 PairClassification, 3 Classification, 4 Clustering, 1 Reranking, ArguAna and SummEval tasks of `MTEB(eng, v2)` (`BASELINE_TASKS` in `run_mteb.py`) and the 13 tasks of NanoBEIR. SIF and uSIF are fitted once on the WikiText corpus and frozen. Mean main score × 100 per task type (Retrieval includes NanoBEIR):

| Model | Classification | Clustering | PairClassification | Reranking | Retrieval | STS | Summarization |
|---|---|---|---|---|---|---|---|
| fse Average paranmt-300 | 74.18 | 32.85 | 73.64 | 51.23 | 39.49 | 72.21 | 28.09 |
| fse SIF-10 paranmt-300 | 72.01 | 32.72 | 71.66 | 50.68 | 40.74 | 72.10 | 24.35 |
| fse uSIF paranmt-300 | 72.68 | 32.55 | 72.07 | 50.80 | 40.40 | 72.12 | 25.11 |
| fse uSIF paranmt-300, wv int8 | 72.69 | 33.57 | 72.07 | 50.81 | 40.44 | 72.11 | 25.09 |
| fse uSIF glove-wiki-gigaword-300 | 70.28 | 35.76 | 72.19 | 51.23 | 38.59 | 65.35 | 24.91 |
| potion-base-8M (anchor) | 77.30 | 35.84 | 76.62 | 54.11 | 44.08 | 72.45 | 25.10 |

Each value comes from a single run; Clustering uses k-means, so its values carry run-to-run noise that this baseline did not measure.

## Numerics

Over the WikiText corpus with paranmt-300:

- The Gram matrix from float32 block products differs from an exact float64 one by at most 1.7e-7 relative; the principal angles between the 10 components of both (orthonormalized in float64 before the comparison) are at most 7.8e-5 degrees.
- 1 and 10 workers give bit-identical components.
- `most_similar` recall@10 against float32 sentence vectors: 0.994 for float16, 0.988 for int8 (1,000 queries; the neighbor ids are in `neighbors-1.1.0.npz`).
- Of the float32 top 100 neighbors, on average 36.6 have a cosine of at least 0.5, 4.95 of at least 0.7 and 1.32 of at least 0.9.

The fixture test (`tests/test_v1_reference.py`) compares the 2.0 reference with 1.1.0's `infer` on small random vectors: on Apple M1 (Accelerate), averaging is bit-identical for every word vector format and for FastText, and the removal of components differs by at most 3e-7 of the largest value of a row.

## Training throughput

Synthetic corpus of `bench_train.py` (at 4931f5e; 200,000 sentences of 12 tokens, a Zipf vocabulary of 50,000 words, 10 % unknown words), median of three runs, sentences per second end to end, in millions. 300 dimensions:

| Input | Model | 1 | 2 | 4 | 8 | 10 workers | peak RSS (MiB) |
|---|---|---|---|---|---|---|---|
| IndexedList | Average | 0.92 | 1.00 | 1.69 | 2.41 | 2.29 | 658 |
| IndexedList | SIF | 0.72 | 0.89 | 1.39 | 1.71 | 1.40 | 847 |
| TextList | Average | 0.57 | 0.97 | 1.48 | 2.00 | 2.00 | 694 |
| TextList | SIF | 0.49 | 0.78 | 1.14 | 1.47 | 1.52 | 887 |
| CorpusFile | Average | 0.63 | 1.26 | 2.27 | 3.11 | 3.23 | 652 |
| CorpusFile | SIF | 0.53 | 0.93 | 1.47 | 2.22 | 2.00 | 848 |
| CorpusFile | uSIF | 0.51 | 0.92 | 1.50 | 1.96 | 1.93 | 827 |
| CorpusFile | Average FastText | 0.26 | 0.51 | 0.93 | 1.48 | 1.76 | 952 |
| IndexedLineDocument | Average | 0.45 | 0.46 | 0.48 | 0.48 | 0.48 | 677 |

With 100 dimensions, CorpusFile reaches 6.57 million sentences per second for Average with 10 workers. The full grid (all inputs and models, 100 and 300 dimensions) is in `train-1.1.0-m1pro.json`.

Storage formats, CorpusFile, 300 dimensions, 10 workers:

| Model | wv float32, sv float32 | wv int8, sv float32 | wv float32, sv int8 | wv int8, sv int8 |
|---|---|---|---|---|
| Average | 3.59 | 4.62 | 1.26 | 1.31 |
| SIF | 2.22 | 2.46 | 0.65 | 0.77 |

int8 word vectors are faster, as fewer bytes are read; int8 sentence vectors cut the throughput to about a third, because 1.1 quantizes the windows of rows after the workers. 2.0 quantizes every row in its worker.

## Lifecycle

SIF-10 over glove-wiki-gigaword-300 on the 859,346 WikiText paragraphs, 10 workers (`bench_lifecycle.py` at 4931f5e):

| Phase | sv float32 | sv int8 |
|---|---|---|
| load vectors (pickle / mmap) | 2.39 s / 1.42 s | 1.66 s / 1.46 s |
| train (of it averaging) | 1.28 s (1.04 s) | 2.36 s (2.24 s) |
| peak RSS during train | 2,292 MiB | 2,347 MiB |
| save, bytes on disk | 0.70 s, 1,527 MB | 0.30 s, 757 MB |
| load | 0.39 s | 0.16 s |
| first search, warm search p50 | 1.52 s, 46.1 ms | 0.24 s, 17.9 ms |
| score of 1,000 vectors | 1.94 s | 9.64 s |
| infer of 1 / 1,000 / 100,000 sentences, p50 | 1.6 ms / 14.8 ms / 1.10 s | 1.2 ms / 14.6 ms / 1.08 s |

`infer` runs in one thread in 1.1: 100,000 paragraphs take about a second whatever the number of workers. The float32 first search builds a normalized copy of all rows; int8 keeps inverse norms only, but scores int8 rows more slowly.

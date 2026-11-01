# Fast Sentence Embeddings

**Our goal: be the fastest inference engine for static embeddings on the planet.**

**Final current-runtime headline: pending re-measurement.**

TODO(DR-151): Re-measure the intended headline comparison with executed-runtime identity and an A/A band within 10%.

Historical MASSIVE observations from 1 November 2026 used an Apple M1 Pro and `potion-base-8M` on 151,674 test texts.
Runtime here means the package code actually loaded. These logs did not record identifiers computed from that code's contents.
These figures cannot serve as final current-runtime throughput results.
One fse worker (a processor thread) encoded **878,260 texts/s** **to be re-measured for runtime attribution**.
Model2vec 0.10.0 with tokenizer parallelism off encoded **57,001 texts/s** **to be re-measured for runtime attribution**.
These are the middle rates from three valid paired rounds, with the models already loaded.
An A/A check repeats the same fse configuration to measure timing variation.
Its symmetric band was **0.9695 to 1.0315, within 10%**.
This is an observed variation band, not a confidence interval. It does not resolve the missing runtime attribution.
A tokenizer divides text into words and word pieces. Tables stored 32-bit numbers (`float32`).
Both libraries used the model's default text limits and normalization, which scales vectors to unit length.
See [the method, ranges, A/A audit and raw results](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/97f19344fdcc76f151227b523992b11c58372811/docs/benchmarks/readme-wikitext-20261101/README.md).
Results apply to this model and input.

![An engine turns pages of text into lists of numbers.](https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/233b6d3c0aaee459792938317895790a8bdd862b/media/readme-engine.jpg)

fse turns text into vectors, lists of numbers you can compare, search or classify.
It runs finished **static embeddings**: each token, a word or word piece, has a fixed vector.
fse looks up those vectors, averages them, and can scale the result to unit length.
It runs on the CPU and trains nothing.

**2.0 is in development on `master`.** Its API may change before release.
For the released 1.x API, use the [1.x README](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/1.x/README.md).
Old saved models must be rebuilt for 2.0.

## Quick start

You need standard CPython 3.11 to 3.15, [uv](https://docs.astral.sh/uv/), Git, Make and a C compiler.
Free-threaded Python builds are unsupported.
2.0 is not on PyPI yet. Install from the development branch:

```sh
git clone --branch master https://github.com/oborchers/Fast_Sentence_Embeddings.git
cd Fast_Sentence_Embeddings
UV_PYTHON=3.13 make install
```

Load a finished model from the Hugging Face Hub and encode two texts:

```python
import fse

encoder = fse.Encoder.from_pretrained("minishlab/potion-base-8M")
rows = encoder.encode(["the cat sleeps", "a dog runs"])
encoder.save("potion.safetensors")

# Later, load the local file without downloading the model again.
encoder = fse.Encoder.load("potion.safetensors")
```

`rows` is a NumPy matrix with one float32 vector per text.
The first load downloads model files. The model's license applies separately from fse's license.
Use `revision=` to pin a Hub commit for reproducible work.

## When to use fse

| Use fse when... | Consider another tool when... |
|---|---|
| You already have a static model and want CPU inference. | You need to train or distill a model. Use its training library. |
| You need the same output bits across supported CPU kernels and worker counts. | You need a contextual transformer. Use [sentence-transformers](https://github.com/huggingface/sentence-transformers). |
| You want to encode files, table columns or streams into a disk store. | You need a distributed vector database or an approximate search index. |
| You want to run an existing linear classifier with your encoder. | You need nonlinear models or training inside the inference engine. |

Static embeddings give a token the same vector in every sentence.
They cannot model context the way a transformer does. Test quality on your own task before choosing speed.
The [numerical contract](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/numerical-contract.md) defines fse's exact arithmetic.
Converted models can differ from their original library within float32 rounding.
Quantized models can lose quality. Bit identity does not mean equal quality between models or storage types.

### Keep compute close to your text

For a local application, start by running fse in the process that already has the text.
This avoids moving text and vectors through a service. Measure that transfer cost before adding a service.

One historical [notebook measurement](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/1631f057640c02591092314e922bec5df213bfdd/recipes/07-local-compute-and-http.ipynb), dated 1 November 2026,
used an Apple M1 Pro, macOS 27.0.1, Python 3.13.16 and one worker with the full GloVe-300 table.
Median means the middle measured time. A microsecond is one millionth of a second.
HTTP is the protocol websites use to transfer data. JSON is a text format for data.

| Work measured | Median time |
|---|---:|
| Encode a batch of 10,000 texts, per text | 0.736 microseconds **to be re-measured for runtime attribution** |
| Encode one text in one call | 52.3 microseconds **to be re-measured for runtime attribution** |
| Receive a fixed JSON vector over HTTP on the same computer, reusing one connection | 178.8 microseconds **to be re-measured for runtime attribution** |

The notebook did not record executed-runtime content identifiers. All three timings need re-measurement for current-runtime attribution.
The batch result uses five warm batches. Each batch repeats 1,000 texts from STS-B ten times.
STS-B is a test set of sentence pairs. The other results use 100 calls each.
The one-minute system load, a measure of running and waiting work, was 4.7114 before and after timing.
The HTTP test excludes embedding compute and turning the received text back into numbers.
Each row measures a different operation. A service that also encodes text needs its own test.
One actual normalized vector used 1,200 bytes as 300 float32 values and 6,320 bytes as compact JSON text.
JSON size depends on the values and formatting. Batching, concurrency and real networks change these costs.

## Which model, which dtype?

Start with a model designed for your language and task. The links below lead to its publisher's model card.
These are starting points for evaluation, not a ranking measured by fse.

| Model | Language and use to evaluate |
|---|---|
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) | English general tasks, including classification, with a small table. |
| [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M) | English general tasks when a larger vocabulary is useful. |
| [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | English search, matching queries to documents. |
| [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) | General tasks across languages. Check the card's language coverage. |
| [static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | English search, with a choice of output dimension. |
| [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | Similarity across languages, with a choice of output dimension. |
| GloVe-300 and word2vec-300 | Classic word-vector baselines for comparison and offline recipes. |

A **dtype** is how each number is stored. fse computes with float32 even when the table uses fewer bits.
The ratios below count table values only. Tokenizers, metadata and int8 row scales add space.
They are storage calculations, not measurements of file size or process memory.

| Table dtype | Bytes per value | Value storage relative to float32 | How to choose |
|---|---:|---:|---|
| float32 | 4 | 100% | Start here to establish quality. |
| float16 | 2 | 50% | Try when table memory matters. Check small values and task quality. |
| int8 | 1, plus a float32 scale per row | 25%, plus scales | Try for a tighter memory budget. Check task quality against float32. |

```python
compact = encoder.quantize("int8")  # a new encoder with a smaller table
compact.save("potion-int8.safetensors")
rows = compact.encode(["the cat sleeps"])
```

`encoder.quantize` changes the model table. `encoder.encode(..., dtype="int8")` changes the output storage instead.
A smaller table does not always run faster. On an Apple M1 Pro, measurements dated 30 October 2026
found int8 tables at dimension 1024 ran 1.20 to 1.34 times as fast on WikiText,
and 0.98 to 1.10 times as fast on MASSIVE, compared with float32.
Those runs used a benchmark pipeline, not the current public `Encoder.quantize` API.
Treat these historical pipeline ratios as **to be re-measured for runtime attribution** before current-runtime claims.
See [PERFORMANCE.md](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/PERFORMANCE.md#int8-tables-at-1024-dimensions) for workload details and limits.
See [the format rules](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/formats.md#encoder-1) for rounding and rejected values.

### Complete saved-file sizes

These [file-size observations](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/3b511c89e905368557b5eb864896b5a5d2cba47c/docs/benchmarks/readme-model-sizes-20261101.json)
were recorded on 1 November 2026 on an Apple M1 Pro.
They cover complete saved fse files, including vocabulary, tokenizer data and int8 row scales.
One MiB is 1,048,576 bytes. Each model keeps its original number of values per vector.

| Model | Values per vector | float32, MiB | float16, MiB | int8, MiB |
|---|---:|---:|---:|---:|
| potion-base-8M | 256 | 29.2 | 14.8 | 7.7 |
| potion-base-32M | 512 | 124.1 | 62.5 | 31.9 |
| potion-retrieval-32M | 512 | 124.1 | 62.5 | 31.9 |
| potion-multilingual-128M | 256 | 502.1 | 257.8 | 137.5 |
| static-retrieval-mrl-en-v1 | 1024 | 119.7 | 60.0 | 30.3 |
| static-similarity-mrl-multilingual-v1 | 1024 | 415.1 | 208.3 | 105.3 |
| GloVe-300 | 300 | 463.6 | 234.8 | 121.8 |
| word2vec-300 | 300 | 3,495.5 | 1,778.9 | 932.0 |

The [classic-model observations](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6ddf24d91b446a170b1218d3df4884afe690d09d/docs/benchmarks/readme-classic-model-sizes-20261101.json)
use full GloVe Wiki/Gigaword and Google News word2vec tables, with 400,000 and 3,000,000 rows respectively.
Their recipe is an unweighted mean with `fse_words@1` tokens and normalized outputs.

Use these sizes to plan disk storage. Measure peak memory, speed and quality for your workload.
The source records exact byte counts, model revisions and the environment that prepared the files.

### Task-quality samples

**POTION-8M, float32**

One [quality sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/e11d7b440e82f02516347145a186762f111ce27a/docs/benchmarks/readme-quality-20261101/README.md)
measured POTION-8M with a float32 table and two fse workers on 1 November 2026.
It used an Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4 and MTEB 2.26.25.
MTEB is a collection of tests for text vectors. NanoBEIR is a set of 13 small search tests.

| Task group | Coverage | Mean task score, points |
|---|---|---:|
| English MTEB(eng, v2), completed tasks only | 40 of 41, MIND deferred | 53.9527 |
| NanoBEIR search tasks | All 13 of 13 | 43.8485 |

Each completed task has equal weight. Points are 100 times the native task score.
The English row averages only 40 completed tasks. The coordinator deferred MindSmallReranking (MIND).
It is **not the complete 41-task benchmark score**.
English tasks use different metrics. NanoBEIR uses `ndcg_at_10`, a score for the first ten search results.
The source records every task, metric, dataset revision and the exact saved model identity.
Further model and dtype comparisons, including quality changes from float32, remain pending.

**Later POTION-base-32M sample, float32**

A [later sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ba96e9a663062c86edb1e8eb8aa75774ccf04bb1/docs/benchmarks/readme-quality-potion32-20261101/README.md)
used POTION-base-32M, two fse workers and the same Apple M1 Pro on 1 November 2026.
The recorded evaluation code differs from the POTION-8M run above.
Confirmations of POTION-8M and POTION-base-32M with matching evaluation code are queued after the recording fix.
A controlled model ranking remains pending.

| Task group | Coverage | Mean task score, points |
|---|---|---:|
| English MTEB(eng, v2), completed tasks only | 40 of 41, MIND deferred | 55.2573 |
| NanoBEIR search tasks | All 13 of 13 | 46.1754 |

This English mean also excludes deferred MIND. It is **not the complete 41-task benchmark score**.
The source records each task's metric and dataset revision, the saved model, and the code identifiers.

**POTION-retrieval-32M, float32, with fixed runtime recording**

A [separate quality sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ba96e9a663062c86edb1e8eb8aa75774ccf04bb1/docs/benchmarks/readme-quality-retrieval32-20261101/README.md)
measured POTION-retrieval-32M with two fse workers on the Apple M1 Pro on 1 November 2026.
It records the loaded code's identity after the recording fix.
Its evaluation code differs from the earlier P8/P32 checkpoints, which still await matching-code confirmations.
A controlled model ranking remains pending.

| Task group | Coverage | Mean task score, points |
|---|---|---:|
| English MTEB(eng, v2), completed tasks only | 40 of 41, MIND deferred | 54.5098 |
| NanoBEIR search tasks | All 13 of 13 | 50.3872 |

The English mean excludes MIND and is **not the complete 41-task benchmark score**.
These are task-quality scores. Setup, suspension and shared-job waits do not provide a model compute-time measurement.

### First neighborhood stability check

An [audit of POTION-8M](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/771fc882e1813ec9b57faa9d04435ef2cd850262/benchmarks/baseline/results/dr151-numerics.jsonl)
used all 5,352 nonblank WikiText validation and test texts on 1 November 2026, on an Apple M1 Pro.
It found the ten most similar other vectors for every text, using cosine similarity, which compares vector directions.
It then measured how many of those neighbors each candidate retained, compared with fse's float32 table.

| Candidate | Mean share of the original top ten retained |
|---|---:|
| model2vec 0.10.0, float32 | 100.000% |
| fse, float16 table | 99.994% |
| fse, int8 table | 99.634% |

Every text had a nonzero vector. The audit excludes each text from its own neighbor list.
For equal scores, the earlier text in the input comes first. All outputs are float32.
These results measure neighborhood stability on this input. Further model and dtype task comparisons remain pending.

TODO(DR-151): Add the remaining peak-memory, speed and task-score comparisons.
Include quality change from each model's float32 result. Name the dataset, metric, revision and date.

## Benchmarks

The one-worker POTION-8M float32 cell uses the [preserved WikiText measurements](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/97f19344fdcc76f151227b523992b11c58372811/docs/benchmarks/readme-wikitext-20261101/README.md)
from 1 November 2026: Apple M1 Pro, macOS 27.0.1, Python 3.13.16,
NumPy 2.5.4, tokenizers 0.23.3 and model2vec 0.10.0.
Model2vec tokenizer parallelism is off or on, as labeled.
The runner enables parallel tokenization for the automatic-mode call.
This input is below the 10,000-text batch-pool threshold, so no batch-worker pool runs.
The model revision is `bf8b056651a2c21b8d2565580b8569da283cab23`.
The 5,352 nonblank texts come from the pinned WikiText validation and test splits.

Each of three paired rounds runs fse twice as an A/A control and model2vec once, in rotating order.
Each child process measures three warm calls. Its rate is texts divided by its median call time.
The primary fse run and model2vec each contribute three child rates to their median and range.
The repeated fse runs remain separate A/A controls, matching the published comparison summary.
Numbers in parentheses show the full range. All recorded load endpoints were below 5.
The raw records include the input digest, dataset revision, source commit and A/A variation.
The historical MASSIVE sample has an A/A band within 10%, but lacks executed-runtime identity.
The table below uses a later precision-matrix sample for ten-worker POTION-8M float32,
with one timed call per child as described below.

The [committed A/A audit](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/f7537d970e4d883b4bfcd61d991859a36cb6da19/docs/fse-comparison-aa.md)
labels every complete comparison. For each repeated fse call, compare its time with the primary call.
Let U be the largest ratio or reciprocal across rounds. The symmetric A/A band is 1/U to U.
A band above 1.10 is **noisy, to be re-measured** in a quiet window.
Only re-measured cells with executed-runtime identity and A/A bands within 10% may supply final headline rates.
Record loaded package paths and content fingerprints. Keep code, model, input and settings unchanged within each series.
A checkout commit, package version or model fingerprint alone does not identify the code that ran.
All current comparison rates below are historical and require re-measurement for runtime attribution.
See the [executed-runtime audit](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/4e888dc18aadc3b5bf6da5b7db7afaa5d0e4b944/docs/fse-comparison-review.md).
These empirical A/A bands are not confidence intervals.
Incomplete attempts supply no accepted speed, even when their diagnostic A/A band is small.

TODO(DR-151): Fill the remaining pending cells with paired measurements and source links.
Report sentences/s as median and range for one worker and all available cores.
Each cell must identify float32 / float16 / int8 table results in that order.
The counterpart must encode the same texts. Record token limits, normalization and any model differences.
Name machine, OS, Python, NumPy, library versions, model revisions, worker counts, load and measurement date.

Model2vec labels below inherit the paired fse series' A/A band, not a separate Model2vec A/A test.
In the POTION-8M row, tokenizer off uses the original one-worker float32 series.
Tokenizer on uses the later ten-worker float32 precision-matrix series.

| Model | fse, 1 worker (f32 / f16 / i8) | fse, 10 workers (f32 / f16 / i8) | Counterpart, float32 (modes below) |
|---|---|---|---|
| potion-base-8M | 219,333 (199,745 to 221,304) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.8998 to 1.1114) / 206,438 (202,631 to 214,714) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.8899 to 1.1237) / pending | 914,709 (827,901 to 928,079) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.5117 to 1.9543) / pending / pending | model2vec tokenizer off: 8,906 (8,879 to 8,956) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.8998 to 1.1114); on: 25,074 (22,589 to 27,134) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.5117 to 1.9543) |
| potion-base-32M | pending | pending | model2vec: pending |
| potion-retrieval-32M | pending | pending | model2vec: pending |
| potion-multilingual-128M | pending | pending | model2vec: pending |
| static-retrieval-mrl-en-v1 | pending | pending | sentence-transformers: pending |
| static-similarity-mrl-multilingual-v1 | pending | pending | sentence-transformers: pending |

The classic models have fse-only measurements:

| Model | fse, 1 worker (f32 / f16 / i8) | fse, 10 workers (f32 / f16 / i8) |
|---|---|---|
| GloVe-300 | 212,783 (203,964 to 219,669) **to be re-measured for runtime attribution** / pending / 210,198 (197,759 to 212,257) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.8730 to 1.1455) | 828,045 (708,401 to 901,244) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.7850 to 1.2739) / pending / pending |
| word2vec-300 | pending / pending / 212,606 (211,846 to 213,981) **to be re-measured for runtime attribution** | pending / pending / 924,000 (886,643 to 932,838) **to be re-measured for runtime attribution** |

The GloVe float32 and word2vec int8 cells use [four completed series](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/97f19344fdcc76f151227b523992b11c58372811/docs/benchmarks/readme-classic-timings-20261101/README.md)
from the same date, machine and WikiText input. Each has three paired rounds,
with one warm timed call per child after a full-corpus warmup.
All load endpoints were below 5.
The source reports A/A variation separately, including a time-ratio range of 0.7850 to 1.1500 for ten-worker GloVe.

The one-worker POTION-8M float16, GloVe int8 and ten-worker POTION-8M float32 cells use
[three completed precision-matrix series](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/97f19344fdcc76f151227b523992b11c58372811/docs/benchmarks/readme-matrix-20261101/README.md).
They use the same machine, date and full WikiText input, with three paired rounds.
Each child times one call after a full-corpus warmup. All load endpoints were below 5.
The source records separate A/A controls. In one ten-worker POTION round, the repeated fse call
took 1.9543 times as long as the primary call. This wide variation limits speed conclusions.
These series do not establish a dtype speed ratio or task quality.

The largest process memory peaks among each series' three primary fse runs were:

| Model and table | 1 worker, MiB | 10 workers, MiB |
|---|---:|---:|
| GloVe-300, float32 | 661.3 **to be re-measured for runtime attribution** | 662.0 **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.7850 to 1.2739) |
| GloVe-300, int8 | 319.1 **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.8730 to 1.1455) | pending |
| word2vec-300, int8 | 1,552.5 **to be re-measured for runtime attribution** | 1,552.7 **to be re-measured for runtime attribution** |
| POTION-8M, float16 | 166.6 **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.8899 to 1.1237) | pending |
| POTION-8M, float32 | pending | 182.7 **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.5117 to 1.9543) |

These peaks include imports, model and corpus loading, warmup, encoding and outputs.
Their A/A labels refer to timing variation, not memory variation.
Runtime-attribution markers also apply to these historical process peaks.
They measure process memory, not saved-file size or memory added by one call.
The linked raw records and summary retain exact byte counts and separate control-process peaks.
Other model, dtype and task-quality results remain pending.

**Reference controls**

The classic-model series also ran Model2Vec with POTION-8M on the same texts.
POTION-8M produces 256 values per vector, rather than 300, and uses different text rules.
These controls measure a different model. They do not establish an equal-model library speed ratio.
Each cell shows the median and range of three control-process rates.
Labels inherit the paired fse series' timing classification, not a separate Model2Vec A/A test.

| Series containing the control | Model2Vec POTION-8M, tokenizer off, texts/s | Model2Vec POTION-8M, tokenizer on, texts/s |
|---|---:|---:|
| GloVe float32 series | 8,915 (8,896 to 8,935) **to be re-measured for runtime attribution** | 24,694 (24,511 to 27,057) **to be re-measured for runtime attribution** **noisy, to be re-measured** (A/A band 0.7850 to 1.2739) |
| word2vec int8 series | 8,944 (8,927 to 8,951) **to be re-measured for runtime attribution** | 27,280 (23,108 to 27,714) **to be re-measured for runtime attribution** |

The off/on controls accompanied the one/ten-worker fse series, respectively.
No Model2Vec batch-worker pool ran. Its exact tokenizer thread count was not measured.

The historical [one-worker MASSIVE sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/233b6d3c0aaee459792938317895790a8bdd862b/docs/benchmarks/readme-wikitext-20261101/dr151-model2vec-massive-1.jsonl) records these historical rates.
Across 151,674 test texts, fse processed 878,260 texts/s (861,892 to 903,738) **to be re-measured for runtime attribution**,
and model2vec processed 57,001 (56,620 to 57,191) **to be re-measured for runtime attribution**, with the same three-round method.
Its symmetric A/A band is 0.9695 to 1.0315, within 10%, but runtime attribution still requires re-measurement.
This is a separate input sample. It does not fill the pending model or dtype cells.

The old 1.x STS results belong to a different API and recipe.
See the [1.x results](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/1.x/README.md#results).
They do not measure these 2.0 models.

## Integrations

- **Model2Vec:** `fse.StaticModel` provides a compatibility API. It accepts a string or a list through `encode`.
  `StaticModel.save_pretrained` and `StaticModel.push_to_hub` export fse model folders.
  Reload them with `fse.StaticModel.from_pretrained`. Model2Vec 0.10.0 cannot read these folders.
  Publishing requires a recorded model license.
  See the [supported calls and differences](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/hub.md#model2vec-compatibility-fsestaticmodel).
- **sentence-transformers:** `Encoder.from_pretrained` reads supported models containing one StaticEmbedding module.
  It does not run arbitrary transformer pipelines. See [conversion limits](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/hub.md#sentence-transformers-models).
- **Hugging Face Hub:** load with `from_pretrained`, save a local folder with `save_pretrained`, or publish with `push_to_hub`.
  Publishing needs your credentials and a model license. See [Hub folders](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/hub.md#hub-folders).
- **gensim:** convert `KeyedVectors`, `Word2Vec` or `FastText` with `fse.gensim.from_gensim`.
  This adapter needs the optional `fse[gensim]` extra. See [word vectors](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/formats.md#word_vectors-1).
- **Tables and streams:** encode lists, text file paths, iterators and Arrow string columns.
  Arrow lets table libraries share a column without copying its text into Python strings.
  See [text inputs](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/text.md).
- **scikit-learn:** import supported fitted linear heads with `Classifier.from_sklearn` or `Regressor.from_sklearn`.
  fse runs the head and does not fit it. See [linear heads](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/formats.md#linear_head-1).
- **Files:** models use safetensors. Vector stores use NumPy `.npy` shards.
  See [formats](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/formats.md) and [disk stores](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/store.md).

Model2Vec conversion defaults to its token and character limits. Sentence-transformers conversion follows that model's tokenizer rules.
The [Hub reader documentation](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/hub.md) records the tested library versions and differences.
Use those rules when comparing outputs.

## Recipes and development

The [recipe notebooks](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/README.md) cover offline model preparation, including SIF and uSIF,
and practical workflows for files, streams, linear heads and integrations.
fse runs the finished table after model preparation.

Read [CONTRIBUTING.md](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/CONTRIBUTING.md) before proposing a change.
It covers AI assistance, reproducible measurements and the review evidence we need.
[ARCHITECTURE.md](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/ARCHITECTURE.md) defines the contracts.
[CHANGELOG.md](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/CHANGELOG.md) records changes. Report vulnerabilities through [SECURITY.md](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/SECURITY.md).

## License and citation

fse 2.0 uses the [MIT License](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/LICENSE). The `1.x` branch and existing 1.x releases remain GPL-3.0.
Model and dataset licenses apply separately. See [NOTICE](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/NOTICE) for third-party notices.

Built by [Dr. Oliver Borchers](https://www.drborchers.com).
If you use fse in research, cite the repository:

```bibtex
@misc{Borchers2019,
  author = {Borchers, Oliver},
  title = {Fast Sentence Embeddings},
  year = {2019},
  howpublished = {\url{https://github.com/oborchers/Fast_Sentence_Embeddings}}
}
```

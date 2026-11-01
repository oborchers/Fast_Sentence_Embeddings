# Fast Sentence Embeddings

**Our goal: be the fastest inference engine for static embeddings on the planet.**

On 1 November 2026, an Apple M1 Pro encoded 5,352 WikiText texts with `potion-base-8M`.
One fse worker (a processor thread) encoded **219,333 texts/s**.
Model2vec 0.10.0 with tokenizer parallelism off encoded **8,906 texts/s**.
Ten fse workers encoded **909,270 texts/s**. Model2vec with tokenizer parallelism on encoded **26,048 texts/s**.
These are the middle measured rates. A tokenizer divides text into words and word pieces.
Tables stored 32-bit numbers (`float32`). Both libraries used the model's default text limits and normalization,
which scales vectors to unit length. The models were already loaded.
See [the method, ranges and raw results](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/233b6d3c0aaee459792938317895790a8bdd862b/docs/benchmarks/readme-wikitext-20261101/README.md).
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

One [notebook measurement](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/07-local-compute-and-http.ipynb), dated 1 November 2026,
used an Apple M1 Pro, macOS 27.0.1, Python 3.13.16 and one worker with the full GloVe-300 table.
Median means the middle measured time. A microsecond is one millionth of a second.
HTTP is the protocol websites use to transfer data. JSON is a text format for data.

| Work measured | Median time |
|---|---:|
| Encode a batch of 10,000 texts, per text | 0.736 microseconds |
| Encode one text in one call | 52.3 microseconds |
| Receive a fixed JSON vector over HTTP on the same computer, reusing one connection | 178.8 microseconds |

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
These results measure neighborhood stability on this input. Classification and retrieval task scores remain pending.

TODO(DR-151): Add the remaining peak-memory and speed comparisons, plus all task-score comparisons.
Include quality change from each model's float32 result. Name the dataset, metric, revision and date.

## Benchmarks

The first row uses the [preserved WikiText measurements](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/233b6d3c0aaee459792938317895790a8bdd862b/docs/benchmarks/readme-wikitext-20261101/README.md)
from 1 November 2026: Apple M1 Pro, macOS 27.0.1, Python 3.13.16,
NumPy 2.5.4, tokenizers 0.23.3 and model2vec 0.10.0.
Model2vec uses its one-worker mode or its automatic mode.
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

TODO(DR-151): Fill the remaining pending cells with paired measurements and source links.
Report sentences/s as median and range for one worker and all available cores.
Each cell must identify float32 / float16 / int8 table results in that order.
The counterpart must encode the same texts. Record token limits, normalization and any model differences.
Name machine, OS, Python, NumPy, library versions, model revisions, worker counts, load and measurement date.

| Model | fse, 1 worker (f32 / f16 / i8) | fse, 10 workers (f32 / f16 / i8) | Counterpart, float32 (modes below) |
|---|---|---|---|
| potion-base-8M | 219,333 (199,745 to 221,304) / pending / pending | 909,270 (830,368 to 944,311) / pending / pending | model2vec tokenizer off: 8,906 (8,879 to 8,956); on: 26,048 (22,894 to 26,880) |
| potion-base-32M | pending | pending | model2vec: pending |
| potion-retrieval-32M | pending | pending | model2vec: pending |
| potion-multilingual-128M | pending | pending | model2vec: pending |
| static-retrieval-mrl-en-v1 | pending | pending | sentence-transformers: pending |
| static-similarity-mrl-multilingual-v1 | pending | pending | sentence-transformers: pending |
The classic models have fse-only measurements:

| Model | fse, 1 worker (f32 / f16 / i8) | fse, 10 workers (f32 / f16 / i8) |
|---|---|---|
| GloVe-300 | 212,783 (203,964 to 219,669) / pending / pending | 828,045 (708,401 to 901,244) / pending / pending |
| word2vec-300 | pending / pending / 212,606 (211,846 to 213,981) | pending / pending / 924,000 (886,643 to 932,838) |

The classic-model cells use [four completed series](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/74f2883fe74a1e1c188cbdffa22562f4e3d7b1d1/docs/benchmarks/readme-classic-timings-20261101/README.md)
from the same date, machine and WikiText input. Each has three paired rounds,
with one warm timed call per child after a full-corpus warmup.
All load endpoints were below 5.
The source reports A/A variation separately, including a time-ratio range of 0.7850 to 1.1500 for ten-worker GloVe.

The largest process memory peaks among each series' three primary fse runs were:

| Classic model and table | 1 worker, MiB | 10 workers, MiB |
|---|---:|---:|
| GloVe-300, float32 | 661.3 | 662.0 |
| word2vec-300, int8 | 1,552.5 | 1,552.7 |

These peaks include imports, model and corpus loading, warmup, encoding and outputs.
They measure process memory, not saved-file size or memory added by one call.
The linked raw records and summary retain exact byte counts and separate control-process peaks.
Other model, dtype and task-quality results remain pending.

**Reference controls**

The classic-model series also ran Model2Vec with POTION-8M on the same texts.
POTION-8M produces 256 values per vector, rather than 300, and uses different text rules.
These controls measure a different model. They do not establish an equal-model library speed ratio.
Each cell shows the median and range of three control-process rates.

| Series containing the control | Model2Vec POTION-8M, tokenizer off, texts/s | Model2Vec POTION-8M, tokenizer on, texts/s |
|---|---:|---:|
| GloVe float32 series | 8,915 (8,896 to 8,935) | 24,694 (24,511 to 27,057) |
| word2vec int8 series | 8,944 (8,927 to 8,951) | 27,280 (23,108 to 27,714) |

The off/on controls accompanied the one/ten-worker fse series, respectively.
No Model2Vec batch-worker pool ran. Its exact tokenizer thread count was not measured.

The same float32 POTION-8M model also has a [one-worker MASSIVE sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/233b6d3c0aaee459792938317895790a8bdd862b/docs/benchmarks/readme-wikitext-20261101/dr151-model2vec-massive-1.jsonl).
Across 151,674 test texts, fse processed 878,260 texts/s (861,892 to 903,738),
and model2vec processed 57,001 (56,620 to 57,191), with the same three-round method.
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

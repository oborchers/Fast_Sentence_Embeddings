# Fast Sentence Embeddings

**Our goal: be the fastest inference engine for static embeddings on the planet.**

TODO(DR-151): Add the measured headline comparison, with machine, date, texts,
worker counts, package versions and a link to the raw results.

One existing measurement: on 31 October 2026, an AMD EPYC 9655P pod encoded
11,903,145 synthetic sentences/s with 16 workers, float32 tables at dimension 300, and normalization disabled.
One worker encoded 1,652,196 sentences/s. These are single shared-host observations,
without a model2vec or sentence-transformers counterpart. They do not establish a speed record.
See [the data and limits](benchmarks/baseline/dr190_runpod_20261031.json)
and [PERFORMANCE.md](PERFORMANCE.md#runpod-linux-x86-observations).

![An engine turns pages of text into lists of numbers.](media/readme-engine.jpg)

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
The [numerical contract](docs/architecture/numerical-contract.md) defines fse's exact arithmetic.
Converted models can differ from their original library within float32 rounding.
Quantized models can lose quality. Bit identity does not mean equal quality between models or storage types.

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
See [PERFORMANCE.md](PERFORMANCE.md#int8-tables-at-1024-dimensions) for workload details and limits.
See [the format rules](docs/architecture/formats.md#encoder-1) for rounding and rejected values.

TODO(DR-151): Add measured file size, peak memory, speed and task scores for each model and dtype,
including quality change from that model's float32 result. Name the dataset, metric, revision and date.

## Benchmarks

TODO(DR-151): Replace pending cells below with the paired measurements and source links.
Report sentences/s as median and range for one worker and all available cores.
Each cell must identify float32 / float16 / int8 table results in that order.
The counterpart must encode the same texts, with matching token limits and normalization.
Name machine, OS, Python, NumPy, library versions, model revisions, worker counts, load and measurement date.

| Model | fse, 1 worker (f32 / f16 / i8) | fse, all cores (f32 / f16 / i8) | Counterpart on the same texts |
|---|---|---|---|
| potion-base-8M | pending | pending | model2vec: pending |
| potion-base-32M | pending | pending | model2vec: pending |
| potion-retrieval-32M | pending | pending | model2vec: pending |
| potion-multilingual-128M | pending | pending | model2vec: pending |
| static-retrieval-mrl-en-v1 | pending | pending | sentence-transformers: pending |
| static-similarity-mrl-multilingual-v1 | pending | pending | sentence-transformers: pending |
| GloVe-300 | pending | pending | source and counterpart: pending |
| word2vec-300 | pending | pending | source and counterpart: pending |

The old 1.x STS results belong to a different API and recipe.
See the [1.x results](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/1.x/README.md#results).
They do not measure these 2.0 models.

## Integrations

- **Model2Vec:** `fse.StaticModel` provides a compatibility API. It accepts a string or a list through `encode`.
  `StaticModel.save_pretrained` and `StaticModel.push_to_hub` export fse model folders.
  Reload them with `fse.StaticModel.from_pretrained`. Model2Vec 0.10.0 cannot read these folders.
  Publishing requires a recorded model license.
  See the [supported calls and differences](docs/architecture/hub.md#model2vec-compatibility-fsestaticmodel).
- **sentence-transformers:** `Encoder.from_pretrained` reads supported models containing one StaticEmbedding module.
  It does not run arbitrary transformer pipelines. See [conversion limits](docs/architecture/hub.md#sentence-transformers-models).
- **Hugging Face Hub:** load with `from_pretrained`, save a local folder with `save_pretrained`, or publish with `push_to_hub`.
  Publishing needs your credentials and a model license. See [Hub folders](docs/architecture/hub.md#hub-folders).
- **gensim:** convert `KeyedVectors`, `Word2Vec` or `FastText` with `fse.gensim.from_gensim`.
  This adapter needs the optional `fse[gensim]` extra. See [word vectors](docs/architecture/formats.md#word_vectors-1).
- **Tables and streams:** encode lists, text file paths, iterators and Arrow string columns.
  Arrow lets table libraries share a column without copying its text into Python strings.
  See [text inputs](docs/architecture/text.md).
- **scikit-learn:** import supported fitted linear heads with `Classifier.from_sklearn` or `Regressor.from_sklearn`.
  fse runs the head and does not fit it. See [linear heads](docs/architecture/formats.md#linear_head-1).
- **Files:** models use safetensors. Vector stores use NumPy `.npy` shards.
  See [formats](docs/architecture/formats.md) and [disk stores](docs/architecture/store.md).

Model2Vec conversion defaults to its token and character limits. Sentence-transformers conversion follows that model's tokenizer rules.
The [Hub reader documentation](docs/architecture/hub.md) records the tested library versions and differences.
Use those rules when comparing outputs.

## Recipes and development

The [recipe notebooks](recipes/README.md) cover offline model preparation, including SIF and uSIF,
and practical workflows for files, streams, linear heads and integrations.
fse runs the finished table after model preparation.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change.
It covers AI assistance, reproducible measurements and the review evidence we need.
[ARCHITECTURE.md](ARCHITECTURE.md) defines the contracts.
[CHANGELOG.md](CHANGELOG.md) records changes. Report vulnerabilities through [SECURITY.md](SECURITY.md).

## License and citation

fse 2.0 uses the [MIT License](LICENSE). The `1.x` branch and existing 1.x releases remain GPL-3.0.
Model and dataset licenses apply separately. See [NOTICE](NOTICE) for third-party notices.

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

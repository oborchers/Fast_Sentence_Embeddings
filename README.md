<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/master/media/logo/fse-logo-dark.svg">
<img alt="fse" width="160" src="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/master/media/logo/fse-logo.svg">
</picture>
</p>

<h2 align="center"><strong>The fastest inference engine for static embeddings on the planet*</strong></h2>
<p align="center"><sub>* That's our goal, at least.</sub></p>

<p align="center">
<a href="https://github.com/oborchers/Fast_Sentence_Embeddings/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/oborchers/Fast_Sentence_Embeddings/actions/workflows/ci.yml/badge.svg?branch=master"></a>
<a href="https://pypi.org/project/fse/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fse.svg"></a>
<a href="https://pypi.org/project/fse/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/fse.svg"></a>
<a href="https://pepy.tech/project/fse"><img alt="Downloads" src="https://pepy.tech/badge/fse"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
<a href="https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/oborchers/Fast_Sentence_Embeddings.svg?style=flat"></a>
</p>

**Compute nearly 3 million text embeddings per second.**

100 million short texts (28.7 tokens on average) embedded and written to disk in 35 seconds, int8, 32 vCPU AMD EPYC 7713, RunPod, 1 November 2026.

[How we measured](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/5c54d65885f1cc7caa56785c142a521a5303c923/docs/benchmarks/readme-evidence.md)

Fast Sentence Embeddings turns text into vectors for search, clustering and classification.
It runs ready-made static models on your CPU. No GPU needed.
Use your Model2Vec model, a supported static Sentence Transformers model, or existing word vectors.
Bring a trained linear classifier to label text in the same application.

![An engine turns pages of text into lists of numbers.](https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/233b6d3c0aaee459792938317895790a8bdd862b/media/readme-engine.jpg)

## Get started

```sh
pip install "fse>=2.0"
```

Moving an existing application? Read the [migration guide from fse 1.x](docs/migrating-from-1x.md) before upgrading.

Python 3.11 to 3.15. Load a model and make a vector for each text:

```python
from fse import Encoder

encoder = Encoder.from_pretrained("minishlab/potion-base-8M")
vectors = encoder.encode(["the cat sleeps", "a dog runs"])
```

A vector is a list of numbers that represents a text. Here you get a NumPy array with one row per text.
For document vectors, pass your documents as strings. The model's text-length limits still apply.
The first load downloads the model. Its license applies separately.
For complete applications, explore the nine [recipes](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/README.md) below.

### Already using Model2Vec?

Switch the import for the common loading and encoding calls:

```python
from fse import StaticModel  # replaces: from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")
vectors = model.encode(["the cat sleeps", "a dog runs"])
```

The supported API is similar, but conversion can change float32 rounding.
`save_pretrained` and `push_to_hub` export fse folders. Reload them with `fse.StaticModel.from_pretrained`.
They are not Model2Vec 0.10.0 export files.

### Using a static Sentence Transformers model?

```python
from fse import Encoder

encoder = Encoder.from_pretrained("sentence-transformers/static-retrieval-mrl-en-v1")
vectors = encoder.encode(["How do I reset my password?", "Open your account settings."])
```

This supports compatible static models. It does not run contextual transformers such as MiniLM or BGE.

## Choose a model and memory budget

Start with POTION-base-8M for a small English model. Use retrieval models for search and multilingual models for other languages.
Compare quality and size below, then check the model on your own data.

| Model (Hub ID) | Quality[^quality] | Float32 file size |
|---|---:|---:|
| [`minishlab/potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M) | 55.26 | 124.1 MiB |
| [`minishlab/potion-retrieval-32M`](https://huggingface.co/minishlab/potion-retrieval-32M) | 54.51 | 124.1 MiB |
| [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) | 53.95 | 29.2 MiB |
| [`sentence-transformers/static-retrieval-mrl-en-v1`](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | 51.89 | 119.7 MiB |
| [`sentence-transformers/static-similarity-mrl-multilingual-v1`](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | 51.43 | 415.1 MiB |
| [`fse/fasttext-crawl-subwords-300-sif10`](https://huggingface.co/fse/fasttext-crawl-subwords-300-sif10) | 51.42 | 4.50 GiB |
| [`fse/paragram-300-sl999-sif10`](https://huggingface.co/fse/paragram-300-sl999-sif10) | 50.98 | 1.93 GiB |
| [`fse/paranmt-300-raw`](https://huggingface.co/fse/paranmt-300-raw) | 50.69 | 89.5 MiB |
| [`minishlab/potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M) | 49.96 | 502.1 MiB |
| [`fse/glove-wiki-gigaword-300-sif10`](https://huggingface.co/fse/glove-wiki-gigaword-300-sif10) | 49.79 | 463.6 MiB |
| [`fse/word2vec-google-news-300-sif10`](https://huggingface.co/fse/word2vec-google-news-300-sif10) | 49.26 | 3.41 GiB |

[^quality]: MTEB(eng, v2), measured on 1 November 2026 with MTEB 2.26.25. Scores are the unweighted mean over 40 of 41 English tasks, multiplied by 100. MIND was not computed. Classic-model scores use the historical benchmark recipe. SIF-10 scores use a WikiText-103 fit, whereas the linked Hub SIF-10 models use wordfreq weights and a FineWeb fit. See each Hub card for that version's own results. Sizes refer to float32 fse files. [Archived benchmark results](https://github.com/oborchers/Fast_Sentence_Embeddings/tree/660309c4c88930254288495a0bb025fe428870f7/benchmarks/baseline).

Start with float32. Try float16 or int8 when memory matters, then check quality again.

```python
from fse import Encoder

compact = Encoder.from_pretrained("minishlab/potion-base-8M").quantize("int8")
compact.save("compact.safetensors")
```

Classic Hub models have `-raw` and `-sif10` versions, also with `-float16` or `-int8` suffixes.
`-raw` in a classic Hub model name means no SIF weighting. It does not mean unnormalized output.

You can also use your own GloVe, word2vec or FastText vectors with the optional `fse[gensim]` adapter.
The [SIF and uSIF recipe](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/01-sif-usif.ipynb) shows how to prepare classic word vectors for sentence search.

## Recipes: from a model to a working application

These nine recipes show the full workflow, including preparation, execution and checks.
Start with the job you want to do:

| Use case | Recipe |
|---|---|
| Start here: fse 101 | [Turn sentences into vectors, find similar text, and save your results.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/00-fse-101.ipynb) |
| Sentence search | [Make classic word vectors more useful for sentence search.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/01-sif-usif.ipynb) |
| Text classification | [Train a text classifier, compare its accuracy, and save it for your app.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/02-linear-heads.ipynb) |
| Files with resume | [Turn a text file into vectors and continue after an interruption.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/03-files-with-resume.ipynb) |
| Queues and Qdrant | [Turn queued news into searchable vectors and replay safely.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/04-kafka-and-qdrant.ipynb) |
| Folder listener | [Pick up completed files and add each one to your search store.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/05-folder-listener.ipynb) |
| Integrations | [Use one local encoder with your search, topic, and table libraries.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/06-integrations.ipynb) |
| Local compute and HTTP | [See when a local HTTP request costs more than encoding the text.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/07-local-compute-and-http.ipynb) |
| Folded classifier | [Fold a trained classifier into word vectors, then save a smaller model for your app.](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/08-folded-classifier.ipynb) |

The [recipe index](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/README.md) has setup instructions and recorded results.
The queue recipe uses a fake queue and local Qdrant. The folder recipe polls for completed files.
These are worked examples you can adapt, not hosted services.

## Label text with a linear classifier

Route support requests, filter documents or classify a dataset.
Train a supported linear model in scikit-learn, then let fse run it directly on text.
The model must be trained on this encoder's vectors.
`normalize=False` works for every supported model, including our classic Hub models under `fse/`.
Our classic Hub models load normalized by default. For folding, load with `normalize=False` before training the linear head.
Install `scikit-learn` first: `pip install scikit-learn`.

```python
from sklearn.linear_model import LogisticRegression
from fse import Classifier, Encoder

encoder = Encoder.from_pretrained("minishlab/potion-base-8M", normalize=False)
texts = ["I cannot log in", "My password is lost", "Send me a quote", "What does it cost?"]
head = LogisticRegression().fit(encoder.encode(texts), ["support", "support", "sales", "sales"])
classifier = Classifier.from_sklearn(encoder, head)
labels = classifier.predict(["I need a new password"])
classifier.save("router.safetensors")
```

Train on real labeled data for your application.
Load the saved model with `Classifier.load("router.safetensors")` and predict without scikit-learn.

The [classification recipe](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/02-linear-heads.ipynb) compares this with Model2Vec's own classifier, TF-IDF and fastText.

### Why folding works

A sentence vector averages the vectors of its words or word pieces.
A linear head multiplies that average by weights W and adds a bias b.
Multiplying and averaging can swap order:

`W · mean(v) + b = mean(W · v) + b`

fse computes W · v once for each word and stores the scores.
A four-label classifier can then keep 4 scores per word instead of 256 values.
Prediction looks up those scores, averages them and adds the bias.
There is no full sentence vector or head matrix multiplication per text.

The [folding recipe](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/08-folded-classifier.ipynb) found identical AG News labels, with probability differences up to about 0.000003 from rounding.
The complete POTION-8M file uses 29.2 MiB, compared with 0.86 MiB for the standalone file.

POTION-8M, AG News, Apple M1 Pro, 1 November 2026.

This works for linear classification and regression heads trained on plain averages, including logistic regression, `LinearSVC` and `Ridge`, without requesting sentence vectors.
fse runs linear heads trained on normalized vectors explicitly, because normalization prevents this fold.
Neural heads with hidden layers cannot use this fold, and fse does not run them.

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/71b69be83efdd8adbf071ada302be8bf1db68b24/media/readme/vectors-and-labels-dark.png">
<img alt="Your texts enter fse on the CPU. An encoder gives vectors for search and clustering. Add a previously trained linear model to get labels for routing and filtering. Training happens outside fse." src="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/71b69be83efdd8adbf071ada302be8bf1db68b24/media/readme/vectors-and-labels-light.png">
</picture>

## When to use fse

**Choose fse when throughput matters more than squeezing out the highest possible model quality.**
Static models give each word or word piece a fixed vector. They do less work than contextual transformers.
That makes them useful for large collections, local applications and repeated classification.
A good contextual transformer may handle meaning, ambiguity and difficult search tasks better.
Check the quality on your own data before choosing the faster model.

This trade-off concerns the model you choose. Switching a Model2Vec static model to fse does not mean choosing a weaker model.
fse runs finished models; it does not train or distill them.

| Tool | A good fit when you want to... |
|---|---|
| **fse** | Run static embeddings, fitted linear classifiers and file or stream jobs on a CPU. |
| **Model2Vec** | Create, distill or fine-tune static models, as well as run them. |
| **Sentence Transformers** | Run or train contextual transformers, or use its static models and broader model ecosystem. |
| **gensim / Doc2Vec** | Train document vectors or word-vector models in gensim. |

## Why is fse so fast?

**We build and tune the whole path, from reading text to writing results.**
That includes splitting text into tokens, finding their vectors, combining them and saving the output.
The goal is to keep useful work together and avoid overhead between steps.

| Design choice | What it does |
|---|---|
| Prepare once | Reuse a finished model's fixed word vectors for every text. |
| Keep work together | The CPU fast path groups tokenization and vector calculation in compiled code. |
| Share the work | CPU workers process batches in parallel. |
| Control the output path | Files and streams can write to fse's vector store without collecting all results in memory. |

## More

- [Documentation](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/ARCHITECTURE.md): supported APIs, model formats, exports and guarantees.
- [Contributing](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/CONTRIBUTING.md): AI-assisted contributions are welcome with reproducible checks and evidence.
- [Changelog](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/CHANGELOG.md) and [security policy](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/SECURITY.md).

fse 2.0 uses the [MIT License](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/LICENSE).
The 1.x releases remain GPL-3.0. Model and dataset licenses apply separately.
Built by [Dr. Oliver Borchers](https://www.drborchers.com).

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

**Fast Sentence Embeddings turns text into vectors for search, clustering and classification.**
It runs ready-made static models on your CPU. No GPU needed.
Use your Model2Vec model, a supported static Sentence Transformers model, or existing word vectors.
Bring a trained linear classifier to label text in the same application.

![An engine turns pages of text into lists of numbers.](https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/233b6d3c0aaee459792938317895790a8bdd862b/media/readme-engine.jpg)

## Speed first

**918,299 texts per second. On an Apple M1 Pro.**
POTION-8M, float32, 10 workers, WikiText, 1 November 2026.
This measures encoding with the model already loaded. Text length and model choice change the result.

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/71b69be83efdd8adbf071ada302be8bf1db68b24/media/readme/throughput-dark.png">
<img alt="POTION-8M on WikiText: float32, one worker, 222,701 texts/s; float32, ten workers, 918,299; float16, ten workers, 946,796. Three-round medians on Apple M1 Pro, 1 November 2026. Bars compare fse settings, not libraries." src="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/71b69be83efdd8adbf071ada302be8bf1db68b24/media/readme/throughput-light.png">
</picture>

The chart uses all 5,352 nonblank WikiText validation and test texts, in three paired rounds.
Float16 uses less space for the model. This is a separate measurement, not a promise of equal quality.
A new library speed comparison and classification benchmark are pending.
The current Model2Vec control rates need re-measurement before a final comparison.
[Measurement details and full model tables](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/2d85d281a5b6ba5e942bb8df691aa694ebb10f6f/docs/benchmarks/readme-evidence.md) retain ranges, quality scores and pending comparisons.

### Classification speed

We will compare complete text-to-label pipelines, including folding.

| Linear classifier | fse, texts/s | Reference pipeline, texts/s | fse speedup |
|---|---:|---|---:|
| On normalized vectors | pending | Model2Vec encoding + the same linear head: pending | pending |
| Folded into raw word-vector averages | pending | fse encoding + the same explicit linear head: pending | pending |

TODO(DR-151): Fill this table and its speed chart from committed classification benchmarks.
Each row must use the same encoder, classifier weights, texts and settings.
The chart will show these measured rates and speedups.

## Get started

```sh
pip install "fse>=2.0"
```

Python 3.11 to 3.15. Load a model and make a vector for each text:

```python
from fse import Encoder

encoder = Encoder.from_pretrained("minishlab/potion-base-8M")
vectors = encoder.encode(["the cat sleeps", "a dog runs"])
```

A vector is a list of numbers that represents a text. Here you get a NumPy array with one row per text.
The first load downloads the model. Its license applies separately.
For complete applications, explore the eight [recipes](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/README.md) below.

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

### Need document vectors, as in a Doc2Vec workflow?

Use a ready-made model to encode your documents, without training a new Doc2Vec model:

```python
from fse import Encoder

encoder = Encoder.from_pretrained("minishlab/potion-base-32M")
documents = ["A customer needs help signing in.", "A visitor asks about pricing."]
vectors = encoder.encode(documents)
```

This is an alternative workflow, not a loader for trained Doc2Vec models. Model text-length limits still apply.

## Label text with a linear classifier

Route support requests, filter documents or classify a dataset.
Train a supported linear model in scikit-learn, then let fse run it directly on text.
The model must be trained on this encoder's vectors.

```python
from sklearn.linear_model import LogisticRegression
from fse import Classifier, Encoder

encoder = Encoder.from_pretrained("minishlab/potion-base-8M")
texts = ["I cannot log in", "My password is lost", "Send me a quote", "What does it cost?"]
head = LogisticRegression().fit(encoder.encode(texts), ["support", "support", "sales", "sales"])
classifier = Classifier.from_sklearn(encoder, head)
labels = classifier.predict(["I need a new password"])
classifier.save("router.safetensors")
```

Install `scikit-learn` for this training example. The four texts illustrate the API; train on representative data for real use.
Later, load the saved classifier with `Classifier.load("router.safetensors")` and predict without scikit-learn.
For large datasets, pass batches instead of building one huge list.
The recipe below covers training, classification, regression and saving the finished model.

The classification recipe compares fse's linear head with Model2Vec 0.10.0's fine-tuned MLP, a small neural network.
In this recipe, Model2Vec also updates its word vectors during training.

Its [default head](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/train/classifier.py) has one hidden layer of 512 units with ReLU, which clips negative values to zero.
Training uses PyTorch. This recipe also evaluates the trained PyTorch model.
The exported [prediction head](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/inference/mlp.py) runs in NumPy, with softmax or sigmoid for probabilities.
This compares different trained models, not just two inference engines.

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/71b69be83efdd8adbf071ada302be8bf1db68b24/media/readme/vectors-and-labels-dark.png">
<img alt="Your texts enter fse on the CPU. An encoder gives vectors for search and clustering. Add a previously trained linear model to get labels for routing and filtering. Training happens outside fse." src="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/71b69be83efdd8adbf071ada302be8bf1db68b24/media/readme/vectors-and-labels-light.png">
</picture>

## Recipes: from a model to a working application

These eight recipes show the full workflow, including preparation, execution and checks.
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

The [recipe index](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/README.md) has setup instructions and recorded results.
The queue recipe uses a fake queue and local Qdrant. The folder recipe polls for completed files.
These are worked examples you can adapt, not hosted services.

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

This compares workflows, not measured speed or quality rankings.

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

This is why fse is an inference engine, not a training framework.
It controls its own processing and storage path. External services and databases still have their own costs.

## Choose a model and memory budget

### Available now

Load these models directly from their publishers. We do not publish copies of them.

| Start with... | For... | Saved float32 file |
|---|---|---:|
| [POTION-base-8M](https://huggingface.co/minishlab/potion-base-8M) | Small English general-purpose models | 29.2 MiB |
| [POTION-base-32M](https://huggingface.co/minishlab/potion-base-32M) | English tasks with a larger vocabulary | 124.1 MiB |
| [POTION-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | English search | 124.1 MiB |
| [POTION-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) | Tasks across languages | 502.1 MiB |
| [static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | English search with adjustable vector size | 119.7 MiB |
| [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | Similarity across languages | 415.1 MiB |

These are models to evaluate, not a quality ranking. Sizes cover complete saved fse files, not process memory.
Start with float32. Try float16 or int8 when memory matters, then check quality again.

```python
compact = Encoder.from_pretrained("minishlab/potion-base-8M").quantize("int8")
compact.save("compact.safetensors")
```

### Classic models planned for the 2.0 release

These Hub models are **planned, not yet available**:

| Model | Original word vectors | Prepared with SIF-10 |
|---|---|---|
| GloVe-300 | `fse/glove-wiki-gigaword-300-raw` | `fse/glove-wiki-gigaword-300-sif10` |
| word2vec-300 | `fse/word2vec-google-news-300-raw` | `fse/word2vec-google-news-300-sif10` |

`raw` keeps the original word vectors. SIF-10 weights words and removes ten common directions from their vectors.
That preparation uses WikiText-103 and happens before saving. Encoding then uses the finished table, without an extra SIF step.
SIF-10 keeps the same table shape as raw. We have no published file sizes for these new Hub variants yet.

For each planned model, float32 uses the ID above. Float16 and int8 variants add `-float16` or `-int8`.
FastText Hub models are still under consideration.

TODO(DR-151): Add MTEB quality comparisons for the classic raw and SIF-10 models when measured.

You can already use your own GloVe, word2vec or FastText word vectors through the optional `fse[gensim]` adapter.
For SIF and uSIF preparation, use [the SIF and uSIF recipe](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/01-sif-usif.ipynb).
uSIF stays a recipe, with no planned Hub upload. Try it on your data rather than assuming it improves every model.

## More

- [Documentation](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/ARCHITECTURE.md): supported APIs, model formats, exports and guarantees.
- [Contributing](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/CONTRIBUTING.md): AI-assisted contributions are welcome with reproducible checks and evidence.
- [Changelog](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/CHANGELOG.md) and [security policy](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/SECURITY.md).

fse 2.0 uses the [MIT License](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/LICENSE).
The 1.x releases remain GPL-3.0. Model and dataset licenses apply separately.
Built by [Dr. Oliver Borchers](https://www.drborchers.com).

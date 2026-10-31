<p align="center">
<a href="https://github.com/oborchers/Fast_Sentence_Embeddings/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/oborchers/Fast_Sentence_Embeddings/actions/workflows/ci.yml/badge.svg?branch=master"></a>
<a href="https://pypi.org/project/fse/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fse.svg"></a>
<a href="https://pypi.org/project/fse/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/fse.svg"></a>
<a href="https://pepy.tech/project/fse"><img alt="Downloads" src="https://pepy.tech/badge/fse"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
<a href="https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/oborchers/Fast_Sentence_Embeddings.svg?style=flat"></a>
</p>
<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/master/media/logo/fse-logo-dark.svg">
<img alt="fse" width="200" src="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/master/media/logo/fse-logo.svg">
</picture>
</p>

Fast Sentence Embeddings
==================================

Status
------------

**fse 2.0 is in development on master** and not released yet; this README describes it, and its API may still change before 2.0.0. The current release is [fse 1.1.0 on PyPI](https://pypi.org/project/fse/), in maintenance mode on the branch `1.x` (bug and compatibility fixes only), with its own [README](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/1.x/README.md) and notebooks. 2.0 changes the API and the file format of saved models. Feature requests are closed until then; bug reports are welcome.

If you start a new project, also look at actively maintained libraries such as [sentence-transformers](https://github.com/huggingface/sentence-transformers) or [model2vec](https://github.com/MinishLab/model2vec).

About
------------

fse 2.0 encodes text with finished static embedding models: a model (`fse.Encoder`) is one `.safetensors` file that holds a table of word rows, its tokenizer and whether every row is normalized. A sentence becomes the mean of the rows of its words, computed in parallel by a native kernel, bit for bit the same on every CPU. fse 2.0 trains nothing: weights, removed components or projections of a model (such as SIF or uSIF) are folded into its rows before it is saved. Vectors go into memory, or into a store on disk that `fse.VectorStore` searches.

Installation
------------

2.0 is not on PyPI yet. From a checkout of master, with [uv](https://docs.astral.sh/uv/) and a C compiler:

    make install

`fse.gensim` (converting gensim word vectors) needs the extra `fse[gensim]`.

On Apple silicon with macOS 14 or newer, the extra `fse[mlx]` adds the backend `"mlx"`, which averages the word vectors on the GPU within a small tolerance of the CPU's rows (usually with their bits): `encoder.encode(texts, backend="mlx")`. For tiny inputs it is slower than the default `"cpu"`.

Usage
------------

Load a finished model and encode:

```python
import fse

encoder = fse.Encoder.load("model.safetensors")
rows = encoder.encode(["the cat sleeps", "a dog runs"])  # one float32 row per text
```

Use a Model2Vec model of the Hub, converted when it is loaded:

```python
import fse

encoder = fse.Encoder.from_pretrained("minishlab/potion-base-8M")
rows = encoder.encode(["the cat sleeps", "a dog runs"])  # normalized, 256 values per text
encoder.save("potion-base-8M.safetensors")  # an fse encoder, loaded without the Hub
multilingual = fse.Encoder.from_pretrained("minishlab/potion-multilingual-128M")  # unigram@1
```

fse gives the token ids of the model's tokenizer (Hugging Face tokenizers 0.23.3) and the vectors of model2vec 0.10.0 within float32 rounding (not bit for bit: the sums run in another order). Like model2vec, it reads the first 512 tokens of the first 512 times the median token length characters of a text; `from_pretrained(..., truncate=False)` reads texts whole, and `max_length=` sets another token limit. Only float32 models are read, and a row whose sum of squares underflows or overflows float32 raises where model2vec returns a row that is not of unit length. Files of the Hub are checked against the commit they come from (docs/architecture/hub.md, "Model2Vec models and from_pretrained"); the model's license applies to its use.

### Switch from model2vec

`fse.StaticModel` has the API of `model2vec.StaticModel`, so code written for it needs only another import:

```python
from fse import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")
row = model.encode("the cat sleeps")                 # a vector of 256 float32 values
rows = model.encode(["the cat sleeps", "a dog runs"], max_length=128, normalize=False)
tokens = model.encode_as_sequence("the cat sleeps")  # one row of the table per token
ids = model.tokenize(["the cat sleeps"])              # [[...]], the unknown id dropped
model.max_length = 64                                 # the default of encode from now on
```

Models are read with the rules of model2vec 0.10.0 (its folder layouts, its config, `[UNK]` dropped, the cut at `max_length` tokens and at `max_length` times the median token length characters), also the static models of sentence-transformers, which `fse.Encoder.from_pretrained` reads with the rules of sentence-transformers instead. Rows equal model2vec's within float32 rounding. `model.encoder` is the `fse.Encoder` of the model's settings, for `ShardSink`, `VectorStore` and the heads.

| model2vec | fse.StaticModel |
|---|---|
| unknown keywords to `encode` are ignored (`**kwargs`) | raise `TypeError`; `convert_to_numpy=True`, `convert_to_tensor=False` and `device=None` or `"cpu"` are accepted |
| `show_progress_bar`, `batch_size`, `use_multiprocessing`, `multiprocessing_threshold` | accepted and checked; they change no row (`use_multiprocessing=False` encodes in one thread) |
| `quantize_to` float16, int8, float64 | raise: float32 tables only |
| `vocabulary_quantization`, models with `weights` or `mapping` | raise |
| `max_length=0`, `dimensionality=0` (ignored or empty) | raise |
| a row whose sum of squares under- or overflows float32 | raises |
| `embedding` is writable | read-only |
| `encode([])` raises in numpy | gives a matrix of shape `(0, dim)` |
| `encode_as_sequence` of a text without tokens is a float64 `(0, dim)` array | float32 `(0, dim)`; the other rows are model2vec's bit for bit |
| `tokenizer` is a `tokenizers.Tokenizer` | a read-only facade: `encode` and `encode_batch` give `.ids` (unknown id kept, cut at `max_length`), `get_vocab`, `get_vocab_size`, `token_to_id`, `id_to_token`; no offsets, no `.tokens`, and `add_special_tokens` must be `False` |
| `save_pretrained`, `push_to_hub` | not yet provided |

Checks of the type (`isinstance(model, model2vec.StaticModel)`) or of its name in other libraries need a change there.

A static model of sentence-transformers (one StaticEmbedding) is converted the same way, with the rules of its `encode`: every token id is pooled, `[UNK]` included, texts are read whole, and `truncate_dim` keeps the first dimensions of a Matryoshka model, as sentence-transformers' own `truncate_dim` does:

```python
import fse

encoder = fse.Encoder.from_pretrained(
    "sentence-transformers/static-retrieval-mrl-en-v1", truncate_dim=256
)
rows = encoder.encode(["what is a panda?", "can lavender grow indoors?"])  # 256 values each
```

The rows equal those of sentence-transformers 6.2.0 within float32 rounding. Models with a Normalize or Dense module after the StaticEmbedding, or with prompts, are refused with the reason; `normalize=True` normalizes the rows with fse's own `normalize@1` after the cut.

Make a model from word vectors (every word weighs 1) and save it:

```python
import fse
from fse.gensim import from_gensim
from gensim.models import KeyedVectors

wv = from_gensim(KeyedVectors.load("glove-wiki-gigaword-300.model"))
encoder = fse.Encoder.from_word_vectors(wv)  # tokenizer fse_words@1, rows normalized
encoder.save("glove.safetensors")  # one file with the table
```

Share a model as a Hugging Face folder (`config.json`, `model.safetensors` and a `README.md` model card). Any fse model can take this way: one converted from a Model2Vec or sentence-transformers model, one made from word vectors, or a classifier or regressor on such an encoder. Convert it, save or push it, and load it back from the folder or the repository:

```python
import fse

encoder = fse.Encoder.from_pretrained("minishlab/potion-base-8M")  # converted into fse's format
encoder.save_pretrained("potion-base-8M-fse")  # a new folder, written atomically
encoder.push_to_hub("you/potion-base-8M-fse")  # one commit; the license of the source comes along
encoder = fse.Encoder.from_pretrained("you/potion-base-8M-fse")  # the same fingerprint and rows
clf.push_to_hub("you/my-classifier", license="apache-2.0", extra_files={"LICENSE": "LICENSE"})
clf = fse.Classifier.from_pretrained("you/my-classifier", revision="main")
```

The card states the license the model records from its source (a converted model), or `license=` if it records none; `push_to_hub` needs one. It publishes the folder in one commit on top of the commit it read, leaves files of the same content and license files already there alone, refuses other content under the same names and other files unless `replace=True`, and checks every file at the new commit. `fse.Classifier` and `fse.Regressor` have the same three methods (`standalone=True` saves the folded table).

Which call to use when:

| You want to | Call |
|---|---|
| load a model | `encoder = fse.Encoder.load("model.safetensors")` (`mmap=True` maps it read-only) |
| share a model | `encoder.save_pretrained("folder")`, `encoder.push_to_hub("you/model")`; read with `from_pretrained` |
| make a model of word vectors | `fse.Encoder.from_word_vectors(wv)`, or `fse.Encoder.from_arrays(vectors, words=words)` |
| keep the mean as it is | `normalize=False` |
| encode texts in a list | `rows = encoder.encode(texts)` |
| encode token lists you made yourself | a model with `tokenizer="external"` and lists of lists of str |
| encode a text file, one text per line | `encoder.encode(Path("texts.txt"))` |
| encode a column of a table, read in place | `encoder.encode(table["text"])` for a pyarrow column, a pandas 3 or Polars Series, or `dataset.with_format("arrow")["text"]` of Hugging Face datasets |
| encode a stream (a generator, a queue) | `encoder.encode(iterator)`: it is read once |
| encode a stream batch by batch and confirm what is safe | `for batch in encoder.encode_batches(polls, sink=sink)`, confirming up to `batch.durable_rows` |
| write the vectors to disk | `with fse.ShardSink("store", encoder, dtype="int8") as sink: encoder.encode(texts, sink=sink)` |
| search them | `with fse.VectorStore.open("store") as store: result = store.search(encoder.encode(["query"]), k=10)` |

- `encode` returns float32 rows by default; `dtype="float16"` gives float16 rows and `dtype="int8"` a `fse.QuantizedRows` (int8 codes and one scale per row).
- `store.search` returns a `fse.SearchResult` with the `ids` (positions in the store) and the cosine `scores`, best first. Its default `mode="auto"` takes the exact arithmetic of fse for up to 8 queries at once and BLAS above, whose bits depend on how many queries are asked together; `mode="exact"` gives the same results for every batch.
- Texts are tokenized by `fse_words@1` by default: lowercase, split at whitespace and punctuation, contractions expanded (docs/architecture/tokenizers.md, "Tokenizers"). Models converted by `from_pretrained` keep their own subword tokenizer: `wordpiece@1` (POTION, the static models of sentence-transformers) or `unigram@1` (potion-multilingual-128M), with the ids of Hugging Face tokenizers 0.23.3.
- FastText word vectors keep their ngram rows: an unknown word adds the rows of its first `max_ngrams` ngrams (default 40) with the weight `oov_weight` (default 1).

### Classify and score text

fse runs linear heads trained elsewhere; it trains none. Train in scikit-learn on fse's vectors, then import the head:

```python
import fse
from sklearn.linear_model import LogisticRegression

encoder = fse.Encoder.load("model.safetensors")  # or from_word_vectors(wv, normalize=False)
x = encoder.encode(train_texts)
head = LogisticRegression(max_iter=1000).fit(x, train_labels)

clf = fse.Classifier.from_sklearn(encoder, head)  # reads coef_, intercept_, classes_
clf.predict(texts)  # labels, as head.predict(encoder.encode(texts)) gives them
clf.predict_proba(texts)  # float32 (n, classes)
for batch in clf.predict_batches(polls, output="proba"):  # a stream, batch by batch
    handle(batch.outputs)
clf.save("classifier.safetensors")  # encoder and head
clf.save("filter.safetensors", standalone=True)  # the folded table and the bias only
```

- Train on raw means (an encoder with `normalize=False`) when you only need the predictions: such a head folds into the table (every word row times A), so predicting reads k values per word instead of d and is faster than `encode`. A head trained on normalized vectors runs on the full mean; `normalized=` says which one a head is, by default the encoder's setting.
- `fse.Regressor` does the same for regression heads (`predict`); `from_arrays(encoder, weight, bias, labels=...)` takes a head of your own.
- `predict` and `predict_proba` also take a `fse.VectorStore` of the encoder's rows (store mode: the texts are not encoded again), and `predict_batches(..., return_vectors=True)` gives the vectors with the predictions from one pooling.
- The predictions are bit-identical across CPU backends, worker counts and runs. The folded head and the vectors agree with scikit-learn within float32 rounding: on Amazon Reviews Polarity (GloVe 300, LogisticRegression on 400,000 reviews, 400,000 test reviews) fse and scikit-learn give the same label for all but one review on raw means (its logit 7.4e-7 from the decision boundary) and for every review on normalized vectors, an accuracy of 0.8299 and 0.8284, and probabilities within 1.2e-6.

What 2.0 no longer has, against fse 1.1.0 and the earlier development versions of 2.0:

- **training**: `Average`, `SIF` and `uSIF` with `train` or `fit`, and any learning from a corpus. SIF and uSIF tables are made offline; a notebook with the 1.1.0 recipe (word weights and the removal of principal components, folded into the table) follows before the release.
- **stored recipes and learned components**: a model holds its finished table, not the recipe, the weights or the components it was made with.
- **word frequencies**: word vectors hold no counts, and `from_gensim` takes none.
- **shared external tables**: every model holds its own table, so two models of the same word vectors hold two copies.
- **`Corpus`**: encode takes lists, paths and streams directly.
- Files written by a development version of 2.0 before this change are refused (`fse.UnsupportedArtifactError`); rebuild them.

### Logging

fse logs through the standard library on five loggers below the logger `fse` (named "fse.artifact", "fse.hub", "fse.backend", "fse.runtime" and "fse.store") and configures nothing itself: without a configuration only its warnings (a slower fallback, a temporary file it could not remove) reach stderr. To see what it does:

```python
import fse

fse.enable_logging()  # or "DEBUG" for plans, summaries and the fields of each record
encoder = fse.Encoder.from_pretrained("minishlab/potion-base-8M")
fse.disable_logging()
```

An application with its own logging attaches its handler to the logger `fse` instead; every record carries the attribute `fse_event` and its fields as `fse_<name>` (docs/architecture/logging.md, "Logging", lists them), and none holds a path, a text, a token of the Hub or a repository id. Downloads of the Hub show the progress bars of huggingface_hub (`HF_HUB_DISABLE_PROGRESS_BARS=1` hides them).

ARCHITECTURE.md indexes docs/architecture/, which describes the design and the numerical contract, CHANGELOG.md every change against 1.1.0.

Results
------------

These are results of fse 1.1.0, not of 2.0 models; the offline SIF and uSIF recipe will be checked against them. Pearson correlation (x100) on the test set of the STS Benchmark, computed with the procedure of [`notebooks/STS-Benchmarks.ipynb` on the branch 1.x](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/1.x/notebooks/STS-Benchmarks.ipynb) (`lang_freq="en"`) with fse 1.1.0 on Python 3.13, numpy 2.5, scipy 1.18, gensim 4.4, wordfreq 3.1 and nltk 3.10. SIF and uSIF compute their principal components exactly since 1.1.0; against the randomized SVD of scikit-learn 1.9 used before, in the same environment, SIF-10 scores move by at most 0.04 points and uSIF scores not at all. Compared with the table published for fse 1.0, uSIF scores rise by 0.2 to 1.4 points. Most of that comes from a fix in 1.1.0 to how uSIF weights the removed principal components, worth 0.2 to 1.3 points when both formulas run in this environment (see [CHANGELOG.md](CHANGELOG.md)). Newer dependency versions shift SIF scores by up to 0.3 points and CBOW scores by up to 0.04 points.

Model | Vectors | params | [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results)
:---: | :---: | :---: | :---:
`CBOW` | `paranmt-300` |  | 79.82
`uSIF` | `paranmt-300` | length=11 | 79.20
`SIF-10` | `paranmt-300` | components=10 | 76.74
`SIF-10` | `paragram-300-sl999` | components=10 | 74.29
`SIF-10` | `paragram-300-ws353` | components=10 | 74.09
`uSIF` | `paragram-300-sl999` | length=11 | 73.60
`SIF-10` | `fasttext-crawl-subwords-300` | components=10 | 73.58
`uSIF` | `paragram-300-ws353` | length=11 | 72.63
`SIF-10` | `fasttext-wiki-news-subwords-300` | components=10 | 72.24
`SIF-10` | `glove-twitter-200` | components=10 | 71.67
`SIF-10` | `glove-wiki-gigaword-300` | components=10 | 71.43
`SIF-10` | `word2vec-google-news-300` | components=10 | 71.18
`SIF-10` | `glove-wiki-gigaword-200` | components=10 | 70.73
`uSIF` | `fasttext-crawl-subwords-300` | length=11 | 70.02
`SIF-10` | `glove-twitter-100` | components=10 | 69.70
`uSIF` | `fasttext-wiki-news-subwords-300` | length=11 | 69.33
`uSIF` | `glove-wiki-gigaword-300` | length=11 | 68.82
`SIF-10` | `glove-wiki-gigaword-100` | components=10 | 68.43
`uSIF` | `glove-wiki-gigaword-200` | length=11 | 68.13
`uSIF` | `glove-twitter-200` | length=11 | 67.60
`uSIF` | `word2vec-google-news-300` | length=11 | 67.60
`uSIF` | `glove-wiki-gigaword-100` | length=11 | 66.35
`SIF-10` | `glove-twitter-50` | components=10 | 65.57
`uSIF` | `glove-twitter-100` | length=11 | 65.41
`uSIF` | `paragram-25` | length=11 | 64.60
`SIF-10` | `glove-wiki-gigaword-50` | components=10 | 64.20
`uSIF` | `glove-wiki-gigaword-50` | length=11 | 63.37
`uSIF` | `glove-twitter-50` | length=11 | 61.80
`CBOW` | `word2vec-google-news-300` |  | 61.54
`SIF-10` | `paragram-25` | components=10 | 59.18
`uSIF` | `glove-twitter-25` | length=11 | 56.30
`CBOW` | `paragram-300-ws353` |  | 54.72
`SIF-10` | `glove-twitter-25` | components=10 | 54.41
`CBOW` | `paragram-300-sl999` |  | 51.46
`CBOW` | `fasttext-crawl-subwords-300` |  | 48.53
`CBOW` | `glove-wiki-gigaword-300` |  | 44.46
`CBOW` | `glove-wiki-gigaword-200` |  | 42.40
`CBOW` | `paragram-25` |  | 40.13
`CBOW` | `glove-wiki-gigaword-100` |  | 38.12
`CBOW` | `glove-wiki-gigaword-50` |  | 37.47
`CBOW` | `glove-twitter-200` |  | 34.94
`CBOW` | `glove-twitter-100` |  | 33.81
`CBOW` | `glove-twitter-50` |  | 30.78
`CBOW` | `glove-twitter-25` |  | 26.15
`CBOW` | `fasttext-wiki-news-subwords-300` |  | 26.08

Literature
-------------

1. Iyyer M, Manjunatha V, Boyd-Graber J, Daumé III H (2015) Deep Unordered
Composition Rivals Syntactic Methods for Text Classification. Proc. 53rd Annu.
Meet. Assoc. Comput. Linguist. 7th Int. Jt. Conf. Nat. Lang. Process., 1681-1691.

2. Arora S, Liang Y, Ma T (2017) A Simple but Tough-to-Beat Baseline for Sentence
Embeddings. Int. Conf. Learn. Represent. (Toulon, France), 1-16.

3. Ethayarajh K (2018) Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline.
Proceedings of the 3rd Workshop on Representation Learning for NLP. (Toulon, France), 91-100.

4. Eneko Agirre, Daniel Cer, Mona Diab, Iñigo Lopez-Gazpio, Lucia Specia. Semeval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation. Proceedings of SemEval 2017.



Changelog
-------------

See [CHANGELOG.md](CHANGELOG.md).

Development
-------------

The project uses [uv](https://docs.astral.sh/uv/) and a Makefile:

    make install   # create the environment and build the extension
    make check     # format check, lint, type check and tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for details and [SECURITY.md](SECURITY.md) for reporting vulnerabilities.


Author
-------------

Built by [Dr. Oliver Borchers](https://www.drborchers.com). For questions that do not belong in a GitHub issue, use the [contact page](https://www.drborchers.com/en/contact/).

Copyright (C) 2019 to 2026 Oliver Borchers, licensed under the [MIT License](LICENSE).

Citation
-------------

If you found this software useful, please cite it in your publication.

	@misc{Borchers2019,
		author = {Borchers, Oliver},
		title = {Fast sentence embeddings},
		year = {2019},
		publisher = {GitHub},
		journal = {GitHub Repository},
		howpublished = {\url{https://github.com/oborchers/Fast_Sentence_Embeddings}},
	}

fse 2.0 uses the MIT license. The `1.x` branch and existing 1.x releases remain GPL-3.0.
Model licenses apply separately: GloVe uses PDDL, word2vec uses Apache-2.0, fastText vectors use CC BY-SA 3.0, and POTION uses MIT. The licenses of paranmt and paragram are unknown. The third-party STS Benchmark data in `evaluation/` keeps the terms in `evaluation/LICENSE.txt`.

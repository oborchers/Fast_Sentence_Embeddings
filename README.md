<p align="center">
<a href="https://github.com/oborchers/Fast_Sentence_Embeddings/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/oborchers/Fast_Sentence_Embeddings/actions/workflows/ci.yml/badge.svg?branch=master"></a>
<a href="https://pypi.org/project/fse/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fse.svg"></a>
<a href="https://pypi.org/project/fse/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/fse.svg"></a>
<a href="https://pepy.tech/project/fse"><img alt="Downloads" src="https://pepy.tech/badge/fse"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
<a href="https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/LICENSE"><img alt="License: GPL-3.0" src="https://img.shields.io/github/license/oborchers/Fast_Sentence_Embeddings.svg?style=flat"></a>
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

**fse 2.0 is in development on master** and not released yet; this README describes it, and its API may still change before 2.0.0. The current release is [fse 1.1.0 on PyPI](https://pypi.org/project/fse/), in maintenance mode on the branch `1.x` (bug and compatibility fixes only), with its own [README](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/1.x/README.md). 2.0 changes the API and the file format of saved models; a way to export 1.x models comes with a later 1.x release. Feature requests are closed until then; bug reports are welcome.

If you start a new project, also look at actively maintained libraries such as [sentence-transformers](https://github.com/huggingface/sentence-transformers) or [model2vec](https://github.com/MinishLab/model2vec).

About
------------

fse computes sentence vectors from word vectors: the average of a sentence's word vectors (`fse.Average`), smooth inverse frequency (`fse.SIF`) and its unsupervised variant (`fse.USIF`). In 2.0, a recipe is fitted once into a frozen `fse.Encoder`, saved as one `.safetensors` file, and the encoder turns sentences into vectors in parallel with a native kernel. The learned steps of SIF and uSIF are folded into the encoder's word vectors by default, so a fitted SIF encoder encodes at the cost of an average. Vectors go into memory, or into a store on disk that `fse.VectorStore` searches.

Installation
------------

2.0 is not on PyPI yet. From a checkout of master, with [uv](https://docs.astral.sh/uv/) and a C compiler:

    make install

`fse.gensim` (converting gensim word vectors) needs the extra `fse[gensim]`.

On Apple silicon with macOS 14 or newer, the extra `fse[mlx]` adds the backend `"mlx"`, which averages the word vectors on the GPU within a small tolerance of the CPU's rows (usually with their bits): `encoder.encode(texts, backend="mlx")`. For tiny inputs it is slower than the default `"cpu"`.

Usage
------------

```python
from pathlib import Path

import fse
from fse.gensim import from_gensim
from gensim.models import KeyedVectors

wv = from_gensim(KeyedVectors.load("glove-wiki-gigaword-300.model"))  # with its word counts
wv.save("glove.safetensors")  # later: fse.WordVectors.load("glove.safetensors")

encoder = fse.SIF().fit(wv, Path("corpus.txt"))  # learns from a corpus, once
rows = encoder.encode(["the cat sleeps", "a dog runs"])  # one float32 row per text
```

Which call to use when:

| You want to | Call |
|---|---|
| fit an encoder that learns nothing from a corpus (Average without `dim`) | `encoder = fse.Average().fit(wv)` |
| fit SIF or uSIF on a corpus | `encoder = fse.SIF().fit(wv, texts)`; `counts=` gives word counts the vectors lack |
| encode texts in a list | `rows = encoder.encode(texts)` |
| use the same texts more than once (fit, then encode) | `corpus = fse.Corpus(texts)`, then `fit(wv, corpus)` and `encode(corpus)` |
| encode token lists you made yourself | a recipe with `tokenizer="external"` and lists of lists of str (or a `Corpus` of them) |
| encode a text file, one text per line | `encoder.encode(Path("texts.txt"))` |
| encode a stream (a generator, a queue) | `encoder.encode(iterator)`: it is read once, so fit and encode need a `Corpus`, a path or two fresh iterators |
| encode a stream batch by batch and confirm what is safe | `for batch in encoder.encode_batches(polls, sink=sink)`, confirming up to `batch.durable_rows` |
| write the vectors to disk | `with fse.ShardSink("store", encoder, dtype="int8") as sink: encoder.encode(texts, sink=sink)` |
| search them | `with fse.VectorStore.open("store") as store: result = store.search(encoder.encode(["query"]), k=10)` |
| save and load an encoder | `encoder.save("encoder.safetensors")`, `fse.Encoder.load("encoder.safetensors")` |

- `encode` returns float32 rows by default; `dtype="float16"` gives float16 rows and `dtype="int8"` a `fse.QuantizedRows` (int8 codes and one scale per row).
- A folded encoder (`encoder.folded`: by default when the recipe learned components or a projection) holds its own word vectors and loads with `fse.Encoder.load(path)`. An unfolded one, such as `fse.Average()` without `dim` or `fse.USIF(components=0)`, needs the word vectors it was fitted on: `fse.Encoder.load(path, wv)`.
- `store.search` returns a `fse.SearchResult` with the `ids` (positions in the store) and the cosine `scores`, best first. Its default `mode="auto"` takes the exact arithmetic of fse for up to 8 queries at once and BLAS above, whose bits depend on how many queries are asked together; `mode="exact"` gives the same results for every batch.
- Texts are tokenized by `fse_words@1` by default: lowercase, split at whitespace and punctuation, contractions expanded (ARCHITECTURE.md, "Tokenizers").

ARCHITECTURE.md describes the design and the numerical contract, CHANGELOG.md every change against 1.1.0.

Results
------------

Pearson correlation (x100) on the test set of the STS Benchmark, computed with the procedure of `notebooks/STS-Benchmarks.ipynb` (`lang_freq="en"`) with fse 1.1.0 on Python 3.13, numpy 2.5, scipy 1.18, gensim 4.4, wordfreq 3.1 and nltk 3.10. SIF and uSIF compute their principal components exactly since 1.1.0; against the randomized SVD of scikit-learn 1.9 used before, in the same environment, SIF-10 scores move by at most 0.04 points and uSIF scores not at all. Compared with the table published for fse 1.0, uSIF scores rise by 0.2 to 1.4 points. Most of that comes from a fix in 1.1.0 to how uSIF weights the removed principal components, worth 0.2 to 1.3 points when both formulas run in this environment (see [CHANGELOG.md](CHANGELOG.md)). Newer dependency versions shift SIF scores by up to 0.3 points and CBOW scores by up to 0.04 points.

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

Copyright (C) 2019 to 2026 Oliver Borchers, licensed under the [GNU General Public License v3.0](LICENSE).

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

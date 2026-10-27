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

Maintenance mode
------------

**fse is no longer actively developed.** It stays available and installable, and it receives compatibility fixes and maintainer-planned scaling work when time allows, but no feature work on request.

- Version 1.1.0 is tested on Python 3.11 to 3.13 with current numpy, scipy, gensim and huggingface-hub releases.
- Python 3.14 and newer are not supported yet, because gensim does not build there (see [gensim#3640](https://github.com/piskvorky/gensim/issues/3640)).
- Bug reports about compatibility are welcome. Feature requests will be closed.

If you start a new project, also look at actively maintained libraries such as [sentence-transformers](https://github.com/huggingface/sentence-transformers) or [model2vec](https://github.com/MinishLab/model2vec).

About
------------

Fast Sentence Embeddings is a Python library that serves as an addition to Gensim. This library is intended to compute *sentence vectors* for large collections of sentences or documents with as little hassle as possible:

```python
from fse import Vectors, Average, IndexedList

vecs = Vectors.from_pretrained("glove-wiki-gigaword-50")
model = Average(vecs)

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model.train(IndexedList(sentences))

model.sv.similarity(0,1)
```

Audience
------------

This package builds upon Gensim and is intended to compute sentence/paragraph vectors for large databases. Use this package if:
- (Sentence) Transformers are too slow
- Your dataset is too large for existing solutions (spacy)
- Using GPUs is not an option.

The average (online) inference time for a well optimized (and batched) sentence-transformer is around 1ms-10ms per sentence. If that is not enough and you are willing to sacrifice a bit in terms of quality, this is your package.

Features
------------

Find the corresponding blog post(s) here (code may be outdated):

- [Visualizing 100,000 Amazon Products](https://towardsdatascience.com/vis-amz-83dea6fcb059)
- [Sentence Embeddings. Fast, please!](https://towardsdatascience.com/fse-2b1ffa791cf9)

**fse** implements three algorithms for sentence embeddings. You can choose
between *unweighted sentence averages*,  *smooth inverse frequency averages*, and *unsupervised smooth inverse frequency averages*.

Key features of **fse** are:

**[X]** Up to 500.000 sentences / second (1)

**[X]** Provides HUB access to various pre-trained models for convenience

**[X]** Supports Average, SIF, and uSIF Embeddings

**[X]** Full support for Gensims Word2Vec and all other compatible classes

**[X]** Full support for Gensims FastText with out-of-vocabulary words

**[X]** Induction of word frequencies for pre-trained embeddings

**[X]** Incredibly fast Cython core routines

**[X]** Dedicated input file formats for easy usage (including disk streaming)

**[X]** Ram-to-disk training for large corpora

**[X]** Disk-to-disk training for even larger corpora

**[X]** Many fail-safe checks for easy usage

**[X]** Simple interface for developing your own models

**[X]** Extensive documentation of all functions

**[X]** Optimized Input Classes

(1) May vary significantly from system to system (i.e. by using swap memory) and processing.
I regularly observe 300k-500k sentences/s for preprocessed data on my Macbook (2016).
Visit **Tutorial.ipynb** for an example.

Installation
------------

Pre-built wheels are available for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x86_64) on Python 3.11 to 3.13:

    pip install -U fse

or, with [uv](https://docs.astral.sh/uv/):

    uv add fse

On other platforms pip builds the Cython extension from the source distribution. This needs a C compiler; Cython and numpy are installed automatically during the build. To build from a clone of this repository:

    uv pip install .

If the C extension cannot be loaded, fse falls back to a much slower numpy implementation and emits a `RuntimeWarning`.

Usage
-------------

Using pre-trained models with **fse** is easy. You can just use them from the hub and download them accordingly.
They will be stored locally so you can re-use them later.

```python
from fse import Vectors, Average, IndexedList
vecs = Vectors.from_pretrained("glove-wiki-gigaword-50")
model = Average(vecs)

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model.train(IndexedList(sentences))

model.sv.similarity(0,1)
```

If your vectors are large and you don't have a lot of RAM, you can supply the `mmap` argument as follows to read the vectors from disk instead of loading them into RAM:

```python
Vectors.from_pretrained("glove-wiki-gigaword-50", mmap="r")
```

To check which vectors are on the hub, please check: https://huggingface.co/fse. For example, you will find:
- glove-twitter-25
- glove-twitter-50
- glove-twitter-100
- glove-twitter-200
- glove-wiki-gigaword-100
- glove-wiki-gigaword-300
- word2vec-google-news-300
- paragram-25
- paranmt-300
- paragram-300-sl999
- paragram-300-ws353
- fasttext-wiki-news-subwords-300
- fasttext-crawl-subwords-300 (Use with `FTVectors`)

**Only load models you trust.** `Vectors.from_pretrained`, `FTVectors.from_pretrained` and the `load` methods of all models unpickle the downloaded or stored files, and unpickling untrusted data can execute arbitrary code.

In order to use **fse** with a custom model you must first estimate a Gensim model which contains a
gensim.models.keyedvectors.KeyedVectors class, for example *Word2Vec* or *Fasttext*. Then you can proceed to compute sentence embeddings for a corpus as follows:

```python
from gensim.models import FastText
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
ft = FastText(sentences, min_count=1, vector_size=10)

from fse import Average, IndexedList
model = Average(ft)
model.train(IndexedList(sentences))

model.sv.similarity(0,1)
```

### Choosing an input class

Every model trains on `(tokens, index)` tuples; the input classes produce them. Training uses one thread unless you pass `workers`; on a machine with several cores, pass `workers=os.cpu_count()`. With two or more workers, `IndexedList`, `CIndexedList`, `SplitIndexedList`, `TextList` and `CorpusFile` are read by every worker for its own part of the input, so they scale with `workers`; `IndexedLineDocument`, `SplitCIndexedList` and the classes with your own splitting are read in Python by one producer thread, which limits them from about two to four workers on. While a model with several workers trains, fse limits the threads of BLAS (OpenBLAS, MKL, Accelerate) to share the cores, for the whole process.

Measured on 16 vCPUs of an AMD EPYC (2 million sentences of 20 tokens, 300 dimensions; the vectors are bit-identical to those of the single producer thread), in sentences per second:

| Input | Model | 1 worker | 8 workers | 16 workers |
|---|---|---|---|---|
| `IndexedList` | Average | 439k | 1.5M | 1.7M |
| `SplitIndexedList` | Average | 284k | 939k | 1.1M |
| `CorpusFile` | Average | 311k | 1.6M | 2.2M |
| `IndexedList` | SIF | 239k | 611k | 1.4M |
| `CorpusFile` | SIF | 179k | 624k | 1.7M |

With the single producer thread before #88, 8 workers reached 325k (`IndexedList`, Average) and 191k (`IndexedList`, SIF) sentences per second on the same machine. The numbers vary with the load of the machine; measure your own setup with `benchmarks/bench_train.py`.

| Your data | Input class | Splitting | Index |
|---|---|---|---|
| Tokenized sentences in memory | `IndexedList(sentences)` | already split | position |
| Raw strings in memory | `TextList(strings)` | like `str.split()`, native | position |
| A local UTF-8 text file, one sentence per line | `CorpusFile(path)` | like `str.split()`, native | line number |
| A compressed local file (`.gz`, `.bz2`, ... via smart_open) | `IndexedLineDocument(path)` | `str.split()` in Python | line number |
| Your own splitting or indices | `CSplitIndexedList`, `CIndexedList`, `CSplitCIndexedList` | your function | yours |

```python
import os

from fse import Average, CorpusFile, TextList

model = Average(ft, workers=os.cpu_count())
model.train(CorpusFile("sentences.txt"))  # line i of the file is sentence i
vectors = model.infer(TextList(["cat say meow", "dog say woof"]))
```

`CorpusFile` and `TextList` give exactly the sentences of `IndexedLineDocument` and `SplitIndexedList`: a line ends at `\n`, text is decoded strictly as UTF-8, and words are split at every character for which `str.isspace()` is true. Invalid UTF-8 raises a `UnicodeDecodeError` that names the sentence. The file must not change from the scan to the end of training: changes that alter its size, modification time or inode raise an error, but truncating a memory-mapped file while it is read can end the process on Linux.

To keep only close matches, pass `min_similarity` to `most_similar` or the `similar_by_*` methods:

```python
model.sv.most_similar(0, topn=10, min_similarity=0.8)
```

### Smaller sentence vectors

`sv_dtype="float16"` halves the memory of the sentence vectors and `sv_dtype="int8"` quarters it. The vectors are still computed in float32; they are only stored smaller, and every read (`sv[i]`, `get_vector`, `get_vectors`, `iter_blocks`) returns float32. Training, updates, `infer`, the similarity searches, `save`/`load` and memory-mapped files (`sv_mapfile_path`) work as with float32.

```python
from fse import SIF, CorpusFile

model = SIF(ft, sv_dtype="int8", sv_mapfile_path="vectors", workers=8)
model.train(CorpusFile("sentences.txt"))
model.sv.most_similar(0, topn=10)

model.sv.convert("float16")  # change the format of trained or loaded vectors
```

int8 stores one float32 scale per vector, so the cosine of two stored int8 vectors equals the cosine of the vectors they represent. On the STS Benchmark (all 15 vector sets of the table below, with CBOW, SIF-10 and uSIF), float16 changes the scores by at most 0.01 points and int8 by at most 0.02 points. Both round the vectors, so close neighbours can swap places: float16 changes at most 0.4 % and int8 up to 7 % of the 10 nearest neighbours of a sentence. Keep float32 where neighbour lists must match it exactly.

With index i for the i-th sentence (every input class except the `C*` lists with custom indices) and the compiled extension, training holds a ring of float32 windows of about 256 MB each instead of a float32 matrix of all sentences, and stores every finished window right away. Jobs hold at most 65,536 sentences, so the ring has two windows for up to 1,024 dimensions and a few more beyond. SIF and uSIF read the input twice for that, once to learn the principal components and once to store the vectors. Other inputs, and subclasses that replace the training hooks, train into a float32 matrix of all new rows first, and the log says why.

To apply a model to all vectors, for example a logistic regression, use `score`. It reads float16 and int8 vectors as stored, in parallel, and can write into a memmap:

```python
import numpy as np

logits = model.sv.score(clf.coef_.T, clf.intercept_)  # X @ W + b, in blocks of rows
probabilities = 1 / (1 + np.exp(-logits))

for first_row, block in model.sv.iter_blocks():  # float32 blocks, for anything else
    ...
```

### Smaller word vectors

`wv_dtype="float16"` or `"int8"` makes the model keep the word vectors it averages (and the ngram vectors of FastText models) in half or a quarter of the memory. The model quantizes a copy once when it is created; the word vectors you pass in stay unchanged. To keep their float32 values out of RAM as well, load them memory-mapped, for example `Vectors.from_pretrained("word2vec-google-news-300", mmap="r")`. With `wv_mapfile_path` the codes are written to a file and memory-mapped.

```python
from fse import SIF, CorpusFile, Vectors

vectors = Vectors.from_pretrained("word2vec-google-news-300", mmap="r")
model = SIF(vectors, wv_dtype="int8", sv_dtype="int8", workers=8)
model.train(CorpusFile("sentences.txt"))
```

On the STS Benchmark (all 15 vector sets, with CBOW, SIF-10 and uSIF), float16 word vectors change the scores by at most 0.01 points and int8 by at most 0.12 points (`glove-twitter-25` with SIF-10; every 300-dimensional set stays within 0.02 points).

The main benefit is memory. Averaging also gets somewhat faster where fse reads the codes with vector instructions: on x86-64 Linux and macOS with AVX2, FMA and F16C (Intel since Haswell, AMD since Zen), and on Apple Silicon and other arm64 CPUs with NEON; fse picks the routines when it is imported. Measured end to end (training from a `CorpusFile` of 2 million sentences, 300 dimensions, Average and SIF), int8 word vectors were, compared with float32 word vectors in fse 1.0:

| Machine | 1 worker | 8 workers |
|---|---|---|
| Apple M1 Pro, 400,000 words | 1.4 to 1.5 times as fast | 1.2 to 1.4 times as fast |
| AMD EPYC (Zen 5, shared cloud machine), 400,000 and 3 million words | 1.0 to 1.4 times as fast | 0.7 to 1.15 times, mostly about the same |

On x86-64 CPUs without AVX2 and on Windows, fse uses portable routines instead: float16 and int8 then save memory but train 20 to 35 % slower (measured on the AMD machine with the portable routines forced). float32 itself keeps its results unchanged; prefetching the rows of the next words made it up to about 1.2 times as fast.

Additional Information
-------------

Within the folder notebooks you can find the following guides. They were written for older versions of fse and gensim and have not been re-run against 1.1.0, so some calls may need small changes:

**Tutorial.ipynb** offers a detailed walk-through of some of the most important functions fse has to offer.

**STS-Benchmarks.ipynb** contains an example of how to use the library with pre-trained models to
replicate the STS Benchmark results [4] reported in the papers.

**Speed Comparision.ipynb** compares the speed between the numpy and the cython routines.

In order to use the **fse** model, you first need some pre-trained gensim
word embedding model, which is then used by **fse** to compute the sentence embeddings.

After computing sentence embeddings, you can use them in supervised or
unsupervised NLP applications, as they serve as a formidable baseline.

The models presented are based on
- Deep-averaging embeddings [1]
- Smooth inverse frequency embeddings [2]
- Unsupervised smooth inverse frequency embeddings [3]

Credits to Radim Řehůřek and all contributors for the **awesome** library
and code that [Gensim](https://github.com/piskvorky/gensim) provides. A whole lot of the code found in this lib is based on Gensim.

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

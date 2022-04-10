<p align="center">
<a href="https://travis-ci.com/oborchers/Fast_Sentence_Embeddings"><img alt="Build Status" src="https://travis-ci.com/oborchers/Fast_Sentence_Embeddings.svg?branch=master"></a>
<a href="https://coveralls.io/github/oborchers/Fast_Sentence_Embeddings?branch=master"><img alt="Coverage Status" src="https://coveralls.io/repos/github/oborchers/Fast_Sentence_Embeddings/badge.svg?branch=master"></a>
<a href="https://pepy.tech/project/fse"><img alt="Downloads" src="https://pepy.tech/badge/fse"></a>
<a href="https://lgtm.com/projects/g/oborchers/Fast_Sentence_Embeddings/context:python"><img alt="Language grade: Python" src="https://img.shields.io/lgtm/grade/python/g/oborchers/Fast_Sentence_Embeddings.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://img.shields.io/github/license/oborchers/Fast_Sentence_Embeddings.svg?style=flat"><img alt="License: GPL3" src="https://img.shields.io/github/license/oborchers/Fast_Sentence_Embeddings.svg?style=flat"></a>
</p>
<p align="center">
<a><img alt="fse" src="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/develop/media/fse.png"></a>
</p>

Fast Sentence Embeddings
==================================

Fast Sentence Embeddings is a Python library that serves as an addition to Gensim. This library is intended to compute *sentence vectors* for large collections of sentences or documents with as little hassle as possible:

```
from fse import Vectors, Average, IndexedList

vecs = Vectors.from_pretrained("glove-wiki-gigaword-50")
model = Average(vecs)

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model.train(IndexedList(sentences))

model.sv.similarity(0,1)
```

If you want to support fse, take a quick [survey](https://forms.gle/8uSU323fWUVtVwcAA) to improve it.

Audience
------------

This package builds upon Gensim and is intenteded to compute sentence/paragraph vectors for large databases. Use this package if:
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

This software depends on NumPy, Scipy, Scikit-learn, Gensim, and Wordfreq. 
You must have them installed prior to installing fse.

As with gensim, it is also recommended you install a BLAS library before installing fse.

The simple way to install **fse** is:

    pip install -U fse

In case you want to build from source, just run:

    python setup.py install

If building the Cython extension fails (you will be notified), try:

    pip install -U git+https://github.com/oborchers/Fast_Sentence_Embeddings

Usage
-------------

Using pre-trained models with **fse** is easy. You can just use them from the hub and download them accordingly.
They will be stored locally so you can re-use them later.

```
from fse import Vectors, Average, IndexedList
vecs = Vectors.from_pretrained("glove-wiki-gigaword-50")
model = Average(vecs)

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model.train(IndexedList(sentences))

model.sv.similarity(0,1)
```

If your vectors are large and you don't have a lot of RAM, you can supply the `mmap` argument as follows to read the vectors from disk instead of loading them into RAM:

```
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

In order to use **fse** with a custom model you must first estimate a Gensim model which contains a
gensim.models.keyedvectors.BaseKeyedVectors class, for example *Word2Vec* or *Fasttext*. Then you can proceed to compute sentence embeddings for a corpus as follows:

```
from gensim.models import FastText
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
ft = FastText(sentences, min_count=1, vector_size=10)

from fse import Average, IndexedList
model = Average(ft)
model.train(IndexedList(sentences))

model.sv.similarity(0,1)
```

fse offers multi-thread support out of the box. However, for most applications a *single thread will most likely be sufficient*.

Additional Information
-------------

Within the folder nootebooks you can find the following guides:

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
and code that [Gensim](https://github.com/RaRe-Technologies/gensim) provides. A whole lot of the code found in this lib is based on Gensim.

To install **fse** on Colab, check out: https://colab.research.google.com/drive/1qq9GBgEosG7YSRn7r6e02T9snJb04OEi 

Results
------------

Model | Vectors | params | [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results)
:---: | :---: | :---: | :---:
`CBOW` | `paranmt-300` |  | 79.82
`uSIF` | `paranmt-300` | length=11 | 79.00
`SIF-10` | `paranmt-300` | components=10 | 76.72
`SIF-10` | `paragram-300-sl999` | components=10 | 74.21
`SIF-10` | `paragram-300-ws353` | components=10 | 74.03
`SIF-10` | `fasttext-crawl-subwords-300` | components=10 | 73.38
`uSIF` | `paragram-300-sl999` | length=11 | 73.04
`SIF-10` | `fasttext-wiki-news-subwords-300` | components=10 | 72.29
`uSIF` | `paragram-300-ws353` | length=11 | 71.84
`SIF-10` | `glove-twitter-200` | components=10 | 71.62
`SIF-10` | `glove-wiki-gigaword-300` | components=10 | 71.35
`SIF-10` | `word2vec-google-news-300` | components=10 | 71.12
`SIF-10` | `glove-wiki-gigaword-200` | components=10 | 70.62
`SIF-10` | `glove-twitter-100` | components=10 | 69.65
`uSIF` | `fasttext-crawl-subwords-300` | length=11 | 69.40
`uSIF` | `fasttext-wiki-news-subwords-300` | length=11 | 68.63
`SIF-10` | `glove-wiki-gigaword-100` | components=10 | 68.34
`uSIF` | `glove-wiki-gigaword-300` | length=11 | 67.60
`uSIF` | `glove-wiki-gigaword-200` | length=11 | 67.11
`uSIF` | `word2vec-google-news-300` | length=11 | 66.99
`uSIF` | `glove-twitter-200` | length=11 | 66.67
`SIF-10` | `glove-twitter-50` | components=10 | 65.52
`uSIF` | `glove-wiki-gigaword-100` | length=11 | 65.33
`uSIF` | `paragram-25` | length=11 | 64.22
`uSIF` | `glove-twitter-100` | length=11 | 64.13
`SIF-10` | `glove-wiki-gigaword-50` | components=10 | 64.11
`uSIF` | `glove-wiki-gigaword-50` | length=11 | 62.06
`CBOW` | `word2vec-google-news-300` |  | 61.54
`uSIF` | `glove-twitter-50` | length=11 | 60.41
`SIF-10` | `paragram-25` | components=10 | 59.07
`uSIF` | `glove-twitter-25` | length=11 | 55.06
`CBOW` | `paragram-300-ws353` |  | 54.72
`SIF-10` | `glove-twitter-25` | components=10 | 54.16
`CBOW` | `paragram-300-sl999` |  | 51.46
`CBOW` | `fasttext-crawl-subwords-300` |  | 48.49
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

Changelog
-------------

1.0.0:
- Added support for gensim>=4. This library is no longer compatible with gensim<4. For migration, see the [README](https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4).
- `size` argument is now `vector_size`
- Added docs

0.2.0:
- Added `Vectors` and `FTVectors` class and hub support by `from_pretrained`
- Extended benchmark
- Fixed zero division bug for uSIF
- Moved tests out of the main folder
- Moved sts out of the main folder

0.1.17:
- Fixed dependency issue where you cannot install fse properly
- Updated readme
- Updated travis python versions (3.6, 3.9)

0.1.15 from 0.1.11:
- Fixed major FT Ngram computation bug
- Rewrote the input class. Turns out NamedTuple was pretty slow. 
- Added further unittests
- Added documentation
- Major speed improvements
- Fixed division by zero for empty sentences
- Fixed overflow when infer method is used with too many sentences
- Fixed similar_by_sentence bug

Literature
-------------

1. Iyyer M, Manjunatha V, Boyd-Graber J, Daumé III H (2015) Deep Unordered 
Composition Rivals Syntactic Methods for Text Classification. Proc. 53rd Annu. 
Meet. Assoc. Comput. Linguist. 7th Int. Jt. Conf. Nat. Lang. Process., 1681–1691.

2. Arora S, Liang Y, Ma T (2017) A Simple but Tough-to-Beat Baseline for Sentence
Embeddings. Int. Conf. Learn. Represent. (Toulon, France), 1–16.

3. Ethayarajh K (2018) Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline.
Proceedings of the 3rd Workshop on Representation Learning for NLP. (Toulon, France), 91–100.

4. Eneko Agirre, Daniel Cer, Mona Diab, Iñigo Lopez-Gazpio, Lucia Specia. Semeval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation. Proceedings of SemEval 2017.


Copyright
-------------

**Disclaimer**: I am working full time. Unfortunately, I have yet to find time to add all the features I'd like to see. Especially the API needs some overhaul and we need support for gensim 4.0.0.

I am looking for active contributors to keep this package alive. Please feel free to ping me at <o.borchers@oxolo.com> if you are interested.

Author: Oliver Borchers

Copyright (C) 2022 Oliver Borchers

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

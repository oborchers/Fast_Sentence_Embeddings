Fast Sentence Embeddings (fse)
==================================

Fast Sentence Embeddings is a Python library that serves as an addition to Gensim. This library is intended to compute *summary vectors* for large collections of sentences or documents. 


Features
------------

**fse** implements two algorithms for sentence embeddings. You can choose
between unweighted sentence averages and smooth inverse frequency averages.
In order to use the **fse** model, you first need some pre-trained embedding
model, which is then used by **fse** to compute the sentence embeddings.

After computing sentence embeddings, you can use them in supervised or
unsupervised NLP applications, as they serve as a formidable baseline.

The models here are based on the the smooth inverse frequency embeddings [1]
and the deep-averaging networks [2].


Installation
------------

This software depends on [NumPy, Scipy, Scikit-learn, Gensim, and Wordfreq]. 
You must have them installed prior to installing fse.

As with gensim, it is also recommended you install a fast BLAS library
before installing fse.

The simple way to install gensim is:

    pip install -U fse

Or, if you have instead downloaded and unzipped the [source tar.gz]
package, you’d run:

    python setup.py install

Exemplary application
-------------

In order to use **fse** you must first estimate a Gensim model which containes a
gensim.models.keyedvectors.BaseKeyedVectors class, for example 
*Word2Vec* or *Fasttext*. Then you can proceed to compute sentence embeddings
for a corpus.

The current version does not offer multi-core support out of the box.

	from gensim.models import Word2Vec
	sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
	model = Word2Vec(sentences, min_count=1)

	from fse.models import Sentence2Vec
	se = Sentence2Vec(model)
	sentences_emb = se.train(sentences)

ToDos
-------------
**[ ]** Bugfixes
**[ ]** Multi Core Implementation
**[ ]** Direct to disc-estimation to to avoid RAM overflow
**[ ]** Add as a gensim feature

Literature
-------------
1. Arora S, Liang Y, Ma T (2017) A Simple but Tough-to-Beat Baseline for Sentence
Embeddings. Int. Conf. Learn. Represent. (Toulon, France), 1–16.

2. Iyyer M, Manjunatha V, Boyd-Graber J, Daumé III H (2015) Deep Unordered 
Composition Rivals Syntactic Methods for Text Classification. Proc. 53rd Annu. 
Meet. Assoc. Comput. Linguist. 7th Int. Jt. Conf. Nat. Lang. Process., 1681–1691.

Copyright
-------------

Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
Copyright (C) 2019 Oliver Borchers

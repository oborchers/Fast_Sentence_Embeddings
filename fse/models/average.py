#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

"""This module implements the base class to compute average representations for sentences, using highly optimized C routines,
data streaming and Pythonic interfaces.

The implementation is based on Iyyer et al. (2015): Deep Unordered Composition Rivals Syntactic Methods for Text Classification.
For more information, see <https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf>.

The training algorithms is based on the Gensim implementation of Word2Vec, FastText, and Doc2Vec. 
For more information, see: :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.fasttext.FastText`, or
:class:`~gensim.models.doc2vec.Doc2Vec`.

Initialize and train a :class:`~fse.models.sentence2vec.Sentence2Vec` model

.. sourcecode:: pycon

        >>> from gensim.models.word2vec import Word2Vec
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>> model = Word2Vec(sentences, min_count=1, size=20)

        >>> from fse.models.average import Average        
        >>> avg = Average(model)
        >>> avg.train([(s, i) for i, s in enumerate(sentences)])
        >>> avg.sv.vectors.shape
        (2, 20)

"""

from __future__ import division 

from fse.models.base_s2v import BaseSentence2VecModel

from gensim.models.keyedvectors import BaseKeyedVectors
from gensim.models.utils_any2vec import ft_ngram_hashes

from numpy import ndarray, float32 as REAL, sum as np_sum, multiply as np_mult,\
    zeros, max as np_max

from typing import List

import logging

logger = logging.getLogger(__name__)

def train_average_np(model:BaseSentence2VecModel, indexed_sentences:List[tuple], target:ndarray, memory:ndarray) -> [int,int]:
    """Training on a sequence of sentences and update the target ndarray.

    Called internally from :meth:`~fse.models.average.Average._do_train_job`.

    Warnings
    --------
    This is the non-optimized, pure Python version. If you have a C compiler,
    fse will use an optimized code path from :mod:`fse.models.average_inner` instead.

    Parameters
    ----------
    model : :class:`~fse.models.base_s2v.BaseSentence2VecModel`
        The BaseSentence2VecModel model instance.
    indexed_sentences : iterable of tuple
        The sentences used to train the model.
    target : ndarray
        The target ndarray. We use the index from indexed_sentences
        to write into the corresponding row of target.
    memory : ndarray
        Private memory for each working thread

    Returns
    -------
    int, int
        Number of effective sentences (non-zero) and effective words in the vocabulary used 
        during training the sentence embedding.

    """
    size = model.wv.vector_size
    vocab = model.wv.vocab

    w_vectors = model.wv.vectors
    w_weights = model.word_weights

    s_vectors = target

    is_ft = model.is_ft

    mem = memory[0]
    subwords_idx = memory[1]

    if is_ft:
        # NOTE: For Fasttext: Use wv.vectors_vocab
        # Using the wv.vectors from fasttext had horrible effects on the sts results
        # I suspect this is because the wv.vectors are based on the averages of
        # wv.vectors_vocab + wv.vectors_ngrams, which will all point into very
        # similar directions.
        max_ngrams = model.batch_ngrams
        w_vectors = model.wv.vectors_vocab
        ngram_vectors = model.wv.vectors_ngrams
        min_n = model.wv.min_n
        max_n = model.wv.max_n
        bucket = model.wv.bucket
        oov_weight = np_max(w_weights)

    eff_sentences, eff_words = 0, 0

    if not is_ft:
        for obj in indexed_sentences:
            mem.fill(0.)
            sent = obj[0]
            sent_adr = obj[1]
            
            word_indices = [vocab[word].index for word in sent if word in vocab]
            eff_sentences += 1
            if not len(word_indices):
                continue
            eff_words += len(word_indices)

            mem += np_sum(np_mult(w_vectors[word_indices],w_weights[word_indices][:,None]) , axis=0)
            mem *= 1/len(word_indices)
            s_vectors[sent_adr] = mem.astype(REAL)
    else:
        for obj in indexed_sentences:
            mem.fill(0.)
            sent = obj[0]
            sent_adr = obj[1]
            
            if not len(sent):
                continue
            mem = zeros(size, dtype=REAL)

            eff_sentences += 1
            eff_words += len(sent) # Counts everything in the sentence

            for word in sent:
                if word in vocab:
                    word_index = vocab[word].index
                    mem += w_vectors[word_index] * w_weights[word_index]
                else:
                    ngram_hashes = ft_ngram_hashes(word, min_n, max_n, bucket, True)[:max_ngrams]
                    if len(ngram_hashes) == 0:
                        continue
                    mem += oov_weight * (np_sum(ngram_vectors[ngram_hashes], axis=0) / len(ngram_hashes))
                # Implicit addition of zero if oov does not contain any ngrams
            s_vectors[sent_adr] = mem / len(sent)

    return eff_sentences, eff_words

try:
    from fse.models.average_inner import train_average_cy
    from fse.models.average_inner import FAST_VERSION, MAX_WORDS_IN_BATCH, MAX_NGRAMS_IN_BATCH
    train_average = train_average_cy
except ImportError:
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000
    MAX_NGRAMS_IN_BATCH = 40
    train_average = train_average_np

class Average(BaseSentence2VecModel):
    """ Train, use and evaluate averaged sentence vectors.

    The model can be stored/loaded via its :meth:`~fse.models.average.Average.save` and
    :meth:`~fse.models.average.Average.load` methods.

    Some important attributes are the following:

    Attributes
    ----------
    wv : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
        This object essentially contains the mapping between words and embeddings. After training, it can be used
        directly to query those embeddings in various ways. See the module level docstring for examples.
    
    sv : :class:`~fse.models.sentencevectors.SentenceVectors`
        This object contains the sentence vectors inferred from the training data. There will be one such vector
        for each unique docusentence supplied during training. They may be individually accessed using the index.
    
    prep : :class:`~fse.models.base_s2v.BaseSentence2VecPreparer`
        The prep object is used to transform and initialize the sv.vectors. Aditionally, it can be used
        to move the vectors to disk for training with memmap.
    
    """

    def __init__(self, model:BaseKeyedVectors, sv_mapfile_path:str=None, wv_mapfile_path:str=None, workers:int=1, lang_freq:str=None, **kwargs):
        """ Average (unweighted) sentence embeddings model. Performs a simple averaging operation over all
        words in a sentences without further transformation.

        The implementation is based on Iyyer et al. (2015): Deep Unordered Composition Rivals Syntactic Methods for Text Classification.
        For more information, see <https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf>.

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors` or :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            the wv.vocab and wv.vector elements are required.
        sv_mapfile_path : str, optional
            Optional path to store the sentence-vectors in for very large datasets. Used for memmap.
        wv_mapfile_path : str, optional
            Optional path to store the word-vectors in for very large datasets. Used for memmap.
            Use sv_mapfile_path and wv_mapfile_path to train disk-to-disk without needing much ram.
        workers : int, optional
            Number of working threads, used for multithreading. For most tasks (few words in a sentence)
            a value of 1 should be more than enough.
        lang_freq : str, optional
            Some pre-trained embeddings, i.e. "GoogleNews-vectors-negative300.bin", do not contain information about
            the frequency of a word. As the frequency is required for estimating the word weights, we induce
            frequencies into the wv.vocab.count based on :class:`~wordfreq`
            If no frequency information is available, you can choose the language to estimate the frequency.
            See https://github.com/LuminosoInsight/wordfreq
        
        """

        super(Average, self).__init__(
            model=model, sv_mapfile_path=sv_mapfile_path, wv_mapfile_path=wv_mapfile_path,
            workers=workers, lang_freq=lang_freq,
            batch_words=MAX_WORDS_IN_BATCH, batch_ngrams=MAX_NGRAMS_IN_BATCH,
            fast_version=FAST_VERSION
            )

    def _do_train_job(self, data_iterable:List[tuple], target:ndarray, memory:ndarray) -> [int, int]:
        """ Internal routine which is called on training and performs averaging for all entries in the iterable """
        eff_sentences, eff_words = train_average(model=self, indexed_sentences=data_iterable, target=target, memory=memory)
        return eff_sentences, eff_words

    def _check_parameter_sanity(self, **kwargs):
        """ Check the sanity of all child paramters """
        if not all(self.word_weights == 1.): 
            raise ValueError("All word weights must equal one for averaging")

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training """
        pass

    def _post_train_calls(self, **kwargs):
        """ Function calls to perform after training, such as computing eigenvectors """
        pass
    
    def _post_inference_calls(self, **kwargs):
        """ Function calls to perform after training & inference
        Examples include the removal of components
        """
        pass
    
    def _check_dtype_santiy(self, **kwargs):
        """ Check the dtypes of all child attributes"""
        pass

    
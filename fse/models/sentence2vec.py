#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.keyedvectors import BaseKeyedVectors

from gensim.matutils import unitvec

from sklearn.decomposition import TruncatedSVD
from wordfreq import get_frequency_dict

from six.moves import xrange

import logging
import warnings
import psutil
logger = logging.getLogger(__name__)

from numpy import float32 as REAL, sum as np_sum, vstack, zeros, ones,\
        dtype, sqrt, newaxis

try:
    # Import cython functions  
    CY_ROUTINES = 1
    from fse.models.sentence2vec_inner import s2v_train
except ImportError as e:
    CY_ROUTINES = 0
    logger.warning("ImportError of Cython functions: %s", e)

    def s2v_train(sentences, wv, weights):
        """Train sentence embedding on a list of sentences.
        This methods is numpy only and is much slower than the cython variant.

        Called internally from :meth:`~fse.models.sentence2vec.Sentence2Vec.train`.

        Parameters
        ----------
        sentences : iterable of list of str
            The corpus used to train the model.
        wv : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
            The BaseKeyedVectors instance containing the vectors used for training
        weights : np.ndarray
            Weights used in the summation of the vectors

        Returns
        -------
        np.ndarray
            The sentence embedding matrix of dim len(sentences) * vector_size
        int
            Number of words in the vocabulary actually used for training.
        int 
            Number of sentences used for training.
        """

        vlookup = wv.vocab
        vectors = wv
        output = []

        effective_words = 0
        effective_sentences = 0

        for s in sentences:
            idx = [vlookup[w].index for w in s if w in vlookup]
            v = np_sum(vectors.vectors[idx] * weights[idx][:, None], axis=0)
            if len(idx) > 0:
                effective_words += len(idx)
                effective_sentences += 1
                v *= 1/len(idx)
            output.append(v)
        return vstack(output).astype(REAL), effective_words, effective_sentences
    
class Sentence2Vec():
    """Compute smooth inverse frequency weighted or averaged sentence emeddings.

    This implementation is based on the 2017 ICLR paper (https://openreview.net/pdf?id=SyK00v5xx):
    Arora S, Liang Y, Ma T (2017) A Simple but Tough-to-Beat Baseline for Sentence Embeddings. Int. Conf. Learn. Represent. (Toulon, France), 1–16.
    All corex routines are optimized based on the Gensim routines (https://github.com/RaRe-Technologies/gensim)

    Attributes
    ----------
    model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors` or :class:`~gensim.models.keyedvectors.BaseWordEmbeddingsModel`
        This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
        the wv.vocab and wv.vector elements are required.

    numpy.ndarray : sif_weights
        Contains the pre-computed SIF weights.
    """

    def __init__(self, model, alpha=1e-3, components=1, no_frequency=False, lang="en"):
        """

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors` or :class:`~gensim.models.keyedvectors.BaseWordEmbeddingsModel`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            the wv.vocab and wv.vector elements are required.
        alpha : float, optional
            Parameter which is used to weigh each individual word based on its probability p(w).
            If alpha = 1, train simply computes the averaged sentence representation.
        components : int, optional
            Number of principal components to remove from the sentence embeddings. Independent of alpha.
        no_frequency : bool, optional
            Some pre-trained embeddings, i.e. "GoogleNews-vectors-negative300.bin", do not contain information about
            the frequency of a word. As the frequency is required for estimating the weights, no_frequency induces
            into the wv.vocab.count class based on :class:`~wordfreq`
        lang : str, optional
            If no frequency information is available, you can choose the language to estimate the frequency.
            See https://github.com/LuminosoInsight/wordfreq

        Returns
        -------
        numpy.ndarray 
            Sentence embedding matrix of dim len(sentences) * dimension

        Examples
        --------
        Initialize and train a :class:`~fse.models.sentence2vec.Sentence2Vec` model

        >>> from gensim.models import Word2Vec
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>> model = Word2Vec(sentences, min_count=1)

        >>> from fse.models import Sentence2Vec
        >>> se = Sentence2Vec(model)
        >>> sentences_emb = se.train(sentences)

        """

        if isinstance(model, BaseWordEmbeddingsModel):
            self.model = model.wv
        elif isinstance(model, BaseKeyedVectors):
            self.model = model
        else:
            raise RuntimeError("Model must be child of BaseWordEmbeddingsModel or BaseKeyedVectors.")

        if not hasattr(self.model, 'vectors'):
            raise RuntimeError("Parameters required for predicting sentence embeddings not found.")

        assert alpha >= 0 & components >= 0

        self.alpha = float(alpha)
        self.components = int(components)
        self.no_frequency = bool(no_frequency)
        self.lang = str(lang)

        self.sif_weights = self._precompute_sif_weights(self.model, self.alpha, no_frequency, lang)


    def _compute_principal_component(self, vectors, npc=1):
        """Compute the n principal components for the sentence embeddings

        Notes
        -----
        Adapted from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py

        Parameters
        ----------
        vectors : numpy.ndarray
            The sentence embedding matrix of dim len(sentences) * vector_size.
        npc : int, optional
            The number of principal components to be computed. Default : 1.

        Returns
        -------
        numpy.ndarray
            The principal components as computed by the TruncatedSVD

        """
        logger.info("computing %d principal components", npc)
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0, algorithm="randomized")
        svd.fit(vectors)
        return svd.components_


    def _remove_principal_component(self, vectors, npc=1):
        """Remove the projection from the sentence embeddings

        Notes
        -----
        Adapted from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py

        Parameters
        ----------
        vectors : numpy.ndarray
            The sentence embedding matrix of dim len(sentences) * vector_size.
        npc : int, optional
            The number of principal components to be computed. Default : 1.

        Returns
        -------
        numpy.ndarray
            The sentence embedding matrix of dim len(sentences) * vector size after removing the projection

        """
        pc = self._compute_principal_component(vectors, npc)
        logger.info("removing %d principal components", npc)
        if npc==1:
            vectors_rpc = vectors - vectors.dot(pc.transpose()) * pc
        else:
            vectors_rpc = vectors - vectors.dot(pc.transpose()).dot(pc)
        return vectors_rpc


    def _precompute_sif_weights(self, wv, alpha=1e-3, no_frequency=False, lang="en"):
        """Precompute the weights used in the vector summation

        Parameters
        ----------
        wv : `~gensim.models.keyedvectors.BaseKeyedVectors`
            A gensim keyedvectors child that contains the word vectors and the vocabulary
        alpha : float, optional
            Parameter which is used to weigh each individual word based on its probability p(w).
            If alpha = 0, the model computes the average sentence embedding. Common values range from 1e-5 to 1e-1.
            For more information, see the original paper.
        no_frequency : bool, optional
            Use a the commonly available frequency table if the Gensim model does not contain information about
            the frequency of the words (see model.wv.vocab.count).
        lang : str, optional
            Determines the language of the frequency table used to compute the weights.

        Returns
        -------
        numpy.ndarray
            The vector of weights for all words in the model vocabulary

        """
        logger.info("pre-computing SIF weights")

        if no_frequency:
            logger.info("no frequency mode: using wordfreq for estimation (lang=%s)",lang)
            freq_dict = get_frequency_dict(str(lang), wordlist='best')

            for w in wv.index2word:
                if w in freq_dict:
                    wv.vocab[w].count = int(freq_dict[w] * (2**31 -1))
                else:
                    wv.vocab[w].count = 1

        if alpha > 0:
            corpus_size = 0
            # Set the dtype correct for cython estimation
            sif = zeros(shape=len(wv.vocab), dtype=REAL) 

            for k in wv.index2word:
                # Compute normalization constant
                corpus_size += wv.vocab[k].count

            for idx, k in enumerate(wv.index2word):
                pw = wv.vocab[k].count / corpus_size
                sif[idx] = alpha / (alpha+pw)
        else:
            sif = ones(shape=len(wv.vocab), dtype=REAL)

        return sif

    def _estimate_memory(self, len_sentences, vocab_size, vector_size):
        """Estimate the size of the embedding in memoy

        Notes
        -----
        Directly adapted from gensim

        Parameters
        ----------
        len_sentences : int
            Length of the sentences iterable
        vocab_size : int
            Size of the vocabulary
        vector_size : int
            Vector size of the sentence embedding

        Returns
        -------
        dict
            Dictionary of esitmated sizes

        """
        report = {}
        report["sif_weights"] = vocab_size * dtype(REAL).itemsize
        report["sentence_vectors"] = len_sentences * vector_size * dtype(REAL).itemsize
        report["total"] = sum(report.values())
        mb_size = int(report["sentence_vectors"] / 1024**2)
        logger.info(
            "estimated required memory for %i sentences and %i dimensions: %i MB (%i GB)",
            len_sentences, 
            vector_size,
            mb_size,
            int(mb_size / 1024)
        )

        if report["total"] >= 0.95 * psutil.virtual_memory()[1]:
            warnings.warn("Sentence2Vec: The sentence embeddings will likely not fit into RAM.")

        return report

    def normalize(self, sentence_matrix, inplace=True):
        """Normalize the sentence_matrix rows to unit_length

        Notes
        -----
        Directly adapted from gensim

        Parameters
        ----------
        sentence_matrix : numpy.ndarray
            The sentence embedding matrix of dim len(sentences) * vector_size
        inplace : bool, optional

        Returns
        -------
        numpy.ndarray
            The sentence embedding matrix of dim len(sentences) * vector_size
        """ 
        logger.info("computing L2-norms of sentence embeddings")
        if inplace:
            for i in xrange(len(sentence_matrix)):
                sentence_matrix[i, :] /= sqrt((sentence_matrix[i, :] ** 2).sum(-1))
        else:
            output = (sentence_matrix / sqrt((sentence_matrix ** 2).sum(-1))[..., newaxis]).astype(REAL)
            return output

    def train(self, sentences, **kwargs):
        """Train the model on sentences

        Parameters
        ----------
        sentences : iterable of list of str
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.

        Returns
        -------
        numpy.ndarray
            The sentence embedding matrix of dim len(sentences) * vector_size
        """ 
        if sentences is None:
            raise RuntimeError("Provide sentences object")

        len_sentences = 0
        if not hasattr(sentences, '__len__'):
            len_sentences = sum(1 for _ in sentences)
        else:
            len_sentences = len(sentences)

        if len_sentences == 0:
            raise RuntimeError("Sentences must be non-empty")

        self._estimate_memory(len_sentences, len(self.model.vocab), self.model.vector_size)

        output, no_words, no_sents = s2v_train(sentences, len_sentences, self.model, self.sif_weights)

        logger.info("finished computing sentence embeddings of %i effective sentences with %i effective words", no_sents, no_words)

        if self.components > 0:
            output = self._remove_principal_component(output, self.components)

        return output
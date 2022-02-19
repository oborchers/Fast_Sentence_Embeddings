#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Alessandro Muzzi <alessandromuzzi17@gmail.com>
# Copyright (C) 2021 Alessandro Muzzi
from typing import List

from math import log
from fse.models.average import Average

from gensim.models.keyedvectors import KeyedVectors

from numpy import float32 as REAL
import numpy as np

import logging

logger = logging.getLogger(__name__)


class Idf(Average):

    def __init__(self, model: KeyedVectors, sv_mapfile_path: str = None,
                 wv_mapfile_path: str = None, workers: int = 1, lang_freq: str = None):
        """ Inverse document frequency (Idf)
            Because the term "the" is so common, term frequency will tend to incorrectly emphasize documents
            which happen to use the word "the" more frequently, without giving enough weight to the more meaningful terms "brown" and "cow".
            The term "the" is not a good keyword to distinguish relevant and non-relevant documents and terms,
            unlike the less-common words "brown" and "cow". Hence, an inverse document frequency factor is incorporated
            which diminishes the weight of terms that occur very frequently in the document set and increases
            the weight of terms that occur rarely. Karen Spärck Jones (1972) conceived a statistical interpretation
            of term-specificity called Inverse Document Frequency (idf), which became a cornerstone of term weighting:
                The specificity of a term can be quantified as an inverse function of the number of documents in which it occurs.

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

        self.vocab = {}

        super(Idf, self).__init__(
            model=model, sv_mapfile_path=sv_mapfile_path, wv_mapfile_path=wv_mapfile_path,
            workers=workers, lang_freq=lang_freq)

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training """
        self._compute_idf_weights(kwargs)

    def _check_parameter_sanity(self):
        """ Check the sanity of all paramters """
        if not all(self.word_weights >= 0.):
            raise ValueError("For Idf, all word weights must be 0 <= w_weight <= 1")

    def _compute_idf_weights(self, statistics):
        """ Computes the Idf weights for all words in the vocabulary """
        logger.info(f"pre-computing Idf weights for {len(self.wv)} words")

        words = self.wv.key_to_index
        ret = []
        if len(words) == 0:
            return np.zeros(self.wv.get_dimension())
        for word in words:
            count = self.vocab.get(word, 0)
            if count == 0:
                idf_w = 1
            else:
                idf_w = log(statistics['total_sentences'] / count, 10)

            ret.append(idf_w)

        self.word_weights = np.array(ret).astype(REAL)

    def train(self, sentences: List[tuple] = None, update: bool = False, queue_factor: int = 2, report_delay: int = 5) -> [int, int]:
        """ Perform word count before start training to have the counts in _compute_idf_weights()
            called by _pre_train_calls() """

        for sentence, idx in sentences:
            sent = set(sentence)
            for word in sent:
                self.vocab[word] = self.vocab.get(word, 0) + 1

        return super().train(sentences, update, queue_factor, report_delay)



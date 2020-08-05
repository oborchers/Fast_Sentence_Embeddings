#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <oliver-borchers@outlook.de>
# Copyright (C) 2020 Oliver Borchers
# For License information, see corresponding LICENSE file.

from fse.models.utils import get_ft_word_vector
from typing import List

import logging

logger = logging.getLogger(__name__)

from numpy import(
    ndarray,
    zeros,
    ceil,
    float32 as REAL,
    amax as np_amax,
    max as np_max,
)

def base_iterator(
    model,
    indexed_sentences: List[tuple],
    target: ndarray,
    memory: tuple,
    window_merger : callable,
    sentence_merger : callable,
) -> [int, int]:
    # """Training on a sequence of sentences and update the target ndarray.

    # Called internally from :meth:`~fse.models.pooling.MaxPooling._do_train_job`.

    # Warnings
    # --------
    # This is the non-optimized, pure Python version. If you have a C compiler,
    # fse will use an optimized code path from :mod:`fse.models.pooling_inner` instead.

    # Parameters
    # ----------
    # model : :class:`~fse.models.base_s2v.BaseSentence2VecModel`
    #     The BaseSentence2VecModel model instance.
    # indexed_sentences : iterable of tuple
    #     The sentences used to train the model.
    # target : ndarray
    #     The target ndarray. We use the index from indexed_sentences
    #     to write into the corresponding row of target.
    # memory : tuple
    #     Private memory array(s) for each working thread

    # Returns
    # -------
    # int, int
    #     Number of effective sentences (non-zero) and effective words in the vocabulary used 
    #     during training the sentence embedding.

    # """
    size = model.wv.vector_size
    vocab = model.wv.vocab

    w_vectors = model.wv.vectors
    w_weights = model.word_weights

    s_vectors = target

    is_ft = model.is_ft

    mem = memory[0]
    mem2 = memory[1]

    window_size = model.window_size
    window_stride = model.window_stride

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
        oov_weight = np_amax(w_weights)

    eff_sentences, eff_words = 0, 0

    for obj in indexed_sentences:
        sent = obj[0]
        sent_adr = obj[1]

        if not len(sent):
            continue

        eff_sentences += 1
        
        # In cython we know the length (-OOV) beforehand
        sent_len = sentence_length(
            model,
            sent,
        )
        if not sent_len:
            continue

        # Number of windows to be encountered
        win_count = int(ceil(sent_len / window_size))

        for word_index, _ in enumerate(sent):
            if word_index % window_stride != 0:
                continue

            win_len = 0
            mem.fill(0.) # mem for summation
            mem2.fill(0.)
            for word in sent[word_index : word_index + window_size]:
                eff_words += window_func(
                    model,
                    word,
                    mem,
                ) # -> mem
                # W2V will return 0&1, FT returns 1

                win_len += 1

            # Herein the window will be merged (i.e., rescaled)
            window_merger(
                model,
                win_len,
                mem,
                mem2,
            ) # mem ->  mem2
        
            # Partially constructs the sentence onto sv.vectors
            sentence_merger(
                sent_len,
                mem2,
                sent_adr,
                target,
            )

    return eff_sentences, eff_words

def sentence_length(
        model,
        sent : List,
    ) -> int:
    """ We know the length of the sentence a-priori
    """
    if model.is_ft:
        return len(sent)
    else:
        # Inefficient, but hey, its just the python version anyways
        return sum([1 if token in model.wv.vocab else 0 for token in sent])

def window_func(
        model,
        word : str,
        mem : ndarray,
    ) -> int:
    """ Computes the word vectors for a word
    """
    if word in model.wv.vocab:
        word_index = model.wv.vocab[word].index
        mem += model.wv.vectors[word_index] * model.word_weights[word_index]
        return 1
    else:
        if model.is_ft:
            mem += get_ft_word_vector(word, model) * np_max(model.word_weights)
            return 1
        else:
            return 0 # Word not taken into account

def window_merger(
        model,
        win_len : int,
        mem : ndarray,
        mem2 : ndarray,
    ):
    """ Average window merger.
    Should implement functionality to merge temporary results from 
    mem to mem2 inplace. Depends on model architecture
    """
    pass

def sentence_merger(
        window_length : int,
        mem : ndarray,
        sent_adr : int,
        target : ndarray,
    ):
    pass
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
    sentence_length : callable,
    window_kernel   : callable,
    window_scaler   : callable,
    sentence_kernel : callable,
    sentence_scaler : callable,
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
    #     The BaseSentence2VecModel model instance or a child of it.
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

    mem = memory[0]
    mem2 = memory[1]
    # Do not need ngram vectors here due to numpy ft ngram implementation

    w_vectors = model.wv.vectors
    w_weights = model.word_weights
    s_vectors = target

    is_ft = model.is_ft

    window_size = model.window_size
    window_stride = model.window_stride

    if is_ft:
        # NOTE: For Fasttext: Use wv.vectors_vocab
        # Using the wv.vectors from fasttext had horrible effects on the sts results
        # I suspect this is because the wv.vectors are based on the averages of
        # wv.vectors_vocab + wv.vectors_ngrams, which will all point into very
        # similar directions.
        min_n = model.wv.min_n
        max_n = model.wv.max_n
        bucket = model.wv.bucket
        max_ngrams = model.batch_ngrams
        oov_weight = np_amax(w_weights)

        w_vectors = model.wv.vectors_vocab
        ngram_vectors = model.wv.vectors_ngrams
        
    eff_sentences, eff_words = 0, 0

    for sent, sent_index in indexed_sentences:
        if not len(sent):
            # Skip if sentence is empty, leaves vector empty
            continue
        eff_sentences += 1
        
        # In cython we know the length (-OOV) beforehand
        sent_len = sentence_length(
            model,
            sent,
        )
        if not sent_len:
            continue

        # Number of windows in a sentence. Includes broken windows at the edge
        win_count = int(ceil(sent_len / window_stride))

        for word_index, _ in enumerate(sent):
            if word_index % window_stride != 0:
                continue
            win_len = 0
            mem.fill(0.) # mem for summation
            mem2.fill(0.)

            for word in sent[word_index : word_index + window_size]:
                eff_words += window_kernel(
                    model,
                    word,
                    mem,
                    mem2,
                ) # -> mem
                # W2V will return 0&1, FT returns 1
                win_len += 1

            # Herein the window will be merged (i.e., rescaled)
            window_scaler(
                model,
                win_len,
                mem,
                mem2,
            ) # mem ->  mem2
        
            # Partially constructs the sentence onto sv.vectors
            sentence_kernel(
                sent_len,
                sent_index,
                target,
                mem,
                mem2,
            )

        # Rescales the sentence if necessary
        # Note: If size & stride = 1 -> win_count = sent_len
        sentence_scaler(
            win_count,
            sent_index,
            target,
            mem,
            mem2,
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

def window_kernel(
        model,
        word : str,
        mem : ndarray,
        mem2 : ndarray,
    ) -> int:
    """ Window kernel implements aggregation function for window.
    Does the vector conversion.
    All results will be stored in mem.
    """
    return 1

def window_scaler(
        model,
        window_length : int,
        mem : ndarray,
        mem2 : ndarray,
    ) -> None:
    """ Window scaler implements scaling function for window result.
    All results will be stored in mem2.
    """
    pass

def sentence_kernel(
        sent_length : int,
        sent_index : int,
        target : ndarray,
        mem : ndarray,
        mem2 : ndarray,
    ) -> int:
    """ Sentence kernel implements aggregation function for all windows.
    All results will be stored in target.
    """
    pass

def sentence_scaler(
        win_count : int,
        sent_adr : int,
        target : ndarray,
        mem : ndarray,
        mem2 : ndarray,
    ) -> None:
    """ Sentence scaler implements scaling function for accumulated window result.
    All results will be stored in target.
    """
    pass
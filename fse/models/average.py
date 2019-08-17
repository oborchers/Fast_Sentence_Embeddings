#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from fse.models.base_s2v import BaseSentence2VecModel
from fse.models.inputs import IndexedSentence
from gensim.models.keyedvectors import BaseKeyedVectors

from numpy import ones, float32 as REAL, sum as np_sum, multiply as np_mult

from typing import List

import logging

logger = logging.getLogger(__name__)

def train_average_np(model:BaseSentence2VecModel, indexed_sentences:List[IndexedSentence]) -> [int,int]:
        size = model.wv.vector_size
        vlookup = model.wv.vocab

        w_vectors = model.wv.vectors
        s_vectors = model.sv.vectors
        w_weights = model.word_weights

        eff_sentences, eff_words = 0, 0

        for obj in indexed_sentences:
            sent_index = obj.index
            sent = obj.words

            word_indices = [vlookup[word].index for word in sent if word in vlookup]
            if not len(word_indices):
                continue

            eff_sentences += 1
            eff_words += len(word_indices)

            v = np_sum(np_mult(w_vectors[word_indices],w_weights[word_indices][:,None]) , axis=0)
            v *= 1/len(word_indices)
            s_vectors[sent_index] = v.astype(REAL)

        return eff_sentences, eff_words

try:
    from fse.models.average_inner import train_average_cy
    from fse.models.average_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
    train_average = train_average_cy
except ImportError:
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000
    train_average = train_average_np

class Average(BaseSentence2VecModel):

    def __init__(self, model:BaseKeyedVectors, mapfile_path:str=None, workers:int=2, lang_freq:str=None, wv_from_disk:bool=False):

        super(Average, self).__init__(
            model=model, mapfile_path=mapfile_path, workers=workers, 
            lang_freq=lang_freq, wv_from_disk=wv_from_disk,
            batch_words=MAX_WORDS_IN_BATCH, fast_version=FAST_VERSION)

    def _do_train_job(self, sentences:List[IndexedSentence]) -> [int,int]:
        eff_sentences, eff_words = train_average(model=self, indexed_sentences=sentences)
        return eff_sentences, eff_words

    def _check_parameter_sanity(self):
        if not all(self.word_weights == 1.): 
            raise ValueError("For averaging, all word weights must be one")

    def _pre_train_calls(self):
        pass
    
    def _check_dtype_santiy(self):
        pass

    def _post_train_calls(self):
        pass
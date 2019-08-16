#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from fse.models.base_s2v import BaseSentence2VecModel
from fse.models.inputs import IndexedSentence
from gensim.models.keyedvectors import BaseKeyedVectors

from numpy import ones, float32 as REAL, sum as np_sum

from typing import List

import logging

logger = logging.getLogger(__name__)

FAST_VERSION = -1
def average_train_np(model:BaseSentence2VecModel, sentences:List[IndexedSentence]) -> [int,int]:
        size = model.wv.vector_size
        vlookup = model.wv.vocab

        w_vectors = model.wv.vectors
        s_vectors = model.sv.vectors

        eff_sentences, eff_words = 0, 0

        for obj in sentences:
            sent_index = obj.index
            sent = obj.words

            word_indices = [vlookup[word].index for word in sent if word in vlookup]
            if not len(word_indices):
                continue

            eff_sentences += 1
            eff_words += len(word_indices)

            v = np_sum(w_vectors[word_indices], axis=0)
            v *= 1/len(word_indices)
            s_vectors[sent_index] = v.astype(REAL)

        return eff_sentences, eff_words


class Average(BaseSentence2VecModel):

    def __init__(self, model:BaseKeyedVectors, mapfile_path:str=None, workers:int=2, lang_freq:str=None, fast_version:int=0, wv_from_disk:bool=False):

        super(Average, self).__init__(
            model=model, mapfile_path=mapfile_path, workers=workers, 
            lang_freq=lang_freq, wv_from_disk=wv_from_disk, fast_version=FAST_VERSION)

    def _do_train_job(self, sentences:List[IndexedSentence]) -> [int,int]:
        eff_sentences, eff_words = average_train_np(self, sentences)
        return eff_sentences, eff_words

    def _pre_train_calls(self):
        pass

    def _post_train_calls(self):
        pass
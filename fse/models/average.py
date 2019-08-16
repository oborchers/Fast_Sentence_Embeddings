#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from fse.models.base_s2v import BaseSentence2VecModel

from gensim.models.keyedvectors import BaseKeyedVectors

from numpy import ones, float32 as REAL, sum as np_sum

FAST_VERSION = -1
def average_train_np(model, sentences):
        size = model.wv.vector_size
        vlookup = model.wv.vocab

        w_vectors = model.wv.vectors
        w_weights = model.word_weights[:, None]

        s_vectors = model.sv.vectors

        eff_sentences, eff_words = 0, 0

        # Here the enumeration is a problem because it should only work with indexed sentences
        for sentence_index, sentence in enumerate(sentences):
            word_indices = [vlookup[word].index for word in sentence if word in vlookup]
            if not len(word_indices):
                continue

            eff_sentences += 1
            eff_words += len(word_indices)

            v = np_sum(w_vectors[word_indices] * w_weights[word_indices], axis=0)
            v *= 1/len(word_indices)
            s_vectors[sentence_index] = v.astype(REAL)

        return eff_sentences, eff_words


class Average(BaseSentence2VecModel):

    def __init__(self, model:BaseKeyedVectors, mapfile_path:str=None, workers:int=2, lang_freq:str=None):

        super(Average, self).__init__(
            model=model, mapfile_path=mapfile_path, workers=workers, 
            lang_freq=lang_freq, fast_version=FAST_VERSION)

        self.word_weights = ones(len(self.wv.vocab), dtype=REAL)

    def _do_train_job(self, sentences):

        summary = average_train_np(self, sentences)

        return summary
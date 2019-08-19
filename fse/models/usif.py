#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from fse.models.average import Average

from gensim.models.keyedvectors import BaseKeyedVectors

from numpy import ndarray, float32 as REAL, zeros

from sklearn.decomposition import TruncatedSVD

import logging

logger = logging.getLogger(__name__)

class uSIF(Average):

    def __init__(self, model:BaseKeyedVectors, components:int=5, length:int=None, sv_mapfile_path:str=None, wv_mapfile_path:str=None, workers:int=1, lang_freq:str=None):        
        
        self.length = length
        self.components = int(components)
        self.components_vec = None
        self.components_vals = None

        super(Average, self).__init__(
            model=model, sv_mapfile_path=sv_mapfile_path, wv_mapfile_path=wv_mapfile_path,
            workers=workers, 
            lang_freq=lang_freq)

    def _check_parameter_sanity(self):
        pass

    def _pre_train_calls(self, **kwargs):
        self.length = kwargs["average_length"] if self.length is None else self.length
        self._compute_usif_weights()

    def _check_dtype_santiy(self):
        pass

    def _post_train_calls(self, **kwargs):
        svd = self._compute_principal_components(components = self.components)
        proj = lambda a, b: a.dot(b.transpose()) * b

        for i in range(self.components):
            lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
            pc = svd.components_[i]
            p = proj(self.sv.vectors, pc.reshape(1,-1))
            self.sv.vectors -= (lambda_i * p)

    def _compute_usif_weights(self):
        logger.info(f"pre-computing uSIF weights for {len(self.wv.vocab)} words")
        v = len(self.wv.vocab)
        corpus_size = 0
        for word in self.wv.vocab:
            corpus_size += self.wv.vocab[word].count

        pw = zeros(v, dtype=REAL)
        for word in self.wv.vocab:
            pw[self.wv.vocab[word].index] = self.wv.vocab[word].count / corpus_size
        
        threshold = 1- (1-(1/v)) ** self.length
        alpha = sum(pw > threshold) / v
        z = v/2
        a = (1 - alpha)/(alpha * z)

        for word in self.wv.vocab:
            idx = self.wv.vocab[word].index
            self.word_weights[idx] = (a / (a/2 + pw[idx])) 
        
    def _compute_principal_components(self, components:int=1) -> ndarray:
        logger.info(f"computing {components} principal components")
        svd = TruncatedSVD(n_components=components, n_iter=7, random_state=0, algorithm="randomized")
        svd.fit(self.sv.vectors)
        return svd#svd.components_, svd.singular_values_

    # def _remove_principal_components(self, components:int=1):
    #     logger.info(f"removing {components} principal components")
    #     if components==1:
    #         self.sv.vectors -= self.sv.vectors.dot(self.components_vec.transpose()) * self.components_vec
    #     else:
    #         self.sv.vectors -= self.sv.vectors.dot(self.components_vec.transpose()).dot(self.components_vec)
        
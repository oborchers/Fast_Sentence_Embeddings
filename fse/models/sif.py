#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from fse.models.average import Average

from gensim.models.keyedvectors import BaseKeyedVectors

from numpy import ndarray, float32 as REAL

from sklearn.decomposition import TruncatedSVD

import logging

logger = logging.getLogger(__name__)

class SIF(Average):

    def __init__(self, model:BaseKeyedVectors, components:int=1, alpha:float=1e-3, sv_mapfile_path:str=None, wv_mapfile_path:str=None, workers:int=1, lang_freq:str=None):        
        self.alpha = float(alpha)
        self.components = int(components)
        self.components_vec = None

        super(SIF, self).__init__(
            model=model, sv_mapfile_path=sv_mapfile_path, wv_mapfile_path=wv_mapfile_path,
            workers=workers, 
            lang_freq=lang_freq)

    def _check_parameter_sanity(self):
        if not all(self.word_weights <= 1.) and not all(self.word_weights >= 0.): 
            raise ValueError("For SIF, all word weights must be 0 <= w_weight <= 1")
        if self.alpha <= 0.:
            raise ValueError("Alpha must be greater than zero.")
        if self.components < 0.:
            raise ValueError("Components must be greater or equal zero")

    def _pre_train_calls(self, **kwargs):
        self._compute_sif_weights()
    
    def _check_dtype_santiy(self):
        pass

    def _post_train_calls(self, **kwargs):
        if self.components > 0:
            self.components_vec = self._compute_principal_components(components = self.components)
            self._remove_principal_components(components = self.components)
        else:
            logger.info(f"no removal of principal components")

    def _compute_sif_weights(self):
        logger.info(f"pre-computing SIF weights for {len(self.wv.vocab)} words")
        corpus_size = 0

        for word in self.wv.vocab:
            corpus_size += self.wv.vocab[word].count

        for word in self.wv.vocab:
                pw = self.wv.vocab[word].count / corpus_size
                self.word_weights[self.wv.vocab[word].index] = self.alpha / (self.alpha+pw)
    
    def _compute_principal_components(self, components:int=1) -> ndarray:
        logger.info(f"computing {components} principal components")
        svd = TruncatedSVD(n_components=components, n_iter=7, random_state=0, algorithm="randomized")
        svd.fit(self.sv.vectors)
        return svd.components_

    def _remove_principal_components(self, components:int=1):
        logger.info(f"removing {components} principal components")
        if components==1:
            self.sv.vectors -= self.sv.vectors.dot(self.components_vec.transpose()) * self.components_vec
        else:
            self.sv.vectors -= self.sv.vectors.dot(self.components_vec.transpose()).dot(self.components_vec)
        
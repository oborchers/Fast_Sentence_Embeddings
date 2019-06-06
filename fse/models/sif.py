#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.keyedvectors import BaseKeyedVectors

from sklearn.decomposition import TruncatedSVD

import logging
logger = logging.getLogger(__name__)

# Define data types for use in cython
REAL = np.float32 
INT = np.intc

try:
    # Import cython functions  
    CY_ROUTINES = 1
    from fse.models.sif_inner import sif_embeddings
except ImportError as e:
    CY_ROUTINES = 0
    logger.warning("ImportError of Cython functions: %s", e)

    def sif_embeddings(sentences, wv, sif_vectors):
        """ Non-Cython implementation of SIF embeddings
        """
        vlookup = wv.vocab
        vectors = sif_vectors

        output = []
        for s in sentences:
            idx = [vlookup[w].index for w in s if w in vlookup]
            v = np.sum(vectors[idx], axis=0)
            if len(idx) > 0:
                v *= 1/len(idx)
            output.append(v)
        return np.vstack(output).astype(REAL)
    
class SIF():

    def __init__(self, model, alpha=1e-3, components=1):
        """Class for computing the SIF embedding

        Parameters
        ----------
        alpha : float, optional
            Parameter which is used to weigh each individual word based on its probability p(w).
            If alpha = 1 train simply computes the average representation
        components : int, optional
            Number of principal components to remove from the sentence embedding

        Returns
        -------
        numpy.ndarray 
            SIF sentence embedding matrix of dim len(sentences) * dimension
        """

        if isinstance(model, BaseWordEmbeddingsModel):
            self.model = model.wv
        elif isinstance(model, BaseKeyedVectors):
            self.model = model
        else:
            raise RuntimeError("Model must be child of BaseWordEmbeddingsModel")

        self.alpha = float(alpha)
        self.components = int(components)

        self.sif_vectors = None
        self.sif = None


    def compute_principal_component(self, X,npc=1):
        """
        Source: https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        logger.debug("Computing principal component")
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0, algorithm="randomized")
        svd.fit(X)
        return svd.components_


    def remove_principal_component(self, X, npc=1):
        """
        Source: https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        logger.debug("Removing principal component")
        pc = self.compute_principal_component(X, npc)
        if npc==1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX


    def precompute_sif_weights(self, wv, alpha=1e-3):
        """ Precompute the SIF weights

        Parameters
        ----------
        wv : `~gensim.models.keyedvectors.BaseKeyedVectors`
            A gensim keyedvectors child that contains the word vectors and the vocabulary
        alpha : float
            Parameter which is used to weigh each individual word based on its probability p(w).

        """
        logger.debug("Pre-computing SIF weights")

        if alpha > 0:
            corpus_size = 0
            sif = np.zeros(shape=len(wv.vocab), dtype=REAL) 

            for k in wv.index2word:
                corpus_size += wv.vocab[k].count

            for idx, k in enumerate(wv.index2word):
                pw = wv.vocab[k].count / corpus_size
                sif[idx] = alpha / (alpha+pw)
        else:
            sif = np.ones(shape=len(wv.vocab), dtype=REAL)

        return sif


    def precompute_sif_vectors(self, wv, alpha):
        """ Precompute the SIF Vectors

        Parameters
        ----------
        wv : `~gensim.models.keyedvectors.BaseKeyedVectors`
            A gensim keyedvectors child that contains the word vectors and the vocabulary
        """
        logger.debug("Pre-computing SIF vectors")

        if not hasattr(self, 'sif') or self.sif is None:
            self.sif = self.precompute_sif_weights(wv, alpha)

        if not hasattr(self, 'sif_vectors') or self.sif_vectors is None:
            self.sif_vectors = (wv.vectors * self.sif[:, None]).astype(REAL)


    def train(self, sentences, clear=False):
        """ Precompute the SIF Vectors

        Parameters
        ----------
        sentences : iterable
            An iterable which contains the sentences

        Returns
        -------
        numpy.ndarray 
            SIF sentence embedding matrix of dim len(sentences) * dimension
        """ 
        if sentences is None or len(sentences) == 0:
            raise RuntimeError("Sentences must be non-empty")

        self.precompute_sif_vectors(self.model, self.alpha)
        output = sif_embeddings(sentences, self.model, self.sif_vectors)

        if self.components > 0:
            output = self.remove_principal_component(output, self.components)

        if clear:
            logger.debug("Removing sif-vectors and weights")
            self.sif_vectors = None
            self.sif = None

        return output
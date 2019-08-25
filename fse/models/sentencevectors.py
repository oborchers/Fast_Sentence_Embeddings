#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


from __future__ import division

import logging

from numpy import dot, float32 as REAL, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax

from gensim import utils, matutils
from gensim.models.keyedvectors import _l2_norm

from typing import Dict

from pathlib import Path

logger = logging.getLogger(__name__)

class SentenceVectors(utils.SaveLoad):

    def __init__(self, vector_size:int, mapfile_path:str=None):
        self.vector_size = vector_size                      # Size of vectors
        self.vectors = zeros((0, vector_size), REAL)        # Vectors for sentences
        self.vectors_norm = None

        # File for numpy memmap
        self.mapfile_path = Path(mapfile_path) if mapfile_path is not None else None            
        self.mapfile_shape = None

    def __getitem__(self, entities:int) -> ndarray:
        """Get vector representation of `entities`.

        Parameters
        ----------
        entities : {int, list of int}
            Index or sequence of entities.

        Returns
        -------
        numpy.ndarray
            Vector representation for `entities` (1D if `entities` is int, otherwise - 2D).

        """

        if isinstance(entities, (int, integer,)):
            return self.get_vector(entities)

        return vstack([self.get_vector(e) for e in entities])

    def __contains__(self, index:int) -> bool:
        if isinstance(index, (int, integer,)):
            return index < len(self)
        else:
            raise KeyError(f"index {index} is not a valid index")

    def __len__(self) -> int:
        return len(self.vectors)

    def _load_all_vectors_from_disk(self, mapfile_path:Path):
        """ Reads all vectors from disk """
        path = str(mapfile_path.absolute())
        self.vectors = np_memmap(f"{path}.vectors", dtype=REAL, mode='r+', shape=self.mapfile_shape)

    def save(self, *args, **kwargs):
        """Save object.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.load`
            Load object.

        """
        self.mapfile_shape = self.vectors.shape
        ignore = ["vectors_norm"]
        # don't bother storing the cached normalized vectors
        if self.mapfile_path is not None:
            ignore.append("vectors")
        kwargs['ignore'] = kwargs.get('ignore', ignore)
        super(SentenceVectors, self).save(*args, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        # TODO: Unittests
        sv = super(SentenceVectors, cls).load(fname_or_handle, **kwargs)
        path = sv.mapfile_path
        if path is not None:
            sv._load_all_vectors_from_disk(mapfile_path=path)
        return sv

    def get_vector(self, index:int, use_norm:bool=False) -> ndarray:
        """Get sentence representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        index : int
            Input index
        use_norm : bool, optional
            If True - resulting vector will be L2-normalized (unit euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of index.

        Raises
        ------
        KeyError
            If index out of bounds.

        """
        if index in self:
            if use_norm:
                result = self.vectors_norm[index]
            else:
                result = self.vectors[index]

            result.setflags(write=False)
            return result
        else:
            raise KeyError("index {index} not found")

    def init_sims(self, replace:bool=False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True - forget the original vectors and only keep the normalized ones = saves lots of memory!
        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            logger.info("precomputing L2-norms of sentence vectors")
            if not replace and self.mapfile_path is not None:
                self.vectors_norm = np_memmap(
                    self.mapfile_path + '.vectors_norm', dtype=REAL,
                    mode='w+', shape=self.vectors.shape)

            self.vectors_norm = _l2_norm(self.vectors, replace=replace)

    def similarity(self, d1:int, d2:int) -> float:
        """Compute cosine similarity between two sentences from the training set.

        TODO: Accept vectors of out-of-training-set docs, as if from inference.

        Parameters
        ----------
        d1 : {int, str}
            index of sentence / sentence.
        d2 : {int, str}
            index of sentence / sentence.

        Returns
        -------
        float
            The cosine similarity between the vectors of the two sentences.

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def distance(self, d1:int, d2:int) -> float:
        """Compute cosine similarity between two sentences from the training set.

        TODO: Accept vectors of out-of-training-set docs, as if from inference.

        Parameters
        ----------
        d1 : {int, str}
            index of sentence / sentence.
        d2 : {int, str}
            index of sentence / sentence.

        Returns
        -------
        float
            The cosine distance between the vectors of the two sentences.

        """
        return 1 - self.similarity(d1, d2)

    def most_similar(self, positive=None, negative=None, topn:int=10, clip_start:int=0, clip_end=None, indexer=None, indexable=None) -> Dict[int, float]: 
        # TODO
        raise NotImplementedError()

    def infer_sentence(self, sentence, model):
        # TODO
        raise NotImplementedError()

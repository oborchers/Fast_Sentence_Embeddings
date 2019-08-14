#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


from __future__ import division

import logging

from numpy import dot, float32 as REAL, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax
import numpy as np

from gensim import utils, matutils
from gensim.models.keyedvectors import _l2_norm

from six import integer_types

logger = logging.getLogger(__name__)

class SentenceVectors(utils.SaveLoad):

    def __init__(self, vector_size, mapfile_path=None):
        # [ ] Indexed Sentence Class (Index, Sentence)
        # [ ] Document Boundary (DocId, Up, Low)
        # [X] Only int indices for sentences
        # [X] Aceppt lists as input
        # [ ] Implement uSIF Emebddings
        # [ ] Write base Sentence Embedding Class
        # [ ] Multi Core Implementation (Splitted Sentence Queue?)
        # [ ] The LOGGER MUST GO TO DEBUG! Or a logging level. Whatever.
        # [ ] If principal compnents exist, use them for the next train phase --> train + infer
        # [ ] Make a warning when sentences are not passed as [[str, str]] --> validate input
        # [ ] How to best determine length?
        # [ ] Similarity for unseen documents --> Model.infer vector
        # [ ] For outputs, provide an indexable function to map indices to sentences
        # [ ] Check that input is list of list
        # [ ] Initialization with zeros, but on first scan of sentences
        
        # Note to self: Working with dfs where reviews are in the rows would work best
        # with two enumerated tuples (i.e: reviewdIdx, sentIdx)

        self.vector_size = vector_size              # Size of vectors
        self.vectors = zeros((0, vector_size))      # Vectors for sentences
        self.vectors_norm = None
        self.mapfile_path = mapfile_path            # File for numpy memmap

        
    def __getitem__(self, entities):
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

        if isinstance(entities, integer_types + (integer,)):
            return self.get_vector(entities)

        return vstack([self.get_vector(e) for e in entities])

    def __contains__(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < len(self)
        else:
            raise KeyError(f"index {index} is not a valid index")

    def __len__(self):
        return len(self.vectors)

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
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_docs_norm'])
        super(SentenceVectors, self).save(*args, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(SentenceVectors, cls).load(fname_or_handle, **kwargs)

    def get_vector(self, index, use_norm=False):
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

    
    def init_sims(self, replace=False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True - forget the original vectors and only keep the normalized ones = saves lots of memory!

        # TODO: Check if this warning applies?
        # Warnings
        # --------
        # You **cannot continue training** after doing a replace.
        # The model becomes effectively read-only: you can call
        # :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.most_similar`,
        # :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.similarity`, etc., but not train and infer_vector.

        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            logger.info("precomputing L2-norms of sentence vectors")
            if not replace and self.mapfile_path is not None:
                self.vectors_norm = np_memmap(
                    self.mapfile_path + '.vectors_norm', dtype=REAL,
                    mode='w+', shape=self.vectors.shape)

            self.vectors_norm = _l2_norm(self.vectors, replace=replace)

    def most_similar(self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, indexer=None, indexable=None):
        # TODO
        raise NotImplementedError()

    def similarity(self, d1, d2):
        # TODO
        raise NotImplementedError()

    def distance(self, d1, d2):
        # TODO
        raise NotImplementedError()

    def infer_sentence(self, sentence, model):
        # TODO
        raise NotImplementedError()

    def add(self):
        # TODO: Do i need that?
        raise NotImplementedError()

    def __setitem__(self):
        # TODO: Do i need that?
        raise NotImplementedError()
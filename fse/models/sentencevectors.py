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
        # [ ] Indexed Sentence (Index, Sentence)
        # [ ] Document Boundatie (DocId, Up, Low)
        # [ ] Remove Six(?)
        # [ ] Only int indices for sentences
        # [ ] Aceppt lists as input
        # [ ] Implement uSIF Emebddings
        # [ ] Write base Sentence Embedding Class
        # [ ] Multi Core Implementation (Splitted Sentence Queue?)
        # [ ] The LOGGER MUST GO TO DEBUG! Or a logging level. Whatever.
        # [ ] If principal compnents exist, use them for the next train phase --> train + infer
        # [ ] Make a warning when sentences are not passed as [[str, str]] --> validate input
        
        # Note to self: Working with dfs where reviews are in the rows would work best
        # with two enumerated tuples (i.e: reviewdIdx, sentIdx)

        self.vector_size = vector_size              # Size of vectors
        self.vectors = zeros((0, vector_size))      # Vectors for sentences
        self.vectors_norm = None
        self.mapfile_path = mapfile_path            # File for numpy memmap
        self.count = 0

        
    def __getitem__(self, index):
        """Get vector representation of `index`.

        Parameters
        ----------
        index : {int, list of int}
            Index or sequence of indices.

        Returns
        -------
        numpy.ndarray
            Vector representation for `index` (1D if `index` is string, otherwise - 2D).

        """

        if isinstance(index, integer_types + (integer,)):
            index = [index]
        
        output = []
        for i in index:
            if i in self:
                output.append(self.vectors[i])
            else:
                raise KeyError(f"index {index} not in range of known indices")
        return vstack(output)

    def __contains__(self, index):
        return index < self.count

        
    def __len__(self):
        return self.count

if __name__ == '__main__':
    sv_ram = SentenceVectors(2, "test_data/")
    sv_ram.vectors = np.arange(10).reshape(5,2)
    sv_ram.count = len(sv_ram.vectors)

    print(sv_ram[0])

    print(5 in sv_ram)
    print(6 in sv_ram)
    print([5,6] in sv_ram)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


from __future__ import division

from fse.inputs import IndexedList, IndexedLineDocument

from gensim.models.keyedvectors import BaseKeyedVectors

from numpy import dot, float32 as REAL, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax

from gensim import utils, matutils

from typing import List, Tuple

from pathlib import Path

import logging

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

        Parameters
        ----------
        d1 : int
            index of sentence 
        d2 : int
            index of sentence 

        Returns
        -------
        float
            The cosine similarity between the vectors of the two sentences.

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def distance(self, d1:int, d2:int) -> float:
        """Compute cosine similarity between two sentences from the training set.

        Parameters
        ----------
        d1 : int
            index of sentence 
        d2 : int
            index of sentence 

        Returns
        -------
        float
            The cosine distance between the vectors of the two sentences.

        """
        return 1 - self.similarity(d1, d2)

    def most_similar(self, positive:[int,ndarray]=None, negative:[int,ndarray]=None, 
                        indexable:[IndexedList,IndexedLineDocument]=None, topn:int=10, 
                        restrict_size:[int, Tuple[int, int]]=None) -> List[Tuple[int,float]]:

        """Find the top-N most similar sentences.
        Positive sentences contribute positively towards the similarity, negative sentences negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given sentences and the vectors for each sentence in the model.

        Parameters
        ----------
        positive : list of int, optional
            List of indices that contribute positively.
        negative : list of int, optional
            List of indices that contribute negatively.
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
            then similarities for all sentences are returned.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 sentence vectors.
            restrict_vocab=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        if indexable is not None and not hasattr(indexable, "__getitem__"):
            raise RuntimeError("Indexable must provide __getitem__")
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, (int, integer)) and not negative:
            positive = [positive]
        if isinstance(positive, (ndarray)) and not negative:
            if len(positive.shape) == 1:
                positive = [positive]

        positive = [
            (sent, 1.0) if isinstance(sent, (int, integer, ndarray)) else sent
            for sent in positive
        ]
        negative = [
            (sent, -1.0) if isinstance(sent, (int, integer,  ndarray)) else sent
            for sent in negative
        ]

        all_sents, mean = set(), []
        for sent, weight in positive + negative:
            if isinstance(sent, ndarray):
                mean.append(weight * sent)
            else:
                mean.append(weight * self.get_vector(index=sent, use_norm=True))
                if sent in self:
                    all_sents.add(sent)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if isinstance(restrict_size, (int, integer)):
            lo, hi = 0, restrict_size
        elif isinstance(restrict_size, Tuple):
            lo, hi = restrict_size
        else:
            lo, hi = 0, None

        limited = self.vectors_norm if restrict_size is None else self.vectors_norm[lo:hi]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_sents), reverse=True)
        best_off = best + lo
        
        if indexable is not None:
            result = [(indexable[off_idx], off_idx, float(dists[idx])) for off_idx, idx in zip(best_off, best) if off_idx not in all_sents]
        else:
            result = [(off_idx, float(dists[idx])) for off_idx, idx in zip(best_off, best) if off_idx not in all_sents]
        return result[:topn]

    def similar_by_word(self, word:str, wv:BaseKeyedVectors, indexable:[IndexedList,IndexedLineDocument]=None, topn:int=10, 
                        restrict_size:[int,Tuple[int, int]]=None) -> List[Tuple[int,float]]:

        """Find the top-N most similar sentences to a given word.

        Parameters
        ----------
        word : str
            Word
        wv : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
            This object essentially contains the mapping between words and embeddings.
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
            then similarities for all sentences are returned.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 sentence vectors.
            restrict_vocab=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(positive=wv[word], indexable=indexable, topn=topn, restrict_size=restrict_size)

    def similar_by_sentence(self, sentence:List[str], model, indexable:[IndexedList,IndexedLineDocument]=None, topn:int=10,
                            restrict_size:[int,Tuple[int, int]]=None) -> List[Tuple[int,float]]:
        
        """Find the top-N most similar sentences to a given sentence.

        Parameters
        ----------
        sentence : list of str
            Sentence as list of strings
        model : :class:`~fse.models.base_s2v.BaseSentence2VecModel`
            This object essentially provides the infer method used to transform .
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
            then similarities for all sentences are returned.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 sentence vectors.
            restrict_vocab=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        vector = model.infer([(sentence, 0)])
        return self.most_similar(positive=vector, indexable=indexable, topn=topn, restrict_size=restrict_size)
    
    def similar_by_vector(self, vector:ndarray, indexable:[IndexedList,IndexedLineDocument]=None, topn:int=10,
                        restrict_size:[int,Tuple[int, int]]=None) -> List[Tuple[int,float]]:

        """Find the top-N most similar sentences to a given vector.

        Parameters
        ----------
        vector : ndarray
            Vectors
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
            then similarities for all sentences are returned.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 sentence vectors.
            restrict_vocab=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(positive=vector, indexable=indexable, topn=topn, restrict_size=restrict_size)

def _l2_norm(m, replace=False):
    """Return an L2-normalized version of a matrix.

    Parameters
    ----------
    m : np.array
        The matrix to normalize.
    replace : boolean, optional
        If True, modifies the existing matrix.

    Returns
    -------
    The normalized matrix.  If replace=True, this will be the same as m.

    NOTE: This part is copied from Gensim and modified as the call
    m /= dist somtimes raises an exception and sometimes it does not.
    """
    dist = sqrt((m ** 2).sum(-1))[..., newaxis]
    if replace:
        m = m / dist
        return m
    else:
        return (m / dist).astype(REAL)
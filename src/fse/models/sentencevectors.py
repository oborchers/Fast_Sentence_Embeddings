#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers


import logging
from pathlib import Path
from typing import Any, cast

from gensim import matutils, utils
from gensim.models.keyedvectors import KeyedVectors
from numpy import (
    arange,
    argsort,
    array,
    concatenate,
    dot,
    dtype,
    empty,
    flatnonzero,
    floating,
    int64,
    integer,
    isnan,
    issubdtype,
    lexsort,
    ndarray,
    newaxis,
    partition,
    prod,
    sqrt,
    vstack,
    zeros,
)
from numpy import (
    float32 as REAL,
)
from numpy import (
    memmap as np_memmap,
)

from fse._validation import NonNegativeInt, PathLike, Similarity, validate_arguments
from fse.models.utils import REMOVAL_BLOCK_ROWS

logger = logging.getLogger(__name__)


def _most_similar_rows(vectors: ndarray, query: ndarray, count: int) -> tuple[ndarray, ndarray]:
    """The rows with the ``count`` largest dot products with ``query``, largest first.

    The rows are read in blocks of ``max(REMOVAL_BLOCK_ROWS, count)``, and only the best
    ``count`` candidates are kept between blocks, so the memory of the search grows with
    ``count``, not with the number of rows.
    The dot products equal those of a single ``dot(vectors, query)`` up to float32
    rounding, since BLAS may sum a block differently. Of equal products the lower row
    wins; NaN ranks below every number, as in ``argsort``.
    """
    best_rows = empty(0, dtype=int64)
    best_keys = empty(0, dtype=REAL)
    # At least as many rows per block as candidates, so the candidates are copied O(rows)
    # times in total, also for a count close to the number of rows.
    block = max(REMOVAL_BLOCK_ROWS, count)
    for begin in range(0, len(vectors), block):
        # Negated, so the best rows are the smallest keys and NaN sorts after them.
        keys = concatenate([best_keys, -dot(vectors[begin : begin + block], query)])
        rows = concatenate([best_rows, arange(begin, begin + len(keys) - len(best_keys))])
        if len(keys) > count:
            kth = partition(keys, count - 1)[count - 1]
            if isnan(kth):
                # Fewer numbers than places: all of them, then NaN rows fill up.
                better, ties = flatnonzero(~isnan(keys)), flatnonzero(isnan(keys))
            else:
                better, ties = flatnonzero(keys < kth), flatnonzero(keys == kth)
            # Of the rows tied with the count-th, the lowest ones fill the remaining places.
            ties = ties[argsort(rows[ties], kind="stable")[: count - len(better)]]
            keep = concatenate([better, ties])
            keys, rows = keys[keep], rows[keep]
        best_keys, best_rows = keys, rows
    order = lexsort((best_rows, best_keys))
    return best_rows[order], -best_keys[order]


class SentenceVectors(utils.SaveLoad):
    vectors_norm: ndarray | None
    mapfile_shape: tuple[int, int] | None

    @validate_arguments
    def __init__(self, vector_size: NonNegativeInt, mapfile_path: PathLike | None = None):

        self.vector_size = vector_size  # Size of vectors
        self.vectors = zeros((0, vector_size), REAL)  # Vectors for sentences
        self.vectors_norm = None

        # File for numpy memmap
        self.mapfile_path = Path(mapfile_path) if mapfile_path is not None else None
        self.mapfile_shape = None

    def __getitem__(self, entities: int) -> ndarray:
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

        if isinstance(
            entities,
            (
                int,
                integer,
            ),
        ):
            return self.get_vector(entities)

        return vstack([self.get_vector(e) for e in entities])

    def __contains__(self, index: object) -> bool:
        if isinstance(index, bool) or not isinstance(index, (int, integer)):
            return False
        return bool(0 <= index < len(self))

    def __len__(self) -> int:
        return len(self.vectors)

    def _load_all_vectors_from_disk(self, mapfile_path: Path):
        """Reads all vectors from disk"""
        path = str(mapfile_path.absolute())
        self.vectors = np_memmap(f"{path}.vectors", dtype=REAL, mode="r+", shape=self.mapfile_shape)

    def save(self, *args, **kwargs):
        """Save object.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~fse.models.sentencevectors.SentenceVectors.load`
            Load object.

        """
        self.mapfile_shape = self.vectors.shape
        ignore = ["vectors_norm"]
        # don't bother storing the cached normalized vectors
        if self.mapfile_path is not None:
            ignore.append("vectors")
        kwargs["ignore"] = kwargs.get("ignore", ignore)
        super().save(*args, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        # TODO: Unittests
        sv = super().load(fname_or_handle, **kwargs)
        path = sv.mapfile_path
        if path is not None:
            sv._load_all_vectors_from_disk(mapfile_path=path)
        return sv

    def get_vector(self, index: int, use_norm: bool = False) -> ndarray:
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
                self.init_sims()
            result = cast(ndarray, self.vectors_norm)[index] if use_norm else self.vectors[index]

            result.setflags(write=False)
            return result
        else:
            raise KeyError(f"index {index} not found")

    def init_sims(self, replace: bool = False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True, normalize ``vectors`` themselves and keep no separate copy. This saves
            memory, but the original vectors are lost.
        """
        if getattr(self, "vectors_norm", None) is not None and not replace:
            return
        logger.info("precomputing L2-norms of sentence vectors")
        if replace:
            self.vectors = _l2_norm(self.vectors, replace=True)
            self.vectors_norm = self.vectors
        elif self.mapfile_path is not None:
            norm_path = Path(f"{self.mapfile_path}.vectors_norm")
            # Reuse a file of the right size instead of truncating it: another model loaded
            # from the same files may have it mapped, and Windows refuses to truncate then.
            expected_size = int(prod(self.vectors.shape)) * dtype(REAL).itemsize
            exists = norm_path.exists() and norm_path.stat().st_size == expected_size
            self.vectors_norm = np_memmap(
                norm_path,
                dtype=REAL,
                mode="r+" if exists else "w+",
                shape=self.vectors.shape,
            )
            # Normalize in blocks so a memory-mapped matrix never has to fit into RAM.
            block = 100_000
            for start in range(0, len(self.vectors), block):
                self.vectors_norm[start : start + block] = _l2_norm(
                    self.vectors[start : start + block]
                )
        else:
            self.vectors_norm = _l2_norm(self.vectors)

    def similarity(self, d1: int, d2: int) -> float:
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

    def distance(self, d1: int, d2: int) -> float:
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

    @validate_arguments
    def most_similar(
        self,
        positive: Any = None,
        negative: Any = None,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
        """Find the top-N most similar sentences.
        Positive sentences contribute positively towards the similarity, negative sentences negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given sentences and the vectors for each sentence in the model.
        It is an exact search: every query reads all normalized sentence vectors in blocks.
        With a positive `topn`, the search needs memory for one block and `topn` candidates,
        not for a similarity per sentence; the normalized vectors themselves are computed
        once by :meth:`init_sims`, in RAM unless the vectors are memory-mapped. Of equally
        similar sentences the lower
        index comes first. For many queries over very many sentences, build an approximate
        nearest neighbor index (for example FAISS) from the vectors instead.

        Parameters
        ----------
        positive : list of int, optional
            List of indices that contribute positively.
        negative : list of int, optional
            List of indices that contribute negatively.
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

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
        if isinstance(positive, (ndarray)) and not negative and len(positive.shape) == 1:
            positive = [positive]

        positive = [
            (sent, 1.0) if isinstance(sent, (int, integer, ndarray)) else sent for sent in positive
        ]
        negative = [
            (sent, -1.0) if isinstance(sent, (int, integer, ndarray)) else sent for sent in negative
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
        elif isinstance(restrict_size, tuple):
            lo, hi = restrict_size
        else:
            lo, hi = 0, None
        if hi is not None and hi < lo:
            raise ValueError(
                f"restrict_size {restrict_size} has an upper bound below its lower bound"
            )

        if min_similarity is not None and not topn:
            raise ValueError(
                "min_similarity filters the top-N list; it needs a positive topn, not "
                f"topn={topn}, which returns the similarities to all sentences"
            )

        vectors_norm = cast(ndarray, self.vectors_norm)
        limited = vectors_norm if restrict_size is None else vectors_norm[lo:hi]
        if not topn:
            return dot(limited, mean)
        best, similarities = _most_similar_rows(limited, mean, topn + len(all_sents))
        best_off = best + lo

        result: list[tuple[Any, ...]]
        if indexable is not None:
            result = [
                (indexable[off_idx], off_idx, float(sim))
                for off_idx, sim in zip(best_off.tolist(), similarities, strict=True)
                if off_idx not in all_sents
            ]
        else:
            result = [
                (off_idx, float(sim))
                for off_idx, sim in zip(best_off.tolist(), similarities, strict=True)
                if off_idx not in all_sents
            ]
        result = result[:topn]
        if min_similarity is not None:
            # result is sorted by descending similarity.
            result = [item for item in result if item[-1] >= min_similarity]
        return result

    @validate_arguments
    def similar_by_word(
        self,
        word: str,
        wv: KeyedVectors,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
        """Find the top-N most similar sentences to a given word.

        Parameters
        ----------
        word : str
            Word
        wv : :class:`~gensim.models.keyedvectors.KeyedVectors`
            This object essentially contains the mapping between words and embeddings.
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(
            positive=wv[word],
            indexable=indexable,
            topn=topn,
            restrict_size=restrict_size,
            min_similarity=min_similarity,
        )

    @validate_arguments
    def similar_by_sentence(
        self,
        sentence: Any,
        model: Any,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
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
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        infer_op = getattr(model, "infer", None)
        if not callable(infer_op):
            raise RuntimeError(
                "Model does not have infer method. Make sure to pass a BaseSentence2VecModel"
            )

        vector = model.infer([(sentence, 0)])
        return self.most_similar(
            positive=vector,
            indexable=indexable,
            topn=topn,
            restrict_size=restrict_size,
            min_similarity=min_similarity,
        )

    @validate_arguments
    def similar_by_vector(
        self,
        vector: ndarray,
        indexable: Any = None,
        topn: NonNegativeInt | None = 10,
        restrict_size: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] | None = None,
        min_similarity: Similarity | None = None,
    ) -> list[tuple[Any, ...]] | ndarray:
        """Find the top-N most similar sentences to a given vector.

        Parameters
        ----------
        vector : ndarray
            Vectors
        indexable: list, IndexedList, IndexedLineDocument
            Provides an indexable object from where the most similar sentences are read
        topn : int or None, optional
            Number of top-N similar sentences to return, when `topn` is a positive int. When
            `topn` is 0 or None, the similarities to all sentences are returned as an array.
        restrict_size : int or Tuple(int,int), optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_size=10000 would
            only check the first 10000 sentence vectors.
            restrict_size=(500, 1000) would search the sentence vectors with indices between
            500 and 1000.
        min_similarity : float, optional
            Only return sentences whose cosine similarity is at least this value, in [-1, 1].
            Fewer than `topn` sentences are returned when fewer reach it. Requires a positive
            `topn`.

        Returns
        -------
        list of (int, float) or list of (str, int, float)
            A sequence of (index, similarity) is returned.
            When an indexable is provided, returns (str, index, similarity)
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(
            positive=vector,
            indexable=indexable,
            topn=topn,
            restrict_size=restrict_size,
            min_similarity=min_similarity,
        )


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
    The normalized matrix. Rows of zeros stay zero. If replace=True and m holds floats, m is
    normalized in place and returned; other dtypes cannot be normalized in place, so a new
    float32 matrix is returned.
    """
    dist = sqrt((m**2).sum(-1))[..., newaxis]
    # A zero vector has no direction; leave it at zero instead of producing NaN.
    dist[dist == 0] = 1
    if replace and issubdtype(m.dtype, floating):
        m /= dist
        return m
    return (m / dist).astype(REAL)

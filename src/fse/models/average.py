#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""This module implements the base class to compute average representations for sentences, using highly optimized C routines,
data streaming and Pythonic interfaces.

The implementation is based on Iyyer et al. (2015): Deep Unordered Composition Rivals Syntactic Methods for Text Classification.
For more information, see <https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf>.

The training algorithms is based on the Gensim implementation of Word2Vec, FastText, and Doc2Vec.
For more information, see: :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.fasttext.FastText`, or
:class:`~gensim.models.doc2vec.Doc2Vec`.

Initialize and train a :class:`~fse.models.average.Average` model

.. sourcecode:: pycon

        >>> from gensim.models.word2vec import Word2Vec
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>> model = Word2Vec(sentences, min_count=1, vector_size=20)

        >>> from fse.models.average import Average
        >>> avg = Average(model)
        >>> avg.train([(s, i) for i, s in enumerate(sentences)])
        >>> avg.sv.vectors.shape
        (2, 20)

"""

import logging
from collections.abc import Iterable
from typing import Any

from gensim.models.fasttext import ft_ngram_hashes
from numpy import (
    asarray as np_asarray,
)
from numpy import (
    float32 as REAL,
)
from numpy import (
    full,
    ndarray,
    zeros,
)
from numpy import (
    max as np_max,
)
from numpy import (
    multiply as np_mult,
)
from numpy import (
    sum as np_sum,
)

from fse._validation import PathLike, PositiveInt, WordVectorModel, validate_arguments
from fse.models._quantize import StorageDtype, dequantize
from fse.models.base_s2v import BaseSentence2VecModel, _OffsetIndices

logger = logging.getLogger(__name__)


def _weighted_rows(
    vectors: ndarray | None, scales: ndarray | None, index: list[int], weights: Any
) -> Any:
    """``weights[:, None] * rows`` for the stored rows at index, in float32.

    int8 codes are multiplied by ``weights * scales`` as the kernel does, so a tiny scale
    times a weight underflows in both paths alike.
    """
    if vectors is None:
        raise RuntimeError("the model has no stored vectors for these rows")
    weights = np_asarray(weights, dtype=REAL)
    if scales is not None:
        return dequantize(vectors[index], weights * scales[index])
    rows = vectors[index] if vectors.dtype == REAL else dequantize(vectors[index], None)
    return np_mult(rows, weights[:, None])


def train_average_np(
    model: BaseSentence2VecModel,
    indexed_sentences: list[tuple],
    target: ndarray,
    memory: tuple[ndarray, ndarray],
) -> tuple[int, int]:
    """Training on a sequence of sentences and update the target ndarray.

    Called internally from :meth:`~fse.models.average.Average._do_train_job`.

    Warnings
    --------
    This is the non-optimized, pure Python version. If you have a C compiler,
    fse will use an optimized code path from :mod:`fse.models.average_inner` instead.

    Parameters
    ----------
    model : :class:`~fse.models.base_s2v.BaseSentence2VecModel`
        The BaseSentence2VecModel model instance.
    indexed_sentences : iterable of tuple
        The sentences used to train the model.
    target : ndarray
        The target ndarray. We use the index from indexed_sentences
        to write into the corresponding row of target.
    memory : ndarray
        Private memory for each working thread

    Returns
    -------
    int, int
        Number of effective sentences (non-zero) and effective words in the vocabulary used
        during training the sentence embedding.

    """
    size = model.wv.vector_size

    # The rows the kernel reads: float32 vectors, or the float16 or int8 codes of a model
    # with wv_dtype, restored to float32 row by row.
    w_vectors, w_scales, ngram_vectors, ngram_scales = model._word_arrays()
    w_weights = model.word_weights

    s_vectors = target

    is_ft = model.is_ft

    mem = memory[0]

    if is_ft:
        # NOTE: For Fasttext: Use wv.vectors_vocab
        # Using the wv.vectors from fasttext had horrible effects on the sts results
        # I suspect this is because the wv.vectors are based on the averages of
        # wv.vectors_vocab + wv.vectors_ngrams, which will all point into very
        # similar directions.
        max_ngrams = model.batch_ngrams
        min_n = model.wv.min_n
        max_n = model.wv.max_n
        bucket = model.wv.bucket
        oov_weight = np_max(w_weights)

    eff_sentences, eff_words = 0, 0

    if not is_ft:
        for obj in indexed_sentences:
            mem.fill(0.0)
            sent = obj[0]
            sent_adr = obj[1]

            if not sent:
                continue
            word_indices = [
                model.wv.key_to_index[word] for word in sent if word in model.wv.key_to_index
            ]
            eff_sentences += 1
            if not word_indices:
                continue
            eff_words += len(word_indices)

            mem += np_sum(
                _weighted_rows(w_vectors, w_scales, word_indices, w_weights[word_indices]),
                axis=0,
            )
            mem *= 1 / len(word_indices)
            s_vectors[sent_adr] += mem.astype(REAL)
    else:
        for obj in indexed_sentences:
            mem.fill(0.0)
            sent = obj[0]
            sent_adr = obj[1]

            if not len(sent):
                continue
            mem = zeros(size, dtype=REAL)

            eff_sentences += 1
            eff_words += len(sent)  # Counts everything in the sentence

            for word in sent:
                if word in model.wv.key_to_index:
                    word_index = model.wv.key_to_index[word]
                    mem += _weighted_rows(
                        w_vectors, w_scales, [word_index], w_weights[[word_index]]
                    )[0]
                else:
                    # A model without ngram buckets has no vector for an unknown word, as
                    # in the Cython routine.
                    ngram_hashes = (
                        ft_ngram_hashes(word, min_n, max_n, bucket)[:max_ngrams] if bucket else []
                    )
                    if len(ngram_hashes) == 0:
                        continue
                    # Each ngram row weighted by oov_weight / ngrams, as in the kernel.
                    inv_ngram = REAL(1.0) / REAL(len(ngram_hashes)) * REAL(oov_weight)
                    weights = full(len(ngram_hashes), inv_ngram, dtype=REAL)
                    mem += np_sum(
                        _weighted_rows(ngram_vectors, ngram_scales, ngram_hashes, weights), axis=0
                    )
                # Implicit addition of zero if oov does not contain any ngrams
            s_vectors[sent_adr] += mem / len(sent)

    return eff_sentences, eff_words


try:
    from fse.models.average_inner import (
        FAST_VERSION,
        MAX_NGRAMS_IN_BATCH,
        MAX_WORDS_IN_BATCH,
        JobEncoder,
        KernelContext,
        VocabTable,
        run_chunk,
        run_job,
        train_average_cy,
    )

    train_average = train_average_cy
    CY_ROUTINES = True
except ImportError:
    CY_ROUTINES = False
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000
    MAX_NGRAMS_IN_BATCH = 40
    train_average = train_average_np


class _NativeChunk:
    """Sentences of a CorpusFile or TextList for one worker: positions start to end."""

    __slots__ = ("end", "first_row", "n_items", "source", "start", "table")

    def __init__(self, source, table, start, end, first_row, n_items):
        self.source = source
        self.table = table
        self.start = start
        self.end = end
        self.first_row = first_row
        self.n_items = n_items


class _ChunkFeeder:
    """Hands out the chunks of a scanned native source, like JobEncoder hands out jobs."""

    def __init__(self, source, table, chunks, offset):
        self._chunks = iter(chunks)
        self._source = source
        self._table = table
        self._offset = offset

    def next_job(self):
        for start, end, first, count in self._chunks:
            return _NativeChunk(self._source, self._table, start, end, first + self._offset, count)
        return None


class Average(BaseSentence2VecModel):
    """Train, use and evaluate averaged sentence vectors.

    The model can be stored/loaded via its :meth:`~fse.models.average.Average.save` and
    :meth:`~fse.models.average.Average.load` methods.

    Some important attributes are the following:

    Attributes
    ----------
    wv : :class:`~gensim.models.keyedvectors.KeyedVectors`
        This object essentially contains the mapping between words and embeddings. After training, it can be used
        directly to query those embeddings in various ways. See the module level docstring for examples.

    sv : :class:`~fse.models.sentencevectors.SentenceVectors`
        This object contains the sentence vectors inferred from the training data. There will be one such vector
        for each unique docusentence supplied during training. They may be individually accessed using the index.

    prep : :class:`~fse.models.base_s2v.BaseSentence2VecPreparer`
        The prep object is used to transform and initialize the sv.vectors. Aditionally, it can be used
        to move the vectors to disk for training with memmap.

    """

    @validate_arguments
    def __init__(
        self,
        model: WordVectorModel,
        sv_mapfile_path: PathLike | None = None,
        wv_mapfile_path: PathLike | None = None,
        workers: PositiveInt = 1,
        lang_freq: str | None = None,
        sv_dtype: StorageDtype = "float32",
        wv_dtype: StorageDtype = "float32",
    ):
        """Average (unweighted) sentence embeddings model. Performs a simple averaging operation over all
        words in a sentences without further transformation.

        The implementation is based on Iyyer et al. (2015): Deep Unordered Composition Rivals Syntactic Methods for Text Classification.
        For more information, see <https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf>.

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.KeyedVectors` or a gensim model with a ``wv`` attribute, such as :class:`~gensim.models.Word2Vec` or :class:`~gensim.models.FastText`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            its ``key_to_index`` and ``vectors`` are required.
        sv_mapfile_path : str, optional
            Optional path to store the sentence-vectors in for very large datasets. Used for memmap.
        wv_mapfile_path : str, optional
            Optional path to store the word-vectors in for very large datasets. Used for memmap.
            Use sv_mapfile_path and wv_mapfile_path to train disk-to-disk without needing much ram.
        workers : int, optional
            Number of working threads, used for multithreading. For most tasks (few words in a sentence)
            a value of 1 should be more than enough.
        lang_freq : str, optional
            Some pre-trained embeddings, i.e. "GoogleNews-vectors-negative300.bin", do not contain information about
            the frequency of a word. As the frequency is required for estimating the word weights, we induce
            frequencies into the ``count`` attribute of each word based on :class:`~wordfreq`
            If no frequency information is available, you can choose the language to estimate the frequency.
            See https://github.com/LuminosoInsight/wordfreq
        sv_dtype : {"float32", "float16", "int8"}, optional
            Storage format of the sentence vectors: float16 halves and int8 quarters their
            memory. They are computed in float32 and every read returns float32.
        wv_dtype : {"float32", "float16", "int8"}, optional
            Format in which the model keeps the word vectors it averages: float16 halves
            their memory and int8 quarters it, and both make averaging faster. The word
            vectors passed in are not changed.

        """

        super().__init__(
            model=model,
            sv_mapfile_path=sv_mapfile_path,
            wv_mapfile_path=wv_mapfile_path,
            workers=workers,
            lang_freq=lang_freq,
            sv_dtype=sv_dtype,
            wv_dtype=wv_dtype,
            batch_words=MAX_WORDS_IN_BATCH,
            batch_ngrams=MAX_NGRAMS_IN_BATCH,
            fast_version=FAST_VERSION,
        )

    def _uses_kernel(self) -> bool:
        """Encoded jobs need the extension, and they bypass _do_train_job, so a subclass or
        an instance that replaces _do_train_job keeps the job path."""
        return bool(CY_ROUTINES) and not (
            "_do_train_job" in self.__dict__
            or type(self)._do_train_job is not Average._do_train_job
        )

    def _kernel_context(self, target: ndarray) -> Any:
        """Kernel context for encoded jobs, or None for the job path of _do_train_job."""
        return KernelContext(self, target) if self._uses_kernel() else None

    def _job_encoder(self, context: Any, sentences: Iterable[tuple]) -> Any:
        """Batch and encode the input in one pass while holding the GIL, in the producer.

        CorpusFile and TextList hand out chunks of their bytes instead; each worker then
        splits and looks up its chunk itself, without the GIL.
        """
        offset = 0
        if isinstance(sentences, _OffsetIndices):
            offset, base = sentences.offset, sentences.sentences
        else:
            base = sentences
        native = base._native_chunks() if hasattr(base, "_native_chunks") else None
        if native is not None:
            source, chunks = native
            context.sources.append(source)
            return _ChunkFeeder(source, VocabTable(self.wv.key_to_index), chunks, offset)
        return JobEncoder(context, sentences, self.batch_words)

    def _finish_kernel_context(self, context: Any):
        """Close the native sources of a call; every worker has finished by now."""
        if context is not None:
            for source in context.sources:
                source.close()

    def _run_encoded_job(
        self, context: Any, job: Any, memory: tuple[ndarray, ndarray]
    ) -> tuple[int, int]:
        """Average an encoded job or a native chunk without the GIL."""
        if isinstance(job, _NativeChunk):
            counts = run_chunk(
                context,
                job.table,
                job.source.data,
                job.source.offsets,
                job.start,
                job.end,
                job.first_row,
                memory[0],
                job.n_items,
            )
            # A file changed while this chunk was read would have given wrong vectors.
            if job.source.changed():
                raise RuntimeError("the input file changed during training")
        else:
            counts = run_job(context, job, memory[0])
        single_pass = self.__dict__.get("_single_pass")
        if single_pass is not None:
            # The producer checked that these rows belong to this job only; they are final.
            single_pass.rows_done(job.first_row, job.first_row + job.n_items)
        return counts

    def _do_train_job(
        self, data_iterable: list[tuple], target: ndarray, memory: tuple[ndarray, ndarray]
    ) -> tuple[int, int]:
        """Internal routine which is called on training and performs averaging for all entries in the iterable"""
        eff_sentences, eff_words = train_average(
            model=self, indexed_sentences=data_iterable, target=target, memory=memory
        )
        return eff_sentences, eff_words

    def _check_parameter_sanity(self, **kwargs):
        """Check the sanity of all child paramters"""
        if not all(self.word_weights == 1.0):
            raise ValueError("All word weights must equal one for averaging")

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training"""
        pass

    def _post_train_calls(self, **kwargs):
        """Function calls to perform after training, such as computing eigenvectors"""
        pass

    def _post_inference_calls(self, output: ndarray | None = None, **kwargs):
        """Function calls to perform after training & inference
        Examples include the removal of components
        """
        pass

    def _check_dtype_santiy(self, **kwargs):
        """Check the dtypes of all child attributes"""
        pass

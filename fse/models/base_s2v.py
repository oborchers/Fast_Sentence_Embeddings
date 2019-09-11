#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Base class containing common methods for training, using & evaluating sentence embeddings.
A lot of the code is based on Gensim. I have to thank Radim Rehurek and the whole team
for the outstanding library which I used for a lot of my research.

Attributes
----------
wv : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
    This object essentially contains the mapping between words and embeddings. After training, it can be used
    directly to query those embeddings in various ways. See the module level docstring for examples.

sv : :class:`~fse.models.sentencevectors.SentenceVectors`
    This object contains the sentence vectors inferred from the training data. There will be one such vector
    for each unique docusentence supplied during training. They may be individually accessed using the index.

prep : :class:`~fse.models.base_s2v.BaseSentence2VecPreparer`
    The prep object is used to transform and initialize the sv.vectors. Aditionally, it can be used
    to move the vectors to disk for training with memmap.

See Also
--------
:class:`~fse.models.average.Average`.
    Average sentence model.
:class:`~fse.models.sif.SIF`.
    Smooth inverse frequency weighted model.
:class:`~fse.models.usif.uSIF`.
    Unsupervised Smooth inverse frequency weighted model.

"""

from fse.models.sentencevectors import SentenceVectors

from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.keyedvectors import BaseKeyedVectors, FastTextKeyedVectors, _l2_norm
from gensim.utils import SaveLoad
from gensim.matutils import zeros_aligned

from numpy import ndarray, memmap as np_memmap, float32 as REAL, uint32 as uINT, \
    empty, zeros, vstack, dtype, ones, finfo, full

from wordfreq import available_languages, get_frequency_dict

from typing import List, Dict

from time import time
from psutil import virtual_memory

from pathlib import Path

import logging
import warnings

import threading

from queue import Queue

logger = logging.getLogger(__name__)

EPS = finfo(REAL).eps

class BaseSentence2VecModel(SaveLoad):
    
    def __init__(self, model:BaseKeyedVectors, sv_mapfile_path:str=None, wv_mapfile_path:str=None, workers:int=1, lang_freq:str=None, fast_version:int=0, batch_words:int=10000, batch_ngrams:int=40, **kwargs):
        """ Base class for all Sentence2Vec Models. Provides core functionality, such as
        save, load, sanity checking, frequency induction, data checking, scanning, etc.

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors` or :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            the wv.vocab and wv.vector elements are required.
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
            frequencies into the wv.vocab.count based on :class:`~wordfreq`
            If no frequency information is available, you can choose the language to estimate the frequency.
            See https://github.com/LuminosoInsight/wordfreq
        fast_version : {-1, 1}, optional
            Whether or not the fast cython implementation of the internal training methods is available. 1 means it is.
        batch_words : int, optional
            Number of words to be processed by a single job.
        batch_ngrams : int, optional
            Number of maxium ngrams for oov words.
        **kwargs : object
            Key word arguments needed to allow children classes to accept more arguments.

        """

        self.workers = int(workers)
        self.batch_words = batch_words
        self.batch_ngrams = batch_ngrams
        self.wv = None                              
                                                    
        self.is_ft = False                          

        self.wv_mapfile_path = Path(wv_mapfile_path) if wv_mapfile_path is not None else None
        self.wv_mapfile_shapes = {}

        if fast_version < 0:
            warnings.warn(
                "C extension not loaded, training/inferring will be slow. "
                "Install a C compiler and reinstall fse."
            )

        self._check_and_include_model(model)

        if self.wv_mapfile_path is not None:
            self._map_all_vectors_to_disk(self.wv_mapfile_path)

        if lang_freq is not None:
            self._check_language_settings(lang_freq)
            self._induce_frequencies()

        self.sv = SentenceVectors(vector_size=self.wv.vector_size, mapfile_path=sv_mapfile_path)
        self.prep = BaseSentence2VecPreparer()

        self.word_weights = ones(len(self.wv.vocab), REAL)

    def __str__(self) -> str:
        """ Human readable representation of the model's state.

        Returns
        -------
        str
            Human readable representation of the model's state.

        """
        return f"{self.__class__.__name__} based on {self.wv.__class__.__name__}, size={len(self.sv)}"

    def _check_and_include_model(self, model:BaseKeyedVectors):
        """ Check if the supplied model is a compatible model. Performs all kinds of checks and small optimizations.
        
        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors` or :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            The model to inject into this class.

        """
        if isinstance(model, BaseWordEmbeddingsModel):
            self.wv = model.wv
        elif isinstance(model, BaseKeyedVectors):
            self.wv = model
        else:
            raise RuntimeError(f"Model must be child of BaseWordEmbeddingsModel or BaseKeyedVectors. Received {str(model)}")
        self.wv.vectors_norm = None
        
        if isinstance(self.wv, FastTextKeyedVectors):
            self.wv.vectors_vocab_norm = None # Save some space
            self.wv.vectors_ngrams_norm = None
            self.wv.vectors_vocab_norm = None
            self.is_ft = True

            if not self.wv.compatible_hash:
                raise RuntimeError("FastText model requires compatible hash function")
            if not hasattr(self.wv, 'vectors_vocab') or self.wv.vectors_vocab is None:
                raise RuntimeError("vectors_vocab required for sentence embeddings not found.")
            if not hasattr(self.wv, 'vectors_ngrams') or self.wv.vectors_ngrams is None:
                raise RuntimeError("Ngram vectors required for sentence embeddings not found.")
            
        if not hasattr(self.wv, 'vectors') or self.wv.vectors is None:
            raise RuntimeError("Word vectors required for sentence embeddings not found.")
        if not hasattr(self.wv, 'vocab'):
            raise RuntimeError("Vocab required for sentence embeddings not found.")
    
    def _check_language_settings(self, lang_freq:str):
        """ Check if the supplied language is a compatible with the wordfreq package
        
        Parameters
        ----------
        lang_freq : str
            The language used to induce the frequencies into the wv.vocab object.

        """
        if lang_freq in available_languages(wordlist='best'):
            self.lang_freq = str(lang_freq)
            logger.info("no frequency mode: using wordfreq for estimation "
                        f"of frequency for language: {self.lang_freq}")
        else:
            raise ValueError(f"Language {lang_freq} is not available in wordfreq")
    
    def _induce_frequencies(self, domain:int=2**31 - 1):
        """ Induce frequencies for a pretrained model, as not all pretrained models come with frequencies.
        
        Parameters
        ----------
        domain : int
            The cumulative count of the vocabulary.

        """
        freq_dict = get_frequency_dict(self.lang_freq, wordlist='best')
        for word in self.wv.index2word:
            if word in freq_dict:
                self.wv.vocab[word].count = int(freq_dict[word] * domain)
            else:
                self.wv.vocab[word].count = int(1e-8 * domain)

    def _check_input_data_sanity(self, data_iterable:tuple):
        """ Check if the input data complies with the required formats
        
        Parameters
        ----------
        data_iterable : tuple
            The cumulative count of the vocabulary.

        """
        if data_iterable is None:
            raise TypeError("You must provide a data iterable to train on")
        elif isinstance(data_iterable, str):
            raise TypeError("Passed string. Input data must be iterable list of list of tokens or tuple")
        elif not hasattr(data_iterable, "__iter__"):
            raise TypeError("Iterable must provide __iter__ function")

    def _log_train_end(self, eff_sentences:int, eff_words:int, overall_time:float):
        """ Log the end of training.

        Parameters
        ----------
        eff_sentences : int
            Number of effective (non-zero) sentences encountered in training.
        eff_words : int
            Number of effective words used in training (after ignoring unknown words).
        overall_time : float
            Time in seconds for the task to be completed.

        """
        logger.info(
            f"training on {eff_sentences} effective sentences with {eff_words} effective words "
            f"took {int(overall_time)}s with {int(eff_sentences / overall_time)} sentences/s"
        )

    def _check_pre_training_sanity(self, total_sentences:int, total_words:int, average_length:int, **kwargs):
        """ Check if all available objects for training are available and compliant

        Parameters
        ----------
        total_sentences : int
            Number of sentences encountered while scanning
        total_words : int
            Number of words encountered while scanning
        average_length : int
            Average sentence length

        """
        if not hasattr(self, "wv") or self.wv is None: 
            raise RuntimeError("you must first load a valid BaseKeyedVectors object")
        if not len(self.wv.vectors):
            raise RuntimeError("you must initialize vectors before computing sentence vectors")

        if self.is_ft and not len(self.wv.vectors_ngrams):
            raise RuntimeError("you must initialize ngram vectors before computing sentence vectors")
        if self.is_ft and not len(self.wv.vectors_vocab):
            raise RuntimeError("you must initialize vectors_vocab before computing sentence vectors")

        if sum([self.wv.vocab[w].count for w in self.wv.vocab]) == len(self.wv.vocab):
            logger.warning(
                "The sum of the word counts is equal to its length (all word counts are 1). "
                "Make sure to obtain proper word counts by using lang_freq for pretrained embeddings."
            )

        if not hasattr(self.sv, "vectors") or self.sv.vectors is None: 
            raise RuntimeError("initialization of Sentencevectors failed")
        if not hasattr(self, "word_weights") or self.word_weights is None: 
            raise RuntimeError("initialization of word weights failed")
            
        if not len(self.wv.vectors) == len(self.word_weights):
            raise RuntimeError("Number of word vectors and weights does not match")

        if self.wv.vectors.dtype != REAL:
            raise TypeError(f"type of wv.vectors is wrong: {self.wv.vectors.dtype}")
        if self.is_ft and self.wv.vectors_ngrams.dtype != REAL:
            raise TypeError(f"type of wv.vectors_ngrams is wrong: {self.wv.vectors_ngrams.dtype}")
        if self.is_ft and self.wv.vectors_vocab.dtype != REAL:
            raise TypeError(f"type of wv.vectors_vocab is wrong: {self.wv.vectors_vocab.dtype}")
        if self.sv.vectors.dtype != REAL:
            raise TypeError(f"type of sv.vectors is wrong: {self.sv.vectors.dtype}")
        if self.word_weights.dtype != REAL:
            raise TypeError(f"type of word_weights is wrong: {self.word_weights.dtype}")

        if total_sentences is 0 or total_words is 0 or average_length is 0:
            raise ValueError(
                f"scanning the sentences returned invalid values. Check the input."
            )

    def _check_post_training_sanity(self, eff_sentences:int, eff_words:int):
        """ Check if the training results make sense

        Parameters
        ----------
        eff_sentences : int
            Number of effective sentences encountered during training
        eff_words : int
            Number of effective words encountered during training
        
        """
        if eff_sentences is 0 or eff_words is 0:
            raise ValueError(
                f"training returned invalid values. Check the input."
            )
    
    def _check_indexed_sent_valid(self, iterPos:int, obj:tuple, checked:int=False) -> [int, List[str]]:
        """ Performs a check if the passed object contains valid data

        Parameters
        ----------
        iterPos : int
            Position in file/iterable
        obj : tuple
            An tuple object containing the index and sentence
        
        Returns
        -------
        int
            Index of the sentence used to write to (in sv.vectors)
        list
            List of strings containing all words in a sentence

        """

        if isinstance(obj, tuple):
            sent = obj[0]   #Faster than obj.words
            index = obj[1]
        else:
            raise TypeError(f"Passed {type(obj)}: {obj}. Iterable must contain tuple.")

        if not checked:
            if not isinstance(sent, list) or not all(isinstance(w, str) for w in sent):
                raise TypeError(f"At {iterPos}: Passed {type(sent)}: {sent}. tuple.words must contain list of str.")
            if not isinstance(index, int):
                raise TypeError(f"At {iterPos}: Passed {type(index)}: {index}. tuple.index must contain index")
            if index < 0:
                raise ValueError(f"At {iterPos}: Passed negative {index}")
        return index, sent

    def _map_all_vectors_to_disk(self, mapfile_path:Path):
        """ Maps all vectors to disk 

        Parameters
        ----------
        mapfile_path : Path
            Path where to write the vectors to

        """
        path = str(mapfile_path.absolute())

        self.wv_mapfile_shapes["vectors"] = self.wv.vectors.shape
        self.wv.vectors = self._move_ndarray_to_disk(self.wv.vectors, mapfile_path=path, name="wv")
        if self.is_ft:
            self.wv_mapfile_shapes["vectors_vocab"] = self.wv.vectors_vocab.shape
            self.wv_mapfile_shapes["vectors_ngrams"] = self.wv.vectors_ngrams.shape
            self.wv.vectors_vocab = self._move_ndarray_to_disk(self.wv.vectors_vocab, mapfile_path=self.wv_mapfile_path, name="vocab")    
            self.wv.vectors_ngrams = self._move_ndarray_to_disk(self.wv.vectors_ngrams, mapfile_path=self.wv_mapfile_path, name="ngrams")

    def _load_all_vectors_from_disk(self, mapfile_path:Path):
        """ Reads all vectors from disk 

        Parameters
        ----------
        mapfile_path : Path
            Path where to read the vectors from

        """
        path = str(mapfile_path.absolute())

        self.wv.vectors = np_memmap(f"{path}_wv.vectors", dtype=REAL, mode='r', shape=self.wv_mapfile_shapes["vectors"])
        if self.is_ft:
            self.wv.vectors_vocab = np_memmap(
                f"{path}_vocab.vectors", dtype=REAL, mode='r', shape=self.wv_mapfile_shapes["vectors_vocab"])
            self.wv.vectors_ngrams = np_memmap(
                f"{path}_ngrams.vectors", dtype=REAL, mode='r', shape=self.wv_mapfile_shapes["vectors_ngrams"])
        
    def _move_ndarray_to_disk(self, vector:ndarray, mapfile_path:str, name:str="") -> ndarray:
        """ Moves a numpy ndarray to disk via memmap

        Parameters
        ----------
        vector : ndarray
            The vector to write to disk
        mapfile_path : Path
            Path where to write the vector to
        name : str
            Suffix which is appended to the path to distinguish multiple files

        Returns
        -------
        ndarray
            readonly ndarray to be used in further computations

        """
        shape = vector.shape
        path = Path(f"{mapfile_path}_{name}.vectors")

        if not path.exists():
            logger.info(f"writing {name} to {path}")
            memvecs = np_memmap(
                path, dtype=REAL,
                mode='w+', shape=shape)
            memvecs[:] = vector[:]
            del memvecs, vector
        else:
            # If multiple instances of this class exist, all can access the same files
            logger.info(f"loading pre-existing {name} from {path}")

        readonly_memvecs = np_memmap(path, dtype=REAL, mode='r', shape=shape)
        return readonly_memvecs

    def _get_thread_working_mem(self) -> [ndarray, ndarray]:
        """Computes the memory used per worker thread.

        Returns
        -------
        np.ndarray
            Each worker threads private work memory.

        """
        mem = zeros_aligned(self.sv.vector_size, dtype=REAL)
        oov_mem = zeros_aligned((self.batch_words, self.batch_ngrams), dtype=uINT)
        return (mem, oov_mem)

    def _do_train_job(self, data_iterable:List[tuple], target:ndarray, memory:ndarray) -> [int, int]:
        """ Function to be called on a batch of sentences. Returns eff sentences/words """
        raise NotImplementedError()

    def _pre_train_calls(self, **kwargs):
        """ Function calls to perform before training """
        raise NotImplementedError()

    def _post_train_calls(self, **kwargs):
        """ Function calls to perform after training, such as computing eigenvectors """
        raise NotImplementedError()
    
    def _post_inference_calls(self, **kwargs):
        """ Function calls to perform after training & inference
        Examples include the removal of components
        """
        raise NotImplementedError()

    def _check_parameter_sanity(self, **kwargs):
        """ Check the sanity of all child paramters """
        raise NotImplementedError()

    def _check_dtype_santiy(self, **kwargs):
        """ Check the dtypes of all child attributes """
        raise NotImplementedError()

    @classmethod
    def load(cls, *args, **kwargs):
        """ Load a previously saved :class:`~fse.models.base_s2v.BaseSentence2VecModel`.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~fse.models.base_s2v.BaseSentence2VecModel`
            Loaded model.

        """
        # This is kind of an ugly hack because I cannot directly modify the save routine of the
        # correpsonding KeyedVectors Files, as a memmap file makes the npy files irrelvant
        model = super(BaseSentence2VecModel, cls).load(*args, **kwargs)

        if model.wv_mapfile_path is not None:
            model._load_all_vectors_from_disk(model.wv_mapfile_path)
        model.wv_mapfile_shapes = None
        return model

    def save(self, *args, **kwargs):
        """ Save the model.
        This saved model can be loaded again using :func:`~fse.models.base_s2v.BaseSentence2VecModel.load`

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        # Manually removes vectors from the wv class because we cannot modify the save method
        if self.wv_mapfile_path is not None:
            self.wv.vectors = None
            if self.is_ft:
                self.wv.vectors_vocab = None
                self.wv.vectors_ngrams = None
        super(BaseSentence2VecModel, self).save(*args, **kwargs)

    def scan_sentences(self, sentences:List[tuple]=None, progress_per:int=5) -> Dict[str,int]:
        """ Performs an initial scan of the data and reports all corresponding statistics

        Parameters
        ----------
        sentences : (list, iterable)
            An iterable consisting of tuple objects
        progress_per : int
            Number of seconds to pass before reporting the scan progress

        Returns
        -------
        dict
            Dictionary containing the scan statistics
        
        """
        logger.info("scanning all indexed sentences and their word counts")

        current_time = time()
        total_sentences = 0
        total_words = 0
        average_length = 0
        empty_sentences = 0
        max_index = 0
        checked_sentences = 0   # We only check the first item to not constrain runtime so much

        for i, obj in enumerate(sentences):
            index, sent = self._check_indexed_sent_valid(iterPos=i, obj=obj, checked=checked_sentences)
            checked_sentences += 1
            if time() - current_time > progress_per:
                current_time = time()
                logger.info(f"SCANNING : finished {total_sentences} sentences with {total_words} words")

            max_index = max(max_index, index)
            total_sentences += 1
            total_words += len(sent)

            if not len(sent):
                empty_sentences += 1
        
        if empty_sentences:
            logger.warning(f"found {empty_sentences} empty sentences")

        if max_index >= total_sentences:
            raise RuntimeError(f"Index {max_index} is larger than number of sentences {total_sentences}")

        average_length = int(total_words / total_sentences)

        logger.info(
            f"finished scanning {total_sentences} sentences with an average length of {average_length} and {total_words} total words"
        )
        statistics = {
            "total_sentences" : total_sentences,
            "total_words" : total_words,
            "average_length" : average_length,
            "empty_sentences" : empty_sentences,
            "max_index" : max_index + 1
        }
        return statistics
    
    def estimate_memory(self, max_index:int, report:dict=None, **kwargs) -> Dict[str, int]:
        """ Estimate the size of the sentence embedding

        Parameters
        ----------
        max_index : int
            Maximum index found during the initial scan
        report : dict
            Report of subclasses

        Returns
        -------
        dict
            Dictionary of estimated memory sizes

        """
        vocab_size = len(self.wv.vectors)

        report = report or {}
        report["Word Weights"] = vocab_size * dtype(REAL).itemsize
        report["Word Vectors"] = vocab_size * self.wv.vector_size * dtype(REAL).itemsize
        report["Sentence Vectors"] = max_index * self.wv.vector_size * dtype(REAL).itemsize
        if self.is_ft:
            report["Vocab Vectors"] = vocab_size * self.wv.vector_size * dtype(REAL).itemsize
            report["Ngram Vectors"] = self.wv.vectors_ngrams.shape[0] * self.wv.vector_size * dtype(REAL).itemsize
        report["Total"] = sum(report.values())
        mb_size = int(report["Total"] / 1024**2)
        logger.info(
            f"estimated memory for {max_index} sentences with "
            f"{self.wv.vector_size} dimensions and {vocab_size} vocabulary: "
            f"{mb_size} MB ({int(mb_size / 1024)} GB)"
        )
        if report["Total"] >= 0.95 * virtual_memory()[1]:
            logger.warning("The embeddings will likely not fit into RAM. Consider to use mapfile_path")
        return report

    def train(self, sentences:List[tuple]=None, update:bool=False, queue_factor:int=2, report_delay:int=5) -> [int,int]:
        """ Main routine to train an embedding. This method writes all sentences vectors into sv.vectors and is
        used for computing embeddings for large chunks of data. This method also handles post-training transformations,
        such as computing the SVD of the sentence vectors.

        Parameters
        ----------
        sentences : (list, iterable)
            An iterable consisting of tuple objects
        update : bool
            If bool is True, the sentence vector matrix will be updated in size (even with memmap)
        queue_factor : int
            Multiplier for size of queue -> size = number of workers * queue_factor.
        report_delay : int
            Number of seconds between two consecutive progress report messages in the logger.

        Returns
        -------
        int, int
            Count of effective sentences and words encountered

        """
        self._check_input_data_sanity(sentences)
        statistics = self.scan_sentences(sentences)

        self._check_pre_training_sanity(**statistics)

        self.estimate_memory(**statistics)
        self.prep.prepare_vectors(sv=self.sv, total_sentences=statistics["max_index"], update=update)
        
        # Preform post-tain calls (i.e weight computation)
        self._pre_train_calls(**statistics)
        self._check_parameter_sanity()
        self._check_dtype_santiy()
        start_time = time()

        logger.info(f"begin training")

        _, eff_sentences, eff_words = self._train_manager(data_iterable=sentences, total_sentences=statistics["total_sentences"], queue_factor=queue_factor, report_delay=report_delay)

        overall_time = time() - start_time

        self._check_post_training_sanity(eff_sentences=eff_sentences, eff_words=eff_words)

        # Preform post-tain calls (i.e principal component removal)
        self._post_train_calls()

        self._log_train_end(eff_sentences=eff_sentences, eff_words=eff_words, overall_time=overall_time)

        return eff_sentences, eff_words    

    def infer(self, sentences:List[tuple]=None, use_norm=False) -> ndarray:
        """ Secondary routine to train an embedding. This method is essential for small batches of sentences,
        which require little computation. Note: This method does not apply post-training transformations,
        only post inference calls (such as removing principal components).

        Parameters
        ----------
        sentences : (list, iterable)
            An iterable consisting of tuple objects
        use_norm : bool
            If bool is True, the sentence vectors will be L2 normalized (unit euclidean length)

        Returns
        -------
        ndarray
            Computed sentence vectors

        """
        self._check_input_data_sanity(sentences)

        statistics = self.scan_sentences(sentences)

        output = zeros((statistics["max_index"], self.sv.vector_size), dtype=REAL)
        mem = self._get_thread_working_mem()
        
        job_batch, batch_size = [], 0
        for data_idx, data in enumerate(sentences):
            data_length = len(data[0])
            if batch_size + data_length <= self.batch_words:
                job_batch.append(data)
                batch_size += data_length
            else:
                self._do_train_job(data_iterable=job_batch, target=output, memory=mem)
                job_batch, batch_size = [data], data_length
        if job_batch:
            self._do_train_job(data_iterable=job_batch, target=output, memory=mem)

        self._post_inference_calls(output=output)

        if use_norm:
            output = _l2_norm(output)
        return output

    def _train_manager(self, data_iterable:List[tuple], total_sentences:int=None, queue_factor:int=2, report_delay:int=5):
        """ Manager for the multi-core implementation. Directly adapted from gensim
        
        Parameters
        ----------
        data_iterable : (list, iterable)
            An iterable consisting of tuple objects. This will be split in chunks and these chunks will be pushed to the queue.
        total_sentences : int
            Number of sentences found during the initial scan
        queue_factor : int
            Multiplier for size of queue -> size = number of workers * queue_factor.
        report_delay : int
            Number of seconds between two consecutive progress report messages in the logger.

        """
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        # WORKING Threads
        workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(job_queue, progress_queue))
            for _ in range(self.workers)
        ]
        # JOB PRODUCER
        workers.append(
            threading.Thread(
            target=self._job_producer,
            args=(data_iterable, job_queue))
        )

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        jobs, eff_sentences, eff_words = self._log_train_progress(
            progress_queue, total_sentences=total_sentences,
            report_delay=report_delay
        )
        return jobs, eff_sentences, eff_words

    def _worker_loop(self, job_queue, progress_queue):
        """ Train the model, lifting batches of data from the queue.

        This function will be called in parallel by multiple workers (threads or processes) to make
        optimal use of multicore machines.

        Parameters
        ----------
        job_queue : Queue of (list of tuple)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
            Each job is represented as a batch of tuple.
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * Size of job processed
                * Effective sentences encountered in traning
                * Effective words encountered in traning

        """
        mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                # no more jobs => quit this worker
                break  
            eff_sentences, eff_words = self._do_train_job(data_iterable=job, target=self.sv.vectors, memory=mem)
            progress_queue.put((len(job), eff_sentences, eff_words))
            jobs_processed += 1
        logger.debug(f"worker exiting, processed {jobs_processed} jobs")
    
    def _job_producer(self, data_iterable:List[tuple], job_queue:Queue):
        """ Fill the jobs queue using the data found in the input stream.

        Each job is represented as a batch of tuple

        Parameters
        ----------
        data_iterable : (list, iterable)
            An iterable consisting of tuple objects. This will be split in chunks and these chunks will be pushed to the queue.
        job_queue : Queue of (list of tuple)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
            Each job is represented as a batch of tuple.

        """

        job_batch, batch_size = [], 0
        job_no = 0

        for data_idx, data in enumerate(data_iterable):
            data_length = len(data[0])
            if batch_size + data_length <= self.batch_words:
                job_batch.append(data)
                batch_size += data_length
            else:
                job_no += 1
                job_queue.put(job_batch)
                job_batch, batch_size = [data], data_length
        
        if job_batch:
            job_no += 1
            job_queue.put(job_batch)

        for _ in range(self.workers):
            job_queue.put(None)
        logger.debug(f"job loop exiting, total {job_no} jobs")
    
    def _log_train_progress(self, progress_queue:Queue, total_sentences:int=None, report_delay:int=5):
        """ Log the training process after a couple of seconds.

        Parameters
        ----------
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * Size of job processed
                * Effective sentences encountered in traning
                * Effective words encountered in traning
        total_sentences : int
            Number of sentences found during the initial scan
        report_delay : int
            Number of seconds between two consecutive progress report messages in the logger.

        Returns
        -------
        int, int, int
            number of jobs, effective sentences, and effective words in traning

        """
        jobs, eff_sentences, eff_words = 0, 0, 0
        unfinished_worker_count = self.workers
        start_time = time()
        sentence_inc = 0
        while unfinished_worker_count > 0:
            report = progress_queue.get()
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info(f"worker thread finished; awaiting finish of {unfinished_worker_count} more threads")
                continue

            j, s, w = report
            jobs += j
            eff_sentences += s
            eff_words += w
            if time() - start_time >= report_delay:
                start_time = time()

                logger.info("PROGRESS : finished {:3.2f}% with {} sentences and {} words, {} sentences/s".format(
                    100 * (eff_sentences/total_sentences),
                    eff_sentences, eff_words,
                    int((eff_sentences-sentence_inc) / report_delay)
                ))
                sentence_inc = eff_sentences
        
        return jobs, eff_sentences, eff_words

class BaseSentence2VecPreparer(SaveLoad):
    """ Contains helper functions to perpare the weights for the training of BaseSentence2VecModel """

    def prepare_vectors(self, sv:SentenceVectors, total_sentences:int, update:bool=False):
        """Build tables and model weights based on final vocabulary settings."""
        if not update:
            self.reset_vectors(sv, total_sentences)
        else:
            self.update_vectors(sv, total_sentences)

    def reset_vectors(self, sv:SentenceVectors, total_sentences:int):
        """Initialize all sentence vectors to zero and overwrite existing files"""
        logger.info(f"initializing sentence vectors for {total_sentences} sentences")
        if sv.mapfile_path:
            sv.vectors = np_memmap(
                str(sv.mapfile_path) + '.vectors', dtype=REAL,
                mode='w+', shape=(total_sentences, sv.vector_size))
        else:
            sv.vectors = empty((total_sentences, sv.vector_size), dtype=REAL)
        
        for i in range(total_sentences):
            sv.vectors[i] = full(shape=sv.vector_size, fill_value=EPS, dtype=REAL)
        sv.vectors_norm = None

    def update_vectors(self, sv:SentenceVectors, total_sentences:int):
        """Given existing sentence vectors, append new ones"""
        logger.info(f"appending sentence vectors for {total_sentences} sentences")
        sentences_before = len(sv.vectors)
        sentences_after = len(sv.vectors) + total_sentences

        if sv.mapfile_path:
            sv.vectors = np_memmap(
                str(sv.mapfile_path) + '.vectors', dtype=REAL,
                mode='r+', shape=(sentences_after, sv.vector_size))
            for i in range(sentences_before, sentences_after):
                sv.vectors[i] = full(shape=sv.vector_size, fill_value=EPS, dtype=REAL)
        else:
            newvectors = empty((total_sentences, sv.vector_size), dtype=REAL)
            for i in range(total_sentences):
                newvectors[i] = full(shape=sv.vector_size, fill_value=EPS, dtype=REAL)
            sv.vectors = vstack([sv.vectors, newvectors])
        sv.vectors_norm = None
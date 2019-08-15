#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from fse.models.sentencevectors import SentenceVectors
from fse.models.inputs import IndexedSentence

from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.keyedvectors import BaseKeyedVectors, FastTextKeyedVectors
from gensim.utils import SaveLoad

from numpy import ndarray

from wordfreq import available_languages, get_frequency_dict

from typing import List
from types import GeneratorType

from time import time

import logging
import warnings

logger = logging.getLogger(__name__)

class BaseSentence2VecModel(SaveLoad):
    """Base class containing common methods for training, using & evaluating sentence embeddings.

    Attributes
    ----------
    wv : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
        This object essentially contains the mapping between words and embeddings. After training, it can be used
        directly to query those embeddings in various ways. See the module level docstring for examples.
    
    sv : :class:`~fse.models.sentencevectors.SentenceVectors`
        This object contains the sentence vectors inferred from the training data. There will be one such vector
        for each unique docusentence supplied during training. They may be individually accessed using the index.

    See Also
    --------
    :class:`~fse.models.average.Average`.
        Average sentence Model.
    """

    def __init__(self, model:BaseKeyedVectors, mapfile_path:str=None, workers:int=2, min_count:int=0, lang_freq:str=None, fast_version:int=0, **kwargs):
        """
        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors` or :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            the wv.vocab and wv.vector elements are required.
        mapfile_path : str, optional
            Optional path to store the vectors in for very large datasets
        workers : int, optional
            Number of working threads, used for multithreading.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        language : str, optional
            Some pre-trained embeddings, i.e. "GoogleNews-vectors-negative300.bin", do not contain information about
            the frequency of a word. As the frequency is required for estimating the word weights, we induce
            frequencies into the wv.vocab.count based on :class:`~wordfreq`
            If no frequency information is available, you can choose the language to estimate the frequency.
            See https://github.com/LuminosoInsight/wordfreq
        fast_version : {-1, 1}, optional
            Whether or not the fast cython implementation of the internal training methods is available. 1 means it is.
        **kwargs : object
            Key word arguments needed to allow children classes to accept more arguments.
        """
        # [X] Indexed Sentence Class (Index, Sentence)
        # [ ] Document Boundary (DocId, Up, Low)
        # [X] Only int indices for sentences
        # [X] Aceppt lists as input
        # [ ] Implement Average Emebddings
        # [ ] Implement SIF Emebddings
        # [ ] Implement uSIF Emebddings
        # [ ] Write base Sentence Embedding Class
        # [ ] Multi Core Implementation (Splitted Sentence Queue?)
        # [ ] If principal compnents exist, use them for the next train phase --> train + infer
        # [ ] Make a warning when sentences are not passed as [[str, str]] --> validate input
        # [ ] How to best determine length?
        # [ ] Similarity for unseen documents --> Model.infer vector
        # [ ] For outputs, provide an indexable function to map indices to sentences
        # [ ] Check that input is list of list
        # [ ] Initialization with zeros, but on first scan of sentences
        # [ ] Fasttext compatibility for the inner loops
        # [ ] Does sv.vectors & wv.vectors collide during saving without mapfile path?
        # [ ] Decide which attributes to ignore when saving
        # [X] Unittest for inputs_check
        # [ ] Tests for IndexedSentence
        # [ ] rewrite TaggedLineDocument
        # [ ] What happens, if the Indices in IndexedSentence are larger than the matrix?
        
        self.workers = int(workers)
        self.min_count = int(min_count)
        self.wv = None                              # TODO: Check if to ignore wv file when saving
                                                    # TODO: Check what happens to the induced frequency if you ignore wv during saving
        self.subword_information = False            # TODO: Implement Fasttext support

        if fast_version < 0:
            warnings.warn(
                "C extension not loaded, training/inferring will be slow. "
                "Install a C compiler and reinstall fse."
            )

        self._check_and_include_model(model)

        if lang_freq is not None:
            self._check_language_settings(lang_freq)
            self._induce_frequencies()

        self.sv = SentenceVectors(vector_size=self.wv.vector_size, mapfile_path=mapfile_path)

    def _check_and_include_model(self, model:BaseKeyedVectors):
        """Check if the supplied model is a compatible model. """
        if isinstance(model, BaseWordEmbeddingsModel):
            self.wv = model.wv
        elif isinstance(model, BaseKeyedVectors):
            self.wv = model
        else:
            raise RuntimeError(f"Model must be child of BaseWordEmbeddingsModel or BaseKeyedVectors. Received {str(model)}")
        
        if isinstance(self.wv, FastTextKeyedVectors):
            # TODO: Remove after implementation
            raise NotImplementedError()

        if not hasattr(self.wv, 'vectors'):
            raise RuntimeError("Word vectors required for sentence embeddings not found.")
        if not hasattr(self.wv, 'vocab'):
            raise RuntimeError("Vocab required for sentence embeddings not found.")
    
    def _check_language_settings(self, lang_freq:str):
        """Check if the supplied language is a compatible with the wordfreq package"""
        if lang_freq in available_languages(wordlist='best'):
            self.lang_freq = str(lang_freq)
            logger.info("no frequency mode: using wordfreq for estimation"
                        f"of frequency for language: {self.lang_freq}")
        else:
            raise ValueError(f"Language {lang_freq} is not available in wordfreq")
    
    def _induce_frequencies(self, domain:int=2**31 - 1):
        """Induce frequencies for a pretrained model"""
        freq_dict = get_frequency_dict(self.lang_freq, wordlist='best')
        for word in self.wv.index2word:
            if word in freq_dict:
                self.wv.vocab[word].count = int(freq_dict[word] * domain)
            else:
                self.wv.vocab[word].count = int(1e-8 * domain)

    def _check_input_data_sanity(self, data_iterable=None):
        """Check if the input data complies with the required formats"""
        if data_iterable is None:
            raise TypeError("You must provide a data iterable to train on")
        elif isinstance(data_iterable, str):
            raise TypeError("Passed string. Input data must be iterable list of list of tokens or IndexedSentence")        
        elif isinstance(data_iterable, GeneratorType):
            raise TypeError("You can't pass a generator as the iterable. Try a sequence.")
        elif not hasattr(data_iterable, "__iter__"):
            raise TypeError("Iterable must provide __iter__ function")

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved :class:`~fse.models.base_s2v.BaseSentence2VecModel`.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~fse.models.base_s2v.BaseSentence2VecModel`
            Loaded model.

        """
        model = super(BaseSentence2VecModel, cls).load(*args, **kwargs)
        return model

    def save(self, *args, **kwargs):
        """Save the model.
        This saved model can be loaded again using :func:`~fse.models.base_s2v.BaseSentence2VecModel.load`

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        # TODO: Make a decision what to store and what not
        # kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'cum_table'])
        super(BaseSentence2VecModel, self).save(*args, **kwargs)

    def scan_sentences(self, sentences:[List[List[str]], List[IndexedSentence]]=None, progress_per:int=10) -> [int, int, int, int]:
        logger.info("scanning all sentences and their word counts")

        current_time = time()
        total_sentences = 0
        total_words = 0
        average_length = 0
        empty_sentences = 0
        checked_string_types = 0            # Checks only once
        max_index = 0

        for obj_no, obj in enumerate(sentences):
            if isinstance(obj, IndexedSentence):
                index = obj.index
                sent = obj.words
            else:
                index = obj_no
                sent = obj

            if not checked_string_types:
                if not isinstance(sent, list) or not all(isinstance(w, str) for w in sent):
                    raise TypeError(f"Passed {type(sent)}: {sent}. Iterable must contain list of str.")
                checked_string_types += 1
            
            if time() - current_time > progress_per:
                current_time = time()
                logger.info(f"PROGRESS: finished {total_sentences} sentences with {total_words} words")

            max_index = max(max_index, index)
            total_sentences += 1
            total_words += len(sent)

            if not len(sent):
                empty_sentences += 1
        
        if empty_sentences:
            logger.warning(f"found {empty_sentences} empty sentences")

        if max_index >= total_sentences:
            raise RuntimeError(f"Maxium index {max_index} is larger than number of sentences {total_sentences}")

        average_length = int(total_words / total_sentences)

        logger.info(
            f"finished scanning {total_sentences} sentences with an average length of {average_length} and {total_words} total words"
        )
        return total_sentences, total_words, average_length, empty_sentences

    def train(self, sentences:[List[List[str]], List[IndexedSentence]]=None):
        self._check_input_data_sanity(sentences)
        raise NotImplementedError()

    def infer(self, sentences:[List[List[str]], List[IndexedSentence]]=None) -> ndarray:
        raise NotImplementedError()

    def _do_train_job(self):
        raise NotImplementedError()

    def _check_training_sanity(self):
        raise NotImplementedError()

    def _log_progress(self):
        raise NotImplementedError()

    def _log_train_end(self):
        raise NotImplementedError()
    

    def __str__(self):
        raise NotImplementedError()

    

    def estimate_memory(self, vocab_size=None, report=None):
        pass 

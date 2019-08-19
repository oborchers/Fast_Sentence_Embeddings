#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from fse.models.base_s2v import BaseSentence2VecModel
from fse.models.inputs import IndexedSentence

from gensim.models.keyedvectors import BaseKeyedVectors
from gensim.models.utils_any2vec import ft_ngram_hashes

from numpy import ndarray, ones, float32 as REAL, sum as np_sum, multiply as np_mult, zeros, max as np_max

from typing import List

import logging

logger = logging.getLogger(__name__)

def train_average_np(model:BaseSentence2VecModel, indexed_sentences:List[IndexedSentence], target:ndarray) -> [int,int]:
        size = model.wv.vector_size
        vocab = model.wv.vocab

        w_vectors = model.wv.vectors
        w_weights = model.word_weights

        s_vectors = target

        is_ft = model.is_ft

        if is_ft:
            # NOTE: For Fasttext: Ignore wv.vectors_vocab
            # model.wv.vectors is the sum of vectors_vocab + ngram_vectors
            # See :method: `~gensim.models.keyedvectors.FastTextKeyedVectors.adjust_vectors`
            w_vectors = model.wv.vectors_vocab
            ngram_vectors = model.wv.vectors_ngrams
            min_n = model.wv.min_n
            max_n = model.wv.max_n
            bucket = model.wv.bucket
            oov_weight = np_max(w_weights)

        eff_sentences, eff_words = 0, 0

        if not is_ft:
            for obj in indexed_sentences:
                sent_adr = obj.index
                sent = obj.words
                word_indices = [vocab[word].index for word in sent if word in vocab]
                if not len(word_indices):
                    continue

                eff_sentences += 1
                eff_words += len(word_indices)

                vec = np_sum(np_mult(w_vectors[word_indices],w_weights[word_indices][:,None]) , axis=0)
                vec *= 1/len(word_indices)
                s_vectors[sent_adr] = vec.astype(REAL)
        else:
            for obj in indexed_sentences:
                sent_adr = obj.index
                sent = obj.words
                
                if not len(sent):
                    continue
                vec = zeros(size, dtype=REAL)

                eff_sentences += 1
                eff_words += len(sent) # Counts everything in the sentence

                for word in sent:
                    if word in vocab:
                        word_index = vocab[word].index
                        vec += w_vectors[word_index] * w_weights[word_index]
                    else:
                        ngram_hashes = ft_ngram_hashes(word, min_n, max_n, bucket, True)
                        if len(ngram_hashes) == 0:
                            continue
                        vec += oov_weight * (np_sum(ngram_vectors[ngram_hashes], axis=0) / len(ngram_hashes))
                    # Implicit addition of zero if oov does not contain any ngrams
                s_vectors[sent_adr] = vec / len(sent)

        return eff_sentences, eff_words

# try:
#     from fse.models.average_inner import train_average_cy
#     from fse.models.average_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
#     train_average = train_average_cy
# except ImportError:
FAST_VERSION = -1
MAX_WORDS_IN_BATCH = 10000
train_average = train_average_np

class Average(BaseSentence2VecModel):

    def __init__(self, model:BaseKeyedVectors, sv_mapfile_path:str=None, wv_mapfile_path:str=None, workers:int=1, lang_freq:str=None):

        super(Average, self).__init__(
            model=model, sv_mapfile_path=sv_mapfile_path, wv_mapfile_path=wv_mapfile_path,
            workers=workers, lang_freq=lang_freq,
            batch_words=MAX_WORDS_IN_BATCH, fast_version=FAST_VERSION
            )

    def _do_train_job(self, data_iterable:List[IndexedSentence], target:ndarray) -> [int, int]:
        eff_sentences, eff_words = train_average(model=self, indexed_sentences=data_iterable, target=target)
        return eff_sentences, eff_words

    def _check_parameter_sanity(self):
        if not all(self.word_weights == 1.): 
            raise ValueError("For averaging, all word weights must be one")

    def _pre_train_calls(self):
        pass
    
    def _check_dtype_santiy(self):
        pass

    def _post_train_calls(self):
        pass
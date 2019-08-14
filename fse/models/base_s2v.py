#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers


class BaseSentence2VecModel(utils.SaveLoad):

    def __init__(self, model, workers=2, no_frequency=False, lang="en"):
        pass

    def _do_train_job(self):
        raise NotImplementedError()

    def _check_training_sanity(self):
        raise NotImplementedError()

    def _log_progress(self):
        raise NotImplementedError()

    def _log_train_end(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def build_vocab(self):
        raise NotImplementedError()

    def estimate_memory(self, vocab_size=None, report=None):
        pass 

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <oliver-borchers@outlook.de>
# Copyright (C) 2020 Oliver Borchers

from fse.models.base_iterator import base_iterator
from fse.test.model_shared_imports import *

from fse.models.average import Average

class TestSentenceIterator(unittest.TestCase):
    def setUp(self):
        self.sentences = IndexedLineDocument(CORPUS)
        self.model = Average(W2V_DET, lang_freq="en")

        self.model.prep.prepare_vectors(
            sv=self.model.sv, total_sentences=len(ENUM_SENTENCES), update=False
        )
        self.model.window_size=2
        self.model.window_stride=2

        self.model._pre_train_calls()
        self.mem = self.model._get_thread_working_mem()

    def test_iterator_w2v_det(self):
        def window_merger(*args, **kwargs): pass
        def sentence_merger(*args, **kwargs): pass

        it = base_iterator(
            self.model,
            indexed_sentences = ENUM_SENTENCES,
            target = self.model.sv.vectors,
            memory = self.mem,
            window_merger=window_merger,
            sentence_merger=sentence_merger,
        )
        
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
    )
    unittest.main()

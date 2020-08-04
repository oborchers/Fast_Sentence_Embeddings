import logging
import unittest

import numpy as np

from pathlib import Path

from gensim.models import Word2Vec, FastText

from fse.models.base_s2v import EPS
from fse.inputs import IndexedLineDocument

logger = logging.getLogger(__name__)

# Global objs
CORPUS = Path("fse/test/test_data/test_sentences.txt")
with open(CORPUS, "r") as f:
    SENTENCES = [l.split() for i, l in enumerate(f)]
ENUM_SENTENCES = [(l, i) for i, l in enumerate(SENTENCES)]

# Models
DIM = 5

# Deterministic W2V
W2V_DET = Word2Vec(min_count=1, size=DIM)
W2V_DET.build_vocab(SENTENCES)
W2V_DET.wv.vectors[:,] = np.arange(len(W2V_DET.wv.vectors), dtype=np.float32)[:, None]

# Random W2V
W2V_RNG = Word2Vec(min_count=1, size=DIM)
W2V_RNG.build_vocab(SENTENCES)

# Random W2V Large
W2V_RNG_LRG = Word2Vec(min_count=1, size=50)
W2V_RNG_LRG.build_vocab(SENTENCES)

# Deterministic FT
FT_DET = FastText(min_count=1, size=DIM)
FT_DET.build_vocab(SENTENCES)
FT_DET.wv.vectors = FT_DET.wv.vectors_vocab = np.ones_like(
    FT_DET.wv.vectors, dtype=np.float32
)
FT_DET.wv.vectors_ngrams[:,] = np.arange(
    len(FT_DET.wv.vectors_ngrams), dtype=np.float32
)[:, None]

# Random FT
FT_RNG = FastText(min_count=1, size=DIM)
FT_RNG.build_vocab(SENTENCES)

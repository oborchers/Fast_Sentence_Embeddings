"""The gensim types the 1.x models accept, kept apart so that fse._validation needs no gensim."""

from typing import TypeAlias

from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors

#: Word vectors or a gensim model that holds them in ``wv``.
WordVectorModel: TypeAlias = KeyedVectors | Word2Vec | FastText

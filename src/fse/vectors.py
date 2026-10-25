#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Class to obtain BaseKeyedVector from."""

from pathlib import Path

from gensim.models.fasttext import FastTextKeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

_SUFFIX: str = ".model"


def _download(model: str) -> str:
    """Download a pre-trained model from the hub and return the path to its file."""
    try:
        path = Path(snapshot_download(repo_id=f"fse/{model}"))
    except RepositoryNotFoundError as err:
        raise ValueError(f"model {model} does not exist") from err

    if not path.exists():
        raise FileNotFoundError(f"download of model {model} did not produce {path}")

    return (path / (model + _SUFFIX)).as_posix()


class Vectors(KeyedVectors):
    """Class to instantiates vectors from pretrained models."""

    @classmethod
    def from_pretrained(cls, model: str, mmap: str = None):
        """Method to load vectors from a pre-trained model.

        Parameters
        ----------
        model : :str: of the model name to load from the bug. For example: "glove-wiki-gigaword-50"
        mmap : :str: If to load the vectors in mmap mode.

        Returns
        -------
        Vectors
            An object of pretrained vectors.
        """
        return super().load(_download(model), mmap=mmap)


class FTVectors(FastTextKeyedVectors):
    """Class to instantiates FT vectors from pretrained models."""

    @classmethod
    def from_pretrained(cls, model: str, mmap: str = None):
        """Method to load vectors from a pre-trained model.

        Parameters
        ----------
        model : :str: of the model name to load from the bug. For example: "glove-wiki-gigaword-50"
        mmap : :str: If to load the vectors in mmap mode.

        Returns
        -------
        Vectors
            An object of pretrained vectors.
        """
        return super().load(_download(model), mmap=mmap)

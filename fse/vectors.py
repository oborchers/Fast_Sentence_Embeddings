#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Class to obtain BaseKeyedVector from."""

from pathlib import Path

from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
from huggingface_hub import snapshot_download
from requests import HTTPError

_SUFFIX: str = ".model"


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
        try:
            path = Path(snapshot_download(repo_id=f"fse/{model}"))
        except HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError(f"model {model} does not exist")
            raise

        assert path.exists(), "something went wrong. the file wasn't downloaded."

        return super(Vectors, cls).load(
            (path / (model + _SUFFIX)).as_posix(), mmap=mmap
        )


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
        try:
            path = Path(snapshot_download(repo_id=f"fse/{model}"))
        except HTTPError as err:
            if err.response.status_code == 404:
                raise ValueError(f"model {model} does not exist")
            raise

        assert path.exists(), "something went wrong. the file wasn't downloaded."

        return super(FTVectors, cls).load(
            (path / (model + _SUFFIX)).as_posix(), mmap=mmap
        )

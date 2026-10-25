#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Automated tests for loading pretrained vectors."""

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from fse.vectors import FTVectors, Vectors


def _not_found(*args, **kwargs):
    raise RepositoryNotFoundError("not found", response=MagicMock())


@pytest.mark.network
class TestVectorsFromHub:
    def test_from_pretrained(self):
        """Test the pretrained vectors."""
        vectors = Vectors.from_pretrained("glove-wiki-gigaword-50")
        assert vectors.vector_size == 50
        assert vectors.vectors.shape == (400000, 50)

        vectors = Vectors.from_pretrained("glove-wiki-gigaword-50", mmap="r")
        assert vectors.vector_size == 50
        assert vectors.vectors.shape == (400000, 50)

    def test_missing_model_on_hub(self):
        """A model that does not exist on the hub raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            Vectors.from_pretrained("unittest")


@pytest.mark.parametrize("cls", [Vectors, FTVectors])
@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [(_not_found, ValueError), (RuntimeError, RuntimeError)],
)
def test_missing_model(cls, side_effect, expected):
    """A failing download is reported as ValueError or passed through."""
    with patch("fse.vectors.snapshot_download") as mock:
        mock.side_effect = side_effect
        with pytest.raises(expected):
            cls.from_pretrained("unittest")


def test_missing_download(tmp_path):
    """A download that did not produce a directory raises FileNotFoundError."""
    with patch("fse.vectors.snapshot_download") as mock:
        mock.return_value = (tmp_path / "does_not_exist").as_posix()
        with pytest.raises(FileNotFoundError):
            Vectors.from_pretrained("unittest")


def test_ft_from_pretrained(test_data_dir):
    """Test the pretrained vectors."""
    with (
        patch("fse.vectors.snapshot_download") as mock,
        patch("fse.vectors.FastTextKeyedVectors.load"),
    ):
        mock.return_value = test_data_dir.as_posix()
        FTVectors.from_pretrained("ft")

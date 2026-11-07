#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse._store._vectorstore_open``: opening a store from a path or from several (a view)."""

from __future__ import annotations

import numpy as np
import pytest

import fse
from fse import VectorStore
from fse._store import _vectorstore_open
from support.encoder import finished
from support.runtime import texts

ENCODER = finished()


def _write(path, sentences):
    with fse.ShardSink(path, ENCODER, shard_rows=16) as sink:
        ENCODER.encode(sentences, sink=sink)
    return path


def test_open_store_is_what_the_class_method_gives(tmp_path):
    path = _write(tmp_path / "a", texts(40))
    with (
        _vectorstore_open.open_store(VectorStore, path, encoder=ENCODER) as direct,
        VectorStore.open(path, encoder=ENCODER) as via,
    ):
        assert isinstance(direct, VectorStore)
        assert len(direct) == len(via) == 40
        np.testing.assert_array_equal(direct.get_vectors(), via.get_vectors())


def test_open_store_view_joins_the_members(tmp_path):
    first = _write(tmp_path / "a", texts(10))
    second = _write(tmp_path / "b", texts(7))
    with _vectorstore_open.open_store_view(VectorStore, [first, second]) as view:
        assert len(view) == 17
        assert view.path is None


def test_open_store_rejects_an_unknown_backend(tmp_path):
    path = _write(tmp_path / "a", texts(5))
    with pytest.raises(ValueError, match="backend must be one of"):
        _vectorstore_open.open_store(VectorStore, path, backend="nope")


def test_open_store_rejects_an_encoder_of_another_kind(tmp_path):
    path = _write(tmp_path / "a", texts(5))
    with pytest.raises(TypeError, match="encoder must be an Encoder"):
        _vectorstore_open.open_store(VectorStore, path, encoder=object())

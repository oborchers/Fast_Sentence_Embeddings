#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers


"""Automated tests for checking the input methods."""

import numpy as np
import pytest

from fse.inputs import (
    BaseIndexedList,
    CIndexedList,
    CSplitCIndexedList,
    CSplitIndexedList,
    IndexedLineDocument,
    IndexedList,
    SplitCIndexedList,
    SplitIndexedList,
)


@pytest.fixture
def list_a():
    return ["the dog is good", "it's nice and comfy"]


@pytest.fixture
def capitalized():
    return ["The Dog is good", "it's nice and comfy"]


def _lower_split(text):
    return text.lower().split()


def _assert_immutable(il):
    with pytest.raises(NotImplementedError):
        il.__delitem__(0)
    with pytest.raises(NotImplementedError):
        il.__setitem__(0, "the")
    with pytest.raises(NotImplementedError):
        il.insert(0, "the")
    with pytest.raises(NotImplementedError):
        il.append("the")
    with pytest.raises(NotImplementedError):
        il.extend(["the", "dog"])


class TestBaseIndexedList:
    @pytest.fixture
    def list_b(self):
        return ["lorem ipsum dolor", "si amet"]

    @pytest.fixture
    def base(self, list_a):
        return BaseIndexedList(list_a)

    @pytest.fixture
    def multi(self, list_a, list_b):
        return BaseIndexedList(list_a, list_b, [s.split() for s in list_a])

    def test_init(self, list_a):
        _ = BaseIndexedList(list_a)

    def test_init_mult_arg(self, multi):
        assert len(multi.items) == 6

    def test_init_ndarray(self, list_a):
        _ = BaseIndexedList(np.array(list_a))

    @pytest.mark.parametrize("value", [1, "Hello"])
    def test__check_list_type(self, base, value):
        with pytest.raises(TypeError):
            base._check_list_type(value)

    def test__check_str_type(self, base):
        assert base._check_str_type("Hello") == 1

    @pytest.mark.parametrize("value", [1, []])
    def test__check_str_type_wrong_type(self, base, value):
        with pytest.raises(TypeError):
            base._check_str_type(value)

    def test__len(self, base):
        assert len(base) == 2

    def test__str(self, base):
        assert str(base) == "['the dog is good', \"it's nice and comfy\"]"

    def test__getitem(self, base):
        with pytest.raises(NotImplementedError):
            base[0]

    def test__delitem(self, multi):
        multi.__delitem__(0)
        assert len(multi) == 5

    def test__setitem(self, multi):
        multi.__setitem__(0, "is it me?")
        assert multi.items[0] == "is it me?"

    def test_append(self, multi):
        multi.append("is it me?")
        assert multi.items[-1] == "is it me?"

    def test_extend(self, multi, list_a):
        multi.extend(list_a)
        assert len(multi) == 8

        with pytest.raises(TypeError, match="A set has no order"):
            multi.extend({"hello there", "its a set"})
        assert len(multi) == 8

    def test_extend_ndarr(self):
        base = BaseIndexedList(np.array([str(i) for i in [1, 2, 3, 4]]))
        base.extend(np.array([str(i) for i in [1, 2, 3, 4]]))
        assert len(base) == 8


class TestIndexedList:
    def test_init(self, list_a):
        _ = IndexedList(list_a)

    def test_getitem(self, list_a):
        il = IndexedList(list_a, [s.split() for s in list_a])
        assert il[0] == ("the dog is good", 0)

    def test_split(self, list_a):
        base = SplitIndexedList(list_a)
        assert base[0][0] == ["the", "dog", "is", "good"]


class TestCIndexedList:
    @pytest.fixture
    def il(self, capitalized):
        return CIndexedList(capitalized, custom_index=[1, 1])

    def test_cust_index(self, il):
        assert il[0][1] == 1

    def test_wrong_len(self, capitalized):
        with pytest.raises(RuntimeError):
            CIndexedList(capitalized, custom_index=[1])

    def test_mutable_funcs(self, il):
        _assert_immutable(il)


class TestCSplitIndexedList:
    def test_getitem(self, capitalized):
        il = CSplitIndexedList(capitalized, custom_split=_lower_split)
        assert il[0][0] == ["the", "dog", "is", "good"]


class TestSplitCIndexedList:
    @pytest.fixture
    def il(self, capitalized):
        return SplitCIndexedList(capitalized, custom_index=[1, 1])

    def test_getitem(self, il):
        assert il[0] == (["The", "Dog", "is", "good"], 1)

    def test_mutable_funcs(self, il):
        _assert_immutable(il)


class TestCSplitCIndexedList:
    @pytest.fixture
    def il(self, capitalized):
        return CSplitCIndexedList(capitalized, custom_split=_lower_split, custom_index=[1, 1])

    def test_getitem(self, il):
        assert il[0] == (["the", "dog", "is", "good"], 1)

    def test_mutable_funcs(self, il):
        _assert_immutable(il)


class TestIndexedLineDocument:
    @pytest.fixture
    def doc(self, corpus_path):
        return IndexedLineDocument(corpus_path)

    def test_getitem(self, doc):
        assert doc[0] == "Good stuff i just wish it lasted longer"
        assert doc[19] == "Save yourself money and buy it direct from lg"
        assert doc[-1] == "I am not sure if it is a tracfone problem or the battery"

    def test_yield(self, doc):
        first = (["Good", "stuff", "i", "just", "wish", "it", "lasted", "longer"], 0)
        last = (
            [
                "I",
                "am",
                "not",
                "sure",
                "if",
                "it",
                "is",
                "a",
                "tracfone",
                "problem",
                "or",
                "the",
                "battery",
            ],
            99,
        )
        items = list(doc)
        assert len(items) == 100
        assert items[0] == first
        assert items[99] == last


def test_indexed_list_rejects_set():
    """A set has no order, so the sentence indices would be arbitrary."""
    from fse import IndexedList

    with pytest.raises(TypeError, match="A set has no order"):
        IndexedList({"the dog", "a cat"})


def test_indexed_line_document_validates_arguments(corpus_path):
    from pydantic import ValidationError

    from fse import IndexedLineDocument

    assert len(list(IndexedLineDocument(corpus_path))) == 100
    with pytest.raises(ValidationError):
        IndexedLineDocument(corpus_path, get_able="yes")

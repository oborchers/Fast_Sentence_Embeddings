#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

from collections.abc import Callable, Iterator, MutableSequence
from operator import methodcaller
from pathlib import Path
from typing import Any

from gensim.utils import any2unicode
from numpy import concatenate, ndarray
from smart_open import open

from fse._validation import PathLike, validate_arguments

# Calls .split() on each item like __getitem__ does, also for numpy str_ items.
_split = methodcaller("split")


class BaseIndexedList(MutableSequence):
    def __init__(self, *args: list | ndarray):
        """Base object to be used for feeding in-memory stored lists of sentences to the
        training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single container. Can be single or multiple
            list/ndarray objects. A single list or ndarray is used as is, without a copy,
            so later changes to it are visible here.
        """

        self.items: Any = list()

        if len(args) == 1:
            self._check_list_type(args[0])
            self.items = args[0]
        else:
            for arg in args:
                self.extend(arg)

        super().__init__()

    def _check_list_type(self, obj: object):
        """Checks input validity."""
        if isinstance(obj, set):
            # A set has no order, so the sentence indices would be arbitrary.
            raise TypeError("Arg must be a list or ndarray. A set has no order; pass sorted(...)")
        if isinstance(obj, (list, ndarray)):
            return 1
        else:
            raise TypeError(f"Arg must be a list or ndarray. Got {type(obj)}")

    def _check_str_type(self, obj: object):
        """Checks input validity."""
        if isinstance(obj, str):
            return 1
        else:
            raise TypeError(f"Arg must be str type. Got {type(obj)}")

    def __len__(self):
        """List length.

        Returns
        -------
        int
           Length of the IndexedList
        """
        return len(self.items)

    def __str__(self):
        """Human readable representation of the object's state, used for debugging.

        Returns
        -------
        str
           Human readable representation of the object's state (words and tags).
        """
        return str(self.items)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple ([str], int)
            Returns the core object, a tuple, for every sentence embedding model.
        """
        raise NotImplementedError()

    def __delitem__(self, i: int):  # type: ignore[override]
        """Delete an item."""
        del self.items[i]

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        """Sets an item."""
        self._check_str_type(item)
        self.items[i] = item

    def insert(self, i: int, item: str):
        """Inserts an item at a position."""
        self._check_str_type(item)
        self.items.insert(i, item)

    def append(self, item: str):
        """Appends item at last position."""
        self._check_str_type(item)
        self.insert(len(self.items), item)

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        """Extens list."""
        self._check_list_type(arg)

        if not isinstance(arg, ndarray):
            self.items += arg
        else:
            self.items = concatenate([self.items, arg], axis=0)


class IndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray):
        """Quasi-list to be used for feeding in-memory stored lists of sentences to the
        training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        """
        super().__init__(*args)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i), i)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not IndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(self.items, range(len(self.items)), strict=True)


class CIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray, custom_index: list | ndarray):
        """Quasi-list with custom indices to be used for feeding in-memory stored lists
        of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_index : list, ndarray
            Custom index to support many to one mappings.
        """
        self.custom_index = custom_index

        super().__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(
                f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}"
            )

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i), self.custom_index[i])

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not CIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(self.items, self.custom_index, strict=True)

    def __delitem__(self, i: int):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def insert(self, i: int, item: str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item: str):
        raise NotImplementedError("Method currently not supported")

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")


class SplitIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray):
        """Quasi-list with string splitting to be used for feeding in-memory stored
        lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        """
        super().__init__(*args)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i).split(), i)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not SplitIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(_split, self.items), range(len(self.items)), strict=True)


class SplitCIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray, custom_index: list | ndarray):
        """Quasi-list with custom indices and string splitting to be used for feeding
        in-memory stored lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_index : list, ndarray
            Custom index to support many to one mappings.
        """
        self.custom_index = custom_index

        super().__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(
                f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}"
            )

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i).split(), self.custom_index[i])

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not SplitCIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(_split, self.items), self.custom_index, strict=True)

    def __delitem__(self, i: int):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def insert(self, i: int, item: str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item: str):
        raise NotImplementedError("Method currently not supported")

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")


class CSplitIndexedList(BaseIndexedList):
    def __init__(self, *args: list | ndarray, custom_split: Callable[[str], list[str]]):
        """Quasi-list with custom string splitting to be used for feeding in-memory
        stored lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_split : callable
            Split function to be used to convert strings into list of str.
        """
        self.custom_split = custom_split
        super().__init__(*args)

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.custom_split(self.items.__getitem__(i)), i)

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not CSplitIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(self.custom_split, self.items), range(len(self.items)), strict=True)


class CSplitCIndexedList(BaseIndexedList):
    def __init__(
        self,
        *args: list | ndarray,
        custom_split: Callable[[str], list[str]],
        custom_index: list | ndarray,
    ):
        """Quasi-list with custom indices and ustom string splitting to be used for
        feeding in-memory stored lists of sentences to the training routine.

        Parameters
        ----------
        args : lists, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple lists or arrays; sets are rejected.
        custom_split : callable
            Split function to be used to convert strings into list of str.
        custom_index : list, ndarray
            Custom index to support many to one mappings.
        """
        self.custom_split = custom_split
        self.custom_index = custom_index

        super().__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(
                f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}"
            )

    def __getitem__(self, i: int) -> tuple:  # type: ignore[override]
        """Getitem method.

        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.custom_split(self.items.__getitem__(i)), self.custom_index[i])

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all sentences, faster than indexing each one."""
        if type(self).__getitem__ is not CSplitCIndexedList.__getitem__:
            # A subclass that changes __getitem__ is iterated through it.
            return MutableSequence.__iter__(self)
        return zip(map(self.custom_split, self.items), self.custom_index, strict=True)

    def __delitem__(self, i: int):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def __setitem__(self, i: int, item: str):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")

    def insert(self, i: int, item: str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item: str):
        raise NotImplementedError("Method currently not supported")

    def extend(self, arg: list | ndarray):  # type: ignore[override]
        raise NotImplementedError("Method currently not supported")


class IndexedLineDocument:
    @validate_arguments
    def __init__(self, path: PathLike, get_able: bool = True):
        """Iterate over a file that contains sentences: one line = tuple([str], int).

        Words are expected to be already preprocessed and separated by whitespace. Sentence tags are constructed
        automatically from the sentence line number.

        Parameters
        ----------
        path : str
            The path of the file to read and return lines from
        get_able : bool, optional
            Use to determine if the IndexedLineDocument is indexable.
            This functionality is required if you want to pass an indexable to
            :meth:`~fse.models.sentencevectors.SentenceVectors.most_similar`.

        """
        self.path = Path(path)
        self.line_offset: list[int] = []
        self.get_able = get_able

        if self.get_able:
            self._build_offsets()

    def _build_offsets(self):
        """Builds an offset table to index the file."""
        with open(self.path, "rb") as f:
            offset = f.tell()
            for line in f:
                self.line_offset.append(offset)
                offset += len(line)

    def __getitem__(self, i):
        """Returns the line indexed by i. Primarily used for.

        :meth:`~fse.models.sentencevectors.SentenceVectors.most_similar`

        Parameters
        ----------
        i : int
            The line index used to index the file

        Returns
        -------
        str
            line at the current index
        """
        if not self.get_able:
            raise RuntimeError("To index the lines, you must contruct with get_able=True")

        with open(self.path, "rb") as f:
            f.seek(self.line_offset[i])
            output = f.readline()
            f.seek(0)
            return any2unicode(output).rstrip()

    def __iter__(self):
        """Iterate through the lines in the source.

        Yields
        ------
        tuple : (list[str], int)
            Tuple of list of string and index
        """
        with open(self.path, "rb") as f:
            for i, line in enumerate(f):
                yield (any2unicode(line).split(), i)

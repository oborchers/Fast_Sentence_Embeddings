#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from typing import NamedTuple, List, MutableSequence

from gensim.utils import any2unicode
from smart_open import open

from pathlib import Path

from numpy import ndarray, concatenate

class BaseIndexedList(MutableSequence):

    def __init__(self, *args:[list, set, ndarray]):
        """ Base object to be used for feeding in-memory stored lists of sentences to
        the training routine.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set/ndarray objects.

        """

        self.items = list()
        
        if len(args) == 1:
            self._check_list_type(args[0])
            self.items = args[0]
        else:
            for arg in args:
                self.extend(arg)

        super().__init__()

    def _check_list_type(self, obj:object):
        """ Checks input validity """
        if isinstance(obj, (list, set, ndarray)):
            return 1
        else:
            raise TypeError(f"Arg must be list/set type. Got {type(obj)}")

    def _check_str_type(self, obj:object):
        """ Checks input validity """
        if isinstance(obj, str):
            return 1
        else:
            raise TypeError(f"Arg must be str type. Got {type(obj)}")

    def __len__(self):
        """ List length 
        
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

    def __getitem__(self, i:int) -> tuple:
        """  Getitem method
        
        Returns
        -------
        tuple ([str], int)
            Returns the core object, a tuple, for every sentence embedding model.
        """
        raise NotImplementedError()

    def __delitem__(self, i:int):
        """ Delete an item """
        del self.items[i]
        
    def __setitem__(self, i:int, item:str):
        """ Sets an item """
        self._check_str_type(item)
        self.items[i] = item

    def insert(self, i:int, item:str):
        """ Inserts an item at a position """
        self._check_str_type(item)
        self.items.insert(i, item)

    def append(self, item:str):
        """ Appends item at last position"""
        self._check_str_type(item)
        self.insert(len(self.items), item)
    
    def extend(self, arg:[list, set, ndarray]):
        """ Extens list """
        self._check_list_type(arg)

        if not isinstance(arg, ndarray):
            self.items += arg
        else:
            self.items = concatenate([self.items, arg], axis=0)

class IndexedList(BaseIndexedList):

    def __init__(self, *args:[list, set, ndarray]):
        """ Quasi-list to be used for feeding in-memory stored lists of sentences to
        the training routine.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.

        """
        super(IndexedList, self).__init__(*args)

    def __getitem__(self, i:int) -> tuple:
        """  Getitem method
        
        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i), i)

class CIndexedList(BaseIndexedList):

    def __init__(self, *args:[list, set, ndarray], custom_index:[list, ndarray]):
        """ Quasi-list with custom indices to be used for feeding in-memory stored lists of sentences to
        the training routine.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.
        custom_index : list, ndarray
            Custom index to support many to one mappings.

        """
        self.custom_index = custom_index

        super(CIndexedList, self).__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}")

    def __getitem__(self, i:int) -> tuple:
        """  Getitem method
        
        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i), self.custom_index[i])

    def __delitem__(self, i:int):
        raise NotImplementedError("Method currently not supported")
        
    def __setitem__(self, i:int, item:str):
        raise NotImplementedError("Method currently not supported")

    def insert(self, i:int, item:str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item:str):
        raise NotImplementedError("Method currently not supported")
    
    def extend(self, arg:[list, set, ndarray]):
        raise NotImplementedError("Method currently not supported")

class SplitIndexedList(BaseIndexedList):

    def __init__(self, *args:[list, set, ndarray]):
        """ Quasi-list with string splitting to be used for feeding in-memory stored lists of sentences to
        the training routine.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.

        """
        super(SplitIndexedList, self).__init__(*args)

    def __getitem__(self, i:int) -> tuple:
        """  Getitem method
        
        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i).split(), i)

class SplitCIndexedList(BaseIndexedList):

    def __init__(self, *args:[list, set, ndarray], custom_index:[list, ndarray]):
        """ Quasi-list with custom indices and string splitting to be used for feeding in-memory stored lists of sentences to
        the training routine.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.
        custom_index : list, ndarray
            Custom index to support many to one mappings.

        """
        self.custom_index = custom_index

        super(SplitCIndexedList, self).__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}")


    def __getitem__(self, i:int) -> tuple:
        """  Getitem method
        
        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.items.__getitem__(i).split(), self.custom_index[i])
    
    def __delitem__(self, i:int):
        raise NotImplementedError("Method currently not supported")
        
    def __setitem__(self, i:int, item:str):
        raise NotImplementedError("Method currently not supported")

    def insert(self, i:int, item:str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item:str):
        raise NotImplementedError("Method currently not supported")
    
    def extend(self, arg:[list, set, ndarray]):
        raise NotImplementedError("Method currently not supported")

class CSplitIndexedList(BaseIndexedList):

    def __init__(self, *args:[list, set, ndarray], custom_split:callable):
        """ Quasi-list with custom string splitting to be used for feeding in-memory stored lists of sentences to
        the training routine.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.
        custom_split : callable
            Split function to be used to convert strings into list of str.

        """
        self.custom_split = custom_split
        super(CSplitIndexedList, self).__init__(*args)

    def __getitem__(self, i:int) -> tuple:
        """  Getitem method
        
        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.custom_split(self.items.__getitem__(i)), i)

class CSplitCIndexedList(BaseIndexedList):

    def __init__(self, *args:[list, set, ndarray], custom_split:callable, custom_index:[list, ndarray]):
        """ Quasi-list with custom indices and ustom string splitting to be used for feeding in-memory stored lists of sentences to
        the training routine.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.
        custom_split : callable
            Split function to be used to convert strings into list of str.
        custom_index : list, ndarray
            Custom index to support many to one mappings.

        """
        self.custom_split = custom_split
        self.custom_index = custom_index
        
        super(CSplitCIndexedList, self).__init__(*args)

        if len(self.items) != len(self.custom_index):
            raise RuntimeError(f"Size of custom_index {len(custom_index)} does not match items {len(self.items)}")

    def __getitem__(self, i:int) -> tuple:
        """  Getitem method
        
        Returns
        -------
        tuple
            Returns the core object, tuple, for every sentence embedding model.
        """
        return (self.custom_split(self.items.__getitem__(i)), self.custom_index[i])

    def __delitem__(self, i:int):
        raise NotImplementedError("Method currently not supported")
        
    def __setitem__(self, i:int, item:str):
        raise NotImplementedError("Method currently not supported")

    def insert(self, i:int, item:str):
        raise NotImplementedError("Method currently not supported")

    def append(self, item:str):
        raise NotImplementedError("Method currently not supported")
    
    def extend(self, arg:[list, set, ndarray]):
        raise NotImplementedError("Method currently not supported")

class IndexedLineDocument(object):

    def __init__(self, path, get_able=True):
        """ Iterate over a file that contains sentences: one line = tuple([str], int).

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
        self.line_offset = list()
        self.get_able = bool(get_able)

        if self.get_able:
            self._build_offsets()
    
    def _build_offsets(self):
        """ Builds an offset table to index the file """
        with open(self.path, "rb") as f:
            offset = f.tell()
            for line in f:
                self.line_offset.append(offset)
                offset += len(line)
    
    def __getitem__(self, i):
        """ Returns the line indexed by i. Primarily used for 
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
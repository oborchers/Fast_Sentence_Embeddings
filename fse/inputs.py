#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from typing import NamedTuple, List, MutableSequence

from gensim.utils import any2unicode
from smart_open import open

from pathlib import Path

from numpy import ndarray

class IndexedSentence(NamedTuple):
    words: List[str]
    index: int

    def __str__(self):
        """Human readable representation of the object's state, used for debugging.

        Returns
        -------
        str
           Human readable representation of the object's state (words and tags).

        """
        return f"{self.__class__.__name__}({self.words}, {self.index})"

class IndexedList(MutableSequence):
    
    def __init__(self, *args:[list, set, ndarray], split:bool=True, split_func:bool=None, pre_splitted:bool=False, custom_index:bool=None):
        """ Quasi-list to be used for feeding in-memory stored lists of sentences to
        the training routine as IndexedSentence.

        Parameters
        ----------
        args : lists, sets, ndarray
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.
        split : bool, optional
            If true performs a split function on the strings contained in the list.
        split_func : function, optional
            A user definable split function which turns a string into a list of strings.
        pre_splitted : bool, optional
            Determines if the input is already splitted in the format of ["token0", "token1"]

        """
        self.pre_splitted = bool(pre_splitted)
        self.split = bool(split) if not self.pre_splitted else False
        self.split_func = split_func
        self._check_kwargs_sanity()
        
        self.items = list()

        if len(args) == 1:
            self._check_list_type(args[0])
            self.items = args[0]
        else:
            for arg in args:
                self._check_list_type(arg)
                self.items += arg

        self.custom_index = custom_index
        if custom_index is not None:
            # Custom index for many-to-one averages
            self.custom_index = custom_index
            if len(self.items) != len(self.custom_index):
                RuntimeError(f"Custom index has wrong length {len(self.custom_index)}")

        self._set_get_meth()

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
    
    def _check_kwargs_sanity(self):
        """ Checks argument validity """
        if self.split and self.split_func is not None:
            raise RuntimeError("You must provide either split=True or a split_func, not both")
        if (self.split or self.split_func is not None) and self.pre_splitted:
            raise RuntimeError("Split function and pre_splitted are not compatible")

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

    def _get_presplitted(self, i:int) -> IndexedSentence:
        """ Get with presplitted list. Fastest. """
        return IndexedSentence(self.items.__getitem__(i), i)

    def _get_not_splitted(self, i:int) -> IndexedSentence:
        """ Get with regular split func """
        return IndexedSentence(self.items.__getitem__(i).split(), i)

    def _get_splitfunc(self, i:int) -> IndexedSentence:
        """ Get with custom split func """
        return IndexedSentence(self.split_func(self.items.__getitem__(i)), i)

    def _get_presplitted_cust_idx(self, i:int) -> IndexedSentence:
        """ Get with presplitted list """
        return IndexedSentence(self.items.__getitem__(i), self.custom_index.__getitem__(i))

    def _get_not_splitted_cust_idx(self, i:int) -> IndexedSentence:
        """ Get with regular split func """
        return IndexedSentence(self.items.__getitem__(i).split(), self.custom_index.__getitem__(i))

    def _get_splitfunc_cust_idx(self, i:int) -> IndexedSentence:
        """ Get with custom split func. Slowest."""
        return IndexedSentence(self.split_func(self.items.__getitem__(i)), self.custom_index.__getitem__(i))

    def _set_get_meth(self):
        """ Sets the __getitem__ method so that we only have to check this once"""
        if self.custom_index is not None:
            if self.pre_splitted:
                self._getmeth = self._get_presplitted_cust_idx
            if self.split:
                self._getmeth = self._get_not_splitted_cust_idx
            if self.split_func is not None:
                self._getmeth = self._get_splitfunc_cust_idx
        else:
            if self.pre_splitted:
                self._getmeth = self._get_presplitted
            if self.split:
                self._getmeth = self._get_not_splitted
            if self.split_func is not None:
                self._getmeth = self._get_splitfunc
    
    def __getitem__(self, i:int) -> IndexedSentence:
        """  Getitem method
        
        Returns
        -------
        IndexedSentence
            Returns the core object, IndexedSentence, for every sentence embedding model.
        """
        # TODO: Implement this beast.
        #return (self.items.__getitem__(i), i)
        return self._getmeth(i)

    def __delitem__(self, i:int):
        """ Delete an item """
        del self.items[i]
        if self.custom_index is not None:
            del self.custom_index[i]

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
    
    def extend(self, *args:[list, set, ndarray]):
        """ Extens list """
        for arg in args:
            self._check_list_type(arg)
            self.items += arg

class IndexedLineDocument(object):

    def __init__(self, path, get_able=True):
        """ Iterate over a file that contains sentences: one line = :class:`~fse.inputs.IndexedSentence` object.

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
            raise RuntimeError("To index the lines you must contruct with get_able=True")

        with open(self.path, "rb") as f:
            f.seek(self.line_offset[i])
            output = f.readline()
            f.seek(0)
            return any2unicode(output).rstrip()

    def __iter__(self):
        """Iterate through the lines in the source.

        Yields
        ------
        :class:`~fse.inputs.IndexedSentence`
            IndexedSentence from `path` specified in the constructor.

        """
        with open(self.path, "rb") as f:
            for i, line in enumerate(f):
                yield IndexedSentence(any2unicode(line).split(), i)
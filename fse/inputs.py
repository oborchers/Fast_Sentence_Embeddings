#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from typing import NamedTuple, List, MutableSequence

from gensim.utils import any2unicode, open as s_open

from pathlib import Path

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
    
    def __init__(self, *args, split=True, split_func=None):
        """ Quasi-list to be used for feeding in-memory stored lists of sentences to
        the training routine as indexed sentence.

        Parameters
        ----------
        args : lists, sets
            Arguments to be merged into a single contianer. Can be single or multiple list/set objects.
        split : bool, optional
            If true performs a split function on the strings contained in the list.
        split_func : function, optional
            A user definable split function which turns a string into a list of strings.
        """
        self.split = bool(split)
        self.split_func = split_func
        self._check_kwargs_sanity()
        
        self.items = list()

        for arg in args:
            self._check_list_type(arg)
            self.items += arg

        super().__init__()

    def _check_list_type(self, obj):
        """ Checks input validity """
        if isinstance(obj, (list, set)):
            return 1
        else:
            raise TypeError(f"Arg must be list/set type. Got {type(obj)}")
    
    def _check_str_type(self, obj):
        """ Checks input validity """
        if isinstance(obj, str):
            return 1
        else:
            raise TypeError(f"Arg must be str type. Got {type(obj)}")
    
    def _check_kwargs_sanity(self):
        """ Checks argument validity """
        if self.split and self.split_func is not None:
            raise RuntimeError("You must provide either split=True or a split_func, not both")

    def __len__(self):
        """ List length """
        return len(self.items)
    
    def __repr__(self):
        return f"{self.__class__.__name__}, {self.items}"

    def __str__(self):
        return str(self.items)

    def _convert_item(self, item):
        """ Convert sentence to list of tokens """
        if self.split:
            return any2unicode(item).split()
        else:
            return self.split_func(any2unicode(item))

    def __getitem__(self, i):
        """ Get a list item """
        item = self.items[i]
        output = self._convert_item(item)
        return IndexedSentence(output, i)

    def __delitem__(self, i):
        """ Delete an item """
        del self.items[i]

    def __setitem__(self, i, item):
        """ Sets an item """
        self._check_str_type(item)
        self.items[i] = item

    def insert(self, i, item):
        """ Inserts an item at a position """
        self._check_str_type(item)
        self.items.insert(i, item)

    def append(self, item):
        """ Appends item at last position"""
        self._check_str_type(item)
        self.insert(len(self.items), item)
    
    def extend(self, *args):
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
        with s_open(self.path, "rb") as f:
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

        with s_open(self.path, "rb") as f:
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
        with s_open(self.path, "rb") as f:
            for i, line in enumerate(f):
                yield IndexedSentence(any2unicode(line).split(), i)
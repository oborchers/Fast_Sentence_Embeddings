#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from typing import NamedTuple, List

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


# class TaggedLineDocument(object):
#     """Iterate over a file that contains documents: one line = :class:`~gensim.models.doc2vec.TaggedDocument` object.

#     Words are expected to be already preprocessed and separated by whitespace. Document tags are constructed
#     automatically from the document line number (each document gets a unique integer tag).

#     """
#     def __init__(self, source):
#         """

#         Parameters
#         ----------
#         source : string or a file-like object
#             Path to the file on disk, or an already-open file object (must support `seek(0)`).

#         Examples
#         --------
#         .. sourcecode:: pycon

#             >>> from gensim.test.utils import datapath
#             >>> from gensim.models.doc2vec import TaggedLineDocument
#             >>>
#             >>> for document in TaggedLineDocument(datapath("head500.noblanks.cor")):
#             ...     pass

#         """
#         self.source = source

#     def __iter__(self):
#         """Iterate through the lines in the source.

#         Yields
#         ------
#         :class:`~gensim.models.doc2vec.TaggedDocument`
#             Document from `source` specified in the constructor.

#         """
#         try:
#             # Assume it is a file-like object and try treating it as such
#             # Things that don't have seek will trigger an exception
#             self.source.seek(0)
#             for item_no, line in enumerate(self.source):
#                 yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
#         except AttributeError:
#             # If it didn't work like a file, use it as a string filename
#             with utils.open(self.source, 'rb') as fin:
#                 for item_no, line in enumerate(fin):
#                     yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
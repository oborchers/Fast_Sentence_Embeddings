# SentenceVectors

> Auto-generated documentation for [fse.models.sentencevectors](../../../fse/models/sentencevectors.py) module.

- [Fast_sentence_embeddings](../../README.md#fast_sentence_embeddings-index) / [Modules](../../MODULES.md#fast_sentence_embeddings-modules) / [Fse](../index.md#fse) / [Models](index.md#models) / SentenceVectors
    - [SentenceVectors](#sentencevectors)
        - [SentenceVectors().\_\_getitem\_\_](#sentencevectors__getitem__)
        - [SentenceVectors().distance](#sentencevectorsdistance)
        - [SentenceVectors().get_vector](#sentencevectorsget_vector)
        - [SentenceVectors().init_sims](#sentencevectorsinit_sims)
        - [SentenceVectors.load](#sentencevectorsload)
        - [SentenceVectors().most_similar](#sentencevectorsmost_similar)
        - [SentenceVectors().save](#sentencevectorssave)
        - [SentenceVectors().similar_by_sentence](#sentencevectorssimilar_by_sentence)
        - [SentenceVectors().similar_by_vector](#sentencevectorssimilar_by_vector)
        - [SentenceVectors().similar_by_word](#sentencevectorssimilar_by_word)
        - [SentenceVectors().similarity](#sentencevectorssimilarity)

## SentenceVectors

[[find in source code]](../../../fse/models/sentencevectors.py#L40)

```python
class SentenceVectors(utils.SaveLoad):
    def __init__(vector_size: int, mapfile_path: str = None):
```

### SentenceVectors().\_\_getitem\_\_

[[find in source code]](../../../fse/models/sentencevectors.py#L53)

```python
def __getitem__(entities: int) -> ndarray:
```

Get vector representation of `entities`.

Parameters
----------
entities : {int, list of int}
    Index or sequence of entities.

Returns
-------
numpy.ndarray
    Vector representation for `entities` (1D if `entities` is int, otherwise - 2D).

### SentenceVectors().distance

[[find in source code]](../../../fse/models/sentencevectors.py#L202)

```python
def distance(d1: int, d2: int) -> float:
```

Compute cosine similarity between two sentences from the training set.

Parameters
----------
d1 : int
    index of sentence
d2 : int
    index of sentence

Returns
-------
float
    The cosine distance between the vectors of the two sentences.

### SentenceVectors().get_vector

[[find in source code]](../../../fse/models/sentencevectors.py#L133)

```python
def get_vector(index: int, use_norm: bool = False) -> ndarray:
```

Get sentence representations in vector space, as a 1D numpy array.

Parameters
----------
index : int
    Input index
norm : bool, optional
    If True - resulting vector will be L2-normalized (unit euclidean length).

Returns
-------
numpy.ndarray
    Vector representation of index.

Raises
------
KeyError
    If index out of bounds.

### SentenceVectors().init_sims

[[find in source code]](../../../fse/models/sentencevectors.py#L165)

```python
def init_sims(replace: bool = False):
```

Precompute L2-normalized vectors.

Parameters
----------
replace : bool, optional
    If True - forget the original vectors and only keep the normalized ones = saves lots of memory!

### SentenceVectors.load

[[find in source code]](../../../fse/models/sentencevectors.py#L123)

```python
@classmethod
def load(fname_or_handle, **kwargs):
```

### SentenceVectors().most_similar

[[find in source code]](../../../fse/models/sentencevectors.py#L220)

```python
def most_similar(
    positive: Union[int, ndarray] = None,
    negative: Union[int, ndarray] = None,
    indexable: Union[IndexedList, IndexedLineDocument] = None,
    topn: int = 10,
    restrict_size: Union[int, Tuple[int, int]] = None,
) -> List[Tuple[int, float]]:
```

Find the top-N most similar sentences.
Positive sentences contribute positively towards the similarity, negative sentences negatively.

This method computes cosine similarity between a simple mean of the projection
weight vectors of the given sentences and the vectors for each sentence in the model.

Parameters
----------
positive : list of int, optional
    List of indices that contribute positively.
negative : list of int, optional
    List of indices that contribute negatively.
indexable: list, IndexedList, IndexedLineDocument
    Provides an indexable object from where the most similar sentences are read
topn : int or None, optional
    Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
    then similarities for all sentences are returned.
restrict_size : int or Tuple(int,int), optional
    Optional integer which limits the range of vectors which
    are searched for most-similar values. For example, restrict_vocab=10000 would
    only check the first 10000 sentence vectors.
    restrict_vocab=(500, 1000) would search the sentence vectors with indices between
    500 and 1000.

Returns
-------
list of (int, float) or list of (str, int, float)
    A sequence of (index, similarity) is returned.
    When an indexable is provided, returns (str, index, similarity)
    When `topn` is None, then similarities for all words are returned as a
    one-dimensional numpy array with the size of the vocabulary.

#### See also

- [IndexedLineDocument](../inputs.md#indexedlinedocument)
- [IndexedList](../inputs.md#indexedlist)

### SentenceVectors().save

[[find in source code]](../../../fse/models/sentencevectors.py#L101)

```python
def save(*args, **kwargs):
```

Save object.

Parameters
----------
fname : str
    Path to the output file.

See Also
--------
:meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.load`
    Load object.

### SentenceVectors().similar_by_sentence

[[find in source code]](../../../fse/models/sentencevectors.py#L373)

```python
def similar_by_sentence(
    sentence: List[str],
    model,
    indexable: Union[IndexedList, IndexedLineDocument] = None,
    topn: int = 10,
    restrict_size: Union[int, Tuple[int, int]] = None,
) -> List[Tuple[int, float]]:
```

Find the top-N most similar sentences to a given sentence.

Parameters
----------
sentence : list of str
    Sentence as list of strings
model : class `fse.models.base_s2v.BaseSentence2VecModel`
    This object essentially provides the infer method used to transform .
indexable: list, IndexedList, IndexedLineDocument
    Provides an indexable object from where the most similar sentences are read
topn : int or None, optional
    Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
    then similarities for all sentences are returned.
restrict_size : int or Tuple(int,int), optional
    Optional integer which limits the range of vectors which
    are searched for most-similar values. For example, restrict_vocab=10000 would
    only check the first 10000 sentence vectors.
    restrict_vocab=(500, 1000) would search the sentence vectors with indices between
    500 and 1000.

Returns
-------
list of (int, float) or list of (str, int, float)
    A sequence of (index, similarity) is returned.
    When an indexable is provided, returns (str, index, similarity)
    When `topn` is None, then similarities for all words are returned as a
    one-dimensional numpy array with the size of the vocabulary.

#### See also

- [IndexedLineDocument](../inputs.md#indexedlinedocument)
- [IndexedList](../inputs.md#indexedlist)

### SentenceVectors().similar_by_vector

[[find in source code]](../../../fse/models/sentencevectors.py#L422)

```python
def similar_by_vector(
    vector: ndarray,
    indexable: Union[IndexedList, IndexedLineDocument] = None,
    topn: int = 10,
    restrict_size: Union[int, Tuple[int, int]] = None,
) -> List[Tuple[int, float]]:
```

Find the top-N most similar sentences to a given vector.

Parameters
----------
vector : ndarray
    Vectors
indexable: list, IndexedList, IndexedLineDocument
    Provides an indexable object from where the most similar sentences are read
topn : int or None, optional
    Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
    then similarities for all sentences are returned.
restrict_size : int or Tuple(int,int), optional
    Optional integer which limits the range of vectors which
    are searched for most-similar values. For example, restrict_vocab=10000 would
    only check the first 10000 sentence vectors.
    restrict_vocab=(500, 1000) would search the sentence vectors with indices between
    500 and 1000.

Returns
-------
list of (int, float) or list of (str, int, float)
    A sequence of (index, similarity) is returned.
    When an indexable is provided, returns (str, index, similarity)
    When `topn` is None, then similarities for all words are returned as a
    one-dimensional numpy array with the size of the vocabulary.

#### See also

- [IndexedLineDocument](../inputs.md#indexedlinedocument)
- [IndexedList](../inputs.md#indexedlist)

### SentenceVectors().similar_by_word

[[find in source code]](../../../fse/models/sentencevectors.py#L328)

```python
def similar_by_word(
    word: str,
    wv: KeyedVectors,
    indexable: Union[IndexedList, IndexedLineDocument] = None,
    topn: int = 10,
    restrict_size: Union[int, Tuple[int, int]] = None,
) -> List[Tuple[int, float]]:
```

Find the top-N most similar sentences to a given word.

Parameters
----------
word : str
    Word
wv : class `gensim.models.keyedvectors.KeyedVectors`
    This object essentially contains the mapping between words and embeddings.
indexable: list, IndexedList, IndexedLineDocument
    Provides an indexable object from where the most similar sentences are read
topn : int or None, optional
    Number of top-N similar sentences to return, when `topn` is int. When `topn` is None,
    then similarities for all sentences are returned.
restrict_size : int or Tuple(int,int), optional
    Optional integer which limits the range of vectors which
    are searched for most-similar values. For example, restrict_vocab=10000 would
    only check the first 10000 sentence vectors.
    restrict_vocab=(500, 1000) would search the sentence vectors with indices between
    500 and 1000.

Returns
-------
list of (int, float) or list of (str, int, float)
    A sequence of (index, similarity) is returned.
    When an indexable is provided, returns (str, index, similarity)
    When `topn` is None, then similarities for all words are returned as a
    one-dimensional numpy array with the size of the vocabulary.

#### See also

- [IndexedLineDocument](../inputs.md#indexedlinedocument)
- [IndexedList](../inputs.md#indexedlist)

### SentenceVectors().similarity

[[find in source code]](../../../fse/models/sentencevectors.py#L184)

```python
def similarity(d1: int, d2: int) -> float:
```

Compute cosine similarity between two sentences from the training set.

Parameters
----------
d1 : int
    index of sentence
d2 : int
    index of sentence

Returns
-------
float
    The cosine similarity between the vectors of the two sentences.

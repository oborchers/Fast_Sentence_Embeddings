# Inputs

> Auto-generated documentation for [fse.inputs](../../fse/inputs.py) module.

- [Fast_sentence_embeddings](../README.md#fast_sentence_embeddings-index) / [Modules](../MODULES.md#fast_sentence_embeddings-modules) / [Fse](index.md#fse) / Inputs
    - [BaseIndexedList](#baseindexedlist)
        - [BaseIndexedList().\_\_delitem\_\_](#baseindexedlist__delitem__)
        - [BaseIndexedList().\_\_getitem\_\_](#baseindexedlist__getitem__)
        - [BaseIndexedList().\_\_len\_\_](#baseindexedlist__len__)
        - [BaseIndexedList().\_\_setitem\_\_](#baseindexedlist__setitem__)
        - [BaseIndexedList().\_\_str\_\_](#baseindexedlist__str__)
        - [BaseIndexedList().append](#baseindexedlistappend)
        - [BaseIndexedList().extend](#baseindexedlistextend)
        - [BaseIndexedList().insert](#baseindexedlistinsert)
    - [CIndexedList](#cindexedlist)
        - [CIndexedList().\_\_getitem\_\_](#cindexedlist__getitem__)
        - [CIndexedList().append](#cindexedlistappend)
        - [CIndexedList().extend](#cindexedlistextend)
        - [CIndexedList().insert](#cindexedlistinsert)
    - [CSplitCIndexedList](#csplitcindexedlist)
        - [CSplitCIndexedList().\_\_getitem\_\_](#csplitcindexedlist__getitem__)
        - [CSplitCIndexedList().append](#csplitcindexedlistappend)
        - [CSplitCIndexedList().extend](#csplitcindexedlistextend)
        - [CSplitCIndexedList().insert](#csplitcindexedlistinsert)
    - [CSplitIndexedList](#csplitindexedlist)
        - [CSplitIndexedList().\_\_getitem\_\_](#csplitindexedlist__getitem__)
    - [IndexedLineDocument](#indexedlinedocument)
        - [IndexedLineDocument().\_\_getitem\_\_](#indexedlinedocument__getitem__)
        - [IndexedLineDocument().\_\_iter\_\_](#indexedlinedocument__iter__)
    - [IndexedList](#indexedlist)
        - [IndexedList().\_\_getitem\_\_](#indexedlist__getitem__)
    - [SplitCIndexedList](#splitcindexedlist)
        - [SplitCIndexedList().\_\_getitem\_\_](#splitcindexedlist__getitem__)
        - [SplitCIndexedList().append](#splitcindexedlistappend)
        - [SplitCIndexedList().extend](#splitcindexedlistextend)
        - [SplitCIndexedList().insert](#splitcindexedlistinsert)
    - [SplitIndexedList](#splitindexedlist)
        - [SplitIndexedList().\_\_getitem\_\_](#splitindexedlist__getitem__)

## BaseIndexedList

[[find in source code]](../../fse/inputs.py#L15)

```python
class BaseIndexedList(MutableSequence):
    def __init__(*args: List[Union[list, set, ndarray]]):
```

### BaseIndexedList().\_\_delitem\_\_

[[find in source code]](../../fse/inputs.py#L81)

```python
def __delitem__(i: int):
```

Delete an item.

### BaseIndexedList().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L71)

```python
def __getitem__(i: int) -> tuple:
```

Getitem method.

Returns
-------
tuple ([str], int)
    Returns the core object, a tuple, for every sentence embedding model.

### BaseIndexedList().\_\_len\_\_

[[find in source code]](../../fse/inputs.py#L51)

```python
def __len__():
```

List length.

Returns
-------
int
   Length of the IndexedList

### BaseIndexedList().\_\_setitem\_\_

[[find in source code]](../../fse/inputs.py#L85)

```python
def __setitem__(i: int, item: str):
```

Sets an item.

### BaseIndexedList().\_\_str\_\_

[[find in source code]](../../fse/inputs.py#L61)

```python
def __str__():
```

Human readable representation of the object's state, used for debugging.

Returns
-------
str
   Human readable representation of the object's state (words and tags).

### BaseIndexedList().append

[[find in source code]](../../fse/inputs.py#L95)

```python
def append(item: str):
```

Appends item at last position.

### BaseIndexedList().extend

[[find in source code]](../../fse/inputs.py#L100)

```python
def extend(arg: Union[list, set, ndarray]):
```

Extens list.

### BaseIndexedList().insert

[[find in source code]](../../fse/inputs.py#L90)

```python
def insert(i: int, item: str):
```

Inserts an item at a position.

## CIndexedList

[[find in source code]](../../fse/inputs.py#L133)

```python
class CIndexedList(BaseIndexedList):
    def __init__(
        custom_index: Union[list, ndarray],
        *args: Union[list, set, ndarray],
    ):
```

#### See also

- [BaseIndexedList](#baseindexedlist)

### CIndexedList().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L156)

```python
def __getitem__(i: int) -> tuple:
```

Getitem method.

Returns
-------
tuple
    Returns the core object, tuple, for every sentence embedding model.

### CIndexedList().append

[[find in source code]](../../fse/inputs.py#L175)

```python
def append(item: str):
```

### CIndexedList().extend

[[find in source code]](../../fse/inputs.py#L178)

```python
def extend(arg: Union[list, set, ndarray]):
```

### CIndexedList().insert

[[find in source code]](../../fse/inputs.py#L172)

```python
def insert(i: int, item: str):
```

## CSplitCIndexedList

[[find in source code]](../../fse/inputs.py#L280)

```python
class CSplitCIndexedList(BaseIndexedList):
    def __init__(
        custom_split: callable,
        custom_index: Union[list, ndarray],
        *args: Union[list, set, ndarray],
    ):
```

#### See also

- [BaseIndexedList](#baseindexedlist)

### CSplitCIndexedList().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L309)

```python
def __getitem__(i: int) -> tuple:
```

Getitem method.

Returns
-------
tuple
    Returns the core object, tuple, for every sentence embedding model.

### CSplitCIndexedList().append

[[find in source code]](../../fse/inputs.py#L328)

```python
def append(item: str):
```

### CSplitCIndexedList().extend

[[find in source code]](../../fse/inputs.py#L331)

```python
def extend(arg: Union[list, set, ndarray]):
```

### CSplitCIndexedList().insert

[[find in source code]](../../fse/inputs.py#L325)

```python
def insert(i: int, item: str):
```

## CSplitIndexedList

[[find in source code]](../../fse/inputs.py#L254)

```python
class CSplitIndexedList(BaseIndexedList):
    def __init__(custom_split: callable, *args: Union[list, set, ndarray]):
```

#### See also

- [BaseIndexedList](#baseindexedlist)

### CSplitIndexedList().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L269)

```python
def __getitem__(i: int) -> tuple:
```

Getitem method.

Returns
-------
tuple
    Returns the core object, tuple, for every sentence embedding model.

## IndexedLineDocument

[[find in source code]](../../fse/inputs.py#L335)

```python
class IndexedLineDocument(object):
    def __init__(path, get_able=True):
```

### IndexedLineDocument().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L367)

```python
def __getitem__(i):
```

Returns the line indexed by i. Primarily used for.

:meth:`~fse.models.sentencevectors.SentenceVectors.most_similar`

Parameters
----------
i : int
    The line index used to index the file

Returns
-------
str
    line at the current index

### IndexedLineDocument().\_\_iter\_\_

[[find in source code]](../../fse/inputs.py#L393)

```python
def __iter__():
```

Iterate through the lines in the source.

Yields
------
tuple : (list[str], int)
    Tuple of list of string and index

## IndexedList

[[find in source code]](../../fse/inputs.py#L110)

```python
class IndexedList(BaseIndexedList):
    def __init__(*args: Union[list, set, ndarray]):
```

#### See also

- [BaseIndexedList](#baseindexedlist)

### IndexedList().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L122)

```python
def __getitem__(i: int) -> tuple:
```

Getitem method.

Returns
-------
tuple
    Returns the core object, tuple, for every sentence embedding model.

## SplitCIndexedList

[[find in source code]](../../fse/inputs.py#L205)

```python
class SplitCIndexedList(BaseIndexedList):
    def __init__(
        custom_index: Union[list, ndarray],
        *args: Union[list, set, ndarray],
    ):
```

#### See also

- [BaseIndexedList](#baseindexedlist)

### SplitCIndexedList().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L228)

```python
def __getitem__(i: int) -> tuple:
```

Getitem method.

Returns
-------
tuple
    Returns the core object, tuple, for every sentence embedding model.

### SplitCIndexedList().append

[[find in source code]](../../fse/inputs.py#L247)

```python
def append(item: str):
```

### SplitCIndexedList().extend

[[find in source code]](../../fse/inputs.py#L250)

```python
def extend(arg: Union[list, set, ndarray]):
```

### SplitCIndexedList().insert

[[find in source code]](../../fse/inputs.py#L244)

```python
def insert(i: int, item: str):
```

## SplitIndexedList

[[find in source code]](../../fse/inputs.py#L182)

```python
class SplitIndexedList(BaseIndexedList):
    def __init__(*args: Union[list, set, ndarray]):
```

#### See also

- [BaseIndexedList](#baseindexedlist)

### SplitIndexedList().\_\_getitem\_\_

[[find in source code]](../../fse/inputs.py#L194)

```python
def __getitem__(i: int) -> tuple:
```

Getitem method.

Returns
-------
tuple
    Returns the core object, tuple, for every sentence embedding model.

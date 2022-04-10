# Vectors

> Auto-generated documentation for [fse.vectors](../../fse/vectors.py) module.

Class to obtain BaseKeyedVector from.

- [Fast_sentence_embeddings](../README.md#fast_sentence_embeddings-index) / [Modules](../MODULES.md#fast_sentence_embeddings-modules) / [Fse](index.md#fse) / Vectors
    - [FTVectors](#ftvectors)
        - [FTVectors.from_pretrained](#ftvectorsfrom_pretrained)
    - [Vectors](#vectors)
        - [Vectors.from_pretrained](#vectorsfrom_pretrained)

## FTVectors

[[find in source code]](../../fse/vectors.py#L51)

```python
class FTVectors(FastTextKeyedVectors):
```

Class to instantiates FT vectors from pretrained models.

### FTVectors.from_pretrained

[[find in source code]](../../fse/vectors.py#L54)

```python
@classmethod
def from_pretrained(model: str, mmap: str = None):
```

Method to load vectors from a pre-trained model.

Parameters
----------
model : :str: of the model name to load from the bug. For example: "glove-wiki-gigaword-50"
mmap : :str: If to load the vectors in mmap mode.

Returns
-------
Vectors
    An object of pretrained vectors.

## Vectors

[[find in source code]](../../fse/vectors.py#L20)

```python
class Vectors(KeyedVectors):
```

Class to instantiates vectors from pretrained models.

### Vectors.from_pretrained

[[find in source code]](../../fse/vectors.py#L23)

```python
@classmethod
def from_pretrained(model: str, mmap: str = None):
```

Method to load vectors from a pre-trained model.

Parameters
----------
model : :str: of the model name to load from the bug. For example: "glove-wiki-gigaword-50"
mmap : :str: If to load the vectors in mmap mode.

Returns
-------
Vectors
    An object of pretrained vectors.

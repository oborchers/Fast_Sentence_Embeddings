# Utils

> Auto-generated documentation for [fse.models.utils](../../../fse/models/utils.py) module.

- [Fast_sentence_embeddings](../../README.md#fast_sentence_embeddings-index) / [Modules](../../MODULES.md#fast_sentence_embeddings-modules) / [Fse](../index.md#fse) / [Models](index.md#models) / Utils
    - [compute_principal_components](#compute_principal_components)
    - [remove_principal_components](#remove_principal_components)
    - [set_madvise_for_mmap](#set_madvise_for_mmap)

## compute_principal_components

[[find in source code]](../../../fse/models/utils.py#L56)

```python
def compute_principal_components(
    vectors: ndarray,
    components: int = 1,
    cache_size_gb: float = 1.0,
) -> Tuple[ndarray, ndarray]:
```

Method used to compute the first singular vectors of a given (sub)matrix

Parameters
----------
vectors : ndarray
    (Sentence) vectors to compute the truncated SVD on
components : int, optional
    Number of singular values/vectors to compute
cache_size_gb : float, optional
        Cache size for computing the principal components in GB

Returns
-------
ndarray, ndarray
    Singular values and singular vectors

## remove_principal_components

[[find in source code]](../../../fse/models/utils.py#L99)

```python
def remove_principal_components(
    vectors: ndarray,
    svd_res: Tuple[ndarray, ndarray],
    weights: ndarray = None,
    inplace: bool = True,
) -> ndarray:
```

Method used to remove the first singular vectors of a given matrix

Parameters
----------
vectors : ndarray
    (Sentence) vectors to remove components fromm
svd_res : (ndarray, ndarray)
    Tuple consisting of the singular values and components to remove from the vectors
weights : ndarray, optional
    Weights to be used to weigh the components which are removed from the vectors
inplace : bool, optional
    If true, removes the components from the vectors inplace (memory efficient)

Returns
-------
ndarray, ndarray
    Singular values and singular vectors

## set_madvise_for_mmap

[[find in source code]](../../../fse/models/utils.py#L26)

```python
def set_madvise_for_mmap(return_madvise: bool = False) -> object:
```

Method used to set madvise parameters.
This problem adresses the memmap issue raised in https://github.com/numpy/numpy/issues/13172
The issue is not applicable for windows

Parameters
----------
return_madvise : bool
    Returns the madvise object for unittests, se test_utils.py

Returns
-------
object
    madvise object

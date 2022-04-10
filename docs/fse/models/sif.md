# SIF

> Auto-generated documentation for [fse.models.sif](../../../fse/models/sif.py) module.

- [Fast_sentence_embeddings](../../README.md#fast_sentence_embeddings-index) / [Modules](../../MODULES.md#fast_sentence_embeddings-modules) / [Fse](../index.md#fse) / [Models](index.md#models) / SIF
    - [SIF](#sif)

## SIF

[[find in source code]](../../../fse/models/sif.py#L19)

```python
class SIF(Average):
    def __init__(
        model: KeyedVectors,
        alpha: float = 0.001,
        components: int = 1,
        cache_size_gb: float = 1.0,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
        workers: int = 1,
        lang_freq: str = None,
    ):
```

#### See also

- [Average](average.md#average)

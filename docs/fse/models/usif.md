# uSIF

> Auto-generated documentation for [fse.models.usif](../../../fse/models/usif.py) module.

- [Fast_sentence_embeddings](../../README.md#fast_sentence_embeddings-index) / [Modules](../../MODULES.md#fast_sentence_embeddings-modules) / [Fse](../index.md#fse) / [Models](index.md#models) / uSIF
    - [uSIF](#usif)

## uSIF

[[find in source code]](../../../fse/models/usif.py#L23)

```python
class uSIF(Average):
    def __init__(
        model: KeyedVectors,
        length: int = None,
        components: int = 5,
        cache_size_gb: float = 1.0,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
        workers: int = 1,
        lang_freq: str = None,
    ):
```

#### See also

- [Average](average.md#average)

# Proposed public type signatures

Approval artifact for DR-323 (Öffentliche Typen und vollständige statische Rückgabetypen vor 2.1 anbieten).
Based on commit `0d7eacbf` and plan v3, 9 November 2026.
These are proposals only. Implementation starts after fse-main approves this artifact.
All signatures list every parameter, including defaults. `Any` in inputs and broad
fallbacks preserves existing accepted calls. Implementation signatures are included
where the caller-facing overloads differ.

## Public names in `fse.typing`

Exactly `Encoding`, `LengthLevel`, `ScoreHead`, `Stored`, `Tokenizer`.
Static re-exports preserve object identity. No additions to `fse.__all__`.
No `ScoreSchema`, `score_schema`, or Pool internals are exported here.
`Pool` remains public in `fse`, including its use in the signatures below.

```python
LengthLevel = Literal["summary", "per_row"]
Stored = ndarray | tuple[ndarray, ...]

class Encoding:
    ids: list[int]

class ScoreHead:
    name: str
    model: str
    task: str
    rule: str
    link: str
    stored: str
    path: str
    labels: tuple[str, ...] | tuple[int, ...] | None
    width: int
    columns: tuple[int, int]
    flat: bool | None
    @property
    def multilabel(self) -> bool: ...
```

`ScoreHead.entry` and `ScoreHead.threshold_name` become private helpers before
export. Their serialized keys, tensor names and values stay unchanged.
`Tokenizer` and `Encoding` are returned objects. Users import their types rather
than construct them. `Tokenizer` keeps these existing signatures:

```python

def encode(self, sequence: str, add_special_tokens: bool=True) -> Encoding: ...

def encode_batch(self, input: Any, add_special_tokens: bool=True) -> list[Encoding]: ...

def get_vocab(self, with_added_tokens: bool=True) -> dict[str, int]: ...

def get_vocab_size(self, with_added_tokens: bool=True) -> int: ...

def token_to_id(self, token: str) -> int | None: ...

def id_to_token(self, id: int) -> str | None: ...

```

`Tokenizer.encode_batch_fast` remains an alias of `encode_batch`.
`StaticModel.tokenizer -> Tokenizer` and `ScoreStore.heads -> Mapping[str, ScoreHead]`
keep their existing types, now importable through `fse.typing`.

## Factory and persistence signatures

`Self` preserves the calling subclass. `Encoder.quantize` and
`WordVectors.quantize` return their base classes, as they do today. Those two
methods are decorated only after their classes are bound. The validator and
its configuration stay unchanged.


### Encoder

```python

@classmethod

def from_word_vectors(cls, word_vectors: Any, *, tokenizer: str='fse_words@1', normalize: bool=True, max_ngrams: PositiveInt | None=None, oov_weight: float | None=None) -> Self: ...

@classmethod

def from_arrays(cls, vectors: Any, *, words: Any=None, vocab_bytes: Any=None, vocab_offsets: Any=None, scales: Any=None, fasttext: Any=None, ngrams: Any=None, ngram_scales: Any=None, tokenizer: str='fse_words@1', normalize: bool=True, max_ngrams: PositiveInt | None=None, oov_weight: float | None=None) -> Self: ...

@classmethod

def from_pretrained(cls, repo_or_path: PathLike, *, revision: str | None=None, token: str | bool | None=None, cache_dir: PathLike | None=None, local_files_only: bool=False, truncate: bool=True, max_length: PositiveInt | Literal['model'] | None='model', truncate_dim: PositiveInt | Literal['model'] | None='model', normalize: bool | Literal['model']='model', mmap: bool=False) -> Self: ...

def quantize(self, dtype: Literal['float16', 'int8']) -> Encoder: ...

def save(self, path: PathLike) -> None: ...

@classmethod

def load(cls, path: PathLike, mmap: bool=False) -> Self: ...

def verify(self) -> None: ...

```

### WordVectors

```python

@classmethod

def from_arrays(cls, vectors: Any, *, words: Sequence[str] | None=None, vocab_bytes: Any=None, vocab_offsets: Any=None, scales: Any=None, fasttext: FastTextParams | None=None, ngrams: Any=None, ngram_scales: Any=None) -> Self: ...

@classmethod

def load(cls, path: PathLike, mmap: bool=False) -> Self: ...

def save(self, path: PathLike) -> None: ...

def verify(self) -> None: ...

def quantize(self, dtype: Literal['float16', 'int8']) -> WordVectors: ...

```

### Classifier

```python

@classmethod

def from_sklearn(cls, encoder: Any, model: Any, *, normalized: bool | None=None, labels: Any=None, thresholds: ThresholdInput=None) -> Self: ...

@classmethod

def from_arrays(cls, encoder: Any, weight: Any, bias: Any=None, *, labels: Any=None, link: Literal['auto', 'sigmoid', 'softmax', 'ovr', 'none', 'multilabel']='auto', normalized: bool | None=None, thresholds: ThresholdInput=None, temperature: Temperature | None=None) -> Self: ...

def with_thresholds(self, thresholds: ThresholdInput) -> Self: ...

@classmethod

def load(cls, path: PathLike, mmap: bool=False) -> Self: ...

@classmethod

def from_pretrained(cls, repo_or_path: PathLike, *, revision: str | None=None, token: str | bool | None=None, cache_dir: PathLike | None=None, local_files_only: bool=False, mmap: bool=False) -> Self: ...

@classmethod

def from_model2vec(cls, model: Any, *, thresholds: ThresholdInput=None) -> Self: ...

```

### Regressor

```python

@classmethod

def from_sklearn(cls, encoder: Any, model: Any, *, normalized: bool | None=None) -> Self: ...

@classmethod

def from_arrays(cls, encoder: Any, weight: Any, bias: Any=None, *, normalized: bool | None=None) -> Self: ...

@classmethod

def load(cls, path: PathLike, mmap: bool=False) -> Self: ...

@classmethod

def from_pretrained(cls, repo_or_path: PathLike, *, revision: str | None=None, token: str | bool | None=None, cache_dir: PathLike | None=None, local_files_only: bool=False, mmap: bool=False) -> Self: ...

```

### LinearModel

```python

def quantize(self, dtype: Literal['float16', 'int8']) -> Self: ...

def save(self, path: PathLike, standalone: bool=False) -> None: ...

```

### StaticModel

```python

@classmethod

def from_pretrained(cls, path: PathLike, token: str | bool | None=None, normalize: bool | None=None, subfolder: str | None=None, quantize_to: Any=None, dimensionality: PositiveInt | None=None, vocabulary_quantization: PositiveInt | None=None, max_length: PositiveInt | Unset | None=UNSET, force_download: bool=True, *, revision: str | None=None, cache_dir: PathLike | None=None, local_files_only: bool=False) -> Self: ...

```

### ShardSinkBase

```python

def __enter__(self) -> Self: ...

```

Private factory return chains also use `Self`: `Encoder._new`, `_converted`,
`_from_artifact`, `WordVectors._open`, `_new`, `StaticModel._new`, and
`LinearModel._new`, `_build`, `_load`. Their arguments stay unchanged.

## Encoder.encode

Eight overloads. A non-resume call without a sink returns rows. A sink returns
`None`, including resume and shard calls. A length report wraps that result.

```python

@overload
def encode(self, texts: Any, *, dtype: Literal['float32', 'float16'] | None=None, workers: int | None=None, backend: str='cpu', sink: None=None, doc_ids: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', shard: None=None, pool: Pool | None=None) -> ndarray: ...

@overload
def encode(self, texts: Any, *, dtype: Literal['int8'], workers: int | None=None, backend: str='cpu', sink: None=None, doc_ids: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', shard: None=None, pool: Pool | None=None) -> QuantizedRows: ...

@overload
def encode(self, texts: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Sink, doc_ids: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', shard: tuple[int, int] | None=None, pool: Pool | None=None) -> None: ...

@overload
def encode(self, texts: Any, *, dtype: Literal['float32', 'float16'] | None=None, workers: int | None=None, backend: str='cpu', sink: None=None, doc_ids: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', shard: None=None, pool: Pool | None=None) -> tuple[ndarray, LengthReport]: ...

@overload
def encode(self, texts: Any, *, dtype: Literal['int8'], workers: int | None=None, backend: str='cpu', sink: None=None, doc_ids: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', shard: None=None, pool: Pool | None=None) -> tuple[QuantizedRows, LengthReport]: ...

@overload
def encode(self, texts: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Sink, doc_ids: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def encode(self, texts: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Sink | None=None, doc_ids: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', shard: tuple[int, int] | None=None, pool: Pool | None=None) -> ndarray | QuantizedRows | None: ...

@overload
def encode(self, texts: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Sink | None=None, doc_ids: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[ndarray | QuantizedRows | None, LengthReport]: ...

def encode(self, texts: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Any=None, doc_ids: Any=None, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', shard: tuple[int, int] | None=None, pool: Pool | None=None) -> Any: ...

```

## Classifier.predict

```python

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> ndarray: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[True], vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray, ndarray]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> ndarray | tuple[ndarray, ndarray]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[True], vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray, ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray, LengthReport] | tuple[ndarray, ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink | None=None, score_sink: Sink, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink | None=None, score_sink: Sink, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any=None, score_sink: Any, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport] | None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport] | None: ...

def predict(self, texts: Any, *, workers: PositiveInt | None=None, backend: str='cpu', lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> Any: ...

```

## Classifier.predict_proba

```python

@overload
def predict_proba(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> ndarray: ...

@overload
def predict_proba(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Sink, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> None: ...

@overload
def predict_proba(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> tuple[ndarray, LengthReport]: ...

@overload
def predict_proba(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Sink, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict_proba(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> ndarray | None: ...

@overload
def predict_proba(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> tuple[ndarray | None, LengthReport]: ...

def predict_proba(self, texts: Any, *, workers: PositiveInt | None=None, backend: str='cpu', sink: Any=None, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> Any: ...

```

## Classifier.decision_function

```python

@overload
def decision_function(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> ndarray: ...

@overload
def decision_function(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Sink, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> None: ...

@overload
def decision_function(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> tuple[ndarray, LengthReport]: ...

@overload
def decision_function(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Sink, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def decision_function(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> ndarray | None: ...

@overload
def decision_function(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> tuple[ndarray | None, LengthReport]: ...

def decision_function(self, texts: Any, *, workers: PositiveInt | None=None, backend: str='cpu', sink: Any=None, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', pool: Pool | None=None) -> Any: ...

```

## Regressor.predict

```python

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> ndarray: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Sink, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Sink, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> ndarray | None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[True], vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray, ndarray]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> ndarray | tuple[ndarray, ndarray] | None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray | None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[True], vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray, ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: Any=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[ndarray | None, LengthReport] | tuple[ndarray, ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink | None=None, score_sink: Sink, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink | None=None, score_sink: Sink, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any=None, score_sink: Any, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport] | None: ...

@overload
def predict(self, texts: Any, *, workers: int | None=None, backend: str='cpu', sink: None=None, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport] | None: ...

def predict(self, texts: Any, *, workers: PositiveInt | None=None, backend: str='cpu', sink: Any=None, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> Any: ...

```

## Heads.predict

```python

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> Mapping[str, ndarray]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[True], vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[Mapping[str, ndarray], ndarray]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> Mapping[str, ndarray] | tuple[Mapping[str, ndarray], ndarray]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[False]=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[Mapping[str, ndarray], LengthReport]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: Literal[True], vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[Mapping[str, ndarray], ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: None=None, score_sink: None=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> tuple[Mapping[str, ndarray], LengthReport] | tuple[Mapping[str, ndarray], ndarray, LengthReport]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink | None=None, score_sink: Sink, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> None: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink | None=None, score_sink: Sink, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> None: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Sink, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport]: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any=None, score_sink: Any, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport] | None: ...

@overload
def predict(self, texts: Any, *, output: Any=None, workers: int | None=None, backend: str='cpu', lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any, score_sink: None=None, doc_ids: Any=None, shard: tuple[int, int] | None=None, pool: Pool | None=None) -> tuple[None, LengthReport] | None: ...

def predict(self, texts: Any, *, output: Any=None, workers: PositiveInt | None=None, backend: str='cpu', lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', return_vectors: bool=False, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=None, shard: Any=None, pool: Pool | None=None) -> Any: ...

```

## Public generator signatures

These functions already return generators. `Generator[Y, None, None]` exposes
`close()` to type checkers. The internal chain and `streaming` change together.
The inputs remain unchanged.


### Encoder

```python

@overload
def encode_batches(self, batches: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Any=None, doc_ids: bool=False, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[EncodedBatch, None, None]: ...

@overload
def encode_batches(self, batches: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Any=None, doc_ids: bool=False, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[tuple[EncodedBatch, LengthReport], None, None]: ...

def encode_batches(self, batches: Any, *, dtype: str | None=None, workers: int | None=None, backend: str='cpu', sink: Any=None, doc_ids: bool=False, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[Any, None, None]: ...

```

### Classifier

```python

@overload
def predict_batches(self, batches: Any, *, output: Literal['labels', 'proba', 'decision']='labels', return_vectors: bool=False, workers: int | None=None, backend: str='cpu', sink: Any=None, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=False, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[PredictedBatch, None, None]: ...

@overload
def predict_batches(self, batches: Any, *, output: Literal['labels', 'proba', 'decision']='labels', return_vectors: bool=False, workers: int | None=None, backend: str='cpu', sink: Any=None, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=False, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[tuple[PredictedBatch, LengthReport], None, None]: ...

def predict_batches(self, batches: Any, *, output: Literal['labels', 'proba', 'decision']='labels', return_vectors: bool=False, workers: PositiveInt | None=None, backend: str='cpu', sink: Any=None, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=False, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[Any, None, None]: ...

```

### Regressor

```python

@overload
def predict_batches(self, batches: Any, *, output: Literal['values']='values', return_vectors: bool=False, workers: int | None=None, backend: str='cpu', sink: Any=None, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=False, lengths: None=None, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[PredictedBatch, None, None]: ...

@overload
def predict_batches(self, batches: Any, *, output: Literal['values']='values', return_vectors: bool=False, workers: int | None=None, backend: str='cpu', sink: Any=None, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=False, lengths: LengthLevel, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[tuple[PredictedBatch, LengthReport], None, None]: ...

def predict_batches(self, batches: Any, *, output: Literal['values']='values', return_vectors: bool=False, workers: PositiveInt | None=None, backend: str='cpu', sink: Any=None, vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=False, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[Any, None, None]: ...

```

### Heads

```python

def predict_batches(self, batches: Any, *, output: Any=None, return_vectors: bool=False, workers: PositiveInt | None=None, backend: str='cpu', vector_sink: Any=None, score_sink: Any=None, doc_ids: Any=False, lengths: Any=None, count_truncated: Any=False, on_truncation: Any='keep', positions: bool=False, pool: Pool | None=None) -> Generator[HeadsBatch, None, None]: ...

```

### ScoreStore

```python

def iter_batches(self, *, block_rows: PositiveInt=DEFAULT_BLOCK_ROWS, start: NonNegativeInt=0, stop: NonNegativeInt | None=None) -> Generator[ScoreBatch, None, None]: ...

def predict_batches(self, *, output: Any=None, thresholds: Any=None, block_rows: PositiveInt=DEFAULT_BLOCK_ROWS, backend: str | None=None) -> Generator[ScoreBatch, None, None]: ...

```

### VectorStore

```python

def iter_batches(self, *, block_rows: PositiveInt=DEFAULT_BLOCK_ROWS, start: NonNegativeInt=0, stop: NonNegativeInt | None=None) -> Generator[VectorBatch, None, None]: ...

```

## Supporting types

```python
class EncodedBatch:
    rows: ndarray | QuantizedRows | None
    @property
    def vectors(self) -> ndarray | QuantizedRows | None: ...

class PredictedBatch:
    outputs: ndarray | None
    vectors: ndarray | None

class Heads(_HeadsPredicting, Mapping[str, Classifier | Regressor]):
    def __getitem__(self, name: str) -> Classifier | Regressor: ...

class LengthReport:
    lengths: LengthLevel

class LengthSettings:
    lengths: LengthLevel | None

class Sink(Protocol):
    if TYPE_CHECKING:
        @property
        def dim(self) -> int: ...
    else:
        dim: int
    def target(self, first_row: int, n_rows: int) -> Stored: ...
    def commit(self, first_row: int, rows: Stored) -> None: ...

def streaming(function: Callable[P, Generator[T, None, None]]) -> Callable[P, Generator[T, None, None]]: ...
```

The batch fragments show changed fields only. Their other fields, order, names,
NamedTuple behavior and serialization stay unchanged. `Heads.get` and `values`
inherit their types from `Mapping`. `Sink.dim` keeps its runtime annotation so
explicit subclasses can assign `self.dim`.

## Implementation and verification

Mixin moves keep `__slots__ = ()`, layout and existing pydantic error titles.
The internal generator chain includes `_text_batches` and the wrapper in
`streaming`. No numerical code, arithmetic or persistence identifiers change.

Planned checks: positive `assert_type` plus runtime checks for every factory,
all overload branches, sink combinations, union arguments, subclass construction,
base-class quantization, and every public generator path. Explicit sink subclass
assignment, strict validation and moved-method error-title tests cover runtime
compatibility. Minimum pydantic 2.5.0 and Python 3.11 to 3.15 run in CI.

Run targeted tests and conformance, full ruff check and format check over src,
tests, tools and benchmarks, guardrails, and one final `make check`. Heavy work
uses the shared FIFO lock and the approved time window. Add CHANGELOG and docs.
Independent Codex review follows on the committed implementation, at most two rounds.

Claim class: correctness. No timing, no speed claim, no paid run.
Linux comparison: none run. CPU and memory profiles do not apply to these type changes.

Prior art: [PEP 673](https://peps.python.org/pep-0673/),
[overload specification](https://typing.python.org/en/latest/spec/overload.html),
[Python generator annotations](https://docs.python.org/3/library/typing.html#annotating-generators-and-coroutines),
[pydantic validation decorator](https://docs.pydantic.dev/latest/concepts/validation_decorator/).

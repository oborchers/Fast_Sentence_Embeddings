# Base S2v

> Auto-generated documentation for [fse.models.base_s2v](../../../fse/models/base_s2v.py) module.

Base class containing common methods for training, using & evaluating sentence embeddings.
A lot of the code is based on Gensim. I have to thank Radim Rehurek and the whole team
for the outstanding library which I used for a lot of my research.

- [Fast_sentence_embeddings](../../README.md#fast_sentence_embeddings-index) / [Modules](../../MODULES.md#fast_sentence_embeddings-modules) / [Fse](../index.md#fse) / [Models](index.md#models) / Base S2v
    - [BaseSentence2VecModel](#basesentence2vecmodel)
        - [BaseSentence2VecModel().\_\_str\_\_](#basesentence2vecmodel__str__)
        - [BaseSentence2VecModel().estimate_memory](#basesentence2vecmodelestimate_memory)
        - [BaseSentence2VecModel().infer](#basesentence2vecmodelinfer)
        - [BaseSentence2VecModel.load](#basesentence2vecmodelload)
        - [BaseSentence2VecModel().save](#basesentence2vecmodelsave)
        - [BaseSentence2VecModel().scan_sentences](#basesentence2vecmodelscan_sentences)
        - [BaseSentence2VecModel().train](#basesentence2vecmodeltrain)
    - [BaseSentence2VecPreparer](#basesentence2vecpreparer)
        - [BaseSentence2VecPreparer().prepare_vectors](#basesentence2vecpreparerprepare_vectors)
        - [BaseSentence2VecPreparer().reset_vectors](#basesentence2vecpreparerreset_vectors)
        - [BaseSentence2VecPreparer().update_vectors](#basesentence2vecpreparerupdate_vectors)

Attributes
----------
wv : class `gensim.models.keyedvectors.KeyedVectors`
    This object essentially contains the mapping between words and embeddings. After training, it can be used
    directly to query those embeddings in various ways. See the module level docstring for examples.

sv : class `fse.models.sentencevectors.SentenceVectors`
    This object contains the sentence vectors inferred from the training data. There will be one such vector
    for each unique docusentence supplied during training. They may be individually accessed using the index.

prep : class `fse.models.base_s2v.BaseSentence2VecPreparer`
    The prep object is used to transform and initialize the sv.vectors. Aditionally, it can be used
    to move the vectors to disk for training with memmap.

See Also
--------
class `fse.models.average.Average`.
    Average sentence model.
class `fse.models.sif.SIF`.
    Smooth inverse frequency weighted model.
class `fse.models.usif.uSIF`.
    Unsupervised Smooth inverse frequency weighted model.

## BaseSentence2VecModel

[[find in source code]](../../../fse/models/base_s2v.py#L82)

```python
class BaseSentence2VecModel(SaveLoad):
    def __init__(
        model: KeyedVectors,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
        workers: int = 1,
        lang_freq: str = None,
        fast_version: int = 0,
        batch_words: int = 10000,
        batch_ngrams: int = 40,
        **kwargs,
    ):
```

### BaseSentence2VecModel().\_\_str\_\_

[[find in source code]](../../../fse/models/base_s2v.py#L163)

```python
def __str__() -> str:
```

Human readable representation of the model's state.

Returns
-------
str
    Human readable representation of the model's state.

### BaseSentence2VecModel().estimate_memory

[[find in source code]](../../../fse/models/base_s2v.py#L652)

```python
def estimate_memory(
    max_index: int,
    report: dict = None,
    **kwargs,
) -> Dict[str, int]:
```

Estimate the size of the sentence embedding

Parameters
----------
max_index : int
    Maximum index found during the initial scan
report : dict
    Report of subclasses

Returns
-------
dict
    Dictionary of estimated memory sizes

### BaseSentence2VecModel().infer

[[find in source code]](../../../fse/models/base_s2v.py#L768)

```python
def infer(sentences: List[tuple] = None, use_norm=False) -> ndarray:
```

Secondary routine to train an embedding. This method is essential for small batches of sentences,
which require little computation. Note: This method does not apply post-training transformations,
only post inference calls (such as removing principal components).

Parameters
----------
sentences : (list, iterable)
    An iterable consisting of tuple objects
use_norm : bool
    If bool is True, the sentence vectors will be L2 normalized (unit euclidean length)

Returns
-------
ndarray
    Computed sentence vectors

### BaseSentence2VecModel.load

[[find in source code]](../../../fse/models/base_s2v.py#L540)

```python
@classmethod
def load(*args, **kwargs):
```

Load a previously saved class `fse.models.base_s2v.BaseSentence2VecModel`.

Parameters
----------
fname : str
    Path to the saved file.

Returns
-------
class `fse.models.base_s2v.BaseSentence2VecModel`
    Loaded model.

### BaseSentence2VecModel().save

[[find in source code]](../../../fse/models/base_s2v.py#L567)

```python
def save(*args, **kwargs):
```

Save the model.
This saved model can be loaded again using :func:`~fse.models.base_s2v.BaseSentence2VecModel.load`

Parameters
----------
fname : str
    Path to the file.

### BaseSentence2VecModel().scan_sentences

[[find in source code]](../../../fse/models/base_s2v.py#L582)

```python
def scan_sentences(
    sentences: List[tuple] = None,
    progress_per: int = 5,
) -> Dict[str, int]:
```

Performs an initial scan of the data and reports all corresponding statistics

Parameters
----------
sentences : (list, iterable)
    An iterable consisting of tuple objects
progress_per : int
    Number of seconds to pass before reporting the scan progress

Returns
-------
dict
    Dictionary containing the scan statistics

### BaseSentence2VecModel().train

[[find in source code]](../../../fse/models/base_s2v.py#L700)

```python
def train(
    sentences: List[tuple] = None,
    update: bool = False,
    queue_factor: int = 2,
    report_delay: int = 5,
) -> Tuple[int, int]:
```

Main routine to train an embedding. This method writes all sentences vectors into sv.vectors and is
used for computing embeddings for large chunks of data. This method also handles post-training transformations,
such as computing the SVD of the sentence vectors.

Parameters
----------
sentences : (list, iterable)
    An iterable consisting of tuple objects
update : bool
    If bool is True, the sentence vector matrix will be updated in size (even with memmap)
queue_factor : int
    Multiplier for size of queue -> size = number of workers * queue_factor.
report_delay : int
    Number of seconds between two consecutive progress report messages in the logger.

Returns
-------
int, int
    Count of effective sentences and words encountered

## BaseSentence2VecPreparer

[[find in source code]](../../../fse/models/base_s2v.py#L979)

```python
class BaseSentence2VecPreparer(SaveLoad):
```

Contains helper functions to perpare the weights for the training of BaseSentence2VecModel

### BaseSentence2VecPreparer().prepare_vectors

[[find in source code]](../../../fse/models/base_s2v.py#L982)

```python
def prepare_vectors(
    sv: SentenceVectors,
    total_sentences: int,
    update: bool = False,
):
```

Build tables and model weights based on final vocabulary settings.

#### See also

- [SentenceVectors](sentencevectors.md#sentencevectors)

### BaseSentence2VecPreparer().reset_vectors

[[find in source code]](../../../fse/models/base_s2v.py#L991)

```python
def reset_vectors(sv: SentenceVectors, total_sentences: int):
```

Initialize all sentence vectors to zero and overwrite existing files

#### See also

- [SentenceVectors](sentencevectors.md#sentencevectors)

### BaseSentence2VecPreparer().update_vectors

[[find in source code]](../../../fse/models/base_s2v.py#L1008)

```python
def update_vectors(sv: SentenceVectors, total_sentences: int):
```

Given existing sentence vectors, append new ones

#### See also

- [SentenceVectors](sentencevectors.md#sentencevectors)

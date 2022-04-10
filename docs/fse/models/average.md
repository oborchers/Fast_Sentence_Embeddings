# Average

> Auto-generated documentation for [fse.models.average](../../../fse/models/average.py) module.

This module implements the base class to compute average representations for sentences, using highly optimized C routines,
data streaming and Pythonic interfaces.

- [Fast_sentence_embeddings](../../README.md#fast_sentence_embeddings-index) / [Modules](../../MODULES.md#fast_sentence_embeddings-modules) / [Fse](../index.md#fse) / [Models](index.md#models) / Average
    - [Average](#average)
    - [train_average_np](#train_average_np)

The implementation is based on Iyyer et al. (2015): Deep Unordered Composition Rivals Syntactic Methods for Text Classification.
For more information, see <https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf>.

The training algorithms is based on the Gensim implementation of Word2Vec, FastText, and Doc2Vec.
For more information, see: class `gensim.models.word2vec.Word2Vec`, class `gensim.models.fasttext.FastText`, or
class `gensim.models.doc2vec.Doc2Vec`.

Initialize and train a class `fse.models.sentence2vec.Sentence2Vec` model

pycon

```python
>>> from gensim.models.word2vec import Word2Vec
>>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
>>> model = Word2Vec(sentences, min_count=1, vector_size=20)
```

```python
>>> from fse.models.average import Average
>>> avg = Average(model)
>>> avg.train([(s, i) for i, s in enumerate(sentences)])
>>> avg.sv.vectors.shape
(2, 20)

## Average

[[find in source code]](../../../fse/models/average.py#L187)

```python
class Average(BaseSentence2VecModel):
    def __init__(
        model: KeyedVectors,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
        workers: int = 1,
        lang_freq: str = None,
        **kwargs,
    ):
```

Train, use and evaluate averaged sentence vectors.

The model can be stored/loaded via its :meth:`~fse.models.average.Average.save` and
:meth:`~fse.models.average.Average.load` methods.

Some important attributes are the following:

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

#### See also

- [BaseSentence2VecModel](base_s2v.md#basesentence2vecmodel)

## train_average_np

[[find in source code]](../../../fse/models/average.py#L56)

```python
def train_average_np(
    model: BaseSentence2VecModel,
    indexed_sentences: List[tuple],
    target: ndarray,
    memory: ndarray,
) -> Tuple[int, int]:
```

Training on a sequence of sentences and update the target ndarray.

Called internally from :meth:`~fse.models.average.Average._do_train_job`.

Warnings
--------
This is the non-optimized, pure Python version. If you have a C compiler,
fse will use an optimized code path from :mod:`fse.models.average_inner` instead.

Parameters
----------
model : class `fse.models.base_s2v.BaseSentence2VecModel`
    The BaseSentence2VecModel model instance.
indexed_sentences : iterable of tuple
    The sentences used to train the model.
target : ndarray
    The target ndarray. We use the index from indexed_sentences
    to write into the corresponding row of target.
memory : ndarray
    Private memory for each working thread

Returns
-------
int, int
    Number of effective sentences (non-zero) and effective words in the vocabulary used
    during training the sentence embedding.

#### See also

- [BaseSentence2VecModel](base_s2v.md#basesentence2vecmodel)

# Moving from fse 1.x to 2.0

fse 2.0 turns finished static embedding models into sentence vectors.
It separates model preparation from inference, so encoding does not learn or change a model.
The Python API and saved file format have changed.
fse 1.x stays maintained for bug and compatibility fixes on branch `1.x`.

Keep an isolated 1.x environment while moving your application.
Use Python 3.11 to 3.13 for code that needs gensim.
2.0 itself supports Python 3.11 to 3.15, with gensim available through `fse[gensim]` on supported Python versions.
Free-threaded Python builds are unsupported.
Until 2.0 is released, install its development checkout with `UV_PYTHON=3.13 make install`.
Do not replace a production 1.x environment with the development branch.

## Choose the replacement

| Your 1.x code | Your 2.0 code or preparation step |
|---|---|
| `fse.models.Average(kv)` and `train(...)` | Convert finished word vectors, build an `Encoder`, then `encode(...)`. |
| `fse.models.SIF(kv)` | Fit weights and removed directions offline, fold them into word rows, then build an `Encoder`. |
| `fse.models.uSIF(kv)` | The same preparation, with uSIF's learned component weights. |
| Training your own gensim Word2Vec or FastText | Keep training in gensim. Import its finished vectors with `fse.gensim.from_gensim`. |
| `model.infer(...)` | `encoder.encode(...)`. There is no preceding `train`. |
| `model.save(...)` / `Average.load(...)` | `encoder.save(...)` / `Encoder.load(...)`, using a new safetensors file. |
| `Vectors.from_pretrained(...)` / `FTVectors.from_pretrained(...)` | Load trusted legacy word vectors in gensim, then convert them. |
| `workers` in the model constructor | `workers` on each encoding call. |
| `IndexedList` and related input classes | Pass texts, token lists, a `Path`, or a stream directly. |
| `model.sv` similarity methods | NumPy for small arrays, or `ShardSink` and `VectorStore` for saved rows. |

## Average and your own word vectors

These examples use fixed word vectors so they run without a download.
In your application, `kv` can instead be trained gensim `KeyedVectors`, or the `wv` of a Word2Vec model.

1.x:

```python
# 1.x: average
import numpy as np
from gensim.models import KeyedVectors
from fse import IndexedList
from fse.models import Average

words = ["cat", "dog", "sleep"]
table = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)
kv = KeyedVectors(3)
kv.add_vectors(words, table)
tokens = [["cat", "sleep"], ["dog", "sleep"]]
model = Average(kv, workers=1)
model.train(IndexedList(tokens))
rows = model.infer(IndexedList(tokens), use_norm=False)
```

2.0, in a separate environment:

```python
# 2.0: average
import numpy as np
from fse import Encoder

words = ["cat", "dog", "sleep"]
table = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)
tokens = [["cat", "sleep"], ["dog", "sleep"]]
encoder = Encoder.from_arrays(table, words=words, tokenizer="external", normalize=False)
rows = encoder.encode(tokens, workers=1)
assert rows.shape == (2, 3)
```

`normalize=False` requests raw means. New encoders default to normalization to unit length.
Use `tokenizer="external"` when you own tokenization and pass token lists.
Empty token strings are invalid in 2.0. Remove them deliberately during your input preparation.

To import your finished gensim vectors, install the `gensim` extra and replace the array constructor:

```python
# 2.0: gensim
from gensim.models import KeyedVectors
from fse.gensim import from_gensim

kv = KeyedVectors(3)
kv.add_vectors(words, table)
encoder = Encoder.from_word_vectors(from_gensim(kv), tokenizer="external", normalize=False)
rows = encoder.encode(tokens, workers=1)
```

`from_gensim` also accepts a trained Word2Vec or FastText model directly.
It copies the vector data and does not import word frequency counts.
For FastText it imports word rows and ngram rows, with a default limit of 40 ngrams per unknown word.
FastText models without ngram buckets must first be saved as plain KeyedVectors.
Check unknown-word behavior on your model before replacing a FastText application.
The comparison below covers plain word vectors only.

## SIF and uSIF

2.0 has no `SIF`, `uSIF`, `fit`, frequency induction, or component learning API.
Keep these preparation steps offline.
The [SIF and uSIF recipe](../recipes/01-sif-usif.ipynb) shows how to fit a new table.
For an existing trusted 1.x model, export its learned weights and directions in the 1.x environment.
The following example handles plain float32 word vectors, not FastText or quantized tables.

1.x, after fitting either model:

```python
# 1.x: fold
from fse.models import SIF, uSIF

# Set real counts before fitting. These three counts are only an example.
for word, count in zip(words, [20, 10, 5], strict=True):
    kv.set_vecattr(word, "count", count)
model = SIF(kv, components=1, workers=1)
# Or: model = uSIF(kv, components=1, workers=1)
model.train(IndexedList(tokens))
weighted = model.wv.vectors * model.word_weights[:, None]
basis = model.svd_res[1]
factors = model.svd_weights if isinstance(model, uSIF) else np.ones(len(basis), np.float32)
folded = weighted - ((weighted @ basis.T) * factors) @ basis
np.savez("folded.npz", words=np.array(model.wv.index_to_key), vectors=folded)
```

This example requires `components > 0` and an already fitted model.
For `components=0`, export `weighted` directly.
For a saved model, load it with its own 1.x class in the 1.x environment first.
Only load pickle files from a source you trust.

2.0, after copying `folded.npz` to the new environment:

```python
# 2.0: folded
with np.load("folded.npz", allow_pickle=False) as data:
    folded_encoder = Encoder.from_arrays(
        data["vectors"], words=data["words"].tolist(), tokenizer="external", normalize=True
    )
folded_rows = folded_encoder.encode(tokens, workers=1)
```

Folding moves a linear operation before the mean. Floating-point rounding can change the result.
It is not a promise of bit-for-bit reproduction of a 1.x model.
For FastText, preparation also needs a consistent treatment of ngrams and unknown-word weights.
Keep that application on 1.x until you have checked an appropriate offline export.

## Input, inference, and multiple cores

| 1.x | 2.0 |
|---|---|
| `infer(IndexedList(tokens))` | `encode(tokens)` with an external tokenizer |
| `TextList(texts)` or `SplitIndexedList(texts)` | `encode(texts)` with a text tokenizer |
| `CorpusFile(path)` / `IndexedLineDocument(path)` | `encode(Path(path))`, one UTF-8 sentence per line |
| Reiterable indexed input | Ordered collections, or a one-pass stream of strings |
| Custom indices, including repeated indices | One output row per input sentence, in input order. Combine rows yourself if needed. |
| Constructor `workers=1` by default | Usable CPU count by default, with small inputs run inline |

The default text tokenizer `fse_words@1` is not a replacement for every custom 1.x splitter.
Use external token lists to keep your own splitting rules.
A bare string is rejected. Use `[text]` for one sentence and a `Path` for a file.
Streams yield strings, not the old `(tokens, index)` pairs.

```python
# 2.0: inputs
from pathlib import Path

text_encoder = Encoder.from_arrays(table, words=words, normalize=True)
texts = ["cat sleep", "dog sleep"]
text_rows = text_encoder.encode(texts, workers=2)
single = text_encoder.encode(["cat sleep"], workers=1)
streamed = text_encoder.encode((text for text in texts), workers=1)
Path("sentences.txt").write_text("cat sleep\ndog sleep\n", encoding="utf-8")
file_rows = text_encoder.encode(Path("sentences.txt"), workers=1)
np.testing.assert_array_equal(text_rows, streamed)
np.testing.assert_array_equal(text_rows, file_rows)
```

Set `workers=1` per request when a server already handles requests concurrently.
Encoding without a sink holds the full result in memory, including for a stream.
For large collections, use a sink as shown below or process finite batches with `encode_batches`.

## Save and load

1.x:

```python
# 1.x: persistence
model.save("legacy.model")
restored = type(model).load("legacy.model")
```

2.0:

```python
# 2.0: persistence
text_encoder.save("encoder.safetensors")
restored = Encoder.load("encoder.safetensors", mmap=True)
np.testing.assert_array_equal(restored.encode(texts, workers=1), text_rows)
```

The saved encoder includes its word table and tokenizer settings, not previously encoded sentences.
Save sentence rows separately with NumPy or a vector store.
2.0 does not load 1.x pickles and has no automatic model converter.
Use the explicit offline preparation above, or keep 1.x for the old application.
Never rename a legacy file to `.safetensors` to migrate it.

## Old fse Hub models

1.x downloads such as `Vectors.from_pretrained("glove-wiki-gigaword-50")` and
`FTVectors.from_pretrained("fasttext-crawl-subwords-300")` load gensim pickle artifacts from `fse/*` repositories.
`Encoder.from_pretrained` does not load those pickles.

For the same old word vectors, download the trusted `.model` file and all its sidecar arrays together.
Load them with gensim in an environment where it is supported, then convert:

```python
# 2.0: legacy-hub
legacy_kv = KeyedVectors.load("glove-wiki-gigaword-50.model", mmap="r")
legacy_encoder = Encoder.from_word_vectors(from_gensim(legacy_kv), normalize=True)
legacy_rows = legacy_encoder.encode(["cat sleep"], workers=1)
```

The test suite substitutes a tiny local KeyedVectors file for this download.
For legacy FastText downloads, load with `FastTextKeyedVectors.load` instead and review the FastText limits above.
For SIF or uSIF, converting word vectors alone does not preserve learned weights or directions.

Alternatively, choose a finished Model2Vec model, a supported sentence-transformers StaticEmbedding model,
or a 2.0 fse encoder folder. This changes the embedding model and requires rebuilding stored vectors.
This example downloads a pinned Model2Vec model:

```python
# 2.0: hub (network)
hub_encoder = Encoder.from_pretrained(
    "minishlab/potion-base-8M", revision="bf8b056651a2c21b8d2565580b8569da283cab23"
)
hub_rows = hub_encoder.encode(["cat sleep"], workers=1)
assert hub_rows.shape[0] == 1
```

The [Hub contract](architecture/hub.md) lists supported layouts and tokenization limits.
An arbitrary transformer model is not a static embedding table.

## Similarity and saved sentence rows

1.x exposes `model.sv.similarity(0, 1)` and `model.sv.most_similar(0, topn=2)`.
2.0 returns arrays. For these normalized, nonzero rows, their dot product gives cosine similarity:

```python
# 2.0: similarity
similarity = float(text_rows[0] @ text_rows[1])
assert -1.0 <= similarity <= 1.0
```

For saved rows and cosine search:

```python
# 2.0: store
from fse import ShardSink, VectorStore

with ShardSink("sentence-store", text_encoder) as sink:
    text_encoder.encode(texts, workers=1, sink=sink)
with VectorStore.open("sentence-store", encoder=text_encoder) as store:
    found = store.search(text_encoder.encode(["cat sleep"], workers=1), k=2, mode="exact")
    assert found.ids.shape == (1, 2)
    assert found.ids[0, 0] == 0
```

IDs are input positions. Keep your text or application IDs separately.
Search returns the query's own row when it is present, unlike `most_similar` with a row index.
Filter that row yourself for a nearest-neighbor list that excludes it.
Zero query vectors are rejected by store search. Decide how your application handles sentences without known words.
See [files with resume](../recipes/03-files-with-resume.ipynb) before appending or resuming a store.

## Check results before switching

Run both environments on your own inputs and rebuild stored vectors when you change the model or tokenizer.
Do not mix 1.x and 2.0 vectors merely because their dimensions agree.

The repository records a small, offline comparison in [migration-comparison.json](migration-comparison.json).
Its [script](../benchmarks/compare_migration.py) uses the same fixed word table and token lists in both versions.
It checks Average, SIF, and uSIF, with folded tables exported by 1.x.
The recorded run distinguishes known words, an unknown word, an unknown-only sentence, and an empty sentence.
It compares raw outputs and normalized outputs separately. There are no timing measurements.

Reproduce from the repository root, with each environment using its own installed fse:

```sh
/path/to/1x/bin/python benchmarks/compare_migration.py export /tmp/migration.npz \
    --legacy-commit "$(git rev-parse origin/1.x)"
.venv/bin/python benchmarks/compare_migration.py compare /tmp/migration.npz
```

Read the recorded metrics with their scope. This tiny plain-vector example cannot establish equivalence for other models,
FastText, tokenizers, quantization, or older SIF/uSIF component formulas.

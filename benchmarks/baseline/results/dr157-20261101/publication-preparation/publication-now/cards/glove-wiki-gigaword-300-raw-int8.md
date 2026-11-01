---
library_name: "fse"
license: "pddl"
base_model: "fse/glove-wiki-gigaword-300"
base_model_relation: "quantized"
language:
- "en"
pipeline_tag: "feature-extraction"
tags:
- "fse"
- "static-embeddings"
- "sentence-embeddings"
- "gensim"
- "glove"
- "word-embeddings"
---

# fse/glove-wiki-gigaword-300-raw-int8

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 400000
- dimensions of the table: 300 (int8)
- table stored by `quantize_rows@1` from the float32 table `xxh3-128:2e100367bf4862e645b1ff921714eaf6`
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- fingerprint: `xxh3-128:30618248eaf896d418ac1c3efcc784e8`

## Usage

This model requires fse 2.0 or later.

```
pip install "fse>=2.0"
```

```python
import fse

encoder = fse.Encoder.from_pretrained("fse/glove-wiki-gigaword-300-raw-int8")
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/glove-wiki-gigaword-300 at revision `4e5ad402664236b85be45a0d107b95259ba8a909`.

The vectors fse computes are those of the quantized table: on the cpu backend bit for bit as its reference computes them, not those of the model before quantization. How close they come to it is measured, not promised by this file.

License: pddl. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

<!-- fse-classic-card@1 -->

## Choosing a method and storage format

On the 1,379 STS-B test pairs, raw float32 scored 44.46410 Pearson x100 and SIF-10 float32 scored 69.22434. Use raw for a plain mean of the source vectors. Use SIF-10 to apply the fixed frequency weighting and training-corpus component removal measured here. Evaluate the choice on your own data.

Use float32 to retain the prepared table without storage rounding. Use float16 or int8 when the measured size reduction and neighbour changes below meet your needs. fse computes all three in float32.

| This model | Measured value |
|---|---:|
| Model file bytes | 127757762 |
| STS-B Pearson x100 | 44.45861 |
| Difference from same-method float32, points | -0.00549 |
| Recall@10 against same-method float32 | 0.993836 |
| Default tokenizer Pearson x100 | 44.45861 |

The controlled STS-B comparison uses the 1.x notebook's NLTK preprocessing through an external-tokenizer view of this same table. The default-tokenizer row uses the saved `fse_words@1` tokenizer on raw text. Recall@10 compares each of the 2,758 STS-B sentence rows against all other rows, excludes self, and resolves equal similarities by row index. No load-time or speed claim is made.

## Source terms and attribution

The pretrained vectors are released under the [Public Domain Dedication and License](https://opendatacommons.org/licenses/pddl/1-0/), as stated by the [GloVe authors](https://github.com/stanfordnlp/GloVe). The Apache-2.0 license of their implementation is separate from the vector terms.

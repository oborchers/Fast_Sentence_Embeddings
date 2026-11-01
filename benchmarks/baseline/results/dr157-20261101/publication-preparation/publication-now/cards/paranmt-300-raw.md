---
library_name: "fse"
license: "unknown"
base_model: "fse/paranmt-300"
language:
- "en"
pipeline_tag: "feature-extraction"
tags:
- "fse"
- "static-embeddings"
- "sentence-embeddings"
- "gensim"
- "paragram"
- "word-embeddings"
---

# fse/paranmt-300-raw

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 77224
- dimensions of the table: 300 (float32)
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- fingerprint: `xxh3-128:f870e4619c3e5dd082be1809d1fb7786`

## Usage

This model requires fse 2.0 or later.

```
pip install "fse>=2.0"
```

```python
import fse

encoder = fse.Encoder.from_pretrained("fse/paranmt-300-raw")
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/paranmt-300 at revision `a08e15771891207a22f840a3091a00a1657d888b`.

License: unknown. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

<!-- fse-classic-card@1 -->

## Choosing a method and storage format

On the 1,379 STS-B test pairs, raw float32 scored 79.81644 Pearson x100 and SIF-10 float32 scored 78.11215. Use raw for a plain mean of the source vectors. Use SIF-10 to apply the fixed frequency weighting and training-corpus component removal measured here. Evaluate the choice on your own data.

Use float32 to retain the prepared table without storage rounding. Use float16 or int8 when the measured size reduction and neighbour changes below meet your needs. fse computes all three in float32.

| This model | Measured value |
|---|---:|
| Model file bytes | 93858331 |
| STS-B Pearson x100 | 79.81644 |
| Difference from same-method float32, points | +0.00000 |
| Recall@10 against same-method float32 | 1.000000 |
| Default tokenizer Pearson x100 | 79.81644 |

The controlled STS-B comparison uses the 1.x notebook's NLTK preprocessing through an external-tokenizer view of this same table. The default-tokenizer row uses the saved `fse_words@1` tokenizer on raw text. Recall@10 compares each of the 2,758 STS-B sentence rows against all other rows, excludes self, and resolves equal similarities by row index. No load-time or speed claim is made.

## Source terms and attribution

The authors did not state a license for these vectors in the release materials we inspected. `license: unknown` does not grant permission. This conversion is kept for preservation. See [ParaNMT](https://github.com/jwieting/para-nmt-50m) and the [paper](https://aclanthology.org/P18-1042/). Its training data, [CzEng 1.6](https://ufal.mff.cuni.cz/czeng/czeng16), is freely available only for non-commercial research. This does not establish a license for the derived vectors. Rights holders can request removal through the [repository issues](https://github.com/oborchers/Fast_Sentence_Embeddings/issues).

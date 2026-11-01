---
library_name: "fse"
license: "cc-by-sa-3.0"
base_model: "fse/fasttext-crawl-subwords-300"
base_model_relation: "quantized"
language:
- "en"
pipeline_tag: "feature-extraction"
tags:
- "fse"
- "static-embeddings"
- "sentence-embeddings"
- "gensim"
- "fasttext"
- "word-embeddings"
---

# fse/fasttext-crawl-subwords-300-raw-int8

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 2000000
- dimensions of the table: 300 (int8)
- table stored by `quantize_rows@1` from the float32 table `xxh3-128:a7775dd2c0333965666defe0d670ea8e`
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- FastText ngrams: {"bucket":2000000,"max_n":4,"min_n":4}
- fingerprint: `xxh3-128:f67baa976b632633617f4958856214b8`

## Usage

This model requires fse 2.0 or later.

```
pip install "fse>=2.0"
```

```python
import fse

encoder = fse.Encoder.from_pretrained("fse/fasttext-crawl-subwords-300-raw-int8")
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/fasttext-crawl-subwords-300 at revision `552b66a16850576275f894ed738f9e953b45a594`.

The vectors fse computes are those of the quantized table: on the cpu backend bit for bit as its reference computes them, not those of the model before quantization. How close they come to it is measured, not promised by this file.

License: cc-by-sa-3.0. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

<!-- fse-classic-card@1 -->

## Choosing a method and storage format

On the 1,379 STS-B test pairs, raw float32 scored 48.52845 Pearson x100 and SIF-10 float32 scored 72.40582. Use raw for a plain mean of the source vectors. Use SIF-10 to apply the fixed frequency weighting and training-corpus component removal measured here. Evaluate the choice on your own data.

Use float32 to retain the prepared table without storage rounding. Use float16 or int8 when the measured size reduction and neighbour changes below meet your needs. fse computes all three in float32.

| This model | Measured value |
|---|---:|
| Model file bytes | 1247846930 |
| STS-B Pearson x100 | 48.53995 |
| Difference from same-method float32, points | +0.01150 |
| Recall@10 against same-method float32 | 0.994670 |
| Default tokenizer Pearson x100 | 48.53995 |

The controlled STS-B comparison uses the 1.x notebook's NLTK preprocessing through an external-tokenizer view of this same table. The default-tokenizer row uses the saved `fse_words@1` tokenizer on raw text. Recall@10 compares each of the 2,758 STS-B sentence rows against all other rows, excludes self, and resolves equal similarities by row index. No load-time or speed claim is made.

## Source terms and attribution

The source vectors are by Facebook, from [fastText English word vectors](https://fasttext.cc/docs/en/english-vectors.html), under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). All six derived repositories retain CC BY-SA 3.0, including ShareAlike. Changes: conversion to fse safetensors, sentence mean and normalization, SIF weighting and component folding for `sif10`, and storage rounding for float16 or int8. Facebook does not endorse this conversion.

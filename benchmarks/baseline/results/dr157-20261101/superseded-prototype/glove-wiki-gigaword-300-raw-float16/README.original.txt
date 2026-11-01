---
library_name: "fse"
license: "pddl"
base_model: "fse/glove-wiki-gigaword-300"
language:
- "en"
pipeline_tag: "feature-extraction"
tags:
- "fse"
- "static-embeddings"
- "sentence-embeddings"
---

# fse/glove-wiki-gigaword-300-raw-float16

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 400000
- dimensions of the table: 300 (float16)
- table stored by `quantize_rows@1` from the float32 table `xxh3-128:2e100367bf4862e645b1ff921714eaf6`
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- fingerprint: `xxh3-128:808b1ba980661cbe984a5d34f0a79d2b`
- written by: fse 1.1.1.dev917+g81e64c0

## Usage

```
pip install fse
```

```python
import fse

encoder = fse.Encoder.from_pretrained("/Users/oliverborchers/fse-dr157/glove-wiki-gigaword-300-raw-float16")
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/glove-wiki-gigaword-300 at revision `4e5ad402664236b85be45a0d107b95259ba8a909`.

The vectors fse computes are those of the quantized table: on the cpu backend bit for bit as its reference computes them, not those of the model before quantization. How close they come to it is measured, not promised by this file.

License: pddl. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

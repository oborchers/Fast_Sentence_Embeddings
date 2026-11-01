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
---

# fse/glove-wiki-gigaword-300-sif10-int8

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 400000
- dimensions of the table: 300 (int8)
- table stored by `quantize_rows@1` from the float32 table `xxh3-128:9284d839586ca3068758978e3dceb298`
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- fingerprint: `xxh3-128:99aed226779a6e8bd839c8d163a1b229`
- written by: fse 1.1.1.dev917+g81e64c0

## Usage

```
pip install fse
```

```python
import fse

encoder = fse.Encoder.from_pretrained(
    "/Users/oliverborchers/fse-dr157/glove-wiki-gigaword-300-sif10-int8"
)
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/glove-wiki-gigaword-300 at revision `4e5ad402664236b85be45a0d107b95259ba8a909`.

The vectors fse computes are those of the quantized table: on the cpu backend bit for bit as its reference computes them, not those of the model before quantization. How close they come to it is measured, not promised by this file.

License: pddl. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

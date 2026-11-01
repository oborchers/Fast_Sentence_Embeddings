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

# fse/glove-wiki-gigaword-300-sif10

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 400000
- dimensions of the table: 300 (float32)
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- fingerprint: `xxh3-128:62bd0be39bfcd738d7e3cedc7a5b910c`
- written by: fse 1.1.1.dev917+g81e64c0

## Usage

```
pip install fse
```

```python
import fse

encoder = fse.Encoder.from_pretrained(
    "/Users/oliverborchers/fse-dr157/glove-wiki-gigaword-300-sif10"
)
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/glove-wiki-gigaword-300 at revision `4e5ad402664236b85be45a0d107b95259ba8a909`.

License: pddl. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

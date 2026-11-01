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
datasets:
- "HuggingFaceFW/fineweb-edu-llama3-annotations"
---

# fse/fasttext-crawl-subwords-300-sif10-int8

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 2000000
- dimensions of the table: 300 (int8)
- table stored by `quantize_rows@1` from the float32 table `xxh3-128:1f07655fb868118fcf327209bf91e562`
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- FastText ngrams: {"bucket":2000000,"max_n":4,"min_n":4}
- fingerprint: `xxh3-128:075213b2d66190094fb9dc35b84e2f35`

## Usage

This model requires fse 2.0 or later.

```
pip install "fse>=2.0"
```

```python
import fse

encoder = fse.Encoder.from_pretrained("fse/fasttext-crawl-subwords-300-sif10-int8")
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
| STS-B Pearson x100 | 72.40633 |
| Difference from same-method float32, points | +0.00051 |
| Recall@10 against same-method float32 | 0.995975 |
| Default tokenizer Pearson x100 | 72.40633 |

The controlled STS-B comparison uses the 1.x notebook's NLTK preprocessing through an external-tokenizer view of this same table. The default-tokenizer row uses the saved `fse_words@1` tokenizer on raw text. Recall@10 compares each of the 2,758 STS-B sentence rows against all other rows, excludes self, and resolves equal similarities by row index. No load-time or speed claim is made.

## Offline SIF preparation

Word weights are a/(a+p(w)), with a=0.001. Frequencies match fse 1.x `lang_freq='en'`: English wordfreq best-list frequencies, converted to integer counts over 2**31-1 with a 1e-8 default, then normalized in float32.

- wordfreq version: 3.1.1
- component corpus: `HuggingFaceFW/fineweb-edu-llama3-annotations` at `72df4c92fb1b48beceb16016e8f695ec40a6c3a5`
- training paragraphs: 859346; seed: 0
- tokenized corpus checksum: `xxh3-128:88146a7f61a2c157c4710cfaf3e0cb4b`
- fitted weights and components checksum: `xxh3-128:7b81d9963c76f04d3d6be942b20ad3a9`

Ten uncentered components were fitted on that training corpus, never on the evaluation texts, then folded into the weighted table. Quantization follows folding. GloVe selected this corpus by the mean of STS-B Pearson, STS-B Spearman and NanoBEIR macro nDCG@10, each scaled to 0-100. Those benchmarks guided corpus selection, so their scores are not untouched holdout estimates. `preparation.json` records the full fit receipt.

Frequency-data attribution: [wordfreq](https://github.com/rspeer/wordfreq) by Robyn Speer. Its code is Apache-2.0; its frequency data includes CC BY-SA 4.0 material and source-specific credits in [NOTICE.md](https://github.com/rspeer/wordfreq/blob/master/NOTICE.md). Preparation transforms these frequencies into weights and folds them into the table. The source-vector license field does not waive those data terms.

## Source terms and attribution

The source vectors are by Facebook, from [fastText English word vectors](https://fasttext.cc/docs/en/english-vectors.html), under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). All six derived repositories retain CC BY-SA 3.0, including ShareAlike. Changes: conversion to fse safetensors, sentence mean and normalization, SIF weighting and component folding for `sif10`, and storage rounding for float16 or int8. Facebook does not endorse this conversion.

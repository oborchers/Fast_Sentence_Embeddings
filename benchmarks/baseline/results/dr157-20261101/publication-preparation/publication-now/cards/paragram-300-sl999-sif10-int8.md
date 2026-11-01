---
library_name: "fse"
license: "unknown"
base_model: "fse/paragram-300-sl999"
base_model_relation: "quantized"
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
datasets:
- "HuggingFaceFW/fineweb-edu-llama3-annotations"
---

# fse/paragram-300-sl999-sif10-int8

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 1703756
- dimensions of the table: 300 (int8)
- table stored by `quantize_rows@1` from the float32 table `xxh3-128:03b92a656496cada609ac05d18d4c722`
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- fingerprint: `xxh3-128:37fbc7e840861300414e76ec0964f409`

## Usage

This model requires fse 2.0 or later.

```
pip install "fse>=2.0"
```

```python
import fse

encoder = fse.Encoder.from_pretrained("fse/paragram-300-sl999-sif10-int8")
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/paragram-300-sl999 at revision `c99818c1470d5c4e454eb9d59dfce2b179dba455`.

The vectors fse computes are those of the quantized table: on the cpu backend bit for bit as its reference computes them, not those of the model before quantization. How close they come to it is measured, not promised by this file.

License: unknown. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

<!-- fse-classic-card@1 -->

## Choosing a method and storage format

On the 1,379 STS-B test pairs, raw float32 scored 51.45857 Pearson x100 and SIF-10 float32 scored 74.43794. Use raw for a plain mean of the source vectors. Use SIF-10 to apply the fixed frequency weighting and training-corpus component removal measured here. Evaluate the choice on your own data.

Use float32 to retain the prepared table without storage rounding. Use float16 or int8 when the measured size reduction and neighbour changes below meet your needs. fse computes all three in float32.

| This model | Measured value |
|---|---:|
| Model file bytes | 545590639 |
| STS-B Pearson x100 | 74.43709 |
| Difference from same-method float32, points | -0.00085 |
| Recall@10 against same-method float32 | 0.996483 |
| Default tokenizer Pearson x100 | 74.43709 |

The controlled STS-B comparison uses the 1.x notebook's NLTK preprocessing through an external-tokenizer view of this same table. The default-tokenizer row uses the saved `fse_words@1` tokenizer on raw text. Recall@10 compares each of the 2,758 STS-B sentence rows against all other rows, excludes self, and resolves equal similarities by row index. No load-time or speed claim is made.

## Offline SIF preparation

Word weights are a/(a+p(w)), with a=0.001. Frequencies match fse 1.x `lang_freq='en'`: English wordfreq best-list frequencies, converted to integer counts over 2**31-1 with a 1e-8 default, then normalized in float32.

- wordfreq version: 3.1.1
- component corpus: `HuggingFaceFW/fineweb-edu-llama3-annotations` at `72df4c92fb1b48beceb16016e8f695ec40a6c3a5`
- training paragraphs: 859346; seed: 0
- tokenized corpus checksum: `xxh3-128:88146a7f61a2c157c4710cfaf3e0cb4b`
- fitted weights and components checksum: `xxh3-128:7064e3bbdaf1520ff851268ef04e8ba4`

Ten uncentered components were fitted on that training corpus, never on the evaluation texts, then folded into the weighted table. Quantization follows folding. GloVe selected this corpus by the mean of STS-B Pearson, STS-B Spearman and NanoBEIR macro nDCG@10, each scaled to 0-100. Those benchmarks guided corpus selection, so their scores are not untouched holdout estimates. `preparation.json` records the full fit receipt.

Frequency-data attribution: [wordfreq](https://github.com/rspeer/wordfreq) by Robyn Speer. Its code is Apache-2.0; its frequency data includes CC BY-SA 4.0 material and source-specific credits in [NOTICE.md](https://github.com/rspeer/wordfreq/blob/master/NOTICE.md). Preparation transforms these frequencies into weights and folds them into the table. The source-vector license field does not waive those data terms.

## Source terms and attribution

The authors did not state a license for these vectors in the release materials we inspected. `license: unknown` does not grant permission. This conversion is kept for preservation. See [From Paraphrase Database to Compositional Paraphrase Model and Back](https://aclanthology.org/Q15-1025/). Rights holders can request removal through the [repository issues](https://github.com/oborchers/Fast_Sentence_Embeddings/issues).

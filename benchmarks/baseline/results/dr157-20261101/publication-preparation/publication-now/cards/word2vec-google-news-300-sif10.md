---
library_name: "fse"
license: "other"
base_model: "fse/word2vec-google-news-300"
language:
- "en"
pipeline_tag: "feature-extraction"
tags:
- "fse"
- "static-embeddings"
- "sentence-embeddings"
- "gensim"
- "word2vec"
- "word-embeddings"
datasets:
- "HuggingFaceFW/fineweb-edu-llama3-annotations"
license_name: "word2vec-google-news-research-only"
license_link: "https://code.google.com/archive/p/word2vec/"
---

# fse/word2vec-google-news-300-sif10

A static embedding model for fse: the mean of the token rows of a text, normalized to unit length.

## Model

- rows (vocabulary): 3000000
- dimensions of the table: 300 (float32)
- tokenizer: `fse_words@1`
- steps per row: normalize@1
- fingerprint: `xxh3-128:d56902a9d7d77240e264aa99f4d2cf00`

## Usage

This model requires fse 2.0 or later.

```
pip install "fse>=2.0"
```

```python
import fse

encoder = fse.Encoder.from_pretrained("fse/word2vec-google-news-300-sif10")
vectors = encoder.encode(["A sentence."])
```

## Source and license

Converted from the gensim model fse/word2vec-google-news-300 at revision `4274c2efa75c99a64e3f83eabfc5b34d1acabe70`.

License: other. The license of the source model applies to its use.

## About fse

Made with [fse](https://github.com/oborchers/Fast_Sentence_Embeddings), fast sentence embeddings.

<!-- fse-classic-card@1 -->

## Choosing a method and storage format

On the 1,379 STS-B test pairs, raw float32 scored 61.54132 Pearson x100 and SIF-10 float32 scored 70.88642. Use raw for a plain mean of the source vectors. Use SIF-10 to apply the fixed frequency weighting and training-corpus component removal measured here. Evaluate the choice on your own data.

Use float32 to retain the prepared table without storage rounding. Use float16 or int8 when the measured size reduction and neighbour changes below meet your needs. fse computes all three in float32.

| This model | Measured value |
|---|---:|
| Model file bytes | 3665259542 |
| STS-B Pearson x100 | 70.88642 |
| Difference from same-method float32, points | +0.00000 |
| Recall@10 against same-method float32 | 1.000000 |
| Default tokenizer Pearson x100 | 70.88642 |

The controlled STS-B comparison uses the 1.x notebook's NLTK preprocessing through an external-tokenizer view of this same table. The default-tokenizer row uses the saved `fse_words@1` tokenizer on raw text. Recall@10 compares each of the 2,758 STS-B sentence rows against all other rows, excludes self, and resolves equal similarities by row index. No load-time or speed claim is made.

## Offline SIF preparation

Word weights are a/(a+p(w)), with a=0.001. Frequencies match fse 1.x `lang_freq='en'`: English wordfreq best-list frequencies, converted to integer counts over 2**31-1 with a 1e-8 default, then normalized in float32.

- wordfreq version: 3.1.1
- component corpus: `HuggingFaceFW/fineweb-edu-llama3-annotations` at `72df4c92fb1b48beceb16016e8f695ec40a6c3a5`
- training paragraphs: 859346; seed: 0
- tokenized corpus checksum: `xxh3-128:88146a7f61a2c157c4710cfaf3e0cb4b`
- fitted weights and components checksum: `xxh3-128:ab1c7dd640d7f6d79bd8d8aba629466e`

Ten uncentered components were fitted on that training corpus, never on the evaluation texts, then folded into the weighted table. Quantization follows folding. GloVe selected this corpus by the mean of STS-B Pearson, STS-B Spearman and NanoBEIR macro nDCG@10, each scaled to 0-100. Those benchmarks guided corpus selection, so their scores are not untouched holdout estimates. `preparation.json` records the full fit receipt.

Frequency-data attribution: [wordfreq](https://github.com/rspeer/wordfreq) by Robyn Speer. Its code is Apache-2.0; its frequency data includes CC BY-SA 4.0 material and source-specific credits in [NOTICE.md](https://github.com/rspeer/wordfreq/blob/master/NOTICE.md). Preparation transforms these frequencies into weights and folds them into the table. The source-vector license field does not waive those data terms.

## Source terms and attribution

The [original word2vec project](https://code.google.com/archive/p/word2vec/) states that the project is 'released for research purposes only'. The card therefore uses `license: other`. Apache-2.0 covers the word2vec code, not these pretrained vectors. No broader permission is asserted here.

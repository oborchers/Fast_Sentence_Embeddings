# Classic models on the Hub

All 30 models below run in fse and return 300-dimensional vectors.
Open a model card by clicking its Hub ID, or copy the ID into `Encoder.from_pretrained`.
For a loading example, see [Get started](../README.md#get-started).

## Choose a version

- `-raw`: original word vectors without SIF weighting.
- `-sif10`: fixed word weights and ten common directions removed before saving.
- No precision suffix: float32. `-float16` and `-int8` use less space for the model table.

Both raw and SIF-10 models normalize sentence vectors by default.
`-raw` does not mean unnormalized output. For a linear head that folds, load with `normalize=False` before training.
The Hub SIF-10 models use English wordfreq weights and a FineWeb fit. Their model cards contain their own quality results.

## All models

Sizes below cover `model.safetensors`, not the whole repository, checked on 2 November 2026.
MiB and GiB use powers of 1024. Each model card describes its preparation and license.

| Model (copyable Hub ID) | Model file |
|---|---:|
| [`fse/glove-wiki-gigaword-300-raw`](https://huggingface.co/fse/glove-wiki-gigaword-300-raw) | 463.6 MiB |
| [`fse/glove-wiki-gigaword-300-raw-float16`](https://huggingface.co/fse/glove-wiki-gigaword-300-raw-float16) | 234.8 MiB |
| [`fse/glove-wiki-gigaword-300-raw-int8`](https://huggingface.co/fse/glove-wiki-gigaword-300-raw-int8) | 121.8 MiB |
| [`fse/glove-wiki-gigaword-300-sif10`](https://huggingface.co/fse/glove-wiki-gigaword-300-sif10) | 463.6 MiB |
| [`fse/glove-wiki-gigaword-300-sif10-float16`](https://huggingface.co/fse/glove-wiki-gigaword-300-sif10-float16) | 234.8 MiB |
| [`fse/glove-wiki-gigaword-300-sif10-int8`](https://huggingface.co/fse/glove-wiki-gigaword-300-sif10-int8) | 121.8 MiB |
| [`fse/word2vec-google-news-300-raw`](https://huggingface.co/fse/word2vec-google-news-300-raw) | 3.41 GiB |
| [`fse/word2vec-google-news-300-raw-float16`](https://huggingface.co/fse/word2vec-google-news-300-raw-float16) | 1.74 GiB |
| [`fse/word2vec-google-news-300-raw-int8`](https://huggingface.co/fse/word2vec-google-news-300-raw-int8) | 932.0 MiB |
| [`fse/word2vec-google-news-300-sif10`](https://huggingface.co/fse/word2vec-google-news-300-sif10) | 3.41 GiB |
| [`fse/word2vec-google-news-300-sif10-float16`](https://huggingface.co/fse/word2vec-google-news-300-sif10-float16) | 1.74 GiB |
| [`fse/word2vec-google-news-300-sif10-int8`](https://huggingface.co/fse/word2vec-google-news-300-sif10-int8) | 932.0 MiB |
| [`fse/fasttext-crawl-subwords-300-raw`](https://huggingface.co/fse/fasttext-crawl-subwords-300-raw) | 4.50 GiB |
| [`fse/fasttext-crawl-subwords-300-raw-float16`](https://huggingface.co/fse/fasttext-crawl-subwords-300-raw-float16) | 2.26 GiB |
| [`fse/fasttext-crawl-subwords-300-raw-int8`](https://huggingface.co/fse/fasttext-crawl-subwords-300-raw-int8) | 1.16 GiB |
| [`fse/fasttext-crawl-subwords-300-sif10`](https://huggingface.co/fse/fasttext-crawl-subwords-300-sif10) | 4.50 GiB |
| [`fse/fasttext-crawl-subwords-300-sif10-float16`](https://huggingface.co/fse/fasttext-crawl-subwords-300-sif10-float16) | 2.26 GiB |
| [`fse/fasttext-crawl-subwords-300-sif10-int8`](https://huggingface.co/fse/fasttext-crawl-subwords-300-sif10-int8) | 1.16 GiB |
| [`fse/paragram-300-sl999-raw`](https://huggingface.co/fse/paragram-300-sl999-raw) | 1.93 GiB |
| [`fse/paragram-300-sl999-raw-float16`](https://huggingface.co/fse/paragram-300-sl999-raw-float16) | 1001.3 MiB |
| [`fse/paragram-300-sl999-raw-int8`](https://huggingface.co/fse/paragram-300-sl999-raw-int8) | 520.3 MiB |
| [`fse/paragram-300-sl999-sif10`](https://huggingface.co/fse/paragram-300-sl999-sif10) | 1.93 GiB |
| [`fse/paragram-300-sl999-sif10-float16`](https://huggingface.co/fse/paragram-300-sl999-sif10-float16) | 1001.3 MiB |
| [`fse/paragram-300-sl999-sif10-int8`](https://huggingface.co/fse/paragram-300-sl999-sif10-int8) | 520.3 MiB |
| [`fse/paranmt-300-raw`](https://huggingface.co/fse/paranmt-300-raw) | 89.5 MiB |
| [`fse/paranmt-300-raw-float16`](https://huggingface.co/fse/paranmt-300-raw-float16) | 45.3 MiB |
| [`fse/paranmt-300-raw-int8`](https://huggingface.co/fse/paranmt-300-raw-int8) | 23.5 MiB |
| [`fse/paranmt-300-sif10`](https://huggingface.co/fse/paranmt-300-sif10) | 89.5 MiB |
| [`fse/paranmt-300-sif10-float16`](https://huggingface.co/fse/paranmt-300-sif10-float16) | 45.3 MiB |
| [`fse/paranmt-300-sif10-int8`](https://huggingface.co/fse/paranmt-300-sif10-int8) | 23.5 MiB |

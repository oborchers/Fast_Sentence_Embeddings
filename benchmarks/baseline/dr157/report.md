# Classic word-vector preparation report

DR-157 (Ausgewählte Modelle auf den Hugging Face Hub laden), PR [#201 (2.0: Prepare the classic word-vector models for the Hub)](https://github.com/oborchers/Fast_Sentence_Embeddings/pull/201).
All 30 models are local. No Hub write, upload, tag or release was performed.

Raw build, fit, comparison, review and gate evidence is in [archive commit `dac448a925fb3a6c63570be9f3d637b0d948bb53`](https://github.com/oborchers/Fast_Sentence_Embeddings/tree/dac448a925fb3a6c63570be9f3d637b0d948bb53/benchmarks/baseline/results/dr157-20261101).
All 146 files (476,939 bytes) were independently compared with the source bytes and SHA256 before branch cleanup.
The complete original report is [`report-full.md`](https://github.com/oborchers/Fast_Sentence_Embeddings/tree/dac448a925fb3a6c63570be9f3d637b0d948bb53/benchmarks/baseline/results/dr157-20261101/report-full.md). `archive-summary.json` records the transfer proof.

## Approved payload and release gate

The 30 repositories contain 135 files totaling **39,426,324,698 bytes** (39.426 GB, 36.719 GiB).
FastText accounts for 17,023,110,368 bytes. Its separate hold is lifted; DR-146 (Über FastText in 2.0 entscheiden (12f)) is Done.
Oliver approved content, repo names and licenses under Option B. Upload waits for release 2.0.0 and regenerated release cards.
Current manifest SHA256: `ac85cae41b05788bc278d68fd3231489de1615c7c15b8b79229844c3d0dabe69`.
`publication-inventory.md` lists all repo/file sizes and SHA256; `cards/*.txt` preserves exact current card text.
Model bytes, card bytes and the publication manifest are unchanged by evidence relocation.

`RELEASE_CARDS.md` gives the guarded card refresh after remote tag v2.0.0 and installation of actual fse 2.0.0.
Only README Usage changes to `pip install fse` and the released API requirement. New hardlink views preserve all non-README SHA256.
The historical artifact writer remains accurate. New release manifest SHA256 is reported only after that step runs.
`POST_UPLOAD.md` covers fresh isolated downloads, exact files, literal card examples, metadata and the 512-sentence probe.
Its Pearson tolerance is 0.001 points, plus bitwise CPU vectors. Failures never trigger Hub repair writes.

## Component-fit corpus selection

| Corpus | STS-B Pearson x100 | STS-B Spearman x100 | NanoBEIR nDCG@10 x100 | Mean |
|---|---:|---:|---:|---:|
| wikitext | 68.67240 | 65.15750 | 40.59846 | 58.14279 |
| fineweb | 69.22434 | 65.36655 | 40.56338 | 58.38476 |

FineWeb wins the equal-weight mean. NanoBEIR uses all 13 tasks; corpus selection used evaluation scores, so these are not untouched holdouts.
Weights use wordfreq 3.1.1 English best frequencies, a=0.001. Ten uncentered components use 859,346 separate training paragraphs, seed 0.
FineWeb revision: `72df4c92fb1b48beceb16016e8f695ec40a6c3a5`; WikiText-103 revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.
Each source has its own fitted state. Cards and preparation receipts retain corpus, revision, seed, wordfreq version and state hashes.
No core runtime dependency or persistence-schema change was needed. Licenses and attribution remain in every approved card.

## STS-B comparison with fse 1.x

Pearson x100 on all 1,379 test pairs with the controlled 1.x preprocessing.
Recall@10 uses 2,758 rows, excludes self and resolves ties by row id. Historical test-fit SIF is diagnostic only and never shipped.

| Source | Method | Dtype | 1.x | 2.0 | Difference, points | vs float32, points | Recall@10 |
|---|---|---|---:|---:|---:|---:|---:|
| paragram-300-sl999 | raw | float32 | 51.46 | 51.45857 | -0.00143 | +0.00000 | 1.000000 |
| paragram-300-sl999 | raw | float16 | 51.46 | 51.45803 | -0.00197 | -0.00054 | 1.000000 |
| paragram-300-sl999 | raw | int8 | 51.46 | 51.44729 | -0.01271 | -0.01128 | 0.996193 |
| paragram-300-sl999 | sif10 | float32 | 74.29 | 74.43794 | +0.14794 | +0.00000 | 1.000000 |
| paragram-300-sl999 | sif10 | float16 | 74.29 | 74.43796 | +0.14796 | +0.00002 | 0.999927 |
| paragram-300-sl999 | sif10 | int8 | 74.29 | 74.43709 | +0.14709 | -0.00085 | 0.996483 |
| paragram-300-sl999 | sif10-historical-test-fit | float32 | 74.29 | 74.29445 | +0.00445 | n/a | n/a |
| paranmt-300 | raw | float32 | 79.82 | 79.81644 | -0.00356 | +0.00000 | 1.000000 |
| paranmt-300 | raw | float16 | 79.82 | 79.81645 | -0.00355 | +0.00001 | 0.999891 |
| paranmt-300 | raw | int8 | 79.82 | 79.81797 | -0.00203 | +0.00153 | 0.995722 |
| paranmt-300 | sif10 | float32 | 76.74 | 78.11215 | +1.37215 | +0.00000 | 1.000000 |
| paranmt-300 | sif10 | float16 | 76.74 | 78.11211 | +1.37211 | -0.00004 | 0.999855 |
| paranmt-300 | sif10 | int8 | 76.74 | 78.11458 | +1.37458 | +0.00243 | 0.996773 |
| paranmt-300 | sif10-historical-test-fit | float32 | 76.74 | 76.73867 | -0.00133 | n/a | n/a |
| glove-wiki-gigaword-300 | raw | float32 | 44.46 | 44.46410 | +0.00410 | +0.00000 | 1.000000 |
| glove-wiki-gigaword-300 | raw | float16 | 44.46 | 44.46390 | +0.00390 | -0.00020 | 0.999819 |
| glove-wiki-gigaword-300 | raw | int8 | 44.46 | 44.45861 | -0.00139 | -0.00549 | 0.993836 |
| glove-wiki-gigaword-300 | sif10 | float32 | 71.43 | 69.22434 | -2.20566 | +0.00000 | 1.000000 |
| glove-wiki-gigaword-300 | sif10 | float16 | 71.43 | 69.22451 | -2.20549 | +0.00017 | 0.999891 |
| glove-wiki-gigaword-300 | sif10 | int8 | 71.43 | 69.22183 | -2.20817 | -0.00250 | 0.996701 |
| glove-wiki-gigaword-300 | sif10-historical-test-fit | float32 | 71.43 | 71.43245 | +0.00245 | n/a | n/a |
| word2vec-google-news-300 | raw | float32 | 61.54 | 61.54132 | +0.00132 | +0.00000 | 1.000000 |
| word2vec-google-news-300 | raw | float16 | 61.54 | 61.54132 | +0.00132 | +0.00000 | 1.000000 |
| word2vec-google-news-300 | raw | int8 | 61.54 | 61.54394 | +0.00394 | +0.00262 | 0.996483 |
| word2vec-google-news-300 | sif10 | float32 | 71.18 | 70.88642 | -0.29358 | +0.00000 | 1.000000 |
| word2vec-google-news-300 | sif10 | float16 | 71.18 | 70.88629 | -0.29371 | -0.00012 | 1.000000 |
| word2vec-google-news-300 | sif10 | int8 | 71.18 | 70.88874 | -0.29126 | +0.00232 | 0.997027 |
| word2vec-google-news-300 | sif10-historical-test-fit | float32 | 71.18 | 71.17538 | -0.00462 | n/a | n/a |
| fasttext-crawl-subwords-300 | raw | float32 | 48.53 | 48.52845 | -0.00155 | +0.00000 | 1.000000 |
| fasttext-crawl-subwords-300 | raw | float16 | 48.53 | 48.52752 | -0.00248 | -0.00093 | 0.999819 |
| fasttext-crawl-subwords-300 | raw | int8 | 48.53 | 48.53995 | +0.00995 | +0.01150 | 0.994670 |
| fasttext-crawl-subwords-300 | sif10 | float32 | 73.58 | 72.40582 | -1.17418 | +0.00000 | 1.000000 |
| fasttext-crawl-subwords-300 | sif10 | float16 | 73.58 | 72.40556 | -1.17444 | -0.00026 | 0.999927 |
| fasttext-crawl-subwords-300 | sif10 | int8 | 73.58 | 72.40633 | -1.17367 | +0.00051 | 0.995975 |
| fasttext-crawl-subwords-300 | sif10-historical-test-fit | float32 | 73.58 | 73.58036 | +0.00036 | n/a | n/a |

### Differences above 0.5 points

- ParaNMT: historical diagnostic 76.73867 differs from 76.74 by -0.00133 points. Separate-corpus components plus float32 folding add +1.37347 points.
- GloVe: historical diagnostic 71.43245 differs from 71.43 by +0.00245 points. Separate-corpus components plus float32 folding add -2.20811 points.
- FastText: historical diagnostic 73.58036 differs from 73.58 by +0.00036 points. Separate-corpus components plus float32 folding add -1.17454 points.

The dtype differences in the table complete each decomposition. No ablation isolates corpus choice from folding rounding.

## Validation and operations

Code e72f708 passed 19 focused tests and full `make check PYTEST_N=4`: 9,186 passed, 514 skipped.
The final independent review reports Keine Befunde; earlier findings and resolutions are archived.
Genuine FastText tensors and real runtime loads/probes cover all dtypes; metadata, tolerance, bitwise and SIF component mismatches are rejected.
No hot-path code changed, so throughput measurements and profiles do not apply. No Mac timing claims were made.
The archived cleanup journal records 2,244,403,200 allocated bytes removed. Results, final models and shared caches were retained.

Done: local models, quality verification, approved basis payload and tested release-card/readback tools.
Open: release 2.0.0, actual release cards and readback after coordinator uploads.
Needs from you: nothing before release; afterwards the complete upload commit receipts.

# Classic word-vector preparation report

DR-157 (Ausgewählte Modelle auf den Hugging Face Hub laden). All model folders are local. No Hugging Face upload or write was performed.

Draft PR: [#201 (2.0: Prepare the classic word-vector models for the Hub)](https://github.com/oborchers/Fast_Sentence_Embeddings/pull/201).

## Component-fit corpus comparison

| Corpus | STS-B Pearson x100 | STS-B Spearman x100 | NanoBEIR nDCG@10 x100 | Mean |
|---|---:|---:|---:|---:|
| wikitext | 68.67240 | 65.15750 | 40.59846 | 58.14279 |
| fineweb | 69.22434 | 65.36655 | 40.56338 | 58.38476 |

Selected corpus: **fineweb**.

All three scores have equal weight. NanoBEIR is the macro mean of all 13 retrieval tasks. Components use training paragraphs only. Evaluation scores selected the corpus, so they are not untouched holdout estimates.

```json
{
  "corpus": "HuggingFaceFW/fineweb-edu-llama3-annotations",
  "revision": "72df4c92fb1b48beceb16016e8f695ec40a6c3a5",
  "seed": 0,
  "paragraphs": 859346,
  "corpus_xxh3": "xxh3-128:88146a7f61a2c157c4710cfaf3e0cb4b",
  "wordfreq_version": "3.1.1",
  "wordfreq_language": "en",
  "wordfreq_wordlist": "best",
  "a": 0.001,
  "components": 10,
  "aggregation": "fse 2.0 weighted_mean@1, transform_version 2",
  "solver": "scipy.linalg.eigh(driver=evd), float32 block Gram, float64 accumulation",
  "state_xxh3": "xxh3-128:f39211ff056a80af376a9afb7e50064a",
  "candidate_fingerprint": "xxh3-128:264a787bf6f8b6d434b62297e53377a2",
  "tokenizer": {
    "nltk_version": "3.10.4",
    "punkt_tab_english": {
      "abbrev_types.txt": "xxh3-128:0acdbf18bd21d71f604380094bf25402",
      "collocations.tab": "xxh3-128:234e0b2209c617091029602a9025d12c",
      "ortho_context.tab": "xxh3-128:6ef2c67cab33775cdfe9a537e1c6c4ed",
      "sent_starters.txt": "xxh3-128:faa8bab2de114ffc95373f90d0fa7da0"
    },
    "preprocessing_recipe": "fse_1x_nltk@1",
    "preprocessing_code_xxh3": "xxh3-128:e7d052234037c417c1598ab6c7e26beb"
  },
  "source_repo": "fse/glove-wiki-gigaword-300",
  "source_revision": "4e5ad402664236b85be45a0d107b95259ba8a909"
}
```

## Comparison with fse 1.x

Pearson x100 on all 1,379 STS-B test pairs. Controlled preprocessing follows the 1.x notebook. Historical SIF fits components on the test texts solely as a diagnostic and is never shipped. Recall@10 uses the 2,758 sentence rows, excludes self, and breaks ties by row id.

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

- paranmt-300, sif10, float32: +1.37215 points. The historical test-fit diagnostic is 76.73867, a -0.00133-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by +1.37347 points. This dtype changes the prepared float32 score by +0.00000 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- paranmt-300, sif10, float16: +1.37211 points. The historical test-fit diagnostic is 76.73867, a -0.00133-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by +1.37347 points. This dtype changes the prepared float32 score by -0.00004 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- paranmt-300, sif10, int8: +1.37458 points. The historical test-fit diagnostic is 76.73867, a -0.00133-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by +1.37347 points. This dtype changes the prepared float32 score by +0.00243 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- glove-wiki-gigaword-300, sif10, float32: -2.20566 points. The historical test-fit diagnostic is 71.43245, a +0.00245-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by -2.20811 points. This dtype changes the prepared float32 score by +0.00000 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- glove-wiki-gigaword-300, sif10, float16: -2.20549 points. The historical test-fit diagnostic is 71.43245, a +0.00245-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by -2.20811 points. This dtype changes the prepared float32 score by +0.00017 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- glove-wiki-gigaword-300, sif10, int8: -2.20817 points. The historical test-fit diagnostic is 71.43245, a +0.00245-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by -2.20811 points. This dtype changes the prepared float32 score by -0.00250 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- fasttext-crawl-subwords-300, sif10, float32: -1.17418 points. The historical test-fit diagnostic is 73.58036, a +0.00036-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by -1.17454 points. This dtype changes the prepared float32 score by +0.00000 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- fasttext-crawl-subwords-300, sif10, float16: -1.17444 points. The historical test-fit diagnostic is 73.58036, a +0.00036-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by -1.17454 points. This dtype changes the prepared float32 score by -0.00026 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.
- fasttext-crawl-subwords-300, sif10, int8: -1.17367 points. The historical test-fit diagnostic is 73.58036, a +0.00036-point gap from the published value. Moving to separate-training-corpus components plus a folded float32 table changes that diagnostic by -1.17454 points. This dtype changes the prepared float32 score by +0.00051 points. These observed terms sum to the reported difference. No separate ablation isolates corpus choice from folding rounding.

## Built folders

Checksums were recomputed over every file before this report was written. The JSON manifest beside each folder lists all file sizes and checksums.

| Intended repository | Model bytes | Model xxh3-128 | Encoder fingerprint |
|---|---:|---|---|
| fse/paragram-300-sl999-raw | 2072155759 | `xxh3-128:328bf0d8306a00825de8b04adcdcd2da` | `xxh3-128:70ea0079ef30b2c0e3b9918ae66b4296` |
| fse/paragram-300-sl999-raw-float16 | 1049902223 | `xxh3-128:74bf7be8978fa2b576d9edff1af07ce9` | `xxh3-128:d437ec72d880179f652b77366a523b97` |
| fse/paragram-300-sl999-raw-int8 | 545590639 | `xxh3-128:40113e4ab599bbfc7ccf413b7cc79547` | `xxh3-128:811484bfb3fb71ed12bcaeb4744d991b` |
| fse/paragram-300-sl999-sif10 | 2072155759 | `xxh3-128:038887721ed2795d66137f21be75ad2d` | `xxh3-128:c6c96fd66f29e85fa6fefdf30661badd` |
| fse/paragram-300-sl999-sif10-float16 | 1049902223 | `xxh3-128:a1dfc0e61c8ed97fcb89f3dc961eb069` | `xxh3-128:2b2ab794cc974b128746edfe070c320d` |
| fse/paragram-300-sl999-sif10-int8 | 545590639 | `xxh3-128:847515c7c6828600cdbaa35b6812d2b2` | `xxh3-128:37fbc7e840861300414e76ec0964f409` |
| fse/paranmt-300-raw | 93858331 | `xxh3-128:0319fa9e4e0e33533a70c3d18192ee32` | `xxh3-128:f870e4619c3e5dd082be1809d1fb7786` |
| fse/paranmt-300-raw-float16 | 47523995 | `xxh3-128:b305daf79082684560379d588a698891` | `xxh3-128:26b7176aacbdf42b64c8e90791847319` |
| fse/paranmt-300-raw-int8 | 24665819 | `xxh3-128:79980e25c364299b22db142a08ca4f1c` | `xxh3-128:e49c0b0f0d05fcebcb96a99c443d375d` |
| fse/paranmt-300-sif10 | 93858331 | `xxh3-128:3ad3b972d0f07aba6ee8ab69c54def61` | `xxh3-128:4d2fb5d81b9888d2eb7bb8e69ee9c7f6` |
| fse/paranmt-300-sif10-float16 | 47523995 | `xxh3-128:70d942077d66829154b1bdffe8979d2f` | `xxh3-128:3145e0e4282adfa085f0ed3a6581076e` |
| fse/paranmt-300-sif10-int8 | 24665819 | `xxh3-128:da4dd5ae1f7643431ca7eaddea89d6a8` | `xxh3-128:da86203db14c76128eb2f9a8d4be97e6` |
| fse/glove-wiki-gigaword-300-raw | 486157506 | `xxh3-128:d6eec35248772daeb28ef9c9cc8fa348` | `xxh3-128:4e3dee37a6fccb1d36ab6f9b5bc272c2` |
| fse/glove-wiki-gigaword-300-raw-float16 | 246157570 | `xxh3-128:03593e6417690b5e21aee7b21db2c51d` | `xxh3-128:808b1ba980661cbe984a5d34f0a79d2b` |
| fse/glove-wiki-gigaword-300-raw-int8 | 127757762 | `xxh3-128:e471cecf9a495ed2dbc7f98bf73a8371` | `xxh3-128:30618248eaf896d418ac1c3efcc784e8` |
| fse/glove-wiki-gigaword-300-sif10 | 486157506 | `xxh3-128:145490ab6dba7a338ba3a7d487cb9077` | `xxh3-128:74dd7999a3a145152105dcbb1b6dba53` |
| fse/glove-wiki-gigaword-300-sif10-float16 | 246157570 | `xxh3-128:943a86cf7143e1c24234849deacfc7df` | `xxh3-128:f12d7470c438888448cc3bfab9582692` |
| fse/glove-wiki-gigaword-300-sif10-int8 | 127757762 | `xxh3-128:b18e382f5892bbd78d0c4f36391d6cbd` | `xxh3-128:071d69611357292c3a937a8ee8ce3d1f` |
| fse/word2vec-google-news-300-raw | 3665259542 | `xxh3-128:ae442c9ca4507fe07a04a13692163e89` | `xxh3-128:64e4ef70aca1f497f7173a33764825dd` |
| fse/word2vec-google-news-300-raw-float16 | 1865259670 | `xxh3-128:1b122a7153310d7e4df7ebe75f5abc7f` | `xxh3-128:4fbe512db126cfe88b7daa8ef5440a7c` |
| fse/word2vec-google-news-300-raw-int8 | 977259798 | `xxh3-128:cf74b56c5ca1a04e390732c3e01c7efa` | `xxh3-128:1535b380564e1ce4a7dbd553eaddec01` |
| fse/word2vec-google-news-300-sif10 | 3665259542 | `xxh3-128:c07efc677ece6489996cc6ed2367698f` | `xxh3-128:d56902a9d7d77240e264aa99f4d2cf00` |
| fse/word2vec-google-news-300-sif10-float16 | 1865259670 | `xxh3-128:9fd5a1048889c4490390064fab248771` | `xxh3-128:499b36600b36b4a6672d3ccab989b585` |
| fse/word2vec-google-news-300-sif10-int8 | 977259798 | `xxh3-128:16e02c9c2cf1abc36521e481bcc874d6` | `xxh3-128:06ce76785b6c401587536055650438bd` |
| fse/fasttext-crawl-subwords-300-raw | 4831846546 | `xxh3-128:48a8f3c4682e15883e40cf7bef5eec60` | `xxh3-128:4d9f5f4df07caebb5960acdaf7fddaa6` |
| fse/fasttext-crawl-subwords-300-raw-float16 | 2431846610 | `xxh3-128:bd5a830d5eb6dcae2ab74043654754fa` | `xxh3-128:eb03270f4c886d64cfe8629540cde77e` |
| fse/fasttext-crawl-subwords-300-raw-int8 | 1247846930 | `xxh3-128:b75e7fcf134dd82cb551abb3c0faeb70` | `xxh3-128:f67baa976b632633617f4958856214b8` |
| fse/fasttext-crawl-subwords-300-sif10 | 4831846546 | `xxh3-128:47cb6f4b291729113966c41d23c67187` | `xxh3-128:a5ef569a2957cb24870f87d52432f56f` |
| fse/fasttext-crawl-subwords-300-sif10-float16 | 2431846610 | `xxh3-128:06390000e0ea7c0de743f07ed2b2c746` | `xxh3-128:46fe26f184731f35ffba3f9ef7373e7d` |
| fse/fasttext-crawl-subwords-300-sif10-int8 | 1247846930 | `xxh3-128:25f2cf2e3af3ca464d27d7396b525c9d` | `xxh3-128:075213b2d66190094fb9dc35b84e2f35` |

## Provenance and cards

Finished folders: `/Users/oliverborchers/fse-dr157/models`. Card previews: `/Users/oliverborchers/fse-dr157/cards`.

The existing fse 2.0 card template supports generic vector-source provenance, license, language, base_model and int8-only quantized relation. Offline card enrichment adds original source tags, the SIF fit dataset, licensing context, fit hashes and measured dtype comparisons. No core runtime dependency or persistence-schema change is needed for wordfreq. fse.source pins each original vector repository and revision. Each SIF card and preparation.json record corpus, revision, seed, wordfreq version and fitted-state checksum.

Licenses: GloVe pddl; Google News word2vec other/research-only; fastText cc-by-sa-3.0 with Facebook attribution and ShareAlike; ParaNMT and Paragram unknown with preservation and removal-on-request notices. See each card's linked primary sources. The old fse source repositories remain unchanged.

No timings, load-time comparisons or speed claims were made on this Mac.

## Cleanup

| Removed owned intermediate | Allocated bytes removed |
|---|---:|
| `/private/tmp/dr157-review-702f3c0` | 11354112 |
| `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub/src/fse/_native.c` | 6090752 |
| `/Users/oliverborchers/fse-dr157/glove-wiki-gigaword-300-sif10` | 486166528 |
| `/Users/oliverborchers/fse-dr157/glove-wiki-gigaword-300-sif10-float16` | 246169600 |
| `/Users/oliverborchers/fse-dr157/glove-wiki-gigaword-300-sif10-int8` | 127766528 |
| `/private/tmp/dr157-review-f320afa` | 11464704 |
| `/Users/oliverborchers/fse-dr157/comparison/fineweb-seed0-859346.txt` | 174477312 |
| `/Users/oliverborchers/fse-dr157/review-dacdb78` | 11788288 |
| `/Users/oliverborchers/fse-dr157/comparison/glove-wikitext-wordfreq.safetensors` | 486158336 |
| `/Users/oliverborchers/fse-dr157/comparison/glove-fineweb-wordfreq.safetensors` | 486158336 |
| `/Users/oliverborchers/fse-dr157/review-73176a9` | 12447744 |
| `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub/src/fse/_native.c` | 0 |
| `/Users/oliverborchers/fse-dr157/review-2f4c1a8` | 12480512 |
| `/Users/oliverborchers/fse-dr157/review-da79a11` | 12492800 |
| `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub/.mypy_cache` | 60825600 |
| `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub/.ruff_cache` | 86016 |
| `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub/.pytest_cache` | 24576 |

Total allocated size removed: 2145951744 bytes. The journal also records observed filesystem free-space readings. Concurrent workers can affect those readings. Results, evidence, reports and shared caches were retained.

Additional removal: the regenerated Cython intermediate had 6,086,512 logical bytes. Its first journal write failed with PermissionError after removal. The confirmed removal was recorded on retry, with zero allocated-byte credit because that reading was not secured. The compiled extension remains available.

## Delivery checks and open questions

Source 73176a9 passed make check: 9,164 tests passed, 514 skipped. Final source da79a11 passed make check: 9,167 tests passed, 514 skipped in 223.76 seconds. Format, lint, guardrails, type checking and the offline suite passed. The final independent review reports Keine Befunde. All earlier review findings are resolved in committed source and evidence. The interrupted 2f4c1a8 gate is retained without a passing claim.

Oliver approved FastText for 2.0; DR-146 (Über FastText in 2.0 entscheiden (12f)) is Done. The six FastText publication_hold fields are now null. Every repository must be PUBLIC from creation. Oliver must approve exact cards, repo names, licenses and every file before any Hub write. The initial GloVe prototypes were retired after preserving their small evidence.
## PUBLIC approval payload

All 30 repositories contain 135 files, totaling **39,426,324,698 bytes**
(39.426 GB, 36.719 GiB). FastText accounts for 17,023,110,368 bytes and has no separate publication hold. Sizes include all payload files.

Exact inventory, including the table by source, method and dtype:
`/Users/oliverborchers/fse-dr157/publication-inventory.md`.

Immutable manifest: `/Users/oliverborchers/fse-dr157/publication-manifest.json`.
SHA256: `ac85cae41b05788bc278d68fd3231489de1615c7c15b8b79229844c3d0dabe69`.
The same bytes are committed under `benchmarks/baseline/dr157/`.
All 30 literal cards are committed under `cards/*.txt`; local Markdown previews
remain in `/Users/oliverborchers/fse-dr157/cards`.

Oliver approved the content, repo list and licenses of all 30 repos (Option B). Upload must wait until the 2.0.0 release and regenerated release cards. This development-card manifest remains unapproved as a final upload payload. No Hub write has occurred.

## Required post-upload verification

After Oliver approves the payload and the coordinator finishes the public uploads,
run `benchmarks/verify_published_hub.py` using every approved repository and its
exact uploaded commit. `benchmarks/baseline/dr157/POST_UPLOAD.md` gives the receipt
format and invocation. No second benchmark method or pod work is requested.

Each repository uses a new child process and its own empty HF_HOME. All payload
files must match local sizes and SHA256. The literal card example must load through
the public 2.0 API, then metadata and a fixed 512-sentence STS-B probe must match.
Pearson tolerance is 0.001 points, with an additional bitwise CPU vector check.
Failure measurements remain in durable evidence. Cache cleanup follows evidence
publication, verifies ownership before deletion and records removed bytes.

The actual Hub readback has **not run**. Approval and completed upload receipts are
required. The worker will commit the per-repo JSON table and report it to fse-main.
Any failure is reported without Hub overwrite, deletion or repair writes.

## Final validation and handoff

Final code: `da79a11ee70ce0c011a0554abd489d00b516c4ad`.
Final full gate: `make check PYTEST_N=4`, 9,167 passed, 514 skipped.
The log is committed as `make-check-da79a11.txt`.
Independent final review: `reviews/review-da79a11.md`, Keine Befunde.
`reviews/resolution.md` accounts for every accepted earlier finding.

No hot-path code changed, so runtime throughput measurements and profiles do not
apply. No Mac timing claim was made. Remote CI status remains visible on the draft
PR; local results do not claim remote CI success.

The 30 models, exact cards, all sizes/hashes, licensing disclosures and fixed probe
references are ready for Oliver's review. This is not publication authorization.
The coordinator receives the report and manifest directly through Herdr.

Done: local preparation, verification, review, PUBLIC approval inventory and read-only post-upload runner.
Open: actual fresh Hub readback after approval and coordinator uploads, and regenerated release cards for 2.0.0.
Needs from you: release 2.0.0, followed by the coordinator's complete uploaded-repository commit receipts.

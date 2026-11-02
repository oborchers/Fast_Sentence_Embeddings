# Measurement details for the README

The README keeps the introduction short. This guide retains the full model and precision matrix,
quality samples, historical observations, ranges and source records.

## Evidence archive

The [file manifest](readme-evidence-manifest.json) lists every archived README snapshot with its byte size, SHA256 and immutable file URL.
The originals live on the separate `benchmark-evidence-2.0` branch and remain byte-identical.
The [comparison study manifest](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/41d9fc8c54f859cd6c450a94c34c95dcc0d7c84e/benchmarks/baseline/dr151_evidence_manifest_20261101.json) covers its upstream journals, profiles and audits.
This branch is never merged into master. Local study files retain only compact summaries, each below 50 KB.
The [validation summary](readme-validation-20261101.json) identifies the exact source commit covered by its package and CI checks.

## Current speed cohort

Use the [compact speed summary](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6cb13ed72321d1207fc106f6c0e7154935164017/benchmarks/baseline/speed-2026-11.json).
Its [evidence reference](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6cb13ed72321d1207fc106f6c0e7154935164017/benchmarks/baseline/evidence-2026-11.json)
pins the complete archive at `f97e32edbc0f9daa538b56b54f3187d8d3c177d8` on `benchmark-evidence-2.0`.
No new benchmark was run for this README update.

The cohort uses AMD EPYC 9965, 32 assigned logical CPUs, Python 3.13.18, Linux, 1 to 2 November 2026.
Executed fse code is `624251df2269e3099f5c61987be63e718893b58c`, runtime SHA256
`5a4883c1675cef10e5a08bf1c084f0dc57f872ed1cca9074a08b673b15b8fac0`.
The driver source is `0c639113f80383fda461f3e776db38eee0efee71`.

### Dtype table

Cells `dtypes-02` through `dtypes-07` are all publication-eligible.
They share 4,620,000 WikiText-103 passages, input SHA256
`bec2f6d5e127889fce0b29e6123915f083431ac0cd4b7fe3a4f24e43758a7c92` and the same executed fse runtime.
Input digest is `e68e8867eb78a48272a9bcf4765bf97a301f7ce6ecb66126e9c5ce080960a9a6`.
The input samples source positions with replacement, then shuffles them. 814,724 distinct texts appear.
The exact mean length is 119.0559961038961 whitespace words. All outputs are normalized float32 vectors.

| Model precision | 1 worker, texts/s | 16 workers, texts/s |
|---|---:|---:|
| float32 | 139,422 | 1,265,123 |
| float16 | 153,886 | 1,359,540 |
| int8 | 163,779 | 1,379,949 |

Values are primary fse-a medians, rounded to whole texts/s for display.
Three fse-a and three fse-b rounds are retained per cell. Every first and warm call lasts at least 2 seconds.
The README extraction independently recomputed primary medians and global A/A from the ordered warm calls.
A/A upper bounds, in table order, are 1.068446063164378/1.018007185052106,
1.0404600732108225/1.064054384858206 and 1.0231300777602998/1.0306276381377704.
These all pass the 1.10 limit. This run supplies measured 1/16-worker settings, no 32-worker comparison.
Do not infer a 32/16 ratio or reuse the earlier run's hardware-selection result as a measurement of this cohort.

### Headline and library panel remain open

The maintainer requires all README speed numbers to come from one reproducible run.
The older EPYC 7713 showcase is withdrawn from the README. Its archive remains historical evidence only.
The current run's 100M-repeat cell `headline-01` is invalid (global A/A upper 1.1625013207283315).
It supplies no replacement headline. The README keeps a visible draft placeholder until the benchmark decision.
The older local showcase summary is marked withdrawn for current publication without altering its recorded numbers.

Rust `libraries-10` and FastEmbed `libraries-11` are valid one-worker pairs, each using 500,000 identical passages within that pair.
Their paired median ratios are 1.7262630895228022 and 28.489499659793236, with global A/A upper bounds
1.0991783964909603 and 1.098455653340089. Each ratio retains its own fse numerator and reference runtime identity.
These are not paired with the 4.62M dtype input. The library panel remains held for the maintainer's decision.
Model2Vec's one-worker cell is invalid. The ST adapter drops unknown ids and fails output equivalence, so no ST rate is published.
The 16-worker library cells and classification were rejected by the memory preflight before eligible results.
No missing comparison, classification rate or host cohort is inferred.

## Withdrawn short-call speed evidence

**Invalid for speed publication:** the complete Mac matrix (5,352 texts per call), every Pod 2 speed cell
(2,000 to 5,352 texts), and all 30 Pod 3 panel cells (5,352 texts).
The later maintainer decision also withdraws **all Apple M1 Pro speed measurements, regardless of input size**.
This decision overrides every earlier eligible label, runtime attribution and A/A acceptance.
No speedup or rate from those studies may appear in README, PERFORMANCE, model cards or changelog claims.
Their immutable archive records remain audit evidence, not publishable results.
Quality, accuracy, neighborhood stability and saved-file observations remain separate and are not withdrawn.

The current dtype table uses only the current EPYC 9965 cohort above. Earlier timing rows do not fill missing cells.
Quality, accuracy, neighborhood stability and file sizes remain under their own recorded protocols.

## Classifier comparisons: linear head versus fine-tuned MLP

A head turns a text vector into scores or labels. The classification recipe compares these different training methods:

| Method | Word vectors | Trained head | Training | Prediction path |
|---|---|---|---|---|
| fse with an imported linear head | Fixed for head training | Linear | scikit-learn in the recipe | fse CPU backend |
| Model2Vec 0.10.0 classifier | Fine-tuned during training | MLP, by default one hidden layer of 512 units with ReLU | PyTorch | Recipe: PyTorch model. Exported head: plain NumPy. |

An MLP is a small neural network. ReLU clips negative hidden-layer values to zero.
The exported Model2Vec head runs in plain NumPy, with softmax or sigmoid when returning probabilities.
The [linked recipe at its current source](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/a190e2de19e0944b112a44e386b26efc74ecc4ca/recipes/02-linear-heads.ipynb)
calls `predict` on the trained PyTorch model instead. Its accuracy rows are not NumPy-export measurements.
Its [classification defaults](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/train/classifier.py), [trainable embeddings and head construction](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/train/base.py),
and [NumPy inference implementation](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/inference/mlp.py) are pinned to the official 0.10.0 source.

The recipe uses a fixed encoder and linear head for fse, while Model2Vec changes both the word vectors and head.
Accuracy differences here do not isolate the inference engine.
Speed comparisons must also name these heads and the encoder settings rather than treating them as the same computation.

The withdrawn encoding-only records do not run the classifier MLP. No classifier rate may be inferred from them.
Pod 2 accuracy remains recorded below. The corrected folded speed comparison remains pending.

## Folding and the pending classification speed panel

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/1f76884072537cb70a757f02d3e2e37cfa0b4ba4/media/readme/classification-folding-dark.png">
<img alt="A trained linear head prepares a word-score table once. For new texts, fse looks up word scores, averages them, adds the bias and returns labels. This applies to raw, unnormalized averages when sentence vectors are not requested. The corrected folded speed comparison is pending." src="https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/1f76884072537cb70a757f02d3e2e37cfa0b4ba4/media/readme/classification-folding-light.png">
</picture>


This explanatory figure shows preparation of per-word scores for a linear head trained on raw, unnormalized averages.
fse combines the head weights with word vectors once, then averages word scores and adds the bias for each text.
This path supports scores, labels and probabilities without requesting sentence vectors. Normalized heads and requests for full vectors use the full-vector path.
With fewer output scores than vector values, the folded table has shorter rows.

A complete fse classifier already prepares this folded path where legal. Saving standalone preserves the folded table.
It is not a separate faster prediction algorithm and is not plotted as a speedup.
Sources: `src/fse/_runtime/_head.py` (`make_head`, `fold`, `HeadPipeline`) and `src/fse/_head/_model.py` (`save`, `_standalone_artifact`).

The main README now reserves a measured classification table and chart instead of using this explanation as its speed panel.
Pod 2 normalized-head classifier rates are invalid for publication, regardless of their original protocol. The final folded panel still needs committed DR-151 classifier records,
with the exact trained head, normalization, dataset, machine, date and worker count from each row.
The normalized-linear comparison and raw-mean folded comparison stay separate.
Speedup is the matched reference time divided by fse time for the same input, equivalently fse throughput divided by reference throughput.
A separately fine-tuned Model2Vec MLP is a different trained model, so it does not fill a same-head engine comparison.
Its default 512-unit ReLU head and PyTorch/NumPy paths are documented above.
Only complete series with executed-backend attribution and a symmetric A/A upper band at most1.10 qualify.
Encoder-only throughput and retained recipe accuracy rows do not fill this measurement.

### Pod 2 accuracy only, speed withdrawn

Every Pod 2 speed cell is invalid for publication because of its small per-call input.
The prior rate/range table and paired speedup have been removed. Their A/A acceptance does not restore validity.

The Subjectivity accuracy example uses POTION-8M float32 and a 2,000-text test split.
Both models trained on the same fixed selection of 1,000 texts.
fse uses normalized vectors and an imported linear head, so this is not folded inference.
Model2Vec uses frozen vectors plus an independently fitted sklearn linear head, not its fine-tuned MLP.
The coefficient arrays differ. Accuracy does not isolate the inference engine.

| Trained pipeline | Accuracy | Macro-F1 |
|---|---:|---:|
| fse, normalized vectors plus linear head | 0.8880 | 0.887696 |
| Model2Vec, frozen vectors plus independently fitted sklearn linear head | 0.8880 | 0.887696 |

[Archived journal](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/04e5cbeb1af5d36a45384906d91457d423829680/benchmarks/baseline/results/dr151-pod2-20261101/speed-results.jsonl.gz),
series `be8994521d4c496ba64b30b19dd39b2a`, cell `classification/setfit/subj/minishlab/potion-base-8M/1/model2vec`.
Its speed fields are withdrawn. Only the accuracy fields above are retained as results.

### Keep compute close to your text

For a local application, start by running fse in the process that already has the text.
This avoids moving text and vectors through a service. Measure that transfer cost before adding a service.

The local-compute notebook explains transfer overhead. Its earlier timing values are not presented here as accepted speed evidence.
One actual normalized vector used 1,200 bytes as 300 float32 values and 6,320 bytes as compact JSON text.
JSON size depends on the values and formatting. Batching, concurrency and real networks change these costs.

## Which model, which dtype?

Start with a model designed for your language and task. The links below lead to its publisher's model card.
These are starting points for evaluation, not a ranking measured by fse.

| Model | Language and use to evaluate |
|---|---|
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) | English general tasks, including classification, with a small table. |
| [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M) | English general tasks when a larger vocabulary is useful. |
| [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | English search, matching queries to documents. |
| [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) | General tasks across languages. Check the card's language coverage. |
| [static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | English search, with a choice of output dimension. |
| [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | Similarity across languages, with a choice of output dimension. |
| GloVe-300 and word2vec-300 | Classic word-vector baselines for comparison and offline recipes. |

A **dtype** is how each number is stored. fse computes with float32 even when the table uses fewer bits.
The ratios below count table values only. Tokenizers, metadata and int8 row scales add space.
They are storage calculations, not measurements of file size or process memory.

| Table dtype | Bytes per value | Value storage relative to float32 | How to choose |
|---|---:|---:|---|
| float32 | 4 | 100% | Start here to establish quality. |
| float16 | 2 | 50% | Try when table memory matters. Check small values and task quality. |
| int8 | 1, plus a float32 scale per row | 25%, plus scales | Try for a tighter memory budget. Check task quality against float32. |

```python
compact = encoder.quantize("int8")  # a new encoder with a smaller table
compact.save("potion-int8.safetensors")
rows = compact.encode(["the cat sleeps"])
```

`encoder.quantize` changes the model table. `encoder.encode(..., dtype="int8")` changes the output storage instead.
A smaller table does not always run faster. Earlier Mac pipeline speed ratios have been withdrawn.
See [the format rules](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/formats.md#encoder-1) for rounding and rejected values.

### Hub SIF-10 tables and classic benchmark tables

These are different finished tables. Keep their results separate.

| Table family | Word weights | Where the ten components are fitted | Score source |
|---|---|---|---|
| Planned Hub `-sif10` models | English frequencies from wordfreq | A FineWeb sample | The Hub study's own committed results |
| Classic benchmark tables | WikiText-103 frequencies | WikiText-103 training text | The corresponding classic benchmark study |

The maintainer selected the FineWeb preparation for the Hub models.
Hub scores must come from the [Hub preparation study](https://github.com/oborchers/Fast_Sentence_Embeddings/tree/f320afa6036352c3616671a9302d76ac4917bf7c/benchmarks/baseline/dr157), with the model's own fit receipt and artifact identity.
The GloVe/paranmt benchmark rows and the Pod 5 word2vec/FastText/paragram rows use the separate WikiText protocol.
Their scores do not describe the Hub models. The legacy 1.x diagnostic is also a separate protocol.
No Hub score or corpus-selection mean is inferred from a classic benchmark row.

### Complete saved-file sizes

These [file-size observations](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/3b511c89e905368557b5eb864896b5a5d2cba47c/docs/benchmarks/readme-model-sizes-20261101.json)
were recorded on 1 November 2026 on an Apple M1 Pro.
They cover complete saved fse files, including vocabulary, tokenizer data and int8 row scales.
One MiB is 1,048,576 bytes. Each model keeps its original number of values per vector.

| Model | Values per vector | float32, MiB | float16, MiB | int8, MiB |
|---|---:|---:|---:|---:|
| potion-base-8M | 256 | 29.2 | 14.8 | 7.7 |
| potion-base-32M | 512 | 124.1 | 62.5 | 31.9 |
| potion-retrieval-32M | 512 | 124.1 | 62.5 | 31.9 |
| potion-multilingual-128M | 256 | 502.1 | 257.8 | 137.5 |
| static-retrieval-mrl-en-v1 | 1024 | 119.7 | 60.0 | 30.3 |
| static-similarity-mrl-multilingual-v1 | 1024 | 415.1 | 208.3 | 105.3 |
| GloVe-300 | 300 | 463.6 | 234.8 | 121.8 |
| word2vec-300 | 300 | 3,495.5 | 1,778.9 | 932.0 |

The [classic-model observations](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6ddf24d91b446a170b1218d3df4884afe690d09d/docs/benchmarks/readme-classic-model-sizes-20261101.json)
use full GloVe Wiki/Gigaword and Google News word2vec tables, with 400,000 and 3,000,000 rows respectively.
Their recipe is an unweighted mean with `fse_words@1` tokens and normalized outputs.

Use these sizes to plan disk storage. Measure peak memory, speed and quality for your workload.
The source records exact byte counts, model revisions and the environment that prepared the files.

### Task-quality samples

MIND final status: **not computed: MTEB 2.26.25 MindSmallReranking needs more memory than the 64 GB pod (two attempts, status 137 and earlier ENOSPC)**. No further MIND runs are planned.
Every English MTEB(eng, v2) mean below is explicitly based on **40 of 41 tasks**.
The original raw snapshots retain their historical deferred/null records; no missing task score is invented.

**POTION-8M, float32**

One [quality sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/04e5cbeb1af5d36a45384906d91457d423829680/docs/benchmarks/readme-quality-20261101/README.md)
measured POTION-8M with a float32 table and two fse workers on 1 November 2026.
It used an Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4 and MTEB 2.26.25.
MTEB is a collection of tests for text vectors. NanoBEIR is a set of 13 small search tests.

| Task group | Coverage | Mean task score, points |
|---|---|---:|
| English MTEB(eng, v2), completed tasks only | 40 of 41 tasks, MIND not computed | 53.9527 |
| NanoBEIR search tasks | All 13 of 13 | 43.8485 |

Each completed task has equal weight. Points are 100 times the native task score.
The English row averages 40 of 41 tasks. MIND is not computed.
It is **not the complete 41-task benchmark score**.
English tasks use different metrics. NanoBEIR uses `ndcg_at_10`, a score for the first ten search results.
The source records every task, metric, dataset revision and the exact saved model identity.
Further model and dtype comparisons, including quality changes from float32, remain pending.

**Later POTION-base-32M sample, float32**

A [later sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/04e5cbeb1af5d36a45384906d91457d423829680/docs/benchmarks/readme-quality-potion32-20261101/README.md)
used POTION-base-32M, two fse workers and the same Apple M1 Pro on 1 November 2026.
The recorded evaluation code differs from the POTION-8M run above.
Confirmations of POTION-8M and POTION-base-32M with matching evaluation code are queued after the recording fix.
A controlled model ranking remains pending.

| Task group | Coverage | Mean task score, points |
|---|---|---:|
| English MTEB(eng, v2), completed tasks only | 40 of 41 tasks, MIND not computed | 55.2573 |
| NanoBEIR search tasks | All 13 of 13 | 46.1754 |

This English mean covers 40 of 41 tasks and excludes uncomputed MIND. It is **not the complete 41-task benchmark score**.
The source records each task's metric and dataset revision, the saved model, and the code identifiers.

**POTION-retrieval-32M, float32, with fixed runtime recording**

A [separate quality sample](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/04e5cbeb1af5d36a45384906d91457d423829680/docs/benchmarks/readme-quality-retrieval32-20261101/README.md)
measured POTION-retrieval-32M with two fse workers on the Apple M1 Pro on 1 November 2026.
It records the loaded code's identity after the recording fix.
Its evaluation code differs from the earlier P8/P32 checkpoints, which still await matching-code confirmations.
A controlled model ranking remains pending.

| Task group | Coverage | Mean task score, points |
|---|---|---:|
| English MTEB(eng, v2), completed tasks only | 40 of 41 tasks, MIND not computed | 54.5098 |
| NanoBEIR search tasks | All 13 of 13 | 50.3872 |

The English mean covers 40 of 41 tasks, excludes uncomputed MIND and is **not the complete 41-task benchmark score**.
These are task-quality scores. Setup, suspension and shared-job waits do not provide a model compute-time measurement.

### First neighborhood stability check

An [audit of POTION-8M](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/04e5cbeb1af5d36a45384906d91457d423829680/benchmarks/baseline/results/dr151-numerics.jsonl)
used all 5,352 nonblank WikiText validation and test texts on 1 November 2026, on an Apple M1 Pro.
It found the ten most similar other vectors for every text, using cosine similarity, which compares vector directions.
It then measured how many of those neighbors each candidate retained, compared with fse's float32 table.

| Candidate | Mean share of the original top ten retained |
|---|---:|
| model2vec 0.10.0, float32 | 100.000% |
| fse, float16 table | 99.994% |
| fse, int8 table | 99.634% |

Every text had a nonzero vector. The audit excludes each text from its own neighbor list.
For equal scores, the earlier text in the input comes first. All outputs are float32.
These results measure neighborhood stability on this input. Further model and dtype task comparisons remain pending.

TODO(DR-151): Add the remaining peak-memory, speed and task-score comparisons.
Include quality change from each model's float32 result. Name the dataset, metric, revision and date.

## Invalid Mac timing studies

All Mac speed studies are withdrawn from publication, including the former laptop headline, dtype rates,
classic-model throughput and Model2Vec reference controls. Their tables and chart have been removed from this guide.
The [archive manifest](readme-evidence-manifest.json) retains exact original records for audit.
The five local timing summaries carry `speed_publication.valid: false`, overriding their historical acceptance fields.
No archived eligible label permits reuse of these numbers as a claim.
Quality and file-size sections above remain valid under their stated protocols.

## Work kept out of the main README

The main README shows final values only and links here once with “How we measured”.
It omits unfinished comparison rows and moves measurement-process wording here, including controls, A/A, ranges and noise labels.
Withdrawn speed figures are no longer embedded. Original journals remain archived and invalid for speed publication.

TODO(DR-151): Fill the classification comparison table and chart from committed matched-pipeline records.
Keep normalized-linear and raw-mean folded cases separate. Use the same model, weights, texts and settings per row.
Show fse texts/s, matched reference texts/s and a paired speedup. Fine-tuned MLP results remain separate trained-model comparisons.
Do not display an empty classification table in the main README while those values are unavailable.

TODO(DR-151): Add MTEB quality comparisons for classic raw and SIF-10 models when measured.

Pod 4 omits the worker-scaling curve and the 10M write-path stage. The 100M showcase covers writing.
The remaining encoding grid uses 1, 16 and 32 workers for each dtype and library pair.
Keep the corrected folded-classification comparison and large inputs. Every first and warm fse call must last at least two seconds.

The folding explanation in the main README uses the executed recipe 08 result at source 32ff539e19044b69c48d9891b2cb16ddd17e10cc.
The explicit and folded paths agree on all 2,000 AG News labels; maximum probability difference 3.069639205932617e-06.
The complete file (30,668,970 bytes) rounds to 29.2 MiB; standalone (900,522 bytes) rounds to 0.86 MiB.
These are final storage/accuracy observations, not a throughput speedup. Normalized heads run explicitly in fse; hidden-layer neural heads are unsupported.

The earlier Mac headline was replaced by the RunPod showcase at the top of this guide.
The README contains no Mac speed values. Its current caption uses the showcase's measured POTION-token count.

## Earlier Mac sample length

The [committed corpus-length record](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ea174334344c2fb0c97b9211af478d4acda77a40/benchmarks/baseline/dr151_corpus_lengths_20261101.json) records no throughput.
Its `wikitext encode` entry has 5,352 texts and 455,097 whitespace-separated words, averaging 85.03307174887892 words.
This describes the earlier Mac sample only. It is not the RunPod showcase's input length.

Its input digest, `xxh3-128:15151abfbf2f482115b1180347b4295e`, matches the earlier Mac source journals.
Dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3` covers validation and test.
The [counting method](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ea174334344c2fb0c97b9211af478d4acda77a40/benchmarks/quality/text_lengths.py) uses `len(text.split())` on original input before character/token limits.
The separate consumed POTION-token mean is 95.60650224215247. Words and tokenizer pieces are different units.

## Selecting the public core settings

Keep one core. For each model, show 32 only if its accepted float32 primary median is at least 1.20 times its accepted 16-worker median.
Use unrounded medians from three accepted rotating rounds. Otherwise show 16.
Display “More than 16 cores adds little here.” when the ratio falls below 1.20.
This is an editorial threshold for these workloads, not a general CPU-scaling claim.

The generator reads immutable source references and ordered rates from audited `selection_rows` at 1, 16 and 32.
It compares 16 and 32 directly on one shared input per model and requires dtype/reference measurements at the chosen setting.
Dtype cells retain that shared input. Each library bar instead uses its own calibrated paired workload,
with identical input digest/count in both libraries. Counts and digests may differ between bars.
The figure caption names the input size for every bar, and each ratio retains round pairing.
POTION-8M drives its dtype table. MRL library bars require MRL's own decision cells.
`core-selection-evidence.md` records medians, the exact ratio, immutable sources and series IDs.
No six-point curve, interpolation or scaling graphic is required. Missing decision cells fail publication.
The `call_seconds` extraction includes every first and warm fse-a/fse-b call from accepted rounds, each at least two seconds.
See the [row plan](readme-speed-panels.md). Missing data stays out of the public README.

## Compact local summaries

Full journals, rich original summaries and study notes are linked through the file manifest above.

| Study | Compact summary |
|---|---|
| Earlier WikiText and MASSIVE (**speed invalid**) | [Summary](readme-wikitext-20261101/summary.json) |
| Classic model timings (**speed invalid**) | [Summary](readme-classic-timings-20261101/summary.json) |
| Earlier precision matrix (**speed invalid**) | [Summary](readme-matrix-20261101/summary.json) |
| Runtime-attributed Mac window (**speed invalid**) | [Summary](readme-runtime-window5-20261101/summary.json) |
| Earlier int8 comparison (**speed invalid**) | [Summary](readme-int8-20261101/summary.json) |
| POTION-8M quality, English 40 of 41 tasks | [Summary](readme-quality-20261101/summary.json) |
| POTION-32M quality, English 40 of 41 tasks | [Summary](readme-quality-potion32-20261101/summary.json) |
| Retrieval-32M quality, English 40 of 41 tasks | [Summary](readme-quality-retrieval32-20261101/summary.json) |
| 100M-text showcase | [Summary](readme-showcase-20261101/summary.json) |

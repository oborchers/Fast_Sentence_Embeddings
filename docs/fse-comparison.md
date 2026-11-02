# Comparing fse with the alternatives

The November CPU study separates encoding, writing and classification. Quality scores come from the frozen evaluation protocol. A failed measurement provides no speed claim.

## Evidence and reproduction

| Study | Compact source |
| :--- | :--- |
| CPU speed and terminal cell statuses | [speed-2026-11.json](../benchmarks/baseline/speed-2026-11.json) |
| MTEB, NanoBEIR and STS | [quality-2026-11.json](../benchmarks/baseline/quality-2026-11.json) |
| Historical classification accuracy | [classification-quality-2026-11.json](../benchmarks/baseline/classification-quality-2026-11.json) |
| Raw archives, sizes and SHA256 | [evidence-2026-11.json](../benchmarks/baseline/evidence-2026-11.json) |

Raw evidence lives on `benchmark-evidence-2.0`, which is never merged into the code branch. The manifest pins immutable commits. Each compressed archive includes its own per-file size and SHA256 manifest. Source snapshots preserve the exact original scripts and dependencies. Verify the compressed SHA256 before extraction, then verify every extracted member. Historical scripts are evidence, not installation instructions for the current API.

The speed entry point is [speed-run-2026-11.py](../benchmarks/speed-run-2026-11.py). Its two private modules, dependency locks and input specification describe the frozen run. The summary records both the runner commit and the separately pinned library commit. Reproduction requires the original Linux environment and sufficient memory and disk. Do not run this workload on the maintainer's Mac.

## CPU speed

The machine exposed 32 logical CPUs on an AMD EPYC 9965 host, with 64 GB RAM and 150 GB disk. Measurements used one or 16 allowed CPUs, with explicit affinity and thread limits. The run crossed 1 and 2 November 2026 UTC.

Every ordinary encode cell requires at least two seconds per fse call. Library arms receive identical text within each pair. Different pairs may use different input sizes. The table uses one shared 4,620,000-paragraph WikiText input. Paragraphs are sampled and shuffled with a fixed seed. These are passages, not individual sentences.

The primary estimate is the median of fse-a. fse-b is an identical-code control. The global symmetric A/A upper bound must not exceed 1.10. Normal cells use three rounds. Library speedups use the median of paired ratios, not a ratio of unrelated table medians. Output shape, normalization and numerical agreement are checked separately from timing. Every backend records its own installed-code identity.

All six POTION-8M table cells passed: float32, float16 and int8 at one and 16 workers. Rust and FastEmbed comparisons at one worker also passed. The Python Model2Vec comparison failed A/A. The native sentence-transformers comparison failed output verification because the benchmark used the wrong unknown-token policy. Neither rejected comparison supports a speed claim.

There are 26 terminal cell records: 13 valid, five invalid, six failed and two not started. Five valid records are short-call diagnostics, not headline evidence. The 100-million-text writing repeat failed A/A. The classification stage and four 16-worker library cells stopped at the benchmark's conservative memory guard, before inference started. These stops are not operating-system out-of-memory kills.

Peak RSS covers child lifetime, including model loading and output verification. Collection differs between Python and Rust arms. These values do not establish a comparable encode-only memory ranking.

The allocation lasted 8,244 seconds. Estimated compute plus disk cost was USD 2.25. This is a rate-based estimate, not an invoice. The earlier failed setup cost is separate.

## Why Rust can win on small calls

At 16 workers, the accepted size diagnostics put Rust ahead at 5,000 texts per call and fse ahead at 50,000 and 500,000. This brackets an observed crossover, without locating an exact threshold. Short calls are deliberately exempt from the two-second rule for this diagnostic only.

Reusing fse's computation pool reduces part of the small-call overhead but does not explain the whole gap. A nested Rust output is a separate diagnostic whose subsequent flattening is outside its timer. It cannot replace the primary contiguous-output comparison. Two startup controls did not begin before the stage budget expired, so absolute startup costs remain unmeasured.

## Quality

The English mean covers 40 of 41 MTEB(eng, v2) tasks using unmodified MTEB 2.26.25. MindSmallReranking was not computed: the attempts failed first with insufficient disk and later status 137 on the 64 GB machine. Do not label these means as the full leaderboard mean.

All rows below are float32. Scores are multiplied by 100 and rounded only for display. MTEB is the unweighted mean of the same 40 English tasks. NanoBEIR is its separate 13-task mean. STS Benchmark uses Spearman. Full precision and fit identities remain in the linked quality JSON.

| Finished model | MTEB 40/41 | NanoBEIR | STS Benchmark |
| :--- | ---: | ---: | ---: |
| SIF-10-glove-wiki-gigaword-300 | 49.79 | not computed | 62.75 |
| all-MiniLM-L6-v2 | 59.89 | 56.22 | 82.03 |
| potion-multilingual-128M | 49.96 | 36.65 | 71.75 |
| Average-paranmt-300 | 50.69 | 39.39 | 79.07 |
| potion-base-32M | 55.26 | 46.18 | 76.84 |
| potion-base-8M | 53.95 | 43.85 | 75.41 |
| potion-retrieval-32M | 54.51 | 50.39 | 74.10 |
| static-retrieval-mrl-en-v1 | 51.89 | 50.31 | 74.02 |
| static-similarity-mrl-multilingual-v1 | 51.43 | 43.52 | 79.94 |
| word2vec-google-news-300 SIF-10 | 49.26 | 37.66 | 63.69 |
| fasttext-crawl-subwords-300 SIF-10 | 51.42 | 40.65 | 65.46 |
| paragram-300-sl999 SIF-10 | 50.98 | 40.92 | 68.81 |


The quality summary covers six Hub static models, all-MiniLM-L6-v2, plain-mean paranmt-300, and historical SIF-10 GloVe, word2vec, FastText and paragram configurations. NanoBEIR coverage is explicit per model. GloVe has no completed NanoBEIR result in this collection. No missing score is imputed.

Historical SIF-10 uses WikiText fit data, seed zero, corpus frequencies and ten components. FastText includes its actual OOV ngram path. These fitted states differ from later Hub artifacts fitted with wordfreq or FineWeb. Do not attach the scores to a different fitted artifact. Original revisions and artifact identities remain in the summaries and archives.

The original 1.x STS table uses Pearson correlation on the 1,379 STS Benchmark test pairs. The current MTEB result uses Spearman. That metric difference explains part of the GloVe discrepancy. Fit data and frequency policy also differ. No controlled ablation isolates their remaining contribution. Historical M1 quality and accuracy scores remain valid.

## Classification

Historical fse accuracy uses normalized frozen embeddings and a linear head. Model2Vec with the same sklearn linear head is the engine comparison. Model2Vec's own classifier is a different trained model: it fine-tunes embeddings and trains an MLP in PyTorch. By default its head has 512 ReLU hidden units, then softmax or sigmoid. Exported inference runs in NumPy, as implemented in model2vec 0.10.0 `model2vec/inference/mlp.py`.

The historical accuracy summary retains 176 rows. One complete Subjectivity evaluation was checked across Mac and Linux, with equal predictions and accuracy. This does not establish bit-identical trained exports or equivalence on every dataset.

The new raw-mean folded-head measurement failed its memory preflight. It provides no new classification rate or accuracy. Folding a linear head is distinct from replacing it with a fine-tuned MLP. Training cost must remain separate from inference throughput.

## Withdrawn timing

All Apple M1 speed measurements are withdrawn, regardless of input size. Earlier headline comparisons using only about 5,000 texts per call are also withdrawn, even where their A/A check passed. Their original bytes remain archived. Quality, accuracy and numerical regression fixtures are unaffected.

The November study's rejected cells remain rejected. A clean initial probe does not prove a quiet whole run. Missing per-round memory and writeback telemetry prevents a definitive attribution of later temporal drift. Any follow-up requires the maintainer's decision. A different machine is a new cohort and must be labelled accordingly.

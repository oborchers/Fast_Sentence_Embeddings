# README evidence

## Headline and dtype table use different machines

The headline uses the approved 16-worker file-to-vector run on AMD EPYC 9575F with 32 allocated vCPU, RunPod, 2 November 2026.
It embeds 100 million sampled, repeated and shuffled WikiText-103 sentences with POTION-8M (256 dimensions).
There are 3,755,995 distinct source sentences, repeated about 26.6 times.
Average text length is 28.73384302 POTION tokenizer IDs (25.57447294 words).
The model table is float32 and output storage is int8.

Three encoding-and-write calls took 13.40701300802175, 13.48732629197184 and 13.458833114011213 seconds.
The median is 13.458833114011213 seconds, or 7,430,064.638805557 texts/s.
The README rounds the duration to 13.46 seconds and says more than 7 million, without rounding the rate up.
The same machine's 32-worker run took a median of 16.75060166104231 seconds, rounded to 16.75 in the footnote.
The calls include writing through sink close, but exclude model loading and the later explicit filesystem sync.
Durability is `none`; the headline does not claim crash-safe storage at 13.46 seconds.

The collected 16-worker manifest SHA256 is `5fb185267692e6b14cb3831972b4324f9bd599a922a72a37ac3c7bd44d042172`.
The 32-worker manifest SHA256 is `079581ae37e056565d2d60ec01061d65b6fa0fc7ead77bc1037787a092ba8f31`.
Both manifests and all listed member hashes were checked locally against the collected results.
The fixed [sentence-run archive](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/608cf02a9557ef8bade83ae6c0c2fadd5b8d5012/archives/sentence-speed-2026-11.tar.gz) on `benchmark-evidence-2.0` contains both collected studies and `audit/headline-summary.json`.
Archive SHA256: `443e7ff798d38b41d14e3952128d8080b2d698e6005991b147c34a69ea88f1c7`.

The dtype table comes from a separate AMD EPYC 9965 run described below.
These are different machines and workloads. Their rates are not combined or used to derive a speedup.
The library comparisons come from a third allocation, described in the section on library comparisons below. Missing cells are not filled from an older run.

## Dtype table: AMD EPYC 9965

The six README values come only from cells `dtypes-02` through `dtypes-07` of one reproducible run.
The pinned source is [speed-run-2026-11.tar.gz](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/f97e32edbc0f9daa538b56b54f3187d8d3c177d8/archives/speed-run-2026-11.tar.gz)
on the separate `benchmark-evidence-2.0` branch. Its SHA256 is
`743846611953faca7b76cb9ecf09fb63257db47503c117ae6ebb6a5039ec146f`.
The [archive index](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/f97e32edbc0f9daa538b56b54f3187d8d3c177d8/archives.json)
records sizes and hashes. Each archive contains a member manifest for checking the extracted files.
The [compact source](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6cb13ed72321d1207fc106f6c0e7154935164017/benchmarks/baseline/speed-2026-11.json)
provides the same accepted cells and their ordered observations.

The configuration is POTION-8M, AMD EPYC 9965, 32 assigned logical CPUs, Python 3.13.18, Linux, 1 to 2 November 2026.
All six cells share 4,620,000 sampled and shuffled WikiText-103 passages, averaging 119.0559961038961 words.
Input SHA256 is `bec2f6d5e127889fce0b29e6123915f083431ac0cd4b7fe3a4f24e43758a7c92`.
The executed fse runtime SHA256 is `5a4883c1675cef10e5a08bf1c084f0dc57f872ed1cca9074a08b673b15b8fac0`.
The encoder returns normalized float32 vectors. Model storage precision varies by row.

Each cell retains three primary fse-a calls and three repeated fse-b calls.
The displayed rate is the median primary throughput, rounded to whole texts/s.
Every first and warm fse call takes at least 2 seconds, and the global A/A upper band is at most 1.10.
The caller and independent reviewer recomputed the medians and those bands from the saved calls.
No new benchmark was run for the README update.

There is no current 32-worker dtype comparison or classification-speed result.
Earlier Mac and small-input timing evidence is not published here. Old hardware cohorts never fill current missing cells.

## Library comparisons: AMD EPYC 9575F

`readme-speed.json` retains exactly the eight library pairs used by the README tables and graphic.
Each pair uses the same pinned model, input digest/count and normalized float32 output for both libraries.
The table rates are input count divided by each library's median call time.
The speedup is the median of the three paired time ratios, so it need not equal the quotient of the separately rounded rates.
No peak-RSS comparison is made.

Seven pairs come from the WikiText sentence run (manifest `a409cf374700b7dfd8729ce16939dfaefc24b106b1fd9fe8ddc3aaf5369a5a16`).
The failed Sentence Transformers 16-worker call is excluded.
Its replacement uses the recommended reused CPU process pool with batch size 1024 and one native thread per process
(manifest `f2f9bf4300cd4ba4b8f8c41f280c772e03bcb3c712027e73e6c69ba0b0cd1e3a`).
Sentence Transformers with one worker used the default batch size 32.
Model2Vec with 16 allowed CPUs used its default thread path; the coordinator observed about four CPUs in use.
This is the library's default scheduling, not sixteen fully occupied worker processes.

The one-worker pairs each use two million sentences. With sixteen workers, POTION-8M pairs use eighteen million sentences;
the static-retrieval pair uses five million. Each pair has its own input identity, recorded in the summary.
Mean length is about 26 words. All calls use the same AMD EPYC 9575F allocation with 32 vCPU, on 2 November 2026.
POTION-8M has 256 output dimensions. static-retrieval-mrl-en-v1 has 1024; its absolute rates are not compared as if it were POTION-8M.
The chart's two panels use separate, explicitly labelled linear scales.

The independent audit and all eight pairs are in the same [pinned archive](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/608cf02a9557ef8bade83ae6c0c2fadd5b8d5012/archives/sentence-speed-2026-11.tar.gz), under `audit/library-summary.json`.
The compact README summary records its member hash. All original manifests, runtime identities and failed attempts remain in that archive.
The new approved protocol has three paired rounds and no fse A/A duplicate arms; the older A/A eligibility test is not imputed to it.
These results do not replace the separate EPYC 9965 dtype study.

## Quality and file size

[readme-quality.json](readme-quality.json) is the single local summary for all eleven supported models in the README.
It records unrounded 40-task means, the model identities and exact float32 file sizes used for display.
The scores use MTEB 2.26.25, measured 1 November 2026. MTEB(eng, v2) is 40 of 41 tasks because MIND was not computed.
The summary preserves MIND's recorded memory-limit failure (two attempts, ENOSPC and status 137).
It pins the compact quality source and archived source records through their SHA256 identifiers.

Historical SIF-10 benchmark scores use the WikiText-103 fit. Linked Hub SIF-10 models use wordfreq weights and a FineWeb fit.
Those model versions have separate results on their cards. File size is not process memory.

## Example verification

Run the displayed Python examples unchanged with the documented dependencies and real models. Each README section is one run in its own working directory; the examples of "Linear heads" build on each other and share one directory and one interpreter. Examples that need fse 2.1 run against the checked-out master until 2.1 is released.
Check output shape and dtype, the saved int8 table, classifier labels and save/reload behavior.
For folding, check `classifier.normalized is False`; a saved classifier must also load without scikit-learn.
These are documentation checks, not additional library unit tests or speed benchmarks.
The last complete real-model execution is recorded in the PR description. Library API/conformance tests remain in the normal suite.

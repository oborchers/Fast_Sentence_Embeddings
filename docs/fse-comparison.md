# Comparing fse with the alternatives

This guide records the measurement contract and evidence behind the README. The final library comparison has seven accepted pairs from the sentence collection and one accepted ST process-pool follow-up. The two writing configurations form a separate study. Quality and historical accuracy are unchanged.

## Sources and cohorts

| Study | Compact source | Cohort |
| :--- | :--- | :--- |
| Eight paired library comparisons | [library-speed-2026-11.json](../benchmarks/baseline/library-speed-2026-11.json) | EPYC 9575F, sentence inputs, 2 November 2026 UTC |
| 100M native-file writing, 16 and 32 CPUs | [headline-speed-2026-11.json](../benchmarks/baseline/headline-speed-2026-11.json) | Same physical allocation, separate writing study |
| Six accepted table-dtype cells | [speed-2026-11.json](../benchmarks/baseline/speed-2026-11.json) | Earlier EPYC 9965, 4.62M paragraphs, 1/2 November 2026 UTC |
| MTEB, NanoBEIR and STS | [quality-2026-11.json](../benchmarks/baseline/quality-2026-11.json) | Frozen quality protocol |
| Historical classifier accuracy | [classification-quality-2026-11.json](../benchmarks/baseline/classification-quality-2026-11.json) | Accuracy only, no inference-speed claim |
| Immutable raw bundles | [evidence-2026-11.json](../benchmarks/baseline/evidence-2026-11.json) | Per-file sizes and hashes inside each bundle |

The final speed evidence is pinned at [archive commit `608cf02a`](https://github.com/oborchers/Fast_Sentence_Embeddings/commit/608cf02a9557ef8bade83ae6c0c2fadd5b8d5012). Download [the compressed bundle](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/608cf02a9557ef8bade83ae6c0c2fadd5b8d5012/archives/sentence-speed-2026-11.tar.gz), verify SHA256 `443e7ff798d38b41d14e3952128d8080b2d698e6005991b147c34a69ea88f1c7`, then verify every member against `archive-member-manifest.json`. It contains 532 original members. The evidence branch is never merged into the code branch. Historical scripts in the bundle are data, not current installation instructions.

The allocation exposed 32 allowed logical CPUs, 128 GB RAM and 150 GB disk on an AMD EPYC 9575F host. All final library and writing records are dated 2 November 2026 UTC. Raw machine labels are preserved. The maintainer confirmed that `sentence-run-2026-11-02` and `fp09zqzjdg0pgh` refer to the same physical allocation. The ST retry reset its raw allocation timestamp to its incremental budget start. That metadata is retained and does not identify another physical machine.

The library code is pinned separately at `624251df2269e3099f5c61987be63e718893b58c`. Runner pins are `7f2fe04a8f27d881cfcd1e38269066bce25ec8d6` for the original sentence collection, `b2e960f923c67cf0ed1fd4fcfc517836a333a06e` for writing and `6c9fc1a5fd5f42d4b1af0e0756ef9123d493aad9` for the ST process pool. The archive includes all three source snapshots and original invocation logs.

## Paired encoding protocol

Inputs come from the pinned WikiText-103 training split. The sentence rule collapses whitespace, skips headings, splits after punctuation followed by whitespace and keeps pieces with at least five words. Seed 0 selects positions with replacement and shuffles them. Both arms of each pair receive the identical file, truncation and normalized float32 output contract. Different pairs can have different counts and digests. These are pairwise library comparisons, not a fixed-input CPU-scaling curve.

Each pair has three alternating rounds. A full warmup precedes each measured call. Every measured fse call must last at least two seconds. Shape, dtype and 10,000 fixed output rows are checked at unchanged tolerances, `rtol=3e-5` and `atol=3e-6`. The eight saved UNK paragraphs remain unchanged in the ST inputs and their samples. Each backend records its own installed-code identity. Rejected attempts remain archived and supply no speed claim.

The maintainer approved this reduced protocol without an A/A duplicate arm. Its three-minute initial quiet observation is not a per-cell A/A measurement. The summaries therefore state `aa.status=not_run`, rather than inventing a clean control band. The ST follow-up uses a short initial observation. Reported factors are medians of same-round counterpart/fse time ratios, not ratios of unrelated medians.

The following table is an audit of durations. Full-precision rates, ranges, per-round observations, input digests, model revisions and raw member paths are in the compact summary.

| Counterpart | Allowed CPUs | Texts | fse median, s | Counterpart median, s | Median paired factor |
| :--- | ---: | ---: | ---: | ---: | ---: |
| FastEmbed | 1 | 2,000,000 | 2.922560 | 68.294732 | 23.690667 |
| Model2Vec | 1 | 2,000,000 | 2.849297 | 60.487159 | 21.356244 |
| model2vec-rs | 1 | 2,000,000 | 3.195695 | 4.840583 | 1.500028 |
| sentence-transformers | 1 | 2,000,000 | 9.202243 | 107.792172 | 11.813207 |
| FastEmbed | 16 | 18,000,000 | 2.763460 | 62.011209 | 22.308737 |
| Model2Vec | 16 | 18,000,000 | 2.725837 | 341.091340 | 125.780572 |
| model2vec-rs | 16 | 18,000,000 | 2.766894 | 6.485669 | 2.381571 |
| sentence-transformers | 16 | 5,000,000 | 2.421668 | 22.131070 | 9.140731 |

POTION-8M has 256 dimensions and is shared with Python Model2Vec, model2vec-rs and FastEmbed. The ST pair uses native `static-retrieval-mrl-en-v1` with 1024 dimensions. It retains UNK and reads texts whole. This model difference prevents comparing their absolute rates as an engine-only result.

Python Model2Vec 0.10.0 uses its documented serial path at one CPU and its Joblib threading batch path at 16. The latter disables internal tokenizer parallelism. Sixteen allowed CPUs does not mean sixteen processes or full utilization. Rust 0.4.0 uses the pinned source and `encode_flat`, with the global Rayon pool explicitly fixed to the requested count. FastEmbed 0.9.1 uses its NumPy StaticTextEmbedding backend, serial at one CPU and 16 processes with one native thread each. These rows are not ONNX measurements.

ST 6.2.0 uses native StaticEmbedding. Its one-CPU row uses Torch thread count 1 and default batch size 32. The accepted 16-CPU follow-up uses `start_multi_process_pool(target_devices=["cpu"]*16)`, reused for warmup and measurement, with batch size 1024. Native and Torch threads are limited to 1 per process. Batch size 1024 reduces small-call overhead; it is a justified starting point, not a measured optimum. The fse arm, input digest, normalization and tolerance are unchanged.

### ST failure and file-descriptor limit

The original 16-thread ST path did not complete and was manually stopped. Its logs cannot establish whether it stalled in warmup, measured inference or final stacking. It contributes no throughput. The first CPU-process-pool attempt then failed while rebuilding a shared storage descriptor, with `received 0 items of ancdata` and a soft open-file limit of 1024. The hard limit was 524288.

The same code and runner completed after the maintainer raised the soft limit to 524288. This supports a descriptor-pressure explanation for the transport failure, without claiming a recorded EMFILE syscall or an observed peak descriptor count. At 5M rows, roughly 1000 output chunks can approach the low limit before other descriptors. Shared-memory space also needs a separate check. The failed attempt and the accepted retry are both retained. The descriptor-limit analysis is preserved as `notes/st16-file-descriptors.md` inside the final bundle.

## Native-file writing protocol

The two accepted configurations stream the same 100M-line file through POTION-8M into ShardSink. The model table is float32, the persisted vectors are int8 and the encoder output has 256 dimensions. Three rounds use `durability="none"`. Each output is synchronized, verified and deleted before the next round. File and directory synchronization after deletion prevents cleanup from overlapping the next timed operation.

Primary time runs from first input read through sink close. Model loading, input generation and output verification are outside that timer. Sink creation and the subsequent targeted external fsync have separate timings. External fsync does not change the store's recorded `durable_rows` under the none policy. Every round verifies 100M rows, stored checksums, a SHA256 file inventory and 10,000 sampled int8 codes and float32 scales against standalone encoding.

| Allowed CPUs | Median read to close, s | Repeat range, s | Median through external fsync, s |
| ---: | ---: | :--- | ---: |
| 16 | 13.458833114 | 13.407013008 to 13.487326292 | 13.913902076 |
| 32 | 16.750601661 | 16.405571854 to 16.957345100 | 17.281033792 |

Both configurations use input SHA256 `2765496afe36095473776599284b29462fb1798744b3b25abbc06e005fa167ef`, 13,639,687,281 input bytes and 26,400,137,245 logical output bytes per round. The input has 25.57447294 words and 28.73384302 limited tokenizer IDs per text on average. That token count includes UNK and excludes special IDs. The older writing study used another input construction and excluded UNK in its token counter. Do not relabel the old and new digests as the same workload.

The 16-CPU configuration supplies the approved headline. The 32-CPU run preceded it; these were sequential repeats, not an interleaved scaling experiment. The observed 16-CPU advantage does not isolate CPU count from host or storage state. Neither configuration has an A/A arm. Peak RSS covers the inference child before verification, not the complete process tree. Output bytes are logical file sizes, not device write amplification. The full observations remain in `studies/headline16/collected/headline16-results` and `studies/headline32/collected/headline-results` inside [the pinned bundle](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/608cf02a9557ef8bade83ae6c0c2fadd5b8d5012/archives/sentence-speed-2026-11.tar.gz).

## Separate paragraph dtype cohort

The earlier accepted six dtype cells remain unchanged in [their compact source](../benchmarks/baseline/speed-2026-11.json). They used 4,620,000 WikiText paragraphs on EPYC 9965 with 64 GB, float32/float16/int8 tables and one or 16 allowed CPUs. Their older protocol has paired identical-code A/A controls with a symmetric upper bound of 1.10. These conditions do not apply retroactively to the new approved sentence protocol. Keep the machine, input and protocol captions separate. Superseded library rows, rejected controls and short-call diagnostic rates remain in the raw archive only.

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

There are no eligible current classification inference-speed cells. Do not use encoding rates as classifier throughput or draw a measured classification speed chart from these studies. Historical classification accuracy remains in its separate compact summary.

The historical fse linear-head comparison uses frozen embeddings. Model2Vec with the same sklearn linear head is an engine comparison. Model2Vec's own classifier is a different trained model: it fine-tunes embeddings and trains an MLP in PyTorch, by default 512 ReLU hidden units followed by softmax or sigmoid. Exported inference runs in NumPy (`model2vec/inference/mlp.py`, version 0.10.0). Training time must remain separate from inference speed.

The accuracy summary retains 176 historical rows. One full Subjectivity evaluation produced equal predictions and accuracy across Mac and Linux. That does not establish bit-identical trained exports or equivalence on every dataset. The planned folded-head speed measurement failed preflight and provides no new rate.

## Historical profile attribution

These M1 profiles are retained for attribution, not current speed claims. Allocation and RSS peaks cover different instruments and include preparation. They do not rank the current libraries. The [compact profile record](../benchmarks/baseline/thread_pool_profiles_20261031.json) pins the unchanged raw evidence and its limitations.

| Historical variant | Native allocation peak, bytes | RSS peak, bytes | Hottest recorded functions |
| :--- | ---: | ---: | :--- |
| base | 78,198,773 | 154,025,984 | `__psynch_cvwait`, `__pyx_f_3fse_7_native__wp_piece`, `fse_loop_f32_neon` |
| reuse | 80,945,694 | 159,285,248 | `__psynch_cvwait`, `__pyx_f_3fse_7_native__wp_piece`, `fse_loop_f32_neon` |

## Withdrawn timing

All Apple M1 speed measurements are withdrawn, regardless of input size. Earlier headline comparisons with only about 5,000 texts per call remain withdrawn, including rows whose old A/A checks passed. Their original bytes remain archived. They supply no current speed claim. Quality, accuracy and protected numerical regression fixtures are unchanged.

Rejected rows are not replaced by estimates. A missing result remains missing. The code and dependency locks reproduce the recorded workload; every future measurement needs its own machine, input and execution provenance. No further paid run is required by this report.

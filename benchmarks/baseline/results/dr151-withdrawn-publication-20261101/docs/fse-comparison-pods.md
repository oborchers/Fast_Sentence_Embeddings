# Bounded comparison pods

The coordinator launches these scripts. The benchmark tooling never calls RunPod.
Generate the revised MIND script with `python benchmarks/quality/make_mind_pod.py --commit <full SHA> --output <scratchpad>/dr151-pod1.sh`.
Generate Pod 3 with `python benchmarks/quality/make_folded_pod.py --commit <full SHA> --output <scratchpad>/dr151-pod3.sh`.
Keep the already launched Pod 2 script immutable.
All use image `python:3.13`, 32 vCPU and 64 GB RAM. MIND needs **200 GB disk**, the existing Pod 2 uses **60 GB**, and expanded Pod 3 needs **150 GB**.
Setup counts against the hard limit. Computation stops before artifact export.
After export, `ALL_DONE` precedes `sleep infinity`, so the coordinator can collect logs.
The coordinator must terminate the pod after collecting the logs.

## Pod 1: MIND, three-hour hard limit

The pinned, unmodified `mteb==2.26.25` supports a shared loaded task.
The runner calls the public `MindSmallReranking.load_data(num_proc=1)` once.
It passes that same object to successive `mteb.evaluate` calls.
MTEB retains data which was already loaded on entry. The runner checks this after each model.
No loader replacement, task sampling or qrels reduction occurs.
The dataset revision is `227478e3235572039f4f7661840e059f31ef6eb1`.

All eleven float32 configurations run in ascending static artifact size, then MiniLM:

- The six pinned Hub static models in `pod_models.HUB`.
- `fse/paranmt-300`, plain mean, imported through gensim KeyedVectors and `fse.gensim.from_gensim`.
- GloVe-300, Google News word2vec-300 and crawl FastText-300 with folded SIF-10.
- `sentence-transformers/all-MiniLM-L6-v2` last.

The runner separately records shared dataset loading, model loading, encoding and evaluation wall time.
Evaluation time outside `encode` includes scoring and evaluator overhead.
The shared load duration is repeated as context, never charged eleven times.
Expected duration is an **uncalibrated planning estimate of two to three hours**.
There is no successful local MIND calibration. The local loader exhausted RAM.
The three-hour limit can leave models unfinished. Partial artifacts remain explicit.

## Pod 2: speed, four-hour hard limit

Host load covers the shared physical host, so it is recorded without rejecting runs.
This exception applies only to the Linux pod. Local timing remains prohibited by coordinator rule 10.
The initial approximately two-minute probe uses POTION-8M float32, MASSIVE and one fse worker.
It rejects the host if either the within-round envelope or the spread across all controls exceeds 10%.
Model preparation precedes at least 120 seconds of observations, with up to 30 seconds to finish the final round.
Fewer than three valid rounds also rejects the host. Rejected probes stop before Rust and training setup.

Each cell rotates fresh processes with fse A/A controls and requires three valid rounds.
All observations remain in JSONL, including failed or noisy rounds.
Any observed A/A envelope or all-control spread above 10% labels the entire cell incomplete.
Primary fse summaries use `fse-a`; `fse-b` is a separate control.

Run order:

1. Host probe.
2. Model2Vec Python, pinned model2vec-rs and FastEmbed encode comparisons on both full corpora.
3. Raw-file-to-disk comparisons on both corpora, including Arrow input for fse.
4. Float32 table cells on WikiText, then MASSIVE.
5. Int8 table cells on WikiText.
6. Classification training, accuracy and inference comparisons.
7. Existing FAISS and hnswlib lifecycle comparisons, if time remains.
8. Float16 table cells on WikiText.

The table matrix has ten models, three table dtypes and workers 1/32 on WikiText (60 cells).
Float32 adds all ten models and both worker settings on MASSIVE (20 cells).
The ten models are the six Hub models, paranmt mean and the three SIF classics.
Classics use unfitted, same-shape timing proxies, explicitly recorded in their provenance.
SIF proxies retain the fitted recipe's external NLTK tokenization inside encode timing.
FastText retains fse's native OOV ngram path. No score comes from these timing proxies.
Model2Vec rows describe tokenizer and batch-pool settings. Exact tokenizer thread counts are not measured.

End-to-end timing includes input reading, encoding, output writing and writer close.
Models load before that timer. Output verification follows the timer.
fse uses native file input or memory-mapped Arrow, with ShardSink durability `full` and `none`.
Python counterparts use line reading, their library encoder and `np.save`.
Rust uses its library encoder with a buffered NumPy v1 writer.
Counterpart full mode flushes and fsyncs the output file and directories.
None mode makes no durability promise. fse records time until durable only in full mode.
Metrics include wall time, texts/s, file bytes, output digest and peak process RSS.
Rust RSS is the maximum parent or child high-water mark, explicitly labelled, not a sampled tree sum.
CR and LF become spaces in every corpus representation, including encode-only inputs.
No row is stripped or dropped by this conversion. Both original and resulting digests are recorded.

## Classifier protocol

All fifteen disclosed SetFit test sets and all 46,743 held-out FineWeb-Edu pages remain intact.
Frozen fse and Python Model2Vec use the existing sklearn logistic-regression protocols.
FineWeb also compares its supervised fastText filter. SetFit includes fitted TF-IDF.
The direct native comparison pins `model2vec[train]==0.10.0`.
Its documented API is `StaticModelForClassification.from_pretrained(path=...)`, `fit`, then `to_pipeline()`.
The pipeline is saved and reloaded as `StaticModelPipeline` for inference.
Each comparison uses the same POTION source and dataset splits as its frozen counterpart.
Model2Vec 0.10.0 fine-tunes its embeddings and an MLP with one 512-unit ReLU hidden layer using PyTorch. The exported classifier runs in plain NumPy, with softmax or sigmoid for probabilities. Its label path selects from logits. fse's classifier head is linear. Accuracy and speed therefore compare a fine-tuned MLP against a frozen-embedding linear head. [Pinned MLP implementation](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/inference/mlp.py), [pinned training documentation](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/train/README.md).
Training durations are separate and explicitly label sklearn versus torch tool chains.
Native early stopping uses a seed-0 internal training split, a 0.1 fraction capped at 10,000 rows.
This is 100 rows for SetFit and 10,000 for FineWeb. No evaluation row enters this split.
Accuracy and macro-F1 are recorded alongside inference evidence at requested settings 1/32.
The local `subj` accuracy checkpoint awaits comparison with the pod's matching dataset result.

Sources: [Model2Vec training documentation](https://github.com/MinishLab/model2vec/blob/main/model2vec/train/README.md).
Exact API signatures were also checked against the installed 0.10.0 package.

## Fit provenance and exports

`prepare_classics.py` fits empirical vocabulary frequencies and ten uncentered principal components.
It uses all 859,346 cached WikiText-103 training paragraphs, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, seed 0 and SIF parameter `a=1e-3`.
The projection follows `recipes/01-sif-usif.ipynb`: float32 block Gram products,
float64 accumulation, `eigh(driver="evd")`, then offline table folding.
FastText's vocabulary and OOV ngram tables both receive the projection.
Fit receipts record the corpus hash, fitted state hash and finished encoder fingerprint.
The M1 and Pod 1 independently fit with the same script. Cross-platform bit differences remain visible.
FastText's benchmark inclusion makes no statement about its eventual fse 2.0 release status.

Every result JSONL, invocation record and preparation manifest is exported with DR190 artifact markers.
The final manifest labels timeouts, failed invocations and noisy cells as incomplete.
No duration estimate guarantees completion of this expanded queue within four hours.

## Additional Pod 3

Every existing Pod 2 classifier comparison stays intact. Pod 3 adds POTION-8M float32 raw-mean linear inference at workers 1/32 on the same fifteen SetFit datasets and held-out FineWeb pages.

The folded and explicit arms use the same trained float32 coefficients, bias, labels and raw encoder. The explicit arm runs `Encoder.encode` followed by `vectors @ weight.T + bias`. The folded arm uses `Classifier.predict` on its projected table. Each operation records accuracy, size and its own backend identity. The Model2Vec reference is the fine-tuned MLP and exported NumPy pipeline described above.

Raw-mean fse heads already fold internally. Standalone save changes persistence and retained memory, not the folding algorithm. The script records folded-file bytes beside raw-encoder plus linear-head bytes. Floating-point accumulation order can change between folding and explicit matmul, so accuracy and label disagreements are retained. The expanded whole pod has a 150-minute hard limit, including setup, and exports partial results. Its additional stages and budget are specified below. No existing comparison is replaced.

Pod 3 also compares Python Model2Vec with the **same frozen sklearn linear head**, including the identical float32 coefficients and bias. This isolates the encoding engines. Its fine-tuned Model2Vec MLP remains a separate trained-model comparison. AG News linear cells are published first, before MLP training, with raw and summary artifacts.

## MIND rerun storage and SSH

The first 60 GB pod failed during `top_ranked` train preparation, after the 97,006,943-row test qrels cast. It produced no MIND score. Pinned dataset metadata declares 41.34 GB of base Arrow data. The saved models occupied 10.46 GB. Extra casts, downloads and preparation sources also coexist. The deleted pod has no measured per-directory inventory. The [failure audit](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ae79c80c5736706a1e39df2f69a18e5801ae82d3/benchmarks/baseline/dr151_mind_disk_failure_20261101.json) distinguishes declared sizes from observed facts.

The revised pod requests a 200 GB container disk. It requires 140 GiB free before setup and 120 GiB before loading MIND. It removes only raw preparation sources after verifying all ten saved static encoders. Finished artifacts and SIF fit states remain. Hub, datasets, temporary and uv caches use explicit paths under `/dr151-cache`. Xet chunk caching is disabled. No MTEB loader, split or row changes.

The revised MIND script and additional classifier script install SSH first. They consume RunPod’s `PUBLIC_KEY` without shell tracing, start port 22, and retain files in `/dr151-results`. Logs stream stage start/end and disk usage. Their hard limits include setup, with reserved partial-export time. `ALL_DONE` is followed by `sleep infinity` for SSH collection. The coordinator must terminate the pods after collecting results.

## Expanded Pod 3: scaling, write paths and 100M showcase

The additional pod now has a 150-minute workload limit and requests 150 GB of container disk. Existing Pod 2 comparisons stay pinned and unchanged. Pod 3 measures fresh README panels first, then classifier comparisons, the 10M write matrix, curve export and the 100M showcase last. Its workload limit includes setup, Rust compilation and artifact export. Cutoff results remain explicit partial evidence.

The 10M file repeats all qualifying WikiText-103 train chunks, then shuffles source indices with NumPy default_rng seed 0. The raw records label it **repeated and shuffled input**. They retain source-position count, distinct-text count, repetition factors, tokenizer settings, corpus digest and file checksum. The separate 2M benchmark is removed. Recipe 00 length statistics still describe its unrepeated 2M source chunks. Natural duplicate strings are distinguished from artificial source-position repetition.

At 32 workers, the write matrix includes native file encoding without a sink, MemorySink, and ShardSink float32/float16/int8 with durability none/full. The table remains float32 throughout. One-worker rows use encode, MemorySink and float32 ShardSink none. Locked model2vec-rs writes NumPy float32 with and without fsync at 32 workers, and without fsync at one worker. Its library lacks a persisted-vector writer, so the benchmark uses its library plus a buffered NumPy writer. Rotating rounds share identical fse float32/none A/A controls. Each output is checked and deleted before the next operation.

The scaling stage uses POTION-8M float32, the full 5,352-text WikiText validation/test corpus, and the same bench_compare child as the encoding headline. It measures every worker count 1 through 10, plus 16 and 32. Exact artifact header, normalization, source revision, text digest, machine/date, rates/ranges and A/A controls stay in its raw journal and summary. No interpolated points or placeholders are evidence. No local timing is requested.

The final showcase prepares one 100M-text file using the same qualifying source chunks, repetition and fixed-seed shuffle. fse streams the file natively into int8 ShardSink output with durability none and 32 workers. The input is never supplied as a 100M-element Python list. Exact token statistics use source-position multiplicities, outside timing. The showcase uses the maintainer's explicit exception: one timed run plus one A/A repeat. It is labelled a single-pair showcase, not a three-pair matrix cell. Wall time covers first input read through sink close. Row counts, finite values and a streaming output digest are checked after timing.

Each large raw row records texts/s, tokens/s, mean tokens, peak RSS, bytes written, time until durable where applicable, and disk space before output, with output present, and after cleanup. The fse WordPiece path has no sentence-result cache. The pinned [model2vec-rs implementation](https://github.com/MinishLab/model2vec-rs/blob/19f58bef6ce145f2bc08de7016498f28d521feea/src/model.rs) and its fast WordPiece path also have no sentence-result cache. Artifact and OS filesystem caches are separate. Input checksum verification warms the filesystem cache before each operation, which is recorded.

Budget handoff uses the coordinator's supplied planning allowances: $0.45 + $3.85 + $2.90 = $7.20. At the unverified planning rate of $0.96/hour, extrapolated from the [published cpu3c example](https://hi5.sh/guide/compute/runpod/), 150 minutes adds $2.40 compute and about $0.052 container storage (150 GB, [$0.10/GB/month](https://docs.runpod.io/pods/pricing), 30 days). Total workload estimate: $9.65. To keep the supplied total below $10, launch only if actual prior charges are at most $7.30, use a quoted rate at most $1.00/hour, and terminate within 155 minutes of pod creation, including collection. Under those assumptions the upper bound is about $9.94. The script itself prints ALL_DONE and sleeps for SSH collection, so the coordinator must enforce the termination deadline. No pod is launched by the comparison worker.

A conservative check that separately includes the declared storage for Pod 2, MIND and Pod 3 gives about $9.74 at $0.96/hour, using the failed first pod allowance of $0.45. Five minutes of Pod 3 collection raises that estimate to about $9.82. These are planning calculations, not verified account charges.

## Fresh Pod 3 graphics cohort

The first bounded stage contains 30 rotating series. POTION-8M has float32, float16 and int8 table cells at 1, 16 and 32 workers. Its float32 library pairs use Python Model2Vec, model2vec-rs and FastEmbed at those settings. Native sentence-transformers StaticEmbedding uses static-retrieval-mrl-en-v1 with its own same-model fse baseline, at the same three settings. Both models also have complete float32 curves at 1, 2, 4, 8, 16 and 32 workers. No fixed 10-worker pod cell remains.

Each series needs three accepted complete rounds, stable backend-specific loaded-code identity and a global symmetric fse A/A band within 10 percent. Every input output is checked before timing. POTION uses its native normalized output. The native ST static model uses its unnormalized mean output. Cross-library float32 checks use rtol 2e-5 and atol 2e-6, including identical empty-row positions. These third-party checks do not change fse conformance. Rates use fse-a as primary. Speedups use the median of paired ratios.

Curve export reuses these exact series rather than timing them again. The first stage is capped at 20 minutes within the unchanged 150-minute whole-pod limit. Setup and early work share the existing first-stage deadline. A cutoff leaves explicit partial evidence. Actual measured medians determine the renderer setting. Always retain one core. Select 32 only when the unrounded primary median32/median16 is at least 1.20, otherwise select 16. Settings 2, 4 and 8 remain curve-only even when faster. The complete 1/16/32 grid supplies either selected graphic setting.

## Immutable README extraction after collection

Commit the complete Pod 3 paired journal and output-check journal under `benchmarks/baseline/` first. Then `benchmarks/quality/export_readme_panels.py` reads those exact files with `git show <publication-commit>:<path>`. Its required arguments are `--commit`, `--journal`, `--checks`, `--table-caption`, `--comparison-caption` and `--output`. The captions must name the verified machine, model/input/output and date. It never reads an uncommitted journal as publication evidence.

The exporter replays the strict cell summary, checks bound float32-parent lineage and preserves ordered primary rates and exact series IDs. It emits the DR-154 `readme-speed-panels-v1` extraction contract. Excluded series remain listed with reasons. It does not edit the main README, fabricate missing rows, or claim an independent source audit. After independent review, DR-154's renderer applies the complete-curve, fixed 16/32 selection, chosen-row and common-cohort gates before publishing either graphic.

# Bounded comparison pods

The coordinator launches these scripts. The benchmark tooling never calls RunPod.
Generate the scripts from a pushed commit with
`python benchmarks/quality/make_pod_scripts.py --commit <full SHA> --output <scratchpad>`.
Both scripts use image `python:3.13`, 32 vCPU, 64 GB RAM and **60 GB disk**.
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
It rejects the host if any observed symmetric A/A envelope exceeds `[1/1.1, 1.1]`.
Fewer than three valid rounds also rejects the host. Rejected probes stop before Rust and training setup.

Each cell rotates fresh processes with fse A/A controls and requires three valid rounds.
All observations remain in JSONL, including failed or noisy rounds.
Any observed A/A envelope above 10% labels the entire cell incomplete.
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
Native training updates embeddings and a 512-unit hidden layer. Frozen LogReg updates only its linear head.
Training durations are separate and explicitly label sklearn versus torch tool chains.
Native early stopping uses a seed-0 internal 10% training split, never evaluation data.
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

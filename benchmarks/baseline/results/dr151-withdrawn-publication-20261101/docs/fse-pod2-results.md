# Pod 2 result audit, 2026-11-01

The four-hour workload ended with return code **124**. `ALL_DONE` marked artifact collection, not completion of every stage. Execution used source `97b741887dc0bd4730131aa60a7b5242fef8d5ef`, Python 3.13.18 and an AMD EPYC 9655 host. The process affinity was restricted to the assigned 32 vCPUs. Host load was recorded without rejection, as approved.

## Raw evidence

All **184 received text/JSON/log files** are committed at `2501a38`. This includes the complete 4,790-line speed journal. No row was filtered. Three large files use lossless gzip. Every original file was restored from that pushed commit and matched its original SHA256 and byte count.

- [Restoration manifest](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ae79c80c5736706a1e39df2f69a18e5801ae82d3/benchmarks/baseline/dr151_pod2_raw_manifest_20261101.json)
- [Complete speed journal, gzip](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ae79c80c5736706a1e39df2f69a18e5801ae82d3/benchmarks/baseline/results/dr151-pod2-20261101/speed-results.jsonl.gz)
- [Classification quality records](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ae79c80c5736706a1e39df2f69a18e5801ae82d3/benchmarks/baseline/results/dr151-pod2-20261101/classification-quality.jsonl)
- [Independent mechanical audit](../benchmarks/baseline/dr151_pod2_audit_20261101.json)

The original transfer archive SHA256 is `934dfc9595f91bbb100e96d7eb55f5d114fb5465c9bca2b1367609b38ece366f`. The uncompressed speed journal SHA256 is `90fed831b1e7dacb1fafb955e05929428f1624071915c293d16ec88c4cd69883`.

Model/joblib weights, pipeline inputs and the 2.5 GB fastText binary were not transferred. Their recorded identities and sizes remain, but absent payloads were not rehashed. The truncated API log is not used to reconstruct missing results.

## Stage coverage

A valid cell requires at least three complete rotating rounds, stable input/prediction digests, backend-specific runtime identity, the recorded worker budget, identical fse controls and a global symmetric fse A/A band within 10 percent. The audit recalculates these properties from operations and round records. It does not accept `complete: true` on its own. Invalid cells remain in the summary with failed gates and A/A bands.

| Stage | Execution | Valid cells | Missing or ineligible evidence |
|---|---|---:|---|
| Two-minute A/A probe | Complete | 1/1 | None. Global A/A upper 1.056080, 48 complete rounds. |
| Same-model library encoding | Complete | 8/12 | Four cells fail the global A/A gate. |
| File to persisted vectors | Complete | 28/48 | Twenty cells fail the global A/A gate. |
| Float32 table matrix | Complete | 31/40 | Nine cells fail the global A/A gate. |
| Int8 table matrix | Complete | 12/20 | Eight cells fail the global A/A gate. |
| Classifier inference | Partial | 104/192 | 77 completed noisy cells, one unfinished series, ten unstarted cells. |
| FAISS/hnswlib search | Not started | 0/8 | All eight cells. |
| Float16 table matrix | Not started | 0/20 | All twenty cells. |

All fifteen SetFit datasets reached inference for both POTION models and both worker settings. SetFit contributes 104 valid cells from 180 attempted cells. FineWeb contributes no valid timing cell: its completed series fails the global A/A band, its next series is unfinished, and ten series were not started. All **176 classifier quality/preparation records** were produced, including full FineWeb evaluations. Training quality coverage does not imply inference timing coverage.

Pod 2 did not retain full encode arrays for a new cross-library numerical output audit. These counts describe the paired timing/runtime/input gates above. The fresh README encoding graphics still use Pod 3 and its explicit full-output checks, both scaling curves and selected core settings.

## Rust parallelism

**Zero of the 67 Rust operations are affected by `ea17433`.** Every operation records source `97b7418` and the locked model2vec-rs binary identity. In that source:

- `benchmarks/compare_rust_worker.py` sets both `RAYON_NUM_THREADS` and `TOKENIZERS_PARALLELISM` in the worker environment.
- `benchmarks/quality/pod_pipeline.py` sets both before launching the Rust pipeline.

`ea17433` fixes the separate, newly added Pod 3 large-writer path. Pod 2 never executes that module. Each Rust operation has an explicit annotation in the audit, with its logical raw line and configured worker count. These settings do not measure the exact active tokenizer thread count.

## Classifier configuration and eligible example

Pod 2's fse heads run on **normalized vectors**. They are not raw-mean folded heads. Both recorded POTION float32 parent fingerprints match retained artifact headers containing `normalize@1`. The pinned `Classifier.from_sklearn` defaults to the encoder setting. The [source/parent proof](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/ae79c80c5736706a1e39df2f69a18e5801ae82d3/benchmarks/baseline/dr151_pod2_normalization_evidence_20261101.json) records these retained headers and their exact match to transferred Pod 2 manifest entries. Pod 3 remains necessary for folded raw means versus explicit encoding plus the same head.

The frozen Model2Vec plus sklearn row uses its independently fitted linear head. It is not the same coefficient array as fse's head. The separate Model2Vec classifier fine-tunes embeddings and trains an MLP with PyTorch, by default one 512-unit ReLU hidden layer. Its exported inference pipeline runs in plain NumPy in model2vec 0.10.0 (`model2vec/inference/mlp.py`). This is a different trained model from fse's frozen encoder and linear head. Training times stay separate from inference.

These Subjectivity cells use POTION-8M float32, one worker and the full 2,000-row test split. Training used the same fixed 1,000-row selection. Each row below has its own paired fse baseline. The raw date is 2026-11-01 on the Linux pod above. Medians use fse-a, with fse-b retained separately. Speedups use medians of paired ratios.

| Reference | fse primary texts/s | Reference texts/s | Paired speedup | A/A upper | Series ID |
|---|---:|---:|---:|---:|---|
| Model2Vec plus fitted sklearn linear head | 731,485.29 | 28,788.67 | 25.20098 | 1.079353 | `be8994521d4c496ba64b30b19dd39b2a` |
| Fine-tuned Model2Vec MLP, exported NumPy inference | 733,553.84 | 28,298.73 | 25.86384 | 1.090065 | `37669cc4eba8400d8a06233e8731265a` |
| TF-IDF plus sklearn | 739,874.26 | 116,553.44 | 6.30918 | 1.057965 | `e39f8b7166b14a11904a5df6874fc331` |

The fse and frozen Model2Vec linear models both have accuracy 0.8880 and macro-F1 0.887696 on this test set. The fine-tuned Model2Vec MLP has accuracy 0.8900 and macro-F1 0.889887. These are model accuracy values, not inference speed adjustments. All 32-worker Subjectivity/POTION-8M comparison cells fail A/A and are excluded from this example. Per-arm ranges, RSS, sizes, package paths/versions/hashes, all rounds and all other cells remain in the audit and raw journals.

## Local/pod accuracy check

The complete exported POTION-8M MLP Subjectivity evaluation matches the M1 result: accuracy 0.8900, macro-F1 0.889887 and an identical prediction digest for all 2,000 rows. Model/dataset revisions, training/evaluation text and label digests, fit settings and training method file hashes match. Training used two Torch threads on the M1 and sixteen on the pod. Fitted export files differ, so this does not claim bitwise tensor reproducibility across platforms or other datasets.

[Platform check and exact source records](../benchmarks/baseline/dr151_classifier_platform_check_20261101.json).

## FineWeb quality without a speed claim

All 46,743 held-out pages were evaluated with label score >=3, from the pinned FineWeb-Edu annotation revision. The following float32 model rows and the fastText row come from the committed classification-quality journal. Quantized accuracy deltas remain in that same journal.

| Trained model | Accuracy | Macro-F1 |
|---|---:|---:|
| minishlab/potion-base-8M frozen fse linear | 0.924887 | 0.710607 |
| minishlab/potion-base-8M frozen Model2Vec + sklearn linear | 0.924866 | 0.710647 |
| minishlab/potion-base-32M frozen fse linear | 0.927091 | 0.725128 |
| minishlab/potion-base-32M frozen Model2Vec + sklearn linear | 0.927198 | 0.724808 |
| fastText supervised, wordNgrams=2 | 0.934065 | 0.766207 |
| minishlab/potion-base-8M fine-tuned Model2Vec MLP | 0.936140 | 0.770042 |
| minishlab/potion-base-32M fine-tuned Model2Vec MLP | 0.937980 | 0.790157 |

POTION retains its pretrained input limits, while fastText reads the full page. The frozen encoders use separately fitted linear heads. The fine-tuned MLP changes embeddings and uses its 512-unit hidden head, trained with PyTorch and exported to NumPy. This is a task-quality comparison with different model architectures and input limits. Training durations are reported separately in raw records as different tool chains, not as paired inference speedups. No FineWeb inference series passes the required A/A gate on this pod.


## Loading and RSS measurement boundaries

The audit retains per-arm load-time ranges, peak RSS and artifact bytes. These are descriptive fields from the encoding operations. Passing the encoding A/A gate does not separately establish stable loading latency.

Python `load_s` times contender construction inside `python_measure`, after corpus loading and runtime provenance collection. It is not process startup or network download time. Backend imports and constructor work are inside that timer where they have not already occurred. fse's common provenance collector has already imported fse. Rust's timer wraps `StaticModel::from_pretrained` inside the native worker, excluding its process launch and JSON corpus parsing. These boundaries do not support a matched cold-start speedup claim.

Python peak RSS is the encoder interpreter process high-water mark. Rust peak RSS uses the Python parent's `RUSAGE_CHILDREN`, labelled native Rust process; it excludes that Python parent's own memory. Do not treat these as equal-scope total application memory, sum them as simultaneous peaks, or infer that the Rust controller consumes no memory. Artifact bytes are the finished fse artifact versus the counterpart's safetensors weights, not complete installed library or deployment footprints. Full lifecycle and consistent process-tree memory comparisons remain open.

Source boundaries are pinned at [bench_compare.py](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/97b741887dc0bd4730131aa60a7b5242fef8d5ef/benchmarks/bench_compare.py), [Rust launcher](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/97b741887dc0bd4730131aa60a7b5242fef8d5ef/benchmarks/compare_rust_worker.py) and [native worker](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/97b741887dc0bd4730131aa60a7b5242fef8d5ef/benchmarks/compare_rust/src/main.rs). These qualifications do not invalidate the separately bounded encode timings.

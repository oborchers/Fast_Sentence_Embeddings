# Pod 4c ready

Frozen replacement for the failed Pod 4b run. fse-main owns publication, launch, SSH monitoring, collection and deletion. The worker does not launch a pod.

## Pinned launch artifact

- Pushed runtime source: `8174d0e89354930c54523517e811fac16ab93a58`.
- Source URL: https://github.com/oborchers/Fast_Sentence_Embeddings/commit/8174d0e89354930c54523517e811fac16ab93a58
- Script: `benchmarks/baseline/dr151_pod4c_20261101.sh`.
- Absolute path: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/benchmarks/baseline/dr151_pod4c_20261101.sh`.
- SHA256: `ded971e93ddee49a257aa20689b694a66fc223870ff8d397c0422b4c9163752c`.
- Image: `python:3.13`. CPU type: RunPod `cpu3c`, Linux x86_64, 32 assigned vCPUs. Actual CPU model is recorded on the pod.
- RAM: 64 GB. Container disk: 100 GB, with capacity and free-space preflight.
- Whole-script hard limit: 8,400 seconds (140 minutes), including SSH setup, environment installation and export.
- Maximum planning cost: USD 2.55 at compute no greater than USD 1/hour, with at most 10 minutes beyond the script for startup/collection and 100 GB storage at USD 0.10/GB/month prorated over 720 hours. The coordinator checks the console rate. Delete immediately after collection; sleep continues billing.

## Frozen stages

1. SSH-first setup, pinned environment and Rust build with explicit parallelism.
2. Original bounded quiet-host probe: first plus three warm calls, at least two minutes, A/A upper bound at most 1.10. Calibration plus probe is bounded by 15 minutes. The measured probe has its further three-minute bound. Any rejection or missing passing receipt stops later work.
3. All 21 README encoding series: POTION-8M float32/float16/int8 table at 1/16/32; POTION-8M float32 paired with Python Model2Vec, fixed model2vec-rs and FastEmbed at 1/16/32; static-retrieval-mrl-en-v1 float32 paired with native sentence-transformers StaticEmbedding at 1/16/32. Table/selection uses one common input per model. Each library bar has its own identical-within-pair workload, with count and digest recorded.
4. Corrected folded classifier queue: all 15 SetFit datasets plus FineWeb, workers 1/32. Folded fse linear versus explicit fse encode plus the same head and Model2Vec plus the same frozen sklearn head. Separately, fine-tuned Model2Vec MLP exported to NumPy. Original complete held-out accuracy remains separate from balanced repeated inference inputs.
5. Independent receipt validation and export, ALL_DONE once, then sleep for collection.

Published operations use first plus one warm call, with at least three rotating paired rounds and both fse controls. Every actual first/warm fse call, including explicit unfolded encoding, must last at least two seconds. A/A remains at most 1.10. No bounds, models, datasets, dtypes, worker settings or normalization rules are relaxed. Calibration records are sizing diagnostics, not published rates.

## RAM correction and independent review

Memory admission still requires current usage plus conservative additional allocation below 85 percent of the cgroup limit. Calibration reserves another full correctness output. Typed rejection status 78 carries numeric budget diagnostics and triggers bounded size search. Every configuration in the cohort is actually recalibrated. The 2.5-second margin is preferred, not mandatory; two seconds remains mandatory. A feasible candidate is retained and revalidated before fallback acceptance.

MRL is prepared lazily after earlier POTION cells, so its failure cannot erase completed POTION measurements. Failed and missing cells remain explicit and prevent a complete manifest. Only regenerable prepared text files are retired. Their immutable identity JSON is archived at binding, before any pinned measurement can time out. Regeneration verifies identical digest and file checksum. Temporary correctness output flushing/cache advice occurs outside encoding timing; admission never assumes eviction succeeds. Details and primary Linux documentation: `docs/benchmarks/dr151-pod4c-plan.md`.

Independent source review found two P2 edge cases, both fixed. Follow-up review of 8e2cb61..8174d0e found no issues in fallback revalidation, early receipt preservation, ownership-limited retirement or unchanged gates. 54 focused fixture tests, Ruff and guardrails pass. No local model, timing, notebook or quality workload ran. No full local build or make check was run under the local resource restriction.

## Results to collect

Copy all text/JSON/log files recursively under `/dr151-results`, preserving relative paths, including bootstrap.log, bootstrap-invocation.json, ownership/preflight/plan files and completion receipts. Include:

- `readme-panels/probe.json` and `readme-panels/probe/{speed-results,invocations}.jsonl`.
- `readme-panels/{plan,summary}.json`, `{calibration,calibration-decisions,output-checks,output-check-failures,invocations,speed-results}.jsonl` where present.
- Every JSON under `readme-panels/input-identities/`. These retain provenance after prepared text retirement.
- `folded-classification/{classification-plan,classification-summary}.json`, `{classification-quality,calibration,invocations,speed-results}.jsonl`, and every dataset manifest or other text receipt.

Exclude model weights, caches, correctness .npy vectors and every pipeline-inputs directory. All are regenerable. DR190 artifact blocks remain the log fallback. A hard deadline may leave partial coverage; missing, noisy, short or failed cells never become publishable through export.

fse-main watches every two minutes. After receiving SSH coordinates, dr151-compare also watches read only at intervals no greater than five minutes and reports failure/ALL_DONE immediately. On completion/failure the coordinator collects, verifies transfer hashes and deletes without waiting. The owner checks the local copy within 15 minutes, secures manifest-listed raw files to benchmark-evidence-2.0 and reports the commit. No worker controls pod lifecycle.

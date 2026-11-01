# Pod 4 fix-up plan

Implementation and independent review are in progress. The coordinator launches only after the final pinned script and handoff are published. This document launches nothing.

## Resources and spending

Whole-script hard limit: 140 minutes (8,400 seconds), including SSH setup, environment installation and export. Request python:3.13, 32 assigned vCPUs, 64 GB RAM and 100 GB container disk. The preflight requires at least 100 GB capacity and 25 GiB free. Memory admission checks use current cgroup usage and conservative additional budgets before large allocations. Unsafe cells fail rather than overcommitting the pod.

At compute no greater than USD 1/hour, allow 140 minutes plus 10 minutes for startup and collection. With 100 GB storage at USD 0.10/GB/month, prorated over 720 hours, the estimate is USD 2.535. Planning ceiling: USD 2.55. The coordinator checks the actual console rate. Sleep after ALL_DONE continues billing, so copy the results and delete the pod promptly.

## Encoding scope

There are 21 planned encoding series:

- Nine POTION-8M table cells: float32, float16 and int8 at 1, 16 and 32 workers.
- Nine POTION-8M float32 paired library cells: Python Model2Vec, model2vec-rs and FastEmbed at 1, 16 and 32 workers.
- Three native sentence-transformers StaticEmbedding comparisons using static-retrieval-mrl-en-v1 float32 at 1, 16 and 32 workers.

The table and the 16-versus-32 selection use one shared calibrated input per model. Library bars use separately calibrated paired workloads. Every arm within a pair receives the identical ordered input. Bars may have different input counts and digests. The renderer must disclose each count and use paired median ratios. This is the coordinator-approved contract reflected in the README renderer at da4d379.

Inputs are sampled and shuffled WikiText-103 passages from the pinned training split. Receipts record seed 0, dataset revision, full ordered digest, exact count, distinct count, repetition factor and length statistics. They are explicitly labelled as sampled inputs. Calibration is retained separately and never counted as a published observation.

Every actual fse call, first and warm, must take at least two seconds. A good median cannot hide a short call. At least three accepted rotating paired rounds are required, with a symmetric A/A upper bound at most 1.10. Own backend identity, full output agreement, normalization and input identities are checked. Matrix checks use bounded blocks.

A separately bounded host probe runs first, after minimal calibration. Calibration and probe together have a 15-minute limit. The measured probe has a further three-minute limit and runs for at least two minutes. Any nonzero probe status, missing receipt or false receipt stops later workloads. Host load is recorded and does not reject a shared pod run.

## Classification scope

All 15 SetFit datasets and FineWeb remain queued, at 1 and 32 workers. The four inference arms are folded fse linear, explicit fse encode plus the same linear head, Model2Vec plus the same frozen sklearn head, and separately the fine-tuned Model2Vec MLP exported to NumPy. Raw means load through StaticModel.from_pretrained(repo, normalize=False).encoder.

Model2Vec 0.10.0 fine-tunes embeddings and trains an MLP with PyTorch. Its default head has a 512-unit ReLU hidden layer, then softmax or sigmoid. The exported inference path is NumPy. fse uses a linear head. These are separate trained-model comparisons, alongside the same-head engine comparison. See model2vec/inference/mlp.py at the pinned release.

The complete original held-out split supplies accuracy. Balanced repetitions of that full split, shuffled with seed 0, supply inference timing only. Every folded prediction and explicit Encoder.encode call must satisfy the duration rule. Training and inference times remain separate. Allocation checks include decoded input, models, current memory, unfolded embedding matrices and MLP hidden intermediates.

Planned cell receipts, summary eligibility and raw call durations are independently checked before final success. A failed dataset, missing cell or ineligible summary causes a nonzero stage status. Partial evidence is exported. Export/setup failures produce a small failed DR190 manifest. The wrapper emits the final ALL_DONE marker once per execution path.

## Explicit cuts and limitations

The 2/4/8 worker curve points and 10M write-path repeat were removed by maintainer decision. No MIND, quality batch, ten-model matrix, 100M repeat or local measurements are included. The existing Pod 3 showcase remains a separate cohort. The display rule retains one worker and chooses 32 only if its unrounded primary median beats 16 by at least 1.20, otherwise 16.

The hard limit can leave partial coverage, especially classification. No missing or noisy cell becomes publishable through a successful export. Results go under /dr151-results, with DR190 log blocks as fallback. No new paid run is started by the script generator.

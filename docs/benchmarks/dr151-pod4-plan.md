# Pod 4 fix-up plan, not launched

Status: implementation in progress. No launch is authorized by this document. The source pin and script SHA256 will be added after focused checks and review.

## Limit and cost

Plan a four-hour whole-script limit, including setup and result export, 32 assigned vCPUs, 64 GB RAM and 150 GB container disk on python:3.13. At a quoted compute price no greater than USD 1/hour, four hours plus at most five minutes of collection and USD0.10/GB/month disk prorated over 720 hours costs approximately USD 4.17. Use USD 4.20 as the planning ceiling. The stated USD 2.60 balance is short by up to USD 1.60. This is a bounded run, not a guarantee that every cell finishes. Actual console price must be checked by the coordinator before launch. Sleeping after ALL_DONE continues billing.

## Approved contents

Repeat all 30 README encoding series on one new machine. This includes POTION-8M float32/float16/int8 at 1/16/32, all four library comparisons at 1/16/32 with their own matched fse baseline, and both models' curves at 1/2/4/8/16/32. Five accepted Pod 3 series cannot fill a new machine's gaps. Rust workers must set both Rayon count and tokenizer parallelism explicitly.

Use deterministic sampled and shuffled WikiText-103 training passages. Record source revision, seed, sampled row count, distinct count, repetition factor and full ordered input digest. Select a shared input size for every cell and every arm in that cell. Calibrate outside accepted observations and require every measured fse encoding call to last at least 2 seconds. Prefer the smallest qualifying input for slow counterparts. Never shorten only one library's input. Preserve and reject under-duration observations.

Retain full output checks with bounded-memory blocks. Large matrices must not be promoted wholesale to float64 during validation. Output checking and serialization are outside encoding timing. Record actual per-call durations, not a normalized duration that hides many short calls.

Run corrected folded classification: raw means through fse.StaticModel.from_pretrained(normalize=False).encoder, folded fse versus explicit encoding plus the same linear head, Model2Vec plus the same frozen head, and separately fine-tuned Model2Vec MLP exported to NumPy. Preserve original complete held-out accuracy separately from any enlarged inference-only input. Failed datasets produce a nonzero stage status after preserving partial records.

Repeat the 10M write matrix with native file input, every requested output dtype and durability at 32 workers, and the previously retained reduced one-worker cases. No fixed 10-worker pod setting is restored. Every stage reports failed, noisy or unfinished coverage explicitly.

## Removed from this bounded fix-up

No MIND, MTEB quality batches, full ten-model speed matrix,100M showcase repeat, separate 2M showcase, local timing, or local quality run. The existing100Mresult remains its own Pod 3 cohort. No new paid run starts automatically.

## Known original Pod 3 failures

The scaling stage took one second because it exported 12 curve series already measured by the readme stage. It did not skip the measurements; 10 of 12 curve cells failed the A/A gate. The fix-up must measure its own curves and fail if required curve points are ineligible, rather than equating export success with complete measurement coverage.

All 16 folded-classifier datasets failed because Encoder.from_pretrained refuses normalize overrides for models with native rules. The loop swallowed those failures and returned 0. The corrected loader and nonzero stage propagation have tiny offline regressions.

## Local resource restrictions

No local quality job is currently active. Do not resume one merely because disk space was freed. Only light edits, focused tests on tiny fixtures, JSON audits and script generation are authorized here. Shared load rules still apply. All reports and recovered data stay in this worktree or the durable coordinator folder.

# Pod 4c memory fix and final bounded run

Status: independent review in progress. No launch until the final pinned handoff says Pod 4c ready. The coordinator owns publication, launch, read-only SSH monitoring, collection and deletion.

## Observed failure

Pod 4b passed its unchanged quiet-host probe (A/A upper 1.0668062084472236). All ten expected text files are secured on the evidence branch in e8b338363def0f43a9460ea94a66cd2b96e8bb52. The stage then exited 1 while sizing the 1024-dimensional static-retrieval model at 2,210,000 passages. The child rejected the allocation before decoding or allocating the large output. This was a conservative cgroup admission failure, not an observed kernel OOM kill. POTION already had a fitted common count of 4,830,000, but no paired cells ran because both models were prepared before measurement.

## Correction

The cgroup admission rule remains current usage plus conservative additional allocation below 85 percent of the declared limit. Calibration reserves one extra full output matrix for the later correctness check. A memory rejection has a typed exit 78 and a full numeric budget receipt. Bounded input-size search retains every rejected observation and reduces the candidate count. Every shared model cohort uses its one final count and ordered digest. The preferred calibration margin remains 2.5 seconds. A RAM-constrained candidate may be accepted only after all its configurations actually meet the mandatory two seconds. Every subsequent first/warm call is checked again. Infeasible cells fail explicitly.

Prepare MRL lazily when its first pair is reached. A later model preparation failure preserves earlier POTION measurements, is recorded against every affected cell, and leaves final completion false. No model or dataset is silently removed.

Retire only regenerable text inputs after their last use. Pin the common table input until all table cells finish, and pin MRL input across its three worker settings. Preserve identity JSON byte-for-byte in readme-panels/input-identities. Recreated inputs must match their preserved ordered digest and checksum before their existing exact statistics are reused. This avoids keeping obsolete multi-GB files and their cached pages merely to reuse small preparation receipts.

Full-vector correctness outputs remain temporary. Flush and advise away their cached pages before encoding the other arm, outside every encode timer. Linux documents that DONTNEED is advisory and dirty pages require an earlier fsync/fdatasync. The actual admission check still reads current memory and never assumes reclamation succeeded. See [posix_fadvise(2)](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html) and [fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html).

## Repetitions and scope

The quiet-host probe retains its original first plus three warm calls, at least two minutes of observations and unchanged A/A upper limit 1.10. Calibration retains one first and one warm sizing call. Published encoding and classification operations use one first and one warm call in each of at least three rotating paired rounds. Thus the primary summary still has three independent warm observations, both fse controls are retained, and every actual first/warm fse call must last at least two seconds. All raw observations remain available. This halves the main operation calls versus first plus three warm calls, without changing the minimum paired rounds or eligibility bounds. The README renderer explicitly accepts a nonempty warm list.

The frozen workload grid is retained: POTION-8M float32/float16/int8 at 1/16/32, the three POTION library pairs and native StaticEmbedding pair at 1/16/32, followed by the corrected folded classifier queue. Full original held-out accuracy remains separate from repeated inference inputs. No worker curve, 10M write repeat, 100M repeat, MIND or quality batch is added.

## Resources and budget

Use python:3.13, cpu3c Linux x86_64, 32 assigned vCPUs, 64 GB RAM and 100 GB disk. Keep the 8,400-second whole-script limit, including setup/export, and the existing stage order and deadlines. At compute no greater than USD 1/hour, plus 10 minutes for startup/collection and prorated 100 GB storage at USD 0.10/GB/month, the maximum planning cost remains USD 2.55. No worker launches the pod. A quiet-probe rejection stops all later work. Any timeout aborts the workload before another stage can overlap its descendants. Partial coverage is reported honestly if the hard deadline is reached.

Only tiny offline fixture tests and source review run locally. No local model, notebook, timing or quality workload is part of this fix.

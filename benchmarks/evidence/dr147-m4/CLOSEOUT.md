# DR-147 (MLX-Gate entscheiden und auf dem M4 Pro nachmessen): archival closeout

Oliver permanently ended MLX work on 1 November 2026. He reported that MLX had already been removed from master. This branch preserves historical evidence only. It does not restore MLX, integrate scratch changes or propose a merge. The pull request remains Draft.

The maintainer decision supersedes earlier recommendations to keep MLX optional and every previous resume instruction. No further experiment is authorized. This decision is not a proof that every possible GPU architecture would lose to the CPU.

## Recorded outcome

| Evidence | Final scope |
|---|---|
| Original model/dtype matrix | All 60 cells complete, 180 valid pairs, 3636 accepted passes |
| Representative profiles | Six recordings complete; thread sample shares are not wall-time attribution |
| Timestamp probe | CPU stopped at the paired validity gate; no accepted wall decomposition or MLX trace |
| Stage 1, raw task partition for normalization | Complete, 64 blocks and 3840 accepted passes; independently inspected |
| Stage 2, two real public imports plus three reused dummies | Complete, 64 blocks and 5120 accepted passes; independently inspected |
| Stage 3, pack already waiting compatible tasks | Incomplete: WikiText raw 16/16, WikiText norm 16/16, MASSIVE raw 3/16, MASSIVE norm 0/16 |
| Stages 4 and 5 | Not implemented or run |

Measurements used the actual M4 Pro with 8 performance and 4 efficiency cores, 48 GiB and 12 default workers. The quoted 273 GB/s specification is not a measured sustained bandwidth. CPU10 and CPU12 comparisons remain distinct.

### Stage 1

Using the raw partition for normalized MRL1024/MASSIVE changed MLX from 208 tasks/starts to 89. Absolute medians were 94.321396 ms and 47.9532915 ms. The median paired B/A ratio was 0.5040713740. All 32 paired values favored B, in 16 round clusters.

CPU10 B/A was 1.0023 and CPU12 B/A was 0.9993. The CPU counterexample did not show the same gain. MLX B/CPU12-B still had a paired median of 1.3151925127. No CPU victory was established. The first process RSS peak rose from 1392803840 to 1604599808 bytes in the fresh memory witnesses. These are post-hoc process peaks, not a complete runtime-window bound.

The apparent cost per removed task/start is a coupled effect. It includes changed host partitioning and work distribution. Neither the historical 0.2 ms comment nor a later roughly 0.4 ms quotient measures M4 launch latency.

### Stage 2

The public `mx.asarray(copy=False)` path passed actual alias, content and owner-lifetime witnesses. The tested Python proxy did not produce a consistent end-to-end gain. Median paired C/B was 1.0372606612 for WikiText raw, 1.1079897399 for WikiText norm, 1.0349837847 for MASSIVE raw and 1.0627816980 for MASSIVE norm. Every range crossed one.

Both B anchors and all fixed CPU10/CPU12 A/B comparators remain separate in the archived evidence. Proxy and safety costs were included. These results do not isolate intrinsic API overhead or prove that every implementation of input sharing would fail. C was not carried into Stage 3.

### Stage 3, preserved partial result

The candidate packed already taken compatible jobs on top of B, using original imports and the unchanged pooling kernel. Packing and host-view slicing costs were included in encode. There was no artificial waiting period to fill packs. Original jobs, bounds checks, worker copies and output leases remained active.

Four full-corpus audits, eight fresh B/D memory jobs and two actual multi-pack cancellation witnesses completed before timing. Untimed MASSIVE audits observed 25 raw and 31 normalized pool starts instead of 89. Those audit counts are not fixed counts for every timed pass and are not latency measurements.

Only 35 complete timing blocks were retained, containing 2100 bit-validated passes. Four finite load stops left 108 discarded, bit-validated passes. The last actual stop was `load 5.03955078125 is not strictly below 5`. It occurred before the explicit overnight pause. All partial data remains separate. There is no completed MASSIVE comparison, no Stage 3 final acceptance and no invented normalized MASSIVE result.

## Static archive and provenance

[experiments-archive.zip](experiments-archive.zip) contains unchanged Stage 1/2 reports, the Stage 3 pause handoff, existing scratch drivers and safety helpers, all three existing offline replay scripts, historical plans, inspection records, receipts and accounting. [experiments-manifest.json](experiments-manifest.json) records every payload checksum, its source and the ZIP checksum. The scripts are static historical snapshots inside the ZIP, not new executable benchmark modules.

Compact JSON payloads select already recorded ratios, absolute statistics, memory/contract witnesses and retained Stage 3 round minima. No new measurement or causal analysis was performed for this archive. The Stage 3 full-summary replay was prepared before the pause and still requires a complete stage. It was not used to fabricate a final summary.

The experiments used clean head `ef4d6a7bb33ac65a711458365312a5c0faf0b96a`, fse tree `a9349076427d1811bcd85702f1dd7a9eed2639e0` and native checksum `xxh3-128:2d9d0a93859e347a5220443fab372260`. Exact measured script hashes are in the manifest and archived protocol records. Earlier matrix/profile commits retain their own provenance in the existing evidence files.

Replay scripts retain their original machine paths and depend on the preserved raw journals. Experiment drivers also depend on the historical checkout and prepared environment. Their archival presence does not grant permission to run them or make them compatible with current master.

The actual global charge at termination was 1275.997860708987 seconds, including Stage 3's 89.52163095799187 seconds. Historical matrix accounting remains 523.0340705840008 seconds. Oliver removed the old time caps on 31 October, receipt time 16:41:07 MEZ. That historical waiver does not override the permanent termination.

## Deliberately retained locally

Original reports, full journals, reference `.npy` arrays, models, corpus/token arrays, native binaries and large profiles remain outside the worktree. The failed timestamp trace and tracing hooks remain local and are excluded from Git. No local original was removed or overwritten by closeout.

The archive is documentation and evidence only. No runtime source, artifact identifier, numerical version or release changes. No new measurement, model load, test suite, build or CI wait was performed. Existing Git hooks were not bypassed. ZIP CRC and payload SHA256 checks verify copying, not inference behavior.

# M4 Pro MLX matrix evidence

DR-147 (MLX-Gate entscheiden und auf dem M4 Pro nachmessen), measured 31 October 2026 on an Apple M4 Pro using Codex.

The 60-cell matrix and six requested representative profiles are complete. Per-cause end-to-end wall percentages remain unmeasured/not identifiable, explicitly tabulated in PERFORMANCE.md and the full report. MLX policy remains undecided; PR stays Draft by instruction. See profiles.json and profile_offline.py for sample shares and exact memory peaks.

## Replay without measurements

```sh
python benchmarks/report_mlx_gate.py
python benchmarks/report_mlx_gate.py --json
```

These commands use only the standard library and read `matrix.json`. They never import fse/MLX, encode, download, profile, acquire a lock or reserve budget. The first prints all 60 paired comparisons. JSON output also gives all unrounded A/A ratios, throughput, load groups and separate exploratory pairs. Tests include a case where median(pair ratios) differs from ratio(medians), invalid pass/load/acceptance witnesses and the independently audited counts.

## Compact format and audit boundary

`matrix.json` stores each timed pass duration for 180 accepted paired rounds, all side names, pair load extrema/guard counts, encoder/table fingerprints, conversion settings, revisions and full correctness flags. Replay takes the fastest pass per side, divides times within each pair, then reports median/min/max. A/A controls remain separate. Per-side texts/s is computed per pair before aggregation.

The compact file excludes reference arrays, process dumps and full raw guard streams. It preserves raw directory IDs and original argv. Local raw manifests, JSONL journals, arrays and restart logs remain under `/Users/oliver.borchers/Desktop/Code/fse-m4-scratch`. The coordinator independently audited every accepted manifest/result/journal and recomputed ratios from raw pass minima before extraction. Pair load extrema and guard counts are extracted from the actual before/after pass and side guards. Replay checks counts and strict limits, while the complete local journals retain individual observations.

All 3636 accepted pass records remain here as durations. Every accepted cell has three valid pairs, bit-identical full CPU/MLX outputs, worker/restart determinism, actual full-corpus GPU launches after canary preparation, and zero observed/timed fallback events. Tables use public Encoder.quantize for float16/int8 and return float32. The matrix covers five model/dimension variants, two corpora, raw/normalized output and three table dtypes. No dtype was omitted or refused.

## Provenance and hardware

Frozen comparison base: c754efd382f7b42504d140cabf5aab6e40ad51bb. Forty-three accepted cells retain measured driver d8a61a5c1eaa062e886350dd795121eb27548ed7; 17 retain inspected driver 9fc6a93ebda9ce6640b43d79905f6f7db558ef0a. These are measured commits, not the later evidence/report commit.

Complete src/fse tree in both: a9349076427d1811bcd85702f1dd7a9eed2639e0. Loaded native checksum in both: xxh3-128:2d9d0a93859e347a5220443fab372260. Runtime/build inputs and native arithmetic did not change. The newer driver adds only explicit threshold-decision validation. Full interpreter/import/build/package metadata is in `provenance` and local manifests.

Actual hardware is 8 performance plus 4 efficiency cores, 48 GiB, with default 12 workers. CPU1/10/14 and MLX default plus A/A were measured for every cell; CPU14 is oversubscribed. Actual CPU default plus A/A was measured additionally in the two deciding MRL1024/WikiText float32 cells. This is not the m4-pro-14 specification profile. Actual-variant bandwidth is unverified.

## Baseline and Oliver's explicit decision

The idle observation retains all eleven first-value uptime 1-minute readings, fixed at 30-second intervals over five minutes: 3.30, 2.78, 2.39, 2.39, 2.39, 2.46, 2.22, 2.28, 1.97, 3.24, 4.65. Timestamped values are in `idle`. The predeclared median is 2.39; normal min(baseline+1.5,5) gives 3.89.

Oliver explicitly authorized strict limit 5 and automatic retries only after finite load>=5 stops. Coordinator receipt of this decision: 2026-10-31T13:40:11+01:00. Policy dr147-oliver-limit-5@1, actor Oliver. The real baseline remains 2.39. This is an explicit override, never fabricated formula data. Old discarded rounds retain their original acceptance limits. Defender remained active and no services/settings were changed. CPU snapshots do not prove scan causality.

All before/after pass and side observations must be strictly below the applicable limit. Equality rejects. The normal receipt formula is unchanged without the decision. All seven attempts and fourteen budget events remain separate. The first new load-5 attempt stopped at 5.02001953125, retaining one exploratory pair; its authorized retry completed the missing cells. Own terminal locks were verified against exact invocation owner objects and Finder-released, with immediate/delayed duplicate checks clean. Original raw evidence was never deleted.

## Load regimes and limits of interpretation

Per pair: low means every observation <3.89; elevated-only means every observation >=3.89 and <5; mixed means both; missing observations are unknown. Actual counts: 150 low pairs (50 cells), 30 elevated-only pairs (10 cells), no mixed/unknown. All prior 43 complete cells were classified from their actual raw guards. Accepted pair loads range 2.09033203125 to 4.6435546875.

No accepted cell spans regimes. Accepted within-cell load/spread dependence is therefore not identifiable. Cross-cell/model comparisons are confounded. `exploratory` separately retains valid pairs from incomplete historical cells and the one-round pilot; none contributes to acceptance. Old partial cells lack complete/restart acceptance. Replay keeps their n=1 group width null. Same-cell exploratory higher-load ranges are wider in the two comparisons with n>=2 groups, but tiny samples and different attempt/order/threshold history prevent causal attribution to load or Defender. No discarded round is reused and no inconvenient higher-load cell is dropped.

## Matrix-cutoff costs and later profile evidence

Global actual charge: 532.6694807500007/1800 seconds, remaining 1267.3305192499993. Matrix actual: 523.0340705840008/900, remaining 376.9659294159992. Every reservation settled without overrun. Replay recomputes accounts from the original ledger. Remaining budget is not permission for new profiles or measurements.

Against CPU10 only the raw MRL1024/WikiText float32 cell has median MLX time below CPU time: 0.989344, range 0.962579 to 1.440726. That near tie does not demonstrate a robust advantage. Its actual CPU-default comparison is 1.183355, range 1.088285 to 1.660089. Normalized MLX A/A varies from 0.544806 to 1.844942. No policy conclusion follows.

Selected configuration for the later separately authorized six profiles: MRL1024/WikiText/raw/float32, full 5352 texts, CPU actual default 12 versus MLX default (12 here). It is the closest median comparison to parity and has narrower MLX A/A spread than normalized. The later separately authorized six profiles are now complete; see profiles.json for actual functions/peaks and unresolved wall attribution. See PERFORMANCE.md and the local full report `/Users/oliver.borchers/Desktop/fse-m4-results/DR-147-report.md`.

## Completed profile replay

`uv run --no-sync --offline python benchmarks/evidence/dr147-m4/profile_offline.py /path/to/dr147-profiles-3101e4a7fdd8` reads existing local traces only. It maps sample Unix ms from meta.startTime + samples.time into exact encode windows, then uses tools/profile_top.py Symbols/thread_counts. Inclusive and encode-only self sample shares are distinct from wall time. Exact native peaks come from memray.FileReader.metadata.peak_memory. CPU 1308904403 bytes and MLX 1318022310 bytes are whole-process peaks, not cumulative or per-call allocations. RSS and MLX allocator scopes overlap and are not summed. Full per-thread data remains local; profiles.json retains busiest eight per scope, without assuming thread roles.

The earlier matrix accounting is its historical cutoff. The historical combined root charge at the six-profile cutoff was 593.2869275000012, with matrix fixed 523.0340705840008 and separate six-profile charge 60.617446750000454. No additional measurement occurred in that six-profile stage; the later failed CPU probe is documented below. No ablation or policy decision. The full report and PERFORMANCE.md compare the M1 table with workload/filtering caveats and restrict small-dimension inference to WikiText, preserving the MASSIVE counterexample.

## Failed scratch-only timestamp probe

See wall-probe.json and PERFORMANCE.md for the frozen9a4b9f0 CPU first-round failure. M2/A2=1.109003613433869 exceeded1.10 despite aggregate1.047315807024959 being inside the band. A/A0.9188231166490062 is reported as noise/control only, never as an excuse or a causal timer-cost estimate. All20 timed bits matched, no fallback, load4.17041015625. No later round, MLX trace, retry or adaptation. Coverage is non-representative and no wall percentages are accepted. Scripts/raw trace stay scratch-only, not Git. Current global596.1948255419993, matrix fixed523.0340705840008, six-profile60.617446750000454, failed probe2.9078980419981235. No further measurement or ablation is authorized. PR Draft and policy undecided.

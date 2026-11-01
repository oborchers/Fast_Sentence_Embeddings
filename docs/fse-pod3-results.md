# Pod 3 and MIND rerun evidence, 2026-11-01

Raw publication commit: `cbcad5a`. All 20 Pod 3 files and 11 MIND files from the coordinator's durable copies are preserved under `benchmarks/baseline/results/dr151-pod3-20261101/` and `dr151-mind2-20261101/`. Logs use lossless gzip. All 31 pushed Git objects were restored and matched each original byte count and SHA256 in [the transfer manifest](../benchmarks/baseline/dr151_pod3_mind2_transfer_manifest_20261101.json). Manifest SHA256: `4378c684d4a23eb522e035a242faa5d0aeac60a515aa40804e9ae292bd4f4916`.

## Handoff for the README owner

The100M showcase source is `benchmarks/baseline/results/dr151-pod3-20261101/showcase-100m/speed-results.jsonl`, with `summary.json` and `input-identity.json` alongside it. Series 85778f47f4e04ae6a3c5f2531abdc9c8uses the maintainer's one-run plus one-A/A-repeat exception. Primary fse-a reads 100,000,000 repeated/shuffled WikiText sentence chunks from a native text file, writes int8 through ShardSink durability none at 32 workers, and closes in 35.3027726901 seconds: 2,832,638.6961 texts/s. The symmetric A/A upper bound is 1.0453426396. Durable rows are 0 and no time-until-durable claim is made. The output has 100,000,000 published rows and 26,400,136,189 bytes. This is end-to-end file-to-store throughput on Pod 3, not an encode-only library comparison. Full independent result audit remains open.

The original 30 README-series summary marks five eligible. The worker-scaling summary reuses 12 of those earlier series, with no missing planned identifiers; only two curve points are marked eligible. Its one-second export therefore does not prove a complete publishable scaling curve. Keep noisy rows visible in the evidence guide.

The classification invocation journal records 16 failed datasets and no classifier timing output. All fail on the forbidden normalize override, while the stage incorrectly exits 0. The current branch fixes the loader and makes dataset failures fatal after partial-result preservation. No missing classifier rate is inferred from encoding.

## MIND scope decision

MIND is **not computed: MTEB 2.26.25 MindSmallReranking needs more memory than the 64 GB pod (two attempts, status 137 and earlier ENOSPC)**. No further MIND run is planned. English MTEB means continue to cover 40 of 41 tasks and are labelled accordingly. Neither attempt produced a score.

The rerun log proves task loading completed before status 137. A live SSH inspection before pod deletion observed memory.max=64000000000, memory.peak=64000000000 and oom_kill=1. Those temporary diagnostic files were subsequently lost in the scratchpad cleanup; this sentence preserves the observed diagnosis, not a recoverable raw cgroup artifact. The surviving bootstrap log and disk snapshots are committed. The first failure was disk exhaustion; the second reached the memory limit with disk 33 percent used.

Complete DR190 fitted-artifacts, model-order and mind-load blocks were decoded from the preserved log into [the extracted log artifacts](../benchmarks/baseline/dr151_mind2_log_artifacts_20261101.json). The separately rescued model manifests and fit NPZ files were also lost during cleanup and are not represented as committed payloads. Large generated inputs and safetensors weights were not transferred.

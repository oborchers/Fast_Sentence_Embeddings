# Benchmark evidence 2.0 and 2.1

This branch starts from `benchmark-evidence-2.0` and adds the fse 2.1 README speed studies. It is never merged into the code branch.

The 2.1 archives (`release: "2.1"` in `archives.json`) hold the collected results of the AWS EC2 measurements of 14 November 2026: the README allocation and its profiles on m8a.8xlarge, the ST rerun, and the headline and prediction allocations on m8id.8xlarge. They leave out the sampled output vectors (`.npz`) and contain no model weights. The memray profile keeps its capture files (`mem.bin`). Each archive contains `member-manifest.json` with the SHA256 of every file. `docs/benchmarks/readme-evidence.md` on the code branch maps them to the README rows.

The rest of this file describes the 2.0 evidence. The archives retain original study bytes, source snapshots and per-file manifests. Extracted scripts are historical evidence and are not executed by archival tools.

`archives.json` records each compressed archive size, SHA256, member count and original byte total. Each archive also contains its complete member manifest. Historical filenames and immutable source commits remain preserved inside the bundles and in Git history.

The November speed study includes every terminal status, including rejected controls and failed cells. Only eligible large-input cells support headline speed claims. Short-call diagnostics are a separate workload. Earlier M1 speed measurements and earlier small-input headline comparisons are withdrawn; quality scores remain unchanged.

The two large ST diagnostic matrices remain unchanged in the maintainer collection. Their whole-file hashes, checked transfer receipts and bounded numerical analysis are included. The matrices themselves are not stored in Git. Frozen source and input/model revisions permit regeneration.

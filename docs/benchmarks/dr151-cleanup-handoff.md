# Evidence cleanup handoff

Pod 4 and Pod 4b were collected and deleted by the coordinator. Their complete raw copies are archived, with compact rejection/failure summaries on the code branch. Pod 4c is ready under the frozen handoff at `benchmarks/baseline/dr151_pod4c_handoff_20261101.md`. No worker controls pod lifecycle.

The evidence archive is e8b338363def0f43a9460ea94a66cd2b96e8bb52, containing 477 files and 68,097,415 bytes. All 48 README transfer files were independently verified by their owner. The original eight 1.1 baseline files remain unchanged on the code branch.

## Completed addition after Pod 4 collection

These existing master measurements are archived unchanged. Their original code-branch paths now hold compact summaries:

- benchmarks/baseline/store_sync_20261031.json
- benchmarks/baseline/store_none_20261031.json
- benchmarks/baseline/thread_pool_20261031.json
- benchmarks/baseline/thread_pool_profiles_20261031.json
- benchmarks/baseline/dr190_runpod_20261031.json

Each summary includes the raw size, SHA256 and pinned evidence URL. The global manifest retains every original entry. PERFORMANCE.md and the baseline README distinguish summaries from raw files. Preservation tests compare retained values directly with the archive. A reference audit found no runnable Python consumer of these five raw schemas. This addition does not block Pod 4 or the existing transfer. Keep all 1.1 baseline data and make_v1_fixture.py on master. Bring newer master changes in by a normal merge if needed, never rebase or force-push.

Pod 5 belongs to dr151-quality, which has its own branch. Supply quality pins or scripts when asked. Receive its results through a PR or handoff. dr151-compare remains the only writer of its branch and benchmark-evidence-2.0. No further paid run is authorized here.

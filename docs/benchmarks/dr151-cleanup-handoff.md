# Evidence cleanup handoff

The first Pod 4 was deleted after a rejected quiet probe. All nine expected files were checked and secured in b734a9b78829988b52788be3e3707ac6a68c5c10. Pod 4b (l619gzeqjysylz) is the coordinator-owned retry with the same frozen script. Its script hash is afa684d314a39c2811e0df00b7bf28d6fc6210b9006f8ecc99c65647b882a09f. Its runtime pin is e41060dab1b2a7780b52e80b31050e6ddc6f5b86. Do not change or restart the pod. The coordinator records the retry’s checks and collection hashes in `/Users/oliverborchers/fse-dr151-pod-results/pod4b/status.md`. The owner also runs a read-only SSH watcher every four minutes. Failure or ALL_DONE is reported immediately; the coordinator collects, verifies and deletes without idle billing. Completeness is checked on the durable local copy.

After a verified durable copy, dr151-compare checks completeness within 15 minutes and names any missing files. The coordinator no longer waits for the check before deletion. Then secure Pod 4 raw results on benchmark-evidence-2.0 with a compact code-branch summary and manifest. No worker controls pod lifecycle.

The current evidence archive is ae79c80c5736706a1e39df2f69a18e5801ae82d3. It contains 456 verified files, including all original README transfer files. The receipt pinned to 04e5cbeb1af5d36a45384906d91457d423829680 remains valid. Two CRLF working-copy CSV representations are preserved separately without modifying the canonical Git evidence. All eight original 1.1 baselines remain on master and also have archival copies. Their temporary removal was corrected without changing their Git bytes.

## Approved addition, only after Pod 4 is collected

Move these existing master measurements to the evidence branch, inside the existing comparison PR:

- benchmarks/baseline/store_sync_20261031.json
- benchmarks/baseline/store_none_20261031.json
- benchmarks/baseline/thread_pool_20261031.json
- benchmarks/baseline/thread_pool_profiles_20261031.json
- benchmarks/baseline/dr190_runpod_20261031.json

Retain one short summary for each study, with manifest entries containing SHA256 and the pinned evidence commit URL. Update all references in the baseline README, docs, tests and PERFORMANCE.md. Verify no runnable code still reads removed files. This addition does not block Pod 4 or the existing transfer. Keep all 1.1 baseline data and make_v1_fixture.py on master. Bring newer master changes in by a normal merge if needed, never rebase or force-push.

Pod 5 belongs to dr151-quality, which has its own branch. Supply quality pins or scripts when asked. Receive its results through a PR or handoff. dr151-compare remains the only writer of its branch and benchmark-evidence-2.0. No further paid run is authorized here.

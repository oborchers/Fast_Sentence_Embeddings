Two residual gaps remain in durable failure evidence.

1. **Durable staged card recovery: Keine Befunde.**  
   Staged payloads are synchronized before journal publication. Recovery retains checksum validation and synchronized replacement.

2. **Failed probes and durable evidence before cleanup: two findings.**

   **P2: Timeout handling can overwrite an existing child receipt.**  
   [verify_published_hub.py:154](/Users/oliverborchers/fse-dr157/review-73176a9/benchmarks/verify_published_hub.py:154)  
   If the child publishes its receipt but times out before exiting, the handler discards that receipt. Line 168 replaces it with a generic failure row, then deletes the cache. Observed probe values and completed checks disappear.  
   **Fix:** Read any existing valid receipt after a timeout. Preserve its measurements, append the timeout failure, and persist before cleanup.

   **P2: The newly created evidence directory is not durably linked.**  
   [verify_published_hub.py:225](/Users/oliverborchers/fse-dr157/review-73176a9/benchmarks/verify_published_hub.py:225), [hub_readback_evidence.py:20](/Users/oliverborchers/fse-dr157/review-73176a9/benchmarks/hub_readback_evidence.py:20)  
   Atomic receipt writes synchronize the output directory, but its creation never synchronizes its parent. POSIX therefore does not guarantee that the directory survives a system crash, even after receipts are synchronized and caches removed.  
   **Fix:** Synchronize each newly created directory’s parent before processing repositories.

3. **Exact approval payload and receipt bindings: Keine Befunde.**  
   All 30 literal cards, 135 inventory entries, repository totals, and 15 published SIF receipts agree. Build fingerprints, verification hashes, source revisions, and corpus-selection bindings match. Updated comparison evidence reproduces both selection means. Implementation hashes match the recorded historical code.

   Total payload: **39,426,324,698 bytes**. Small payload hashes were recomputed. Model tensors were not read, so their SHA256 values were checked for manifest/inventory agreement only.

4. **Anonymous fresh-cache, read-only verification, metadata, literal example and run probe: Keine Befunde.**  
   Static inspection confirms isolated child caches, disabled implicit authentication, exact revision/file checks, restricted literal examples, metadata comparison, and numerical probe checks.

No files changed. No tests, network requests, or Hub writes occurred.

Done: offline review and committed-evidence audit.  
Open: two P2 findings; actual post-upload verification awaits approval and upload receipts.  
Needs from you: nothing for this review.
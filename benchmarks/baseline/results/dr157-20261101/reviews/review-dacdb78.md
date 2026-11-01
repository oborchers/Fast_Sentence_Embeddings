The 30 local models have consistent cards and receipts, but the delivery is not ready for exact PUBLIC approval. I found five issues.

I reviewed `81e64c0…dacdb78`, surrounding code, all 30 cards, and all 45 top-level model JSON files. I checked small-file and fit-state hashes. I did not modify files, run tests, read large model tables, or upload anything. Earlier review conclusions were not used.

1. **Numerical construction, source-specific SIF provenance, quantization and quality: Keine Befunde.**

   Each source has its own fit-state hash and pinned source revision. Quantization follows weighting and component folding. All card metrics match their verification receipts.

   | Source | Raw float32 Pearson | SIF float32 Pearson |
   |---|---:|---:|
   | GloVe | 44.46410 | 69.22434 |
   | Google News | 61.54132 | 70.88642 |
   | ParaNMT | 79.81644 | 78.11215 |
   | Paragram | 51.45857 | 74.43794 |
   | fastText | 48.52845 | 72.40582 |

   These are recorded results, not measurements rerun during this review. Across both methods, the largest recorded int8 change is 0.01151 Pearson points.

2. **Evidence integrity, recovery and complete delivery: two findings.**

   **Fehler: committed evidence does not identify the delivered fits.**  
   Location: [hub_classic_delivery.py:47](/Users/oliverborchers/fse-dr157/review-dacdb78/benchmarks/hub_classic_delivery.py:47), [committed selection.json:19](/Users/oliverborchers/fse-dr157/review-dacdb78/benchmarks/baseline/dr157/corpus-comparison/selection.json:19).

   Code evidence: `fit.get("selection_xxh3") != digest(selection_path)` raises `"model fit does not reference this corpus selection"`.

   Every generated fit references `xxh3-128:95d35ee34446d86a310e93e563e46425`. The committed selection hashes to `xxh3-128:29245cc411f158ca895dc4eba156c467`. The local selection matches the fits. Committed comparison receipts also lack the implementation identities now required by the validator. Consequently, the committed evidence cannot validate the current delivery.

   **Fix:** commit the exact updated selection, comparison receipts and identity-extension evidence. Include the final small build and verification receipts.

   **Warnung: recovery journals can outlive their staged payloads after a system crash.**  
   Location: [hub_classic_card_publish.py:50](/Users/oliverborchers/fse-dr157/review-dacdb78/benchmarks/hub_classic_card_publish.py:50).

   Code evidence: `(stage / "README.md").write_text(text)` precedes `_write_json(journal, ...)`. The preparation and attributes files use the same unsynchronized pattern.

   The journal is explicitly synchronized, but the staged payload files are not. Directory synchronization alone does not guarantee file-content durability on POSIX. Recovery can encounter a durable journal whose staged bytes fail their recorded checksums.

   **Fix:** flush and synchronize every staged payload before publishing the journal.

3. **Card licenses, attribution, claims and public 2.0 example: Keine Befunde.**

   All 30 examples use the intended public repository and pin installation to the available 2.0 source API. I checked their syntax and API compatibility statically.

   Cards distinguish raw/SIF and storage formats, disclose corpus-selection bias, and make no measured speed claims. GloVe and fastText terms match their primary sources. SIF cards disclose wordfreq’s additional data terms. Unknown vector permissions remain explicitly unknown. See [GloVe](https://github.com/stanfordnlp/GloVe), [fastText](https://fasttext.cc/docs/en/english-vectors.html), and [wordfreq’s notice](https://github.com/rspeer/wordfreq/blob/master/NOTICE.md).

4. **Exact PUBLIC approval inventory, sizes and SHA256 references: one finding.**

   **Fehler: the approval artifacts have not been delivered.**  
   Location: [freeze_hub_publication.py:18](/Users/oliverborchers/fse-dr157/review-dacdb78/benchmarks/freeze_hub_publication.py:18), [POST_UPLOAD.md:29](/Users/oliverborchers/fse-dr157/review-dacdb78/benchmarks/baseline/dr157/POST_UPLOAD.md:29).

   Code evidence: `manifest = freeze_manifest(...)` generates the required artifact. The documented readback command requires `publication-manifest.json`.

   Neither `/Users/oliverborchers/fse-dr157/publication-manifest.json` nor `publication-inventory.md` exists. Existing build receipts provide xxh3 hashes, not the promised complete SHA256 approval inventory. Their recorded payload sizes total **39,426,324,698 bytes** across 30 repositories.

   **Fix:** finish the offline freeze and deliver both immutable artifacts, including a SHA256 reference for the manifest itself. Preserve the fastText publication hold. This requires no upload.

5. **Fresh post-upload readback and durable failure evidence: two findings.**

   **Fehler: failed probes discard the measurements explaining their failure.**  
   Location: [hub_public_readback.py:93](/Users/oliverborchers/fse-dr157/review-dacdb78/benchmarks/hub_public_readback.py:93).

   Code evidence: `delta = pearson - expected["pearson_x100"]` is followed by a `raise ValueError(...)` before returning the probe record.

   A tolerance or bitwise failure therefore loses the observed Pearson, difference and vector hash. The report shows the difference as `"not run"` although the probe ran. Cleanup then removes the downloaded cache.

   **Fix:** record observed and expected measurements before deciding pass/fail. Return the failed probe record or preserve it in a structured exception.

   **Fehler: result rewriting can destroy the only per-repository failure receipt.**  
   Location: [verify_published_hub.py:175](/Users/oliverborchers/fse-dr157/review-dacdb78/benchmarks/verify_published_hub.py:175).

   Code evidence: `result.write_text(json.dumps(row, ...))` runs after `remove_owned_cache(case, owner)`.

   This truncates the child’s existing result. Interruption during rewriting can leave invalid JSON after cache removal. Timeout failures also remain only in memory until this write. Aggregate evidence uses another truncating write.

   **Fix:** atomically publish and synchronize each result before cache cleanup. Atomically update cleanup outcomes and aggregate evidence afterward.

   Otherwise, static inspection found no defect in cache isolation, anonymous access, literal-example restrictions, revision checks, exact file checks or owned-cache cleanup. Actual post-upload execution remains pending, as required.

Done: independent read-only review of all five categories.  
Open: five findings above.  
Needs from you: nothing.
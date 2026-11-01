# Classic word-vector Hub preparation evidence

DR-157 (Ausgewählte Modelle auf den Hugging Face Hub laden), draft PR #201 (2.0: Prepare the classic word-vector models for the Hub).

All 30 local builds and five source verifications are complete. Large model files
and fitted-state arrays remain under `/Users/oliverborchers/fse-dr157/models`.
This committed folder contains the small, durable results and literal card texts.
It excludes downloaded models, caches and corpus inputs.

- `corpus-comparison/`: both complete GloVe fits, STS-B Pearson/Spearman, all 26
  NanoBEIR task results and the selection receipt. FineWeb wins the equal-weight
  mean, 58.38476 versus WikiText's 58.14279. `candidate-audit.txt` records full
  bitwise table reconstruction. `identity-extension.json` binds unchanged measured
  preprocessing and metric implementations to their code hashes.
- `builds/`: all 30 file manifests, five source-specific fit receipts, five
  seven-row verification reports and execution receipts. The historical diagnostic
  fits on evaluation texts only to reproduce 1.x. It is never shipped.
- `cards/*.txt`: the exact 30 generated README texts, stored without Markdown code
  formatting changing their literal examples. Preview originals are also under
  `/Users/oliverborchers/fse-dr157/cards`.
- `superseded-prototype/`: small evidence from the retired initial GloVe prototype.
- `cleanup.json`: targeted removal journal for this worker's own intermediates.
- `reviews/`: completed independent read-only reviews. All findings are resolved. Final review da79a11 reports Keine Befunde;
  `reviews/resolution.md` records the fixes.
- `POST_UPLOAD.md`: exact PUBLIC approval and fresh readback protocol.

All published repositories must be PUBLIC from creation. Oliver's approval of
cards, repo names, licenses and every payload file is the final gate before any
Hub write. FastText is approved for 2.0; DR-146 (Über FastText in 2.0 entscheiden (12f)) is Done. Option B requires release 2.0.0 and regenerated cards before upload. No upload, Hub write, token use or pod lifecycle action occurred here.
Actual fresh Hub readback remains pending until approval and completed uploads.

`report.md` is the final local report. `make-check-da79a11.txt` records 9,167 passing tests and 514 skipped tests. The exact PUBLIC approval payload is `publication-manifest.json`, with its SHA256 in `publication-manifest-sha256.json`; `publication-inventory.md` lists all 135 files and actual sizes. Total: 39,426,324,698 bytes. Content, repo list and licenses are approved under Option B. The final release-card payload is pending release 2.0.0.

Option B is prepared in `RELEASE_CARDS.md`. The current basis manifest changes only
six FastText hold fields; its SHA256 is
`ac85cae41b05788bc278d68fd3231489de1615c7c15b8b79229844c3d0dabe69`.
Source e72f708 passes 19 focused tests and the full gate (9,186 passed, 514 skipped).
The final independent release-runtime review reports Keine Befunde. Production
release rendering and public readback remain gated on release 2.0.0 and uploads.

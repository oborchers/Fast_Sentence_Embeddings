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
- `reviews/`: completed independent read-only reviews. Final source/card review
  and the final repository gate are recorded separately after completion.
- `POST_UPLOAD.md`: exact PUBLIC approval and fresh readback protocol.

All published repositories must be PUBLIC from creation. Oliver's approval of
cards, repo names, licenses and every payload file is the final gate before any
Hub write. FastText additionally awaits DR-146 (Über FastText in 2.0 entscheiden
(12f)). No upload, Hub write, token use or pod lifecycle action occurred here.
Actual fresh Hub readback remains pending until approval and completed uploads.

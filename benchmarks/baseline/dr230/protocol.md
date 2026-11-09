# Civil Comments classifier selection protocol

Work: DR-311 (Toxizitäts-Classifier aus google/civil_comments außerhalb von fse
trainieren und mit Modellkarte vorbereiten). Correctness and quality evidence
only. No speed comparison, no Linux comparison, no paid resources, no upload.

The configuration and training code were committed before the selection run.
The following explicit branch rule was reported to fse-main on 2026-11-09,
before inspecting any candidate result. It documents the rule already executed
by `training.medium` in commit c2fd1ac0.

- Compare the two encoders on the same deterministic 200,000-row training sample
  and the full official validation split, never test.
- For each encoder, select C in {0.1, 1, 10} by greatest validation macro ROC AUC.
  Ties choose the smaller C.
- Bekko wins only if its best macro ROC AUC is at least 0.01 above POTION's best.
- **If Bekko wins, that small Bekko classifier is final.** Full Bekko training is
  not admitted under the 8 GiB memory plan. No full POTION classifier is fitted
  or compared against a winning small Bekko classifier.
- **If POTION wins, only POTION is eligible for full training.** Admission needs
  a successful small numerical import check, a conservative memory plan, at least
  one hour plus 15 minutes for finalization before the earlier of the total
  three-hour budget and 18:00 local time, and 40 GiB free disk.
- A completed full POTION head replaces small POTION only if its validation macro
  ROC AUC is no lower. The C and thresholds remain those fixed during selection.
- A failure or exhausted budget is inconclusive. Keep raw files and do not
  automatically repeat a long run. No finished card is delivered from a failed run.
- The official test split is accessed only after the final model and numerical
  checks are frozen. An exclusive marker prevents a second evaluation.

Targets are annotator fractions binarized with >= 0.5. Inference for the six
retained labels is strictly > 0.5. The sample keeps natural class proportions.
Balanced class weights follow the newer approved ticket comment. No calibration
claim is made. Missing target classes abort. Undefined precision and F1 become
zero. Fewer than 100 test positives mark a label unreliable.

The removed severe_toxicity label's count is measured from the pinned training
split and recorded in the receipt. The model card renders that count and all
evaluation numbers from the JSON, rather than from handwritten results.

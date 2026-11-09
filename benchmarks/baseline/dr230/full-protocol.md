# Full Bekko follow-up protocol

Work: DR-311 (Toxizitäts-Classifier aus google/civil_comments außerhalb von fse
trainieren und mit Modellkarte vorbereiten).

Oliver's decision, relayed by fse-main on 2026-11-09, supersedes the earlier
small-only Bekko branch rule. Train logistic regression on all 1,804,874 training
rows using the same frozen Bekko embeddings. Do not train the encoder. Keep the
full width of 1,024 dimensions. All six existing labels remain in this comparison.

Rules fixed before the full run:

- Admit memory without executing training. The process plan must be at most
  16 GiB and must fit the current macOS availability. Do not stop other apps or
  release resources owned by another worker.
- Use the existing pinned dataset and encoder, natural class proportions,
  balanced weights, seed 17, StandardScaler, and one-vs-rest logistic regression
  with L-BFGS and max_iter=1000. Compare C in {0.1, 1, 10} using full validation
  macro ROC AUC. Ties retain the smaller C. Convergence warnings invalidate a fit.
- Use the saved fse head's float32 probability outputs for full validation. For
  the gain comparison, evaluate the small head on the same probability contract.
  Full replaces small only at a gain of at least 0.01 in macro ROC AUC.
- After C selection, propose each label's maximum-F1 threshold on validation.
  Use strict float32 `score > cutoff`, with the highest cutoff on an F1 tie.
  Report ROC AUC, average precision, precision, recall and F1 for every label,
  for both small and full. Include default and proposed cutoff results.
- Oliver then decides whether to retain six labels or only toxicity and insult.
  Do not choose the labels in this run. The official test stays held.
- New budget: two hours from actual admitted run start, including embedding.
  Stop at safe operation boundaries without extension or automatic retries.
  Finish heavy work before 18:00 local time. Keep partial outputs on failure.

The new command uses its own new external directory, preserving the small run,
its models, selection and active test hold. No finished card or test result is
produced by the full-comparison command.

## Memory design

Installed scikit-learn 1.9.1 accepts float32 for LogisticRegression.fit with
L-BFGS. In its `_linear_loss.py`, weights are cast to X.dtype before matrix
products. Pointwise gradients use the same float32 data. This permits the same
solver without converting the full matrix to float64.

The float32 training matrix needs 7,392,763,904 bytes. A 1.5 GiB reserve covers
the interpreter, target and pointwise arrays, BLAS workspace, and scaling
batches. The conservative peak is 9,003,376,640 bytes (about 8.39 GiB), below
16 GiB. A 9 GiB RSS watchdog is stricter than that upper plan limit.

Parquet is read in batches. Encoding maps, writes, flushes and releases one
output batch at a time. StandardScaler uses partial_fit and in-place transforms
in batches on the float32 file. Each one-vs-rest fit reads one matrix without
an additional scaler transform. The matrix is unmapped before loading Bekko
for validation. No corpus-wide list of Python strings is retained.

float16 storage is permitted by Oliver but unnecessary here: float32 storage
avoids additional feature rounding and still meets the process memory target.
Admission checks actual host availability and the existing 40 GiB disk reserve.
`full-memory-admission.json` records the no-run result. A failed admission means
the full run has not started and its two-hour budget has not begun.

# Streaming SGD follow-up protocol (not started)

Work: DR-311 (Toxizitäts-Classifier aus google/civil_comments außerhalb von fse
trainieren und mit Modellkarte vorbereiten). Oliver's decision was relayed by
fse-main on 2026-11-09. This document proposes the next bounded run. It starts
no training, changes no existing model, and does not release the official test.

The current L-BFGS job keeps its existing deadline of 12:25:34 CET. Its native
watcher remains active. Do not change, extend or restart that job. fse-main
reported an I/O bottleneck from its own observation. This is the reason for
the new streaming design, not a published throughput claim.

## Workload and fixed optimizer

- Same pinned Civil Comments train split, all 1,804,874 rows, frozen full-width
  1,024-dimensional Bekko encoder, six labels in the established order.
- Train six independent binary `SGDClassifier` estimators per alpha. This is
  one-vs-rest for the six indicator columns, not a single multiclass softmax.
- Fixed settings: `loss="log_loss"`, `penalty="l2"`, `fit_intercept=True`,
  `learning_rate="invscaling"`, `eta0=0.01`, `power_t=0.5`, `average=True`,
  `early_stopping=False`, `tol=None`, `shuffle=False`, `random_state=17`,
  `n_jobs=1`. Never call `fit` between `partial_fit` calls.
- Exactly **three complete training epochs** per candidate. `partial_fit`
  processes one supplied batch once; `max_iter` does not define these epochs.
  Do not select an epoch or a learning-rate schedule on validation afterwards.
- Batch size 8,192. A train-only RNG seeded with `17 + epoch` permutes contiguous
  block order and row order within each copied block. All candidates see the
  same order. No full-matrix random gather or corpus-sized Python text list.
- Supply `classes=[0, 1]` at each estimator's first update. A batch need not
  contain both classes. Keep all counters across updates and record rows/epochs.
- Feed each copied batch to all alpha/label estimators before releasing it.
  This shares disk reads rather than rescanning the matrix for each label.
  Fixed decay and coefficient averaging reduce dependence on early updates;
  this is a fixed training choice, not a convergence guarantee.

## Train-only standardization and weights

The existing L-BFGS working `train.npy` is **already standardized in place**.
Its `scaler.joblib` was fitted using `StandardScaler.partial_fit` on the raw
training vectors, before transforming that file. Reuse this pair read-only after
the L-BFGS process exits. Verify shape, dtype, complete train counts, scaler
feature count and `n_samples_seen_`, and record their provenance before SGD.
Do not fit a fresh scaler on standardized values or apply the old scaler twice.
If that provenance cannot be verified, stop and report before training. A fresh
raw matrix would require a separately declared preparation step, not silent reuse.

For each label j, use the complete pinned training counts, never batch or
validation frequencies:

`w_j(1) = N / (2 * positive_j)` and
`w_j(0) = N / (2 * (N - positive_j))`, with `N = 1,804,874`.

Pass the fixed dictionary `{0: w_j(0), 1: w_j(1)}` as that estimator's
`class_weight`. Never pass `class_weight="balanced"` to `partial_fit`, and do
not also apply the same weights as `sample_weight`. This is the balanced
formula on the entire training split. Missing train classes stop admission.

## Alpha grid and decision

Use the three predeclared L2 strengths corresponding to the earlier C grid:

| C reference | alpha = 1 / (C * N) |
|---:|---:|
| 0.1 | 5.5405529693485523e-6 |
| 1 | 5.5405529693485529e-7 |
| 10 | 5.5405529693485526e-8 |

This maps regularization conventions for mean log loss and train-wide balanced
weights. It does not make finite-epoch SGD identical to the L-BFGS solution.

Select alpha only by macro ROC AUC on the entire official validation split,
using fse float32 probability outputs for the comparison to the fixed small
Bekko head. Ties choose larger alpha (the smaller C reference). Replace small
only at a gain of at least 0.01. Report every alpha's status and metrics.

Only candidates with all six estimators trained for all three epochs and a
complete validation evaluation are eligible. Interrupted training, missing
label updates or incomplete validation are `not_finished`. Since training is
interleaved, an interruption before epoch three completes leaves every alpha
unfinished. If validation stops between candidates, compare only completed
ones and identify the grid as incomplete. With no eligible candidate, keep
small and report the full experiment `inconclusive`. No retry or extension.

After alpha selection, tune each selected label's cutoff only on validation:
maximum F1 under strict `float32 probability > float32 cutoff`, highest cutoff
on an F1 tie. Report per-label ROC AUC, average precision, precision, recall,
F1 and cutoff for small and full, with default and tuned threshold results.
These tuned validation results are not independent estimates. Oliver retains
the six-versus-two-label decision. Test stays held until a separate signal.

## Import and numerical checks

Code inspection corrects the proposed importer limitation:
`src/fse/_head/_sklearn.py` already lists `SGDClassifier`; `_sgd_link` maps
`log_loss` to sigmoid. `_sklearn_ovr.py` accepts independent affine binary
sigmoid estimators with classes `[0, 1]`, not only LogisticRegression.

The chosen trainer holds six separately weighted online estimators rather
than a fitted sklearn OneVsRestClassifier object. Do not manufacture wrapper
attributes. Assemble the multilabel head explicitly with
`Classifier.from_arrays(encoder, A64, b64, labels=labels, link="multilabel")`.
Fold the original scaler **in float64 before one float32 rounding**:

`A64[j] = coef_j.astype(float64) / scaler.scale_`

`b64[j] = float64(intercept_j) - A64[j] @ scaler.mean_`

Use each estimator's finished averaged `coef_` and `intercept_`. Reference
predictions stack the six actual sklearn estimators' positive-class outputs,
not a softmax normalization. On a fixed validation probe, compare logits and
probabilities using `tests/support/head_bound.py` (folded/unfolded bounds and
derived sigmoid probability tolerance). Use native float32 encoder vectors
upcast to float64 for the exact affine reference, as the bound defines.
Check masks only outside the bound's ambiguity interval, plus strict equality,
complete/standalone bit-preserving reload, labels and stored thresholds.
Do not weaken the bound or substitute a fixed absolute tolerance.

Before full training, run only the approved small adapter/import probe inside
the new budget. A failed probe stops the run, with no automatic full attempt.

## Resources, estimate and stop rule

New directory and new **one-hour budget from fse-main's next explicit signal**.
The L-BFGS signal is not authorization for SGD. All preparation, probes,
updates and validation count. Heavy FIFO wrapper, nice 10, one CPU thread,
40 GiB disk reserve, no paid resources, no speed claim, no src changes.

Estimated wall time: **20 to 40 minutes**, not measured or guaranteed. The
grid shares three training passes through the matrix, rather than one pass
per optimizer iteration. The fixed epoch count is not increased if time remains.

Map and release one window at a time. One copied 8,192 x 1,024 float32 block is
32 MiB; a row permutation may need a second block. Never retain a full training
mapping across epochs or load the entire matrix into an ndarray. Read labels
in matching windows. Release training buffers before validation and reference
checks. Estimated peak below 3 GiB, with an enforced **4 GiB RSS stop** covering
the executing Python child, including fse conversion/reference work.

Check budget and resources between batches and each label update, before a
new epoch and before validation. At the fixed deadline, stop the verified own
process group. Save per-candidate progress and checkpoints, then mark incomplete
work `not_finished`. Nonfinite inputs abort the run. A nonfinite/divergent
candidate is invalid, cannot be selected, and is not retried. No undocumented
early stopping or reduced epoch count is counted as complete.

## Primary sources

- [scikit-learn scaling strategies](https://scikit-learn.org/stable/computing/scaling_strategies.html):
  streaming input and incremental estimators form the out-of-core pattern.
- [SGDClassifier.partial_fit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.partial_fit):
  one batch epoch per call, explicit first-call classes, caller-controlled
  stopping. Installed 1.9.1 source explicitly rejects the balanced string and
  recommends precomputed class weights for `partial_fit`.
- [compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html):
  the balanced formula used above, including dictionary weights.
- [StandardScaler.partial_fit](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.partial_fit):
  incremental statistics for data too large for one fit.
- [SGD user guide](https://scikit-learn.org/stable/modules/sgd.html):
  feature scaling and averaged SGD. The API documents the chosen inverse-scaling
  schedule. No source guarantees this recipe's convergence or quality.

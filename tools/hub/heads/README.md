# Preparing a Civil Comments classifier

This tool trains outside fse with scikit-learn. It never uploads. The library
continues to run finished models only. All downloads, vectors, estimators and
prepared Hub folders must live outside this checkout.

The reviewed `civil-comments.json` pins the dataset and both MIT encoders.
Use the project environment (`uv sync --all-groups`, Python 3.13 for this run).
Follow the repository's shared heavy-lock workflow for setup and every command
below. Use nice 10. These are correctness runs, without speed claims.

```sh
uv run python tools/hub/heads/run.py preflight --work "$HOME/fse-dr311"
uv run python tools/hub/heads/run.py prepare --work "$HOME/fse-dr311"
uv run python tools/hub/heads/run.py probe --work "$HOME/fse-dr311"
uv run python tools/hub/heads/run.py select --work "$HOME/fse-dr311"
uv run python tools/hub/heads/run.py verify --work "$HOME/fse-dr311"
uv run python tools/hub/heads/run.py finalize --work "$HOME/fse-dr311"
```

`prepare` requires a new output directory. It reads immutable public cached
snapshots when available and otherwise downloads to the run's own cache. It
prepares only train and validation. `probe` checks both adapters, imported
coefficients, persistence and score stores using synthetic labels on validation
texts. These probe scores are not quality results.

`select` takes the 200,000 smallest SHA256 digests of `17:position`, then restores
source row order. It repeats sampling to check determinism. Each retained label
must have positives and negatives in train and validation. It fits a
StandardScaler plus six one-vs-rest logistic regressions, using natural class
proportions and balanced class weights. Convergence warnings invalidate the run.

Selection uses full-validation macro ROC AUC. Each encoder tries C in ascending
order (0.1, 1, 10), and ties retain the smaller C. Bekko replaces POTION only with
at least 0.01 higher macro ROC AUC. Test is unavailable to selection. Full train
is never compared across encoders. If Bekko wins the small comparison, that small
Bekko classifier is final and full POTION is not fitted. If POTION wins, full train
is admitted only for POTION, with at least one hour plus a 15-minute finalization reserve before
the earlier of the budget end and 18:00, and an 8 GiB memory plan. It uses the selected C and replaces the small model only if full-validation
macro ROC AUC is no lower. The receipt records skipped or completed full training.

`verify` imports the final estimator with explicit label names. On validation
texts it checks sklearn logits and probabilities against the data-dependent
bound in `tests/support/head_bound.py`. Complete and standalone reloads must
preserve labels, masks and probability bits. Pure score stores use folded
inference. Combined stores use unfolded inference. Each is compared only to
its matching path, with IDs, thresholds, vectors and store verification.

`finalize` requires unchanged verified selection and artifact hashes. It writes
an exclusive test marker before accessing the official test split, then runs
the saved standalone fse model once. All test metrics come from those fse scores.
Average precision is the non-interpolated precision-recall summary, not
trapezoidal area. Zero predicted positives produce precision and F1 of zero.
An absent target class aborts instead of producing an undefined metric. Labels
with fewer than 100 test positives are marked unreliable.

`civil_comments.py` owns the dataset adapter. `training.fit` and `metrics.metrics`
also accept a one-dimensional binary target with two class labels. This recipe
uses only one six-label head per run. The Civil Comments card remains separate
from these reusable fitting and scoring functions.
Binary metrics require the actual predicted class indices through `predictions=`.
They never infer a class from rounded probabilities that could both equal 0.5.

The three-hour budget starts at preparation's first download. Checks between
downloads, vector batches and fits enforce time, 40 GiB free disk, 8 GiB peak RSS
and an 18:00 local cutoff. A fit already in progress finishes at a safe boundary.
A watchdog also checks peak RSS during fits. Full-train scaling uses batches
in place to avoid a second full matrix. Admission records conservative memory
estimates before vectorization. No automatic retry or budget extension is allowed. Failure retains partial
files, raises an error and produces no completed card. Do not remove exclusive
selection or test markers to repeat a failed run.

Only the atomically published `prepared/` directory is a completed deliverable.
Verification folders have an incomplete card. Failed finalization retains
`pending/` for diagnosis, which must never be published. The final receipt and
manifest become authoritative together when `pending/` is renamed to `prepared/`.

The generated card comes from `metrics.json` only:

```sh
uv run python tools/hub/heads/cards.py "$HOME/fse-dr311/prepared/metrics.json" /tmp/card.md
```

`SHA256SUMS.json` covers both prepared folders, configuration, selection,
verification and metrics. Only small receipts, the manifest and card belong in
`benchmarks/baseline/dr230/`. Keep all large files outside Git. The standalone
folder is the intended distribution artifact, with no `-standalone` name suffix.
Publication and a README model link are separate work.

## Validation-only threshold proposal

The coordinator currently holds the official test while reviewing per-label F1
thresholds. `thresholds.py` uses only the complete official validation split and
the already verified saved standalone model. It writes a proposal receipt and
validation scores outside the checkout. It changes neither saved model.

```sh
uv run python tools/hub/heads/thresholds.py --work "$HOME/fse-dr311" --preflight
# Run under the shared heavy-lock wrapper, as for the other correctness steps:
uv run python tools/hub/heads/thresholds.py --work "$HOME/fse-dr311"
```

Thresholds maximize F1 separately for each label under `score > float32 cutoff`.
Equal scores stay together. Ties in maximal F1 select the highest representable
cutoff, so the rule is deterministic. A small validation probe checks the actual
`with_thresholds` masks and unchanged score bits. These validation values are
optimized on the same split and are not independent holdout estimates.

An active external `test-hold.json` prevents `finalize` from reading test. The
current hold requires an explicit signal from fse-main. This proposal command
does not release it or change the frozen estimator or encoder selection.

Sources: [dataset and CC0 terms](https://huggingface.co/datasets/google/civil_comments),
[annotator fractions](https://www.tensorflow.org/datasets/catalog/civil_comments),
[POTION source](https://huggingface.co/minishlab/potion-base-8M),
[Bekko source](https://huggingface.co/hotchpotch/bekko-static-embedding-v1),
[class weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

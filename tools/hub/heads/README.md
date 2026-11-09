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

## Full Bekko follow-up

The separately approved full-training comparison uses the same frozen Bekko
encoder on every training row, with a new two-hour budget and a reviewed memory
plan. It never reads test or changes the small run's artifacts. It reports both
variants and leaves the six-versus-two-label decision to Oliver.

```sh
uv run python tools/hub/heads/full_bekko.py \
  --parent "$HOME/fse-dr311" --work "$HOME/fse-dr311/full-bekko-01" --preflight
# Only after admission, under the shared heavy-lock wrapper:
uv run python tools/hub/heads/full_bekko.py \
  --parent "$HOME/fse-dr311" --work "$HOME/fse-dr311/full-bekko-01" \
  --signal-at "$DR311_SIGNAL_UNIX"
```

The output directory must be new. An unsuccessful memory admission starts no
work. The command pins sklearn 1.9.1's float32 L-BFGS path, streams input, and
scales one float32 working file in place. It compares C before fitting thresholds.
Only a full-validation macro ROC AUC gain of at least 0.01 replaces the small
head. `benchmarks/baseline/dr230/full-protocol.md` specifies the prior rules.
`DR311_SIGNAL_UNIX` is the Unix time of the coordinator's explicit start signal.
The two-hour budget is anchored to that signal, never to a later process start.

Sources: [dataset and CC0 terms](https://huggingface.co/datasets/google/civil_comments),
[annotator fractions](https://www.tensorflow.org/datasets/catalog/civil_comments),
[POTION source](https://huggingface.co/minishlab/potion-base-8M),
[Bekko source](https://huggingface.co/hotchpotch/bekko-static-embedding-v1),
[class weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## Streaming SGD follow-up

The approved `sgd.json` fixes three epochs and three alpha values, with six
independently weighted online logistic classifiers per alpha. It reuses the
completed train-only scaler and already standardized full Bekko matrix read-only.
Each bounded copied batch feeds every candidate before release. It never loads
the full training matrix into memory or fits a scaler on standardized values.

```sh
uv run python tools/hub/heads/sgd.py preflight \
  --parent "$HOME/fse-dr311" --source "$HOME/fse-dr311/full-bekko-01" \
  --work "$HOME/fse-dr311/sgd-probe-01"
# Under the approved shared heavy-lock wrapper, at most five minutes:
uv run python tools/hub/heads/sgd.py probe \
  --parent "$HOME/fse-dr311" --source "$HOME/fse-dr311/full-bekko-01" \
  --work "$HOME/fse-dr311/sgd-probe-01"
# Only after the coordinator's separate SGD start signal, under the same wrapper:
uv run python tools/hub/heads/sgd.py full \
  --parent "$HOME/fse-dr311" --source "$HOME/fse-dr311/full-bekko-01" \
  --work "$HOME/fse-dr311/sgd-full-01" --signal-at "$DR311_SGD_SIGNAL_UNIX" \
  --probe-receipt "$HOME/fse-dr311/sgd-probe-01/probe.json"
```

The functional probe uses 256 training vectors and 32 training texts. It checks
all three affine imports against the unchanged numerical bound, persistence and
score stores. It reads neither validation nor test, and makes no quality choice.
Full training requires that successful receipt and unchanged tool fingerprints.
The one-hour budget starts at the new explicit signal, including lock waits.
The deadline and 4 GiB peak RSS watchdog terminate incomplete work without retry.

Progress and epoch checkpoints remain outside Git. A candidate is eligible only
after three complete epochs and complete validation. Only completed validation
receipts count in a partial grid. No completed candidate means inconclusive and
keeps the small head. Validation selects alpha by macro ROC AUC, ties favor larger
alpha, and a gain of at least 0.01 is required to replace small. Only then are
per-label maximum-F1 thresholds proposed, with highest cutoff on ties. Existing
artifacts remain unchanged and official test stays held. The full prior rules
and primary sources are in `benchmarks/baseline/dr230/sgd-protocol.md`.

## Approved two-label finalization

After reviewing full validation, the maintainer selected only the existing
`toxicity` and `insult` estimators at alpha 5.540552969348553e-7. No refit or
new alpha choice occurs. `sgd_final.py` imports those two rows with `from_arrays`,
checks the unchanged numerical bound and bit identity with the selected six-label
columns, stores their validation-only maximum-F1 cutoffs and verifies both reloads.

```sh
uv run python tools/hub/heads/sgd_final.py \
  --parent "$HOME/fse-dr311" --source "$HOME/fse-dr311/sgd-full-01" \
  --work "$HOME/fse-dr311/final-toxicity-insult-01" --preflight
# Only with the explicit test signal, under the shared heavy-lock wrapper:
uv run python tools/hub/heads/sgd_final.py \
  --parent "$HOME/fse-dr311" --source "$HOME/fse-dr311/sgd-full-01" \
  --work "$HOME/fse-dr311/final-toxicity-insult-01" --test-approved
```

This finalizer uses the remainder of the signal-anchored one-hour window and the
4 GiB watchdog. It writes the global exclusive `test-started` marker before any
test access, then evaluates the saved standalone model exactly once. The general
test hold remains active for other commands. No retry is permitted after that
marker exists. The final model and stored cutoffs never change after test.
The completed `prepared/` directory contains both artifacts, a card generated
from recorded metrics, MIT licenses and a SHA256 manifest. The card documents
all five removed labels, validation evidence, optimizer and uncalibrated scores.

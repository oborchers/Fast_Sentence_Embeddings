# Preparing two Aegis prompt heads

This recipe trains outside fse. It prepares complete safety and category heads
that share an encoder in `fse.Heads`. There is no upload operation. Data, vectors,
estimators, compatibility environments and weights remain outside the checkout.

The reviewed `aegis-prompts.json` freezes source revisions, SHA256 checksums,
normalization, ten categories, target definitions, subsets, grid and selection.
It retains all eligible rows and exactly recognized AART prompts. Only prompt
strings are encoded. Responses and refusal files are excluded from model input.

Use this worktree's Python 3.13 environment. Run builds, probes and training
under the shared heavy lock with nice 10. Set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` and `MKL_NUM_THREADS=1`.

```sh
uv run python tools/hub/heads/aegis_run.py preflight --work /external/aegis-01
uv run python tools/hub/heads/aegis_run.py prepare --work /external/aegis-01
uv run python tools/hub/heads/aegis_run.py probe --work /external/aegis-01
uv run python tools/hub/heads/aegis_run.py run --work /external/aegis-01
```

Before `run`, read the twenty train-only records in `sample-private.json` locally.
Record each ID with a boolean `category_plausible` in `sample-reviewed.json`.
Keep prompt text out of the review report. This is a plausibility check, not proof
of annotation provenance. The source paper defines the human conversation-level
annotations. Include the review in `data-report.json` before freezing the run.

`prepare` reads all raw splits for structural filters and cross-split leakage.
Test text had already been inspected by the planner for counts and structure.
This access is disclosed rather than claiming an untouched official test split.
The selection phase reads only prepared train and validation. It uses all train
rows, never a training subsample. Validation AP chooses C independently per head.
Bekko replaces POTION only if both specified AP gains reach 0.01.

The probe checks both encoders with synthetic targets and validates the full
working-set RAM estimate before real fitting. Peak resident memory is capped at
3 GiB, free disk must remain at least 40 GiB. The training budget is 90 minutes
from heavy-lock acquisition, excluding queue wait, by maintainer decision on
2026-11-09. Downloads inside the run, vectorization, fits, bootstrap, export and
verification count toward that budget. Check it at safe operation boundaries.
Stop as inconclusive on exhaustion. Never extend or automatically retry.

The `night` command can run missing synthetic probes before fitting, inside the
same 90-minute budget and held lock. Earlier preparation and failed small API
probes are separate preflight work, not additional training attempts. The budget
uses the acquired lock owner's immutable file timestamp, including child startup,
imports and an intervening uv rebuild. Use a `dr318-` worker name for this recipe.

`run` selects, verifies and freezes both complete artifacts. It checks a fresh
PyPI fse 2.0.0 environment on validation probes, then evaluates test once using
saved fse heads. It writes group-bootstrap AP intervals for all three subsets,
for both heads. No model or threshold changes follow the test evaluation.
Exclusive selection and test markers prevent automatic repeat evaluation.

Only a successful run creates `prepared/`, containing complete artifacts, cards,
receipts and `manifest.sha256`. An incomplete staging folder is not a deliverable.
Frozen weight files use hard links during staging to avoid doubling their disk
space. Cards remain separate copies. Never modify a frozen weight file in place.
Copy only small text-free reports and cards to `benchmarks/baseline/dr318/`.
Raw inputs, score matrices, estimators and weights must never enter Git.

Prior art and source contracts, checked 2026-11-09:

- [Aegis paper, conversation annotation](https://arxiv.org/html/2501.09004v1#S4.SS1.SSS1):
  explains why response-bearing labels require a separate prompt-only report.
- [NVIDIA dataset card at the frozen revision](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/blob/d86bb8bedff51d25ac834ab7838f1cc61acb7a2c/README.md):
  raw JSON avoids implicitly mixing refusal files into train and validation.
- [scikit-learn Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html):
  non-interpolated AP defines selection and reporting, rather than trapezoidal area.
- [AART same-commit source copy](https://github.com/xxxx-dddd/aart-ai-safety-dataset/tree/fcdb2a8a3343a992cd8cbfd53eac797ac047c26f):
  pins exact-match identification and preserves the license provenance limitation.

File-only recovery is available after a successful frozen test evaluation:

```sh
uv run python tools/hub/heads/aegis_run.py stage --work /external/aegis-01
```

This command checks the saved selection, frozen weight hashes and the original
one-test marker. It does not fit, predict or evaluate again. It creates a fresh
`pending-staging-02/` and, on success, `prepared/`; the failed `pending/` and all
original receipts remain. Existing targets are refused. This authorized recovery
has a separate five-minute file-only limit and retains the 40 GiB disk floor;
it does not reset or extend the training budget.

Staged JSON uses compact presentation, checked against its decoded original,
including canonical float representations. URLs keep ordinary slashes. The
30-character source scan has one explicit exception: a window wholly contained
in one of the five dataset/license URL literals from the pinned recipe config.
The list freezes before any prompt is read. `text-scan-diagnosis.json` records
every ignored window, source split/index and deliverable occurrence. Other source
text, including material adjacent to those URLs, remains a rejection. This is an
explicitly approved deviation from the original unconditional substring rule.

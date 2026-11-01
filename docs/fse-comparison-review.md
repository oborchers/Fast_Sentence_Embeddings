# Independent comparison methodology review

The read-only Codex CLI review used gpt-6-astra at high reasoning effort.
It reviewed `efa2632` to `1f17754` on 2026-11-01 in a detached checkout.
It ran no tests and changed no files. Pending measurements were outside its findings.

All five findings were verified against the source:

| Finding | Correction |
|---|---|
| Search lifecycle omitted common normalization time | Record preprocessing separately and measure startup through first search. |
| Finished encoders and heads lacked enforced source lineage | Verify bound encoder provenance during preparation and prediction. Bind fitted heads to the float32 training encoder or native weight/config digests. |
| Non-fse quality rows omitted an identity field required by resume | Persist the normalized specification in every completed row. |
| Speed/classifier reports could combine changed configurations | Require stable per-backend identities and identical fse A/A configurations. Keep diagnostic A/A on one compatible setup. |
| Speed/classifier records lacked executed-runtime identity | Record loaded package path and content fingerprint. Require a Rust build manifest tied to source and binary digests. Qualify historical records. |

The reviewer reproduced the published complete-cell medians and both quality means.
It found no further issue with numerical recall, MIND coverage, profile claims,
the tariff scenario or the qualified elapsed-time scenario.
Static inspection found no remaining optional-import portability defect.
This is not execution verification or a completed-results review.
After the corrections, 60 focused regression tests and seven pinned MTEB adapter tests pass. Ruff and guardrails pass. The full check at `4e888dc` passed under the heavy lock with `PYTEST_N=4`: 9,121 passed and 521 skipped in 239.19 seconds. The locked Rust build passed and its source/binary manifest was sealed. [Validation record](../benchmarks/baseline/dr151_validation_20261101.json). A final complete-results gate remains required.
All missing measurements and the final claims review remain open.


## Backend identity follow-up review (68e1bae)

An independent read-only Codex review used gpt-6-astra with high reasoning on
a detached committed snapshot. It ran no tests and changed no files. The full
review is retained locally at `/private/tmp/dr151-runtime-review.last.md`.
The review found no defect in counterpart package selection, installation
path, installed-file hashing or the verified Rust binary identity.

All three reported findings were verified and fixed in `c3005fa`:

- Error: precision reports could group different executing fse runtimes under
  the same cached reference. Precision deltas now require equal package,
  version and content hash. Quality also rechecks the shared reference before
  each model and after each task.
- Error: classifier preparation captured own backend identity only when
  writing completed predictions. Each backend now captures it before loading,
  training and prediction, then verifies the original identity before append.
- Warning: FAISS/hnswlib search verified the search backend but left the common
  fse encoder hash unchecked. Every search record now rechecks that hash.

Eleven focused runtime/protocol tests pass. Ten adapter tests pass through
unittest in the pinned MTEB environment. The attempted pytest command in that
environment failed because pytest is not installed; no dependency was added.
Ruff check, format check and guardrails pass. The full local gate is queued
under heavy.lock after window6. Final full-scope evidence and claims still
require review when collection finishes.

## Pod and native-classifier review, 2026-11-01

An independent read-only gpt-6-astra review at high effort examined source `56c36c6`.
All four findings were verified against executable imports or pinned Model2Vec source.
The classic tokenizer now imports from the same package path as comparison children.
Cell validity includes spread across all fse controls, alongside within-round A/A.
POTION preparation precedes the probe, which requires at least 120 seconds of observations.
Native early-stopping records describe Model2Vec 0.10.0's 10,000-row validation cap.
Regression tests cover package imports, between-round drift and an undersized probe.
The full gate before these corrections passed 9,224 tests with 524 skips.
The corrected source still requires its own focused and full validation receipts.

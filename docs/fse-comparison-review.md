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
Regression checks and the final full gate remain required after these corrections.
All missing measurements and the final claims review remain open.

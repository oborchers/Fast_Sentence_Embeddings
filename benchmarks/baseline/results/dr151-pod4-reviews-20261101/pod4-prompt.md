Review the committed Pod4 implementation at534154e in this detached snapshot. Inspect changes from e1423be to534154e, including surrounding helpers. Do not modify files, run tests, install packages, import project/model libraries, launch benchmarks, or access the network. Standard-library source/data inspection is allowed.

Scope: pod4*.py, make_fixup_pod.py, pod_duration.py, the changed pod_pairs/README output checker/classifier child, Rust worker --lines and buffered output changes, and focused regressions. Ignore the older draft Pod4 plan text: final approved scope is21encoding cells (nine POTION-8M dtype/1,16,32 table cells and12library pairs, including native ST static retrieval), no2/4/8curve, no write stage, and the complete folded classification queue.

Check each category and report findings or explicit No findings:
1. Every accepted fse encode/predict call lasts at least2seconds individually. Calibrations are retained and separate. Per-cell shared input, digest, output checks, native normalization and own backend identity remain valid.
2. Memory and disk safety: large inputs and vector validation, output serialization/temporary buffers, finalizer must never read/export giant input files. Only Linux may execute the workloads. CPU affinity and Rust tokenizer/Rayon settings honor1,16,32budgets.
3. Folded raw means use the valid StaticModel API, same linear coefficients for unfolded and Model2Vec comparisons, full held-out accuracy separate from balanced timing repetitions, fine-tuned NumPy MLP explicitly different. Failure/noisy/missing cells must produce nonzero stage status.
4. The generated standalone script pins source, sets up SSH first inside its hard timeout, keeps results under/dr151-results, emits DR190blocks then ALL_DONE, has8400seconds including setup/export,100GBdisk and no automatic paid launch. The short quiet-host probe must gate later spending. State any paths that can report success despite missing work or fail in practice.

For every finding give severity, file:line, quoted evidence and a concrete fix. Report in English. No invented timing or memory numbers.

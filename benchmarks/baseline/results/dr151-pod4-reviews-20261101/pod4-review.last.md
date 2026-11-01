I found five issues in `e1423be..534154e`. The encoding matrix matches the approved 21 cells.

1. **Call duration: P1, high. The unfolded fse arm bypasses the two-second gate.**

   [pod_pairs.py:97](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-wt/benchmarks/quality/pod_pairs.py:97) checks only `if row["label"] in ("fse-a", "fse-b")`.

   Classification also runs the fse `encode-matmul` arm. Its short calls can enter valid rounds because neither the round gate nor classification summary checks their duration. Furthermore, its timer includes encoding and matrix multiplication, so it cannot establish the duration of the individual `Encoder.encode` call.

   **Fix:** Apply duration validation to every fse arm. Record and validate the unfolded encoder call separately from its matrix multiplication. Add a regression where controls qualify but `encode-matmul` does not.

   **No findings** in returned calibration measurements being retained separately, shared input binding, digests, full output validation, native normalization, or encoding backend identity.

2. **Memory and disk safety: P1, high. Classification sizing does not budget the largest comparator’s memory.**

   [pod4_classify.py:26](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-wt/benchmarks/quality/pod4_classify.py:26) limits only `count > 10_000_000`. Calibration then measures only `"folded"` at line 84.

   The same expanded input reaches the unfolded encoder and NumPy MLP. Those arms materialize full embedding matrices. The MLP also creates full hidden-layer intermediates. The child retains every input string, while the parent retains the original dataset.

   There is no classification memory admission check. The encoding check, `count * width * 4 * 4`, also excludes input strings and existing process memory. Successful folded calibration therefore does not establish that subsequent comparators fit.

   **Fix:** Check available memory before expanding inputs and before each comparator. Include parent memory, decoded strings, model storage, embedding matrices, and overlapping MLP intermediates. Reject an unsafe cell before allocation.

   **No findings** in blockwise vector validation, Rust buffered serialization, or finalizer exclusion of `pipeline-inputs` before reading files. Pod4 entry points enforce Linux. Affinity and Rust tokenizer/Rayon settings respect the requested CPU caps.

3. **Classification correctness and status: P1, high. An ineligible summary can still produce a successful stage and manifest.**

   [pod4_classify.py:124](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-wt/benchmarks/quality/pod4_classify.py:124) records failure only for `if not done["complete"]`. It subsequently calls `publish_summary(args.output)` without checking summary eligibility.

   [pod_folded_summary.py:139](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-wt/benchmarks/quality/pod_folded_summary.py:139) can reject a completed series through `"runtime_stable"` or `"predictions_stable"`. That rejection does not raise. Pod4 then writes `stage-complete.json`.

   The finalizer treats JSON summaries as text. Consequently, a summary containing `eligible: false` can coexist with a final manifest containing `complete: true`.

   **Fix:** Return the summary and require every planned classification cell to exist and be eligible before stage completion. Make the finalizer validate these receipts independently.

   **No findings** in the raw-mean StaticModel APIs, shared linear coefficients, complete held-out evaluation, balanced timing repetitions, or explicit distinction of the fine-tuned NumPy MLP. The queue includes all 15 SetFit datasets and FineWeb.

4. **Standalone wrapper and spending gate: two findings.**

   **P1, high. Missing or failed probes can permit classification spending.**

   [make_fixup_pod.py:30](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-wt/benchmarks/quality/make_fixup_pod.py:30) exits only for `"$readme_status" -eq 75`, then starts classification.

   A calibration exception before the probe, a timeout, or the explicit `readme_status=124` path bypasses that gate. Classification starts without evidence of a successful quiet-host probe.

   The probe is also not independently short. [pod4_readme.py:153](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-wt/benchmarks/quality/pod4_readme.py:153) passes the entire stage deadline alongside `minimum_seconds=120`. Persistent noise can consume the remaining README allocation.

   **Fix:** Run a separately bounded probe after minimal calibration. Require its explicit passing receipt before later workloads. Reject missing receipts and every unsuccessful probe exit.

   **P2, medium. Export failure still produces `ALL_DONE` without a final manifest.**

   The inherited [make_folded_pod.py:113](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-wt/benchmarks/quality/make_folded_pod.py:113) runs the finalizer under `timeout`, then unconditionally executes `printf 'ALL_DONE\\n'`. It ignores the export status. Early cache failures also emit `ALL_DONE` without DR190 blocks.

   Thus, `ALL_DONE` cannot establish successful export. Successful finalization also emits the sentinel twice.

   **Fix:** Capture export status and emit a small failed DR190 manifest on export or early setup failure. Give the wrapper sole ownership of the final sentinel.

   **No findings** in source checkout pinning, SSH preceding environment setup inside the workload timeout, results placement, or automatic paid launch. The wrapper allocates 8,400 seconds including setup/export. The 100 GB disk requirement is declared in its header, not enforced by a capacity preflight.

The focused regressions do not cover these failure paths. I inspected source only, without tests, benchmarks, package installation, project/model imports, or network access.

done: Completed the committed source review.  
open: Five findings above.  
needs from you: nothing.
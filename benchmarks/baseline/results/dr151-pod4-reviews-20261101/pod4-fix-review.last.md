**Do not launch yet. The probe fails with the fresh cache required by the wrapper.** Other failure paths still permit unnecessary spending or incomplete evidence.

I reviewed `bda68c3` and `442c3c6` against `534154e`, including the inherited shell template and string transformations. **Inspection was source-only.** I changed no files, ran no tests, imported no project/model libraries, and started no builds or network/model workloads.

1. **Actual call durations: two P2 findings.**

   **Missing warm receipts can pass.** At [pod_duration.py:14](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/pod_duration.py:14), nested encoder durations extend `values` before its minimum length check. An `encode-matmul` row with `minimum_call_seconds=2`, `first_predict_s=3`, `encoder_call_seconds=[3]`, and no `warm_seconds` passes. The nested first call supplies the second element.

   **Fix:** Validate the required outer call count before adding nested durations. Require matching encoder receipts for every recorded pass.

   **Correctness calls escape the duration gate.** [pod_readme_output.py:100](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/pod_readme_output.py:100) accepts the returned operation without checking `qualifies`. Its fse calls can fall below two seconds while subsequent paired measurements pass. Final validation scans only `speed-results.jsonl`.

   **Fix:** Apply the duration requirement to the correctness-call receipts and include them in final validation.

   No findings on median averaging for complete timing receipts. The unfolded encoder is timed separately. Historical rows retain the zero/default-minimum behavior.

2. **Memory admission: P1, large source allocations precede admission.**

   [pod4_readme.py:141](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/pod4_readme.py:141) loads the corpus before the first admission check. `pod4_inputs.py:54` materializes the complete file with `read_text().splitlines()`.

   Classification has the same gap at [pod4_classify.py:182](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/pod4_classify.py:182). Advancing `inputs()` loads FineWeb through `prepare_fineweb.py:69`, including Arrow data, Python strings and normalized copies. Preparation then allocates training features before the expanded-input admission runs.

   **Impact:** These allocations can exhaust the cgroup before the new checks can reject them.

   **Fix:** Check current cgroup usage and conservative allocation bounds before source decoding and preparation. Use bounded batches where a whole-source bound is unavailable.

   No separate finding in the added timed-child budgets: they include current usage, decoded strings, embedding matrices and NumPy MLP intermediates.

3. **Completion evidence: P1, final validation does not bind raw receipts to expected cells.**

   [pod4_receipts.py:48](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/pod4_receipts.py:48) checks expected cells against saved summary flags. Separately, `long_calls()` accepts any nonempty collection of qualifying fse operations.

   With eligible summaries present, a journal retaining only one qualifying control can pass. The validator does not require every cell’s controls, counterparts, rounds, runtime identities or prediction digests. It also ignores stage-completion files.

   **Impact:** Missing raw evidence can coexist with a successful final manifest.

   **Fix:** Reconstruct eligibility from raw receipts grouped by cell and series. Require the expected arms, rounds, duration counts, identities and completion receipts before emitting success.

4. **Probe and spending control: two P1 findings.**

   **The fresh-cache probe cannot reach measurement.** [pod4_readme.py:142](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/pod4_readme.py:142) calls `calibrate()` before preparing POTION. Calibration first calls `bind_input()`. That reaches `sampled_lengths()` and `text_lengths.counter()`, which requests the model through `cached_snapshot(..., local_files_only=True)`.

   The model download occurs later, inside `calibration()` at line 68. The inherited wrapper requires a fresh `/dr151-cache`, so the earlier lookup fails.

   **Fix:** Restore `ensure_model(POTION, "float32", options)` before the first input preparation.

   **README calibration failures still start classification.** [make_fixup_pod.py:38](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/make_fixup_pod.py:38) stops only for exit status 75. Exceptions from calibration normally produce status 1. Status 124 also continues into classification. The general failure check appears after classification finishes.

   **Fix:** Stop on every nonzero README status before invoking classification. Require its successful completion receipt there. This defect remains from the base.

   No finding in the explicit probe-status and missing-probe-receipt checks: those stop subsequent stages.

5. **Wrapper, export and deadline: two findings.**

   **P1: Nested timeouts can exceed the 8,400-second limit.** [make_fixup_pod.py:23](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/make_fixup_pod.py:23) always starts a 900-second timeout, even when the remaining global allowance is shorter. The inherited Rust timeout similarly uses a fixed 600 seconds.

   GNU `timeout` creates its own process group. These nested groups can survive the outer timeout at [make_folded_pod.py:104](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/make_folded_pod.py:104). Their inherited output pipe can keep `tee` open, delaying finalization beyond the limit. Passing a deadline argument does not bound downloads and preparation before Python checks it.

   **Fix:** Clamp each stage timeout, including its kill grace, to the remaining global allowance. Use supervision that bounds the entire process tree, pipe and export fallback.

   **P2: Failure export depends on an unchecked file write.** [make_fixup_pod.py:61](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-fix-wt/benchmarks/quality/make_fixup_pod.py:61) inserts the finalizer file before disk preflight but never checks whether its creation succeeded. A full filesystem can leave it empty or truncated. The failure paths then invoke that same file and print `ALL_DONE` without producing a failed manifest.

   **Fix:** Check finalizer creation. Provide a small failure emitter through stdin that does not depend on writing a file.

   No findings on the 100 GB capacity check, absence of automatic launch, or single wrapper-owned `ALL_DONE` on paths where finalizer creation succeeds. The sentinel replacements match the inherited template.

Done: Source-only review of both commits and inherited wrapper behavior.  
Open: Eight findings, including the fresh-cache launch blocker.  
Needs from you: Nothing.
Two launch blockers:

- **P1, stale launch artifact:** [benchmarks/baseline/dr151_pod4_20261101.sh:187](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-final-wt/benchmarks/baseline/dr151_pod4_20261101.sh:187) pins `3b4b308`, which lacks the fresh-cache preparation and independent reconstruction fixes. Line 235 also retains the timeout/tee pipeline, and failure paths still depend on `/dr151-finalize.py`. **Minimal fix:** regenerate from the corrected source, then update the handoff’s pin and SHA256.

- **P1, README descendants can overlap classification:** [benchmarks/quality/make_fixup_pod.py:55](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/pod4-final-wt/benchmarks/quality/make_fixup_pod.py:55) makes nested timeouts use `--foreground`, which signals only the direct child. A timed-out README operation can leave descendants running, including the Rust worker launched at `benchmarks/compare_rust_worker.py:38`. Classification then starts at `make_fixup_pod.py:43`. The supervisor cleans the workload group only when the whole script finishes. **Minimal fix:** terminate and reap README descendants within a bounded cleanup period before continuing authorized classification.

Done: reviewed `442c3c6..8f9bc58`, source inspection only.  
Open: two launch blockers.  
Needs from you: nothing. Source-only review complete; no tests, builds, project/model imports, or network/model workloads ran.
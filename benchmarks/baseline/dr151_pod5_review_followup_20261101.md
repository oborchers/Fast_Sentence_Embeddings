**Four findings are resolved. The RAM finding remains open.**

1. **Still open: effective RAM and container peaks.** The preflight checks cgroup limits, but silently uses host RAM and zero usage when expected files are absent. It assumes the container’s cgroup is mounted at the root. See [make_classics_pod.py:59](/private/tmp/dr151-pod5-review-final/benchmarks/quality/make_classics_pod.py:59). Peak collection can likewise return no peak without failing at [pod5.py:32](/private/tmp/dr151-pod5-review-final/benchmarks/quality/pod5.py:32).  
   **Fix:** resolve or explicitly validate the container cgroup layout. Reject missing limit, usage or peak evidence. Account for applicable ancestor limits. No measured 64 GB peak is established.

2. **Resolved: timeout cleanup and durable stage receipts.** The supervisor records the active stage before launch. It gives children separate process groups, sends TERM then KILL, waits for the child, and records exit 124. Its deadline precedes the outer timeout by twenty seconds. See [pod5.py:48](/private/tmp/dr151-pod5-review-final/benchmarks/quality/pod5.py:48) and [pod5.py:71](/private/tmp/dr151-pod5-review-final/benchmarks/quality/pod5.py:71).

3. **Resolved: salvage of partial fitted states.** Finalization copies existing fit files and preparation records before hashing, regardless of workload status. Completion requires both exit zero and a completion receipt. See [generated script:40](/private/tmp/dr151-pod5-review-final/benchmarks/baseline/dr151_pod5_20261101.sh:40) and [generated script:61](/private/tmp/dr151-pod5-review-final/benchmarks/baseline/dr151_pod5_20261101.sh:61).

4. **Resolved: historical evaluation threading.** The script passes two workers at [generated script:151](/private/tmp/dr151-pod5-review-final/benchmarks/baseline/dr151_pod5_20261101.sh:151). The unchanged evaluator applies this to Torch and BLAS at [run_mteb.py:456](/private/tmp/dr151-pod5-review-final/benchmarks/quality/run_mteb.py:456) and [run_mteb.py:469](/private/tmp/dr151-pod5-review-final/benchmarks/quality/run_mteb.py:469). Corpus pins, full paragraph count, fit arithmetic and dependency locks remain unchanged.

5. **Resolved: allocation and collection arithmetic.** The deadline derives from the required allocation epoch, including provisioning. The allowance is 215 minutes through export, ten for collection and fifteen for audit, totaling 240 minutes. Estimated cost is USD 3.9496 at the stated rates. Coordinator rate verification and deletion remain required. See [generated script:4](/private/tmp/dr151-pod5-review-final/benchmarks/baseline/dr151_pod5_20261101.sh:4) and [handoff:17](/private/tmp/dr151-pod5-review-final/benchmarks/baseline/dr151_pod5_handoff_20261101.md:17).

Scope remains three SIF-10 float32 models, each with 40/41 English tasks, including STS, and 13 NanoBEIR tasks. Optional rows are excluded. Completion follows all 159 validated results at [pod5.py:211](/private/tmp/dr151-pod5-review-final/benchmarks/quality/pod5.py:211). No worker lifecycle calls were found.

The committed script exactly matches the generator for source pin `0a79ef7157d64d5f93a6c1c3ef6ee7cd9af88067`. Its SHA256 matches the handoff: `180f03cad0e96772fbaa3d879046a83e945f300ffbaafdaf0cf7dbe61646cbe8`. Pinned executable source matches HEAD.

Inspection encountered sandbox errors creating Git cache and heredoc temporary files (`Operation not permitted`). The read-only checks completed using an alternative invocation.

Done: read-only review; no files changed, tests or workloads run.  
Open: RAM preflight and peak evidence must fail closed.  
Needs from you: nothing.
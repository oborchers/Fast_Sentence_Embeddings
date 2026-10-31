# MLX gate on the M4 Pro

DR-147 (MLX-Gate entscheiden und auf dem M4 Pro nachmessen) uses the frozen base
`c754efd382f7b42504d140cabf5aab6e40ad51bb`. This driver was written on an M4 Pro
using Codex. It changes no inference code. Completed matrix evidence is in
`benchmarks/evidence/dr147-m4/`; `benchmarks/report_mlx_gate.py` replays it without
measurements. Six representative profiles are recorded separately at evidence commit 937860e.
Per-cause wall attribution remains unresolved; MLX policy and final inspection remain open.

`benchmarks/bench_mlx_gate.py list` lists all 60 required cells. Each crosses five
model/dimension variants, two corpora, raw/normalized output and three table dtypes.
Output rows are float32. Real public encoders retain conversion rules and use
`Encoder.quantize`, including for WordPiece models. Models and datasets load only
from the immutable snapshots in the cache receipt.

For every cell the CPU uses 1, 10 and 14 workers, MLX its resolved default, with an
identical A/A control for each. The deciding MRL1024/WikiText float32 cells also
include CPU's actual default. This M4 Pro has 8 performance and 4 efficiency cores;
14 workers are oversubscribed. Bandwidth is unverified. No result supports an
unmeasured best-CPU or 14-core-hardware claim.

## Before execution

The coordinator must inspect the committed driver before authorizing the pilot.
A valid idle receipt is required separately. The first preparation receipt is
invalid and remains discarded evidence. The historical second baseline was
2.59716796875, with strict threshold 4.09716796875. The later Option B baseline
is 2.39, with formula threshold 3.89. Formula-based receipts normally require
`min(idle baseline + 1.5, 5)`. Keep Defender and other services active and unchanged.
No other tests, builds, downloads, measurements or profiles may run concurrently.

Oliver subsequently authorized a strict limit of 5 for the remaining MASSIVE
measurements. This is a recorded decision, not a different idle observation.
A decision-backed `--idle` receipt retains `valid: true`, `baseline: 2.39`, sets
`max_load: 5` and adds exactly:

```json
{
  "threshold_override": {
    "policy": "dr147-oliver-limit-5@1",
    "authorized_by": "Oliver",
    "decided_at": "<actual decision timestamp with timezone>"
  }
}
```

The coordinator supplies the timestamped receipt. Other policies, actors, missing
or naive timestamps, altered baselines and other override limits are refused.
Without this decision, the normal formula is unchanged. The complete receipt is
retained in invocation, job and manifest evidence and used by parent/worker guards.
Limit equality still stops a run. This policy changes only threshold validation;
the driver still stops on a guard failure. A separately authorized coordinator
continuation governs load-only retries and remaining-cell selection within the
same matrix/global caps. Record every pair's load, distinguish all readings below
3.89 from any readings at least 3.89 and strictly below 5, and compare paired
spread descriptively without implying causality. Historical stops remain stopped.

Use the task scratch directory for every transient artifact. On this installation:

```sh
export TMPDIR=/Users/oliver.borchers/Desktop/Code/fse-m4-scratch
export GIT_CEILING_DIRECTORIES="$TMPDIR"
```

The Git ceiling prevents temporary test repositories from discovering the enclosing
Desktop/Code repository. The driver requires a clean committed checkout and verifies
its imports, native extension, base ancestry and recorded build identity.

## Commands after pilot authorization

Run commands from this worktree, with its prepared interpreter. Launch long commands
in the background through the calling harness. These examples each select one cell;
omitting `--cell` requests all 60. Multiple `--cell` arguments preserve their order.

```sh
.venv/bin/python benchmarks/bench_mlx_gate.py list
.venv/bin/python benchmarks/bench_mlx_gate.py measure \
  --scratch "$TMPDIR" --idle "$TMPDIR/dr147-idle-2-receipt.json" \
  --cache "$TMPDIR/dr147-cache-receipt.json" \
  --cell static-retrieval-mrl-en-v1-1024-wikitext-raw-float32 \
  --rounds 3 --passes 3 --budget-s 120
```

Initial historical schedule: five valid rounds, best of ten passes on WikiText and
three on MASSIVE. Sides reverse order on alternate rounds. After the authorized
pilot, project the remaining matrix cost before running it. Reduce passes first,
then rounds to three. Report changes to the schedule. Never drop required cells.
One-round runs are pilot evidence only. They do not supply a decisive sample.

`bench.lock` is acquired atomically. Its owner records branch, commit, PID and a
unique token. Every guard verifies ownership, the live owner process, absence of
`pause-local.marker`, budget and strict load. Every pass and side is checked before
and after it runs. A middle-side breach stops the cell and discards its whole round.
Raw durations and observations are flushed before post-pass acceptance checks.

The driver deliberately retains the lock on exit. Inspect the owner and PID before
releasing the verified own lock externally. On this installation use the prescribed
Finder-based `aos-organization-icloud` workflow, never shell deletion. An existing
lock is never broken. If preflight fails after acquisition, inspect the parent journal
and retained owner before release. Do not launch a second invocation against it.

## Evidence and budget

Each invocation creates an exclusive directory with provenance and selected cells.
Each cell has its job, manifest, raw JSONL journal, worker/restart logs and result.
The manifest records model/corpus revisions, input identity, encoder fingerprints,
default workers, hardware, package versions and import/build paths. Durations are
unrounded. Ratios are formed per pair before median/range aggregation. A/A ranges
must accompany comparisons; a sample of fewer than three pairs is inconclusive.

CPU outputs must retain their bits across worker counts. MLX rows are checked
bitwise where equal, otherwise with independent per-row pooling and normalization
bounds. Empty, tiny and possibly cancelled rows require exact bits. Every timed
pass is compared with its backend's reference outside the clock. Untimed launch
observation follows complete backend preparation, excluding canary starts. It records
corpus launches and that observed run's fallback deltas separately. A canary or
historical fallback count alone cannot establish corpus GPU execution.
Fresh-process checks compare full saved CPU and MLX arrays. Counter units are
backend fallback events, not row counts. Device conformance remains a separate gate.

All invocations share `mlx-gate-budget.jsonl`, capped at 1800 active seconds.
Reservations are charged before execution and settled to actual elapsed time.
A killed parent consumes its full reservation. Do not reset or replace the ledger.
Pilot, loading, warmup, parity, retries, restarts and profiles all consume this budget.
Each child is additionally limited to ten minutes. A timeout ends its process group.
An independent stdlib-only supervisor watches the owner's private pipe and deadline.
Owner SIGTERM takes a controlled cleanup path. Even owner SIGKILL closes that pipe,
so the supervisor terminates the job group and remaining descendants independently
of a blocked encoder or its GIL. The supervisor waits at most 0.25 seconds per
completion check and allows 0.25 seconds for SIGTERM before SIGKILL. Cleanup/reaping
has bounded waits. Each supervised job incurs process startup, an idle monitor and
up to four completion checks per second. This unavoidable activity can affect
timings indirectly. Supervision is identical for paired/A/A sides and runs outside
the encode timer; all its wall time is charged to the shared budget.
Coverage files retain completed and unmeasured cells. Raw journals also survive a
hard timeout. Inspect them before reporting incomplete evidence or retrying.

## Profiles, after the matrix identifies the deciding mode

Select MRL1024/WikiText float32, the deciding mode and CPU/MLX workers explicitly:

```sh
.venv/bin/python benchmarks/bench_mlx_gate.py profile \
  --scratch "$TMPDIR" --idle "$TMPDIR/dr147-idle-2-receipt.json" \
  --cache "$TMPDIR/dr147-cache-receipt.json" \
  --cell static-retrieval-mrl-en-v1-1024-wikitext-raw-float32 \
  --backend cpu --workers 10 --profile-seconds 6 --profiler cpu --budget-s 60
```

Use `--profiler cpu`, `mem` or `rss`. The driver runs the actual worker under samply,
native memray or the process-tree RSS tool, respectively. It then prints the hottest
functions, or renders the temporal memray flamegraph and allocation stats. All
artifacts have exclusive fresh paths. Profiler startup, recording and postprocessing
consume the same shared budget. Never invoke the stock `profile-mem` deletion recipe
or trace only an outer parent whose exec'd child would escape memray.
Before and after each summary subprocess the driver checks pause, live ownership,
load and the global deadline. Expired budgets start no subprocess, including
restart checks or later cells. Absolute deadlines are preserved across supervision.

Report hot functions per thread, allocation/RSS/MLX memory peaks and attribution
limits. CPU samples waiting in MLX do not isolate GPU time or copy time. Separate
initial preparation from steady-state work. Instrumentation and any ablation need
their own untraced control. Existing real-model ablation flags do not construct
arbitrary mode/dtype variants, so do not reuse them without explicit cell checks.

No runtime optimization is authorized before evidence. A proposed optimization
requires at least 2% median gain, all paired time ratios at most 1.0 and improvement
beyond A/A noise, with unchanged bits/fallback behavior, memory and cancellation.
If runtime changes, also compare frozen base/branch `bench_runtime.py throughput`
at 200000 sentences, float32, dimension 300, workers 1 and 10, raw and normalized.

Final delivery remains a recommendation report for Oliver and a ready evidence PR
with PERFORMANCE.md numbers. Oliver's decision is recorded when supplied. Do not
merge a PR, change policy, work another ticket or replace missing measurements with
claims. The current draft is a driver checkpoint, not completed gate evidence.

Primary sources: [MLX evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html),
[MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html),
[Hub revisions](https://huggingface.co/docs/huggingface_hub/guides/download).

Completed profile provenance, guarded scratch target overrides, phase/sample mapping and exact peaks are in `benchmarks/evidence/dr147-m4/profiles.json`. The read-only `profile_offline.py` reproduces inclusive versus encode-only sample shares from local raw traces. These are not wall percentages. Memory/RSS-only limit20 exceptions were explicit but every observed profile guard stayed below5. Ordinary timing policy and idle data were unchanged. No ablation or policy change followed.

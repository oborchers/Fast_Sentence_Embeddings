# Runtime waits at eight workers

Study for DR-442 (Wartezeiten der Laufzeit bei acht Workern messen).
The binding reviewed plan lives in the coordinating session's scratchpad.
This document records the executable contract and the evidence still needed.

## Question and decision

Where does one large float32 encode lose worker activity at eight workers?
Profiles and absolute observations are class descriptive. Contrasts between
variants are preliminary screens. They select the next investigation only.
They do not establish causation or justify keeping a product optimization.
Later product changes have a separate 1.10x throughput boundary and require
independent controls and confirmation. This study changes no product source.

No Linux comparison run. All local observations are non-publishable development
indications. There are no results yet.

## Admission and budget

Do not build, test, time or profile until the coordinator fse-main explicitly
sends the measurement release. Merge master after the prerequisite direct-path
change, then inspect all hook targets again. Never rebase this branch.

The total experimental window is 75 minutes, including preparation, builds,
load/lock waits, profiling and collection. Cost is 0 USD. Stop at the next safe
operation boundary when exhausted. Preserve failures and unstarted cells.
Never extend the budget or restart automatically.

| Phase | Cells or operations | Estimate, minutes |
| :--- | :--- | ---: |
| Preparation | Build, reference probes, memory admission and gates | 10 |
| Profiler access | Check sampling of an explicitly blocked thread | 2 |
| Native profile | Eight-worker allocating baseline, both corpora | 13 |
| Descriptive screens | Nine configurations per corpus, five repetitions each | 45 |
| Collection | Verify receipts, summarize and preserve evidence | 5 |

These estimates are planning assumptions. Every sub-invocation receives the
remaining total budget, including profiling. The profile budget cannot restart
the 75-minute clock. If gates or waits consume the window, report incomplete.

## Workload and variants

`benchmarks/config/runtime-scaling-local.json` uses the existing pinned
potion-base-8M float32 model and complete WikiText and STS-B corpus cycles.
The allocating adapter calls public `encode(..., sink=None)` each time. The
existing `encode-memory` adapter remains the reused-target comparison.
Memory admission reserves two complete output arrays for allocating encode.
Measured process CPU time is differenced around the same repeat loop as wall
time. Their ratio reports mean cores used, without attributing any wait cause.
Initial resident cycles are planning assumptions, subject to memory admission
and a short provenance/size probe before the run. Freeze the same row count
across all variants of a corpus. Each production encode call must take at least
two seconds. A short call is invalid, never repaired by quietly repeating it.
A failed size or memory probe means the plan needs coordinator review.

| Variant | Only changed factor |
| :--- | :--- |
| baseline | Ordinary allocating encode at eight workers |
| joined | Reuse one pre-built StringSource, same bytes and offsets |
| gc-off | Disable Python automatic garbage collection during encode only |
| window-4x | Double the original window, retain the exact original task iterator |
| chunk-4m | Set the executing scan module's chunk cap to 4 MiB |
| chunk-256k | Set that cap to 256 KiB |
| inline-off | Set the executing task module's inline byte threshold to zero |
| workers-10 | Request ten workers, total scaling only, no E-core attribution |
| reused-target | Existing encode-memory target, all other settings held |

The window variant increases the effective original byte allowance by two.
It does this by retaining the original plan and widening its executor window,
so its task sizes cannot change through the planner's memory calculation.
All patches are installed in the executing child and restored in `finally`.
Garbage collection retains its previous enabled state. The runner's explicit
post-call collection remains outside the encode timer for every variant.

Use the existing speed entry with `--diagnose --local-phase descriptive`,
`--development-input benchmarks/config/runtime-scaling-local.json`, an explicit
candidate interpreter, external fresh results/work paths, five descriptive
repetitions and the remaining `--wall-time-seconds`. Keep the shared locks and
load gate. Run a small `--functional` adapter/reference probe first after release.
The runner verifies output shape, dtype and bits against the numpy backend.

## Timeline coverage and limits

A separate untimed encode follows each measured arm when `runtime_timeline`
is true. It must produce the identical full output identity. Its preallocated
fixed event buffer records wall and thread CPU clocks, thread identity,
submission, worker entry/exit, native-call entry/exit, publication callbacks,
commit, planning, join, automatic garbage collection and caller waits. It records actual task row and byte
counts and executor window observations. Overflow invalidates the diagnostic.

The event artifact is separate from the arm's speed summary. Its offline
reducer reports time before first submission, work wall and CPU fractions per
thread, caller waits, observed full-window stalls and last-wave tail length.
A stall requires later results done, an unfinished oldest task, a full window
and idle worker capacity. Pool-place refusals are reported separately.
Snapshot observations are estimates and do not prove causation.

Native-call boundaries include Python checks, GIL-held work and the native
kernel. They cannot measure exact GIL-free duration. Wall minus thread CPU
includes all scheduling and blocking. Samply must first demonstrate whether
it observes an explicitly blocked thread. If it does not, its stack buckets
cannot establish wait proportions. Use the timeline for those observations.

Profile the actual executing child through `speed.py --profile` and its
`profile@1` contract, addressed to the sealed descriptive baseline job. Request
all threads and keep fresh external output/work paths. Keep origin artifacts
and inputs unchanged. Record observed coverage, not just requested coverage.
Summarize compute, GIL-wait symbols, pool-idle symbols, main-thread join,
garbage collection and executor waits only where the stacks support them.

## Completion evidence

Still required: integrated-base hook audit, small functional/reference probes,
samply blocked-thread coverage check, native profiles, fixed-repetition screens,
timeline reductions, next-investigation recommendation, final `make check`,
remote CI and the coordinating Claude review. No benchmark-script tests are
added. This benchmark-only change requires no changelog fragment.

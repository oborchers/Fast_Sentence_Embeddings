# M1 pooling study

DR-440 (Pooling von potion-base-8M und -2M auf dem M1 vermessen) prepares the
evidence for DR-438 (Zeilenholen und Pooling nach dem Vorbild von encoders
beschleunigen). This document declares the study before execution. It contains
no results. The machine is the local Apple M1 Pro, with one inference worker,
float32 tables and float32 output. Results are development indications only,
with `publishable: false`. No Linux comparison is planned.

## Question and decision

Does the larger model mainly lose time to its table working set or to the
dependent accumulator operations? A 64-column NEON block gets a separate
product pull request only if the profile identifies the accumulator chain.
Prefetch changes need their own evidence and ticket. Arithmetic order,
bit identity with `fse._reference`, and persistence names remain unchanged.

The profiling phase uses correctness probes, descriptive stage costs, and
preliminary regression screens. A screen cannot establish an optimization or
equivalence. Stop if pooling contributes less than 30 percent of the larger
model's encode cost, or if no isolated variant shows a preliminary pooling
factor above 1.1. Report uncertainty and conflicting workloads explicitly.

The later product decision requires full optimization evidence on both larger
model pooling cells. Its relevant boundary is factor 1.2 (at least one-sixth
less time). It also requires the existing independent A/A noise control and
confirmation. The smaller model and memory encoding use regression screens
with a declared five-percent slowdown limit. These screens cannot prove
non-regression. The tile budget stays 16 KiB in the conditional product change.

## Frozen input definitions

`benchmarks/config/development-pooling.json` names twelve cells: both models,
both corpora, and tokenizer, pooling, and memory encoding. Model revisions
come from `benchmarks/config/pins.json`.

The STS-B corpus uses the existing pinned preparation. The new
`wikitext-256-words` corpus reads the pinned WikiText test split in source order.
It splits all rows on whitespace, retains heading words, concatenates the
words, and emits consecutive full chunks of 256 words. It drops the final
incomplete chunk and records its word count. Each chunk uses one space between
words. The prepared file and its SHA256 identify the exact texts.

Preparation records each model's actual token and character limits, retained
token minimum, median, maximum and total, and truncated row count. Potion
models must use their 512-token rule. Word count does not imply token count.
Timing reuses the prepared text or token batch through the existing
`benchmarks/speed.py` entry. No separate measurement runner is introduced.

## Execution and limits

Do not start builds, functional inference, profiling, or timing before the
coordinator explicitly releases this study from the concurrent measurement.
After release, use the shared locks, a controller at nice zero, and the
existing host-load admission check. Keep models, outputs, traces, disassembly,
raw observations and environments outside the checkout.

The profiling budget is 5,400 wall seconds, including all preparation, probes,
builds, warmups and coordination waits. The estimated allocation is 900 seconds
for preparation and correctness, 900 for stage costs, 2,400 for isolated variant
builds and screens, 900 for native attribution, and 300 for collection. Cost is
zero. These allocations are estimates, not observed runtimes. Use one shared
deadline across phase invocations. Stop at a safe boundary when exhausted,
preserve partial results, and report inconclusive without automatic extension
or retry. The conditional product phase has a separate 10,800-second budget.

Before stage timing, verify the adapters and executing build. Freeze resident
cycles and repetitions after the functional preparation, before collecting
descriptive samples. Collect five repetitions and report medians and IQRs.
Use repeated-work groups of at least two seconds. Keep diagnostic profile
durations separate from speed summaries.

Collect pointer alignment for the actual mapped artifact and the original
file-import path separately. Record the selected NEON path, dimensions, row
stride, tile capacity, weights and build identity. Do not infer alignment
from an allocation method or classify weights from a different loading path.

An accumulator screen changes only the register block width. A footprint
screen holds dimension, tile capacity, output size and prefetch volume fixed
and records the row remapping. Reducing dimension and changing stride together
does not isolate either cause. Prefetch off must disable prefetch explicitly:
distance zero still prefetches the current row. A tile-size variant remains a
diagnostic experiment. Output first-touch probes must separate allocation,
prove untouched or pre-touched pages, and record page faults.

The addressed `profile@1` contract supports samply on macOS for the executing
child. Its coverage must include native pooling samples, rather than only the
controller. Xcode CPU Counters is outside that contract. Check its availability
after release and report unavailable counters as missing evidence. Do not infer
cache misses or a dependency bottleneck from elapsed time alone. Inspect the
compiled accumulator loop for spills before interpreting a width comparison.

## Delivery evidence

Record accepted, failed, invalid and unstarted cells separately. Preserve the
result manifests, output bit checks, process coverage, package identities and
the actual budget consumption. Report the working-set and accumulator verdict
to the coordinator before implementing a product change. The profiling pull
request receives only a small reviewed summary and provenance, never weights,
raw data, matrices or reconstruction output.

---
name: fse-benchmark
description: "Plan, rehearse, run or audit fse CPU speed and quality measurements. Use for library comparisons, dtype controls, streamed writing, MTEB, STS, SIF fits, recall, measurement budgets and result validity. Uses the repository's two measurement entries."
---

# Measure fse

Read [the measurement guide](../../../docs/benchmarks/measurement.md) before planning a run.
It defines commands, pins, result formats and eleven measurement rules.
Read [lessons and evidence](references/lessons.md) when choosing adapters, memory or lifecycle controls.
Follow repository instructions. This skill does not authorize spending, publication or deletion.

## 1. Establish the question

Resolve missing decisions before allocating resources:

- What claim must the result support: encoding, persisted vectors, quality, or a code change?
- Which machine, CPU, allocation and date form the cohort? Local or rented hardware?
- Sentences or paragraphs? What corpus revision, count, digest, length and repetition policy?
- Which model revision, dimensions, normalization, truncation, table dtype and output format?
- Which workers, CPU affinity and documented parallel path does each library use?
- What are the time, RAM, disk and money limits? Who monitors, collects and ends the allocation?

Recommend one explicit study contract. Never silently shrink only a slow counterpart's input.
Different pairs may use different agreed workloads. Label their counts and digests separately.
Such pairs do not establish a fixed-input worker-scaling curve.

Choose the claim class before the protocol. Record question, decision, workload,
relevant effect boundary, total wall-time budget, estimated cost and stop rule.

| Class | Required evidence |
| :--- | :--- |
| correctness | Function, bits, conformance and quality. No speed decision. |
| descriptive | Candidate-only fixed repetitions, absolute median/IQR. No A/A or confirmation. |
| regression | Paired builds and a relevant slowdown boundary. Short screens remain preliminary without sufficient uncertainty evidence. |
| optimization | Full v2 controls, independent A/A validation and confirmation. Screens never decide keep. |
| publication | Existing Linux publication and release evidence. |

Use the guide's planning suggestions, not universal runtime promises. Reject plans
whose known costs exceed their budgets. Count preparation, warmups and lock/load
waits across all cells. Stop safely as inconclusive on exhaustion, retain raw data
and escalate without automatic extension. Never reinterpret historical or active runs.
A small regression median does not establish equivalence. Comparing descriptive medians does not prove
an optimization. Full numerical and persistence guarantees remain. Linux evidence is
required only for class publication or when the maintainer has explicitly approved a Linux machine
for the ticket.

## 2. Select the existing entry

| Need | Entry | Definition |
| :--- | :--- | :--- |
| Library pairs, dtype controls, streamed writing | `benchmarks/speed.py` | `benchmarks/config/speed.json` |
| MTEB, STS, offline SIF fits, recall, score summaries | `benchmarks/quality.py` | `benchmarks/config/quality.json` |

Start with `python benchmarks/speed.py --describe` or `python benchmarks/quality.py --describe`.
These modes inspect study data without installing optional measurement packages.
Use `--setup`, `--rehearse`, `--run`, `--bootstrap` and `--collect` as documented in the guide.
Quality also provides `--summarize`. Add study data, not another executable entry.
A new Python library adds a native factory and registry entry in `_speed_pipeline.py`.
Do not add repository tests for measurement tools or this skill.

Read `benchmarks/config/pins.json` and the relevant locks before changing a configuration.
Record the runner revision separately from each executing package's version, path and content hash.
Rust also needs source, Cargo lock, compiler, binary hash and actual Rayon thread count.
An fse fingerprint cannot identify a counterpart's installed code.

### Local Unix development

Use the same entries with `--diagnose`, explicit `--base-python` and `--candidate-python`,
an external `--results` and `--work`, and selected `--cells`. `speed.py --local-info`
shows the local contract without optional evaluation packages. Windows is refused.
Local results on macOS, Linux and other Unix systems always say `publishable: false`.
They never replace the Linux publication protocol or justify publishing Mac speeds.

For the class descriptive, use `--local-phase descriptive --candidate-python ...` without a base.
Declare `--descriptive-repetitions` (default five), fixed `iterations` and
`resident_cycles`, and `--wall-time-seconds`. This path has no sizing, A/A or
confirmation. Warmups use the fixed `warmup_calls`, without a duration loop.
Legacy `decision_gate: false` keeps its paired meaning.
For a short regression probe, `--local-phase regression-screen --regression-limit FRACTION`
uses five alternating pairs and always remains inconclusive. No screen PASS rule
or equivalence test is supplied. Both interpreters are required.

Both entries accept total `--wall-time-seconds` and optional `--estimated-seconds`.
Defaults are fifteen minutes for descriptive and two hours otherwise, for the whole invocation.
Estimate first and set an appropriate explicit budget. A declared estimate above
the budget is refused. Active operations may finish past the admission deadline,
then no further operation starts. Record overruns, failed and unstarted cells.
Before long timing blocks, independently probe the adapter and verify actual builds,
inputs and metadata. Retain complete applicable quality checks, reused only for
identical configurations in the current invocation. Apply existing evidence reuse
identity and age checks, never a calendar-day shortcut.

The full optimization v2 local timer uses six-second continuous groups, at least two warmups totaling twelve seconds,
and fifteen alternating pairs per block. A/A adapts through six/twelve/twenty-four
seconds until its same-cell IQR is below two percent, with an independent control
block and a sixty-active-minute calibration limit. Calibration increases repetitions
of the same prepared corpus, never resident rows. Tokenizer and pooling reuse their
prepared input; memory encoding reuses a public Sink target and checks it against
regular encode. Disk groups include a new ShardSink's open, encode and close on each
iteration. Disk usage grows with repetitions; every store is checked and removed
outside the timer. Individual operations may be short. This local repeated-work
contract never satisfies the publication single-call gate. Old v1 controls cannot
be reused. An unresolved cell is inconclusive.
Keep requires saving at least `max(3 * A/A IQR, 0.02)` and a confirmation with the
same positive sign and at least half the effect. Maximum deviation is reported only.
All raw observations remain. Use ABBA explicitly in a cell when appropriate.

`FSE_BENCH_LOCK` and `FSE_HEAVY_LOCK` configure mutually exclusive mkdir locks.
Each lock serves registered waiters in FIFO order; see the measurement guide for queue positions, cancellation and mixed-version rollout.
Use `speed.py --local-heavy --worker-name NAME -- COMMAND ...` for builds/checks.
Timing waits for one-minute load below five before each cell and arm. Foreign CPU
above fifty percent is telemetry only. macOS clears child background policy when
available; Unix priority, Linux affinity and governor observations are capability-based.
Timing requires a controller at nice 0. zsh adds 5 to the nice value of `cmd &` jobs (BG_NICE), so never background a timing controller with `&` in zsh; start it from a process at nice 0 through a harness background shell or after `setopt NO_BG_NICE`. `speed.py --diagnose` refuses a nonzero nice before any work, except with `--diagnostic-rehearsal` or `--local-phase prepare`, where `os.getpriority` is available. See [lessons](references/lessons.md#setup-and-lifecycle-traps).

For a Linux functional probe, build the pinned development image:

```sh
docker build -f benchmarks/docker/Dockerfile -t fse-development .
docker run --rm --cpus 2 --memory 4g \
  --mount type=bind,src="$PWD",dst=/source,readonly \
  --mount type=bind,src=/external/results,dst=/results \
  --mount type=bind,src=/external/work,dst=/work \
  --mount type=bind,src=/external/cache,dst=/cache \
  --mount type=bind,src=/external/coordination,dst=/coordination \
  fse-development /source/benchmarks/speed.py --local-info
```

Create those external directories first. The image builds the extension with
CPython 3.13.18 (the image of `pins.json`) and uv 0.13.0, with no baked models/data and no registry push.
Rebuild it from the same checkout after product changes. Its explicit development
version does not identify a Git commit; package/source hashes identify actual code.
For a functional comparison, replace the final `--local-info` command with:

```sh
/source/benchmarks/speed.py --diagnose --functional \
  --base-python /opt/fse-env/bin/python --candidate-python /opt/fse-env/bin/python \
  --cells potion-stsb-tokenizer potion-stsb-pooling potion-stsb-encode-memory potion-stsb-encode-sink \
  --hf-cache /cache --results /results/probe-01 --work /work/probe-01
```

Use fresh result/work subdirectories, not the existing mount roots. Public downloads
go into the mounted work directory; keep caches and generated output outside Git.
Use `--functional` (alias `--diagnostic-rehearsal`) for functional stage probes: it reduces the input and writes `production-plan.json` and `functional-plan.json`. Run Docker preparation
under heavy coordination and keep Docker off during native host timing sessions.
If a host heavy wrapper covers the complete Docker session, give the container its
own mounted coordination directory. Sharing the held host lock with its inner
controller would make it wait for itself. Other host work still uses the host locks.
See the [local protocol](../../../docs/benchmarks/measurement.md#local-unix-development)
for phases, complete quality checks, resource limits and result reuse.

## 3. Prove correctness and resource feasibility

Load each side through its native API. Match input, truncation, normalization, shape and dtype.
Keep the fixed output sample and saved UNK cases. Library tolerances remain `rtol=3e-5`, `atol=3e-6`.
Dtype controls compare bits. A failed check yields no speed claim. Never relax tolerance to rescue a row.

Calculate peak live memory in bytes against the cgroup limit, not host RAM.
Include text objects, tokenizer buffers, model copies, output, concatenation and sample buffers.
Release warmup outputs before timing, including retained Torch tensors, then collect garbage.
Do not remove real concatenation copies from the budget because warmup output was freed.
The current conservative admission plan uses 85 percent of the cgroup limit.
Record rejected components. Do not call a preflight rejection an observed out-of-memory kill.
Check disk for inputs, caches, overlapping outputs and collection. Keep generated files outside the checkout.

Rehearse every selected cell and lifecycle control in the exact pinned Linux image before a paid run.
Use the same entry's `--rehearse` mode with at most two CPU quota units and four GiB RAM.
Follow the guide's shared-memory and descriptor limits for ST.
Small inputs prove function, not full-size memory capacity or speed.
On a developer Mac or Mac mini, run only functional probes inside bounded Linux containers.
Honor shared `heavy.lock`, `nice` and local worker limits when those coordination rules apply.
Never publish local Mac times or run beside another heavy job.

## 4. Measure under the agreed protocol

Publish speed only from Linux. Size input so every measured fse call lasts at least two seconds.
Aim for three seconds with headroom. If a call falls below the gate, mark it invalid.
Do not reuse withdrawn small-input or M1 speed claims. M1 quality and accuracy remain valid.

Alternate arms for at least three rounds, with a full warmup before each measured call.
Record an initial quiet observation and telemetry before and after every operation.
Report primary medians and the median of same-round counterpart/fse time ratios.
The current library protocol has no duplicate A/A arm: record `not_run`, never a fabricated clean band.
Dtype controls retain the explicit symmetric 10 percent A/A gate.
Never mix machine cohorts, output contracts or protocols into a ratio.

For writes, rotate order and separate first-read-to-sink-close from later external synchronization.
Check row count, codes, scales and file digests. Synchronize and remove owned output before the next arm.
Do not let cleanup writeback overlap the next timer. Follow the owner's deletion authorization.
Report logical file bytes separately from physical device writes and state the RSS measurement scope.
Use telemetry to investigate drift. A quiet start does not prove a quiet run.

Stabilize APIs before timing. For small review fixes, run affected tests and
conformance, then the complete required `make check` on the final state. Repeat
only for new risk or a failed gate. Profile representative changed paths and
resource risks rather than every parameter combination.

For hot-path changes, profile representative changed paths and state the claim classes used, as AGENTS.md requires. A paired Linux float32 comparison is required only for class publication or when the maintainer has explicitly approved a Linux machine for the ticket. Report the actual Linux comparison status and say explicitly when none was run.
Use the profiling targets in AGENTS.md, but verify which processes each profiler captures.
Linux CPU and RSS targets can follow the controller tree. Memray cannot follow an `exec` into a new interpreter.
For memory attribution, instrument the executing Python child directly, using its recorded `--child-job` with fresh diagnostic output paths.
Instrument spawned pool workers separately when needed. Use an appropriate native profiler for Rust.
Keep diagnostic timings out of speed summaries. State process coverage, hottest functions and memory peaks.
A controller-only allocation profile is not an encoder memory profile. See the referenced profiler limitation.

## 5. Preserve quality meaning

Keep MTEB pinned and unmodified. Preserve actual language subsets, splits, seed and complete task rows.
Do not sample evaluation data. Current English coverage is 40 of 41 tasks, excluding MindSmallReranking.
NanoBEIR is separate. Never impute missing tasks or report a partial mean as a full benchmark.
Multilingual tasks keep their benchmark: name a task of several benchmarks as "Benchmark/Task".
Language columns count their tasks; a selection mean is no benchmark mean.
Compare zero-shot only over the common set MTEB confirms for every model.
Set HF_HOME and HF_DATASETS_CACHE explicitly; local runs watch disk and resident set.

Fit SIF only on the declared training corpus. Record frequencies, revision, seed, components and artifact hashes.
Historical WikiText fits and published FineWeb fits are different configurations.
Use explicit published-fit cells when reproducing released SIF-10 models.
Keep wordfreq policy, tokenizer and FastText OOV ngrams explicit. ParaNMT mean is a separate action.
Do not compare historical Pearson with MTEB Spearman as the same metric.
Summarize only complete matching coverage and preserve worker, dtype and fitted-state identities.
Quality evaluation wall time is not inference throughput. Encoding speed is not classifier speed.

## 6. Monitor, collect and report

Observe `cells.jsonl`, `progress.json` and `DONE` continuously at the agreed interval.
A missing file during setup is not a transport failure. `DONE` means finished, including failure.
SIGTERM stops new work at a safe boundary. Admission limits do not kill an already active pair.
Keep failed and not-started cells. Never silently retry under weaker rules.
Verify the manifest and transferred bytes with the entry's `--collect` mode before claiming completeness.
Retain large evidence externally. Git receives only small reviewed summaries and provenance.
Never commit weights, matrices, raw data or reconstruction output, even on an evidence branch.
Inspect `git diff --stat` and file sizes before push. Files above 10 MB need coordinator approval.

Report accepted, invalid, failed and missing cells separately, with source identities and limitations.
Do not edit README, model cards or historical values as a side effect of running a study.

## Conditional: rented machines, including RunPod

Use this section only when a rented machine is available and its lifecycle is authorized.
Agree on compute, disk, setup, warmup, slow-arm and collection costs before starting.
Checked-in billing values are historical examples, not current quotes or spending approval.
CPU flavor and region do not guarantee a processor model. Inspect CPU, affinity and cgroup limits at startup.

Use the guide's bootstrap with an immutable pushed source revision and the exact image digest.
Start SSH before setup if collection requires it. Keep credentials out of logs, studies and results.
Retain the persistent restart guard. End setup and measurement paths in `sleep infinity` after completion handling.
A restart must never launch a second controller or overwrite results.
Use pinned hashed locks and verify installed environments with `uv pip check`.

Assign a read-only watcher and collect incremental files every two minutes.
Distinguish missing result files from SSH errors. Never start another controller from the watcher.
The owner collects and verifies immediately on failure or completion, then promptly deletes the allocation.
Analyze completeness locally after collection. Never retain idle billing for analysis or code preparation.

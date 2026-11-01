# Performance: how close fse comes to the hardware limit

This document says how fast fse can be on a given machine, why, and how close it comes. It holds a small model of the limits (a roofline model), the assumptions behind it, and measurements on an Apple M1 Pro. `benchmarks/roofline.py` computes the same numbers for any encoder, corpus and hardware profile.

## In short

On an M1 Pro (8 performance and 2 efficiency cores), master of 30.10.2026 with PR 136:

- **One worker** runs at 86 to 99 % of the per-core bound: almost all of its time is the work of the stages that more cores share.
- **Eight workers** reach 69 to 97 % of the Amdahl bound and 44 to 78 % of the per-core bound. The gap has two measured parts: each task takes 2 to 11 % more CPU time than alone when all cores run, and the workers are busy only about 70 to 100 % of the parallel time (the last tasks of a call, and waits between tasks).
- **The measurements show no memory bandwidth limit**, also not at 1024 dimensions. static-retrieval-mrl-en-v1 at 1024 dimensions reads 164 to 197 GB/s of table rows (logical bytes) with 8 or 10 workers. That is 82 to 99 % of Apple's 200 GB/s, but more than any reader of the probe got from memory (115.6 GB/s at most). This points to many rows coming from the caches; it is no proof, because the probe gives lower estimates and hardware counters were not read. What the measurements do show: pooling is the largest stage, 72 to 83 % of the time of one worker when timed on its own; that time includes its waits for memory, which these measurements do not separate from its arithmetic.
- **int8 tables at 1024 dimensions** are 1.20 to 1.34 times as fast as float32 on WikiText and 0.98 to 1.10 times on MASSIVE, end to end.

## RunPod Linux x86 observations

DR-190 (Profiling-Durchgang auf Linux x86 bei RunPod durchführen), 31.10.2026,
source `4d92d61`: the primary run used an AMD EPYC 9655P host with 96 physical
cores and a pod affinity of 32 logical CPUs, 64 GB memory limit and an overlay
filesystem. Host load ranged from 4.94 to 10.31. These are single observations
on a shared host, not a stable performance baseline. The
[`runpod-epyc-9655p` profile](benchmarks/roofline_hardware.toml) records the
observed read references. Its host core count does not calibrate the model's
per-core/Amdahl predictions for pod affinity or SMT placement.

| workers | sum read GB/s | random pooling GB/s | runtime float32 sentences/s | normalized sentences/s |
|---|---|---|---|---|
| 1 | 31.59 | 21.35 | 1,652,196 | 1,294,669 |
| 8 | 130.17 | 64.62 | 8,915,302 | 7,870,744 |
| 16 | 182.98 | 113.94 | 11,903,145 | 11,366,842 |
| 32 | 235.47 | 188.23 | 11,043,664 | 9,546,759 |

The probe uses a 1 GiB float32 table at dimension 1024, warm-up and the fastest
of three passes. Runtime uses 200,000 synthetic sentences at dimension 300 and
one round. GB/s counts logical bytes, not measured DRAM traffic. The runtime
peaks at 16 workers in this sample. A WordPiece WikiText trace is 44.68% busy
at 16 workers and 26.58% at 32, with ramp-up the largest idle cause.
`usable_cpus()` reports 32 and counts logical affinity CPUs, not physical cores.
The affinity IDs form 16 pairs separated by 96, consistent with SMT siblings;
no sibling map was captured. Runtime falls 7.22% without normalization and
16.01% with it at 32 versus 16 workers. Evaluate a topology-aware default with
controlled affinity and sustained workloads before changing the default;
SMT and pool/scheduling costs were not isolated.

Native one-worker profiles attribute 46.17% inclusive active samples to
WordPiece's sentence tokenizer and 43.00% to AVX2 pooling. Unigram's token-piece
entry has 71.30%, versus 12.01% pooling. Inclusive totals overlap; active-thread
samples are not wall-time fractions. This supports deferring an AVX-512 pooling
path while investigating tokenization and short-call scheduling. No AVX-512
comparison or float16/int8 input-pooling measurement was made.

Full/none store pairs verified with matching data digests. File/directory sync
p50 was 0.042/0.047 ms. A second observation, on EPYC 9B45 at host load above
1000, gave 55.58/63.41 ms; it measures host/storage variation, not same-host
repeatability or a Linux-wide cost. Two run-2 multiworker recordings could not
be reduced after a post-write `No child process (os error 10)` recorder exit.
A separate profile-only EPYC 9B45 run at host load 395 to 406 supplied those
tables: WordPiece at 32 workers has 25.52% tokenizer and 28.44% pooling inclusive
samples; store at 16 workers has 27.91% pooling. Unresolved symbols account for
at least 35.77% and 32.25% self samples respectively, so their hotspot identities
remain unknown. These shares cannot be applied to run-2 wall timings. The
[evidence summary](benchmarks/baseline/dr190_runpod_20261031.json) preserves both
hosts, results, profile failures and limitations. No paired optimization gain
is claimed.

## The model in plain words

fse turns a text into a vector in a few steps: the strings are joined into one buffer, the buffer is cut into tasks, each task is tokenized and its token rows are averaged (pooled), and the result is normalized. The work grows with the tokens, so the model counts **tokens per second**. A text costs its tokens: texts per second are tokens per second divided by the tokens per text.

Three things limit the rate. Each gives a bound, the largest rate that this limit allows.

| bound | what limits | in one sentence |
|---|---|---|
| per core | the work of the workers | N cores do N times the work of one core. |
| Amdahl | the serial part | Work in the calling thread does not get faster with more cores. |
| memory | reading the table | Each token reads one row of the table from memory, and memory has a bandwidth. |

The **combined bound** puts them together: the serial part, then the parallel part at the slowest of its limits (the work of the workers, memory, or what the calling thread does while they run). Measured throughput divided by a bound is the **share**: how close fse comes to that limit.

```
tokens/s
   ^
   |              memory roof (BW / bytes per token)
   |  ------------------------------------------------
   |                         ___________ Amdahl: serial part + work / C(N)
   |                    ____/
   |               ____/    measured
   |          ____/    ....''''''''''''
   |     ____/ ...'''
   |   _/..''
   +-------------------------------------------------> workers
```

## The formulas

Measured at one worker, in seconds per token: `t_ser`, what the calling thread does before and after the run (join, scan, the rest of `encode`); `t_call`, what it does while the workers run (reserving rows for the tasks and committing them), which overlaps their work; and `t_par`, the stages of the workers (tokenize and lookup, pool, normalize, the rest of the tasks). `C(N)` is the summed speed of the cores N workers use, counted in performance cores:

    C(N)          = min(N, P) + r_E * min(max(N - P, 0), E)        (Hill and Marty 2008)
    per core      = C(N) / t_par
    Amdahl        = 1 / (t_ser + max(t_par / C(N), t_call))          (Amdahl 1967)
    memory        = BW / bytes_per_token                             (Williams et al. 2009)
    combined      = 1 / (t_ser + max(t_par / C(N), t_call, bytes_per_token / BW))

    bytes_per_token = row + weight (4) + id (4) [+ scale (4) for int8]
                      + 4 * dim / tokens_per_text  (the output row, written once)

P and E are the performance and efficiency cores, `r_E` the speed of an efficiency core relative to a performance core, BW a memory bandwidth in bytes per second. A rest below zero (stage times are differences of measured times) counts as zero.

How the stages are timed: `t_par` is the measured time of the tasks themselves (`Pipeline._work`, summed over the tasks of one run), so the bounds use the real path. Since PR 136 a task runs its stages in one native call; the split of `t_par` into tokenize, pool and normalize comes from running the same stages one by one over the whole corpus (`Task.batch`, the pooling kernel, `normalize@1`), which copies the token ids into a `Batch` and meets other cache states. The split attributes the time; it does not enter the bounds except through its sum.

## Assumptions

Each bound holds only under its assumptions. A share above 100 % (the tool flags it from 102 % on, the noise of the stage times) means an assumption failed, and the report says which.

1. **A core works as fast as it did alone.** The per-core and Amdahl bounds scale the time of one worker. With all cores busy a core can run slower (a lower clock, shared L2 caches) or wait (for the GIL, other processes, the scheduler). The tool measures the effect, not its cause: the summed wall and CPU time of the tasks at N workers against one worker.
2. **Threads fill the performance cores first, and an efficiency core counts as a performance core** (`r_E = 1`) unless a profile or `--e-core-ratio` says otherwise. This gives the largest C(N), so the bound stays a ceiling. The real ratio on fse's work is not measured; it matters only for more workers than performance cores.
3. **The calling thread is not counted as a core** for the parallel part.
4. **Every byte comes from memory** in the memory bound. Repeated rows of frequent tokens hit the caches, so real traffic is lower, and a measured rate can exceed this bound. The report prints the rate of row bytes read (tokens per second times row bytes) next to it, and notes when the distinct rows of the corpus fit the L2 cache of one cluster.
5. **Work and memory overlap within a core** in the combined bound (`max`, not the sum).
6. **The scan at N workers is measured at N**, every other stage of the calling thread at one worker.
7. **The calling thread's part during the run overlaps the workers** (`max` with their work, not the sum).

Not in the model: waits of the tasks (they show as the gap between their wall and CPU time), imbalance at the end of a call (measured by `benchmarks/runtime_timeline.py`, "Where the workers idle"), and the system level cache, whose size Apple does not publish.

## Hardware profiles

`benchmarks/roofline_hardware.toml` holds one profile per machine; every number names its source, and a number without one is refused.

| profile | cores | bandwidth | source |
|---|---|---|---|
| `m1-pro-10` | 8 P + 2 E | 200 GB/s (spec) | Apple Newsroom, 18.10.2021 |
| | | 115.6 GB/s (measured, sequential read) | `roofline.py probe`, this machine |
| | | 77.3 GB/s (measured, random rows) | `roofline.py probe`, this machine |
| `m4-pro-14` | 10 P + 4 E | 273 GB/s (spec) | Apple Newsroom, 30.10.2024 |

The topology of the M1 Pro comes from `sysctl` (`hw.perflevel0.physicalcpu` 8, `hw.perflevel1.physicalcpu` 2, 12 MiB L2 per cluster of 4 performance cores). The M4 Pro is a spec profile and was not measured: `roofline.py show result.json --hardware m4-pro-14 --workers 14` gives its bounds with the stage times of the machine that measured them.

### The measured memory roofs

`roofline.py probe` reads a 1 GiB float32 table, far larger than any cache, in 1 to 10 threads (best of 5 passes, load 3.5 to 4.8):

| threads | numpy sum, sequential | fse kernel, consecutive rows | fse kernel, random rows |
|---:|---:|---:|---:|
| 1 | 23.8 GB/s | 14.6 GB/s | 12.9 GB/s |
| 4 | 79.4 | 50.9 | 46.2 |
| 8 | **115.6** | 88.6 | **77.3** |
| 10 | 103.9 | 84.6 | 76.1 |

Two more runs gave at most 115.1 and 104.6 GB/s for the sum and 80.0 and 78.3 GB/s for random rows. The CPU cores read at most about 116 GB/s from memory in this test, well below Apple's 200 GB/s for the chip; whether the CPU cores alone can reach that figure this test does not show. fse's kernel reads about 77 to 80 GB/s when no row is reused. All are lower estimates: a better tuned reader may get more.

## Measured on an M1 Pro

Setup: master 7403a3e (with PR 136) merged into this branch, Python 3.13.16, macOS on an M1 Pro (10 cores, 16 GB). The encoders are loaded with `Encoder.from_pretrained` at the snapshots pinned in `benchmarks/roofline.py`, with their own normalize (POTION normalizes, static-retrieval-mrl-en-v1 does not). The corpora are those of `benchmarks/bench_wordpiece.py`: WikiText-103 validation and test (5,352 texts, 454 bytes and about 95 tokens each) and MASSIVE intent test (151,674 texts, 55 bytes and 13 to 15 tokens each). Per cell, one process: the stages at one worker (the fastest of 15 rounds), then `encode` at 1, 8 and 10 workers in 5 paired rounds of alternating order (the best of 5 passes each, the median of the rounds). Every cell ran under the shared benchmark lock at a load below 5, and rounds with a load of 5 or more before or after were discarded (none in the cells shown; one run of the potion-base-32M MASSIVE cell lost 14 of 15 rounds to the load and was run again). The machine was not quiet: other work kept the load at 2.2 to 4.9. Each cell ran twice; the tables give the range of both runs. "Rows read" counts the bytes of the table rows only; the shares of the memory roofs count all bytes per token of the formula (rows, weights, ids and the output rows).

WikiText:

| model, dims | workers | M tokens/s | rows read, GB/s | of per core | of Amdahl | of memory 200 GB/s (spec) | of read 115.6 GB/s (measured) |
|---|---:|---:|---:|---:|---:|---:|---:|
| model, dims | workers | M tokens/s | rows read, GB/s | of per core | of Amdahl | of memory 200 GB/s (spec) | of read 115.6 GB/s (measured) |
|---|---:|---:|---:|---:|---:|---:|---:|
| potion-base-8M, 256 | 1 | 21.7 to 22.2 | 22 to 23 | 90% to 93% | 96% to 100% | 11% to 12% | 20% |
|  | 8 | 95.2 to 96.3 | 97 to 99 | 50% | 75% to 81% | 50% | 86% to 87% |
|  | 10 | 91.7 to 91.8 | 94 | 38% | 63% to 68% | 48% | 83% |
| potion-base-32M, 512 | 1 | 12.9 | 26 | 97% to 99% | 103% | 13% | 23% |
|  | 8 | 63.8 to 66.6 | 131 to 136 | 61% to 62% | 82% to 91% | 66% to 69% | 115% to 120% |
|  | 10 | 62.3 to 62.4 | 128 | 47% to 48% | 68% to 74% | 65% | 112% |
| static-retrieval-mrl-en-v1, 256 | 1 | 22.0 to 23.3 | 23 to 24 | 86% to 91% | 92% to 98% | 11% to 12% | 20% to 21% |
|  | 8 | 90.1 to 99.0 | 92 to 101 | 44% to 48% | 69% to 80% | 47% to 52% | 81% to 89% |
|  | 10 | 91.7 to 95.3 | 94 to 98 | 36% to 37% | 62% to 68% | 48% to 50% | 83% to 86% |
| static-retrieval-mrl-en-v1, 1024 | 1 | 7.2 to 7.3 | 30 | 97% | 99% to 100% | 15% | 26% |
|  | 8 | 43.0 | 176 | 71% to 72% | 84% to 87% | 89% | 154% |
|  | 10 | 40.1 to 40.6 | 164 to 166 | 54% | 65% to 68% | 83% to 84% | 144% to 145% |
| static-retrieval-mrl-en-v1, 1024, int8 table | 1 | 9.3 to 9.5 | 9 to 10 | 95% to 99% | 98% to 103% | 5% | 9% |
|  | 8 | 54.2 | 56 | 70% | 88% to 92% | 29% | 51% |
|  | 10 | 46.9 to 51.4 | 48 to 53 | 48% to 53% | 64% to 74% | 25% to 28% | 44% to 48% |

MASSIVE:

| model, dims | workers | M tokens/s | rows read, GB/s | of per core | of Amdahl | of memory 200 GB/s (spec) | of read 115.6 GB/s (measured) |
|---|---:|---:|---:|---:|---:|---:|---:|
| model, dims | workers | M tokens/s | rows read, GB/s | of per core | of Amdahl | of memory 200 GB/s (spec) | of read 115.6 GB/s (measured) |
|---|---:|---:|---:|---:|---:|---:|---:|
| potion-base-8M, 256 | 1 | 12.5 to 12.7 | 13 | 96% to 98% | 98% to 100% | 7% | 12% |
|  | 8 | 72.7 | 74 | 70% | 81% to 84% | 40% | 70% |
|  | 10 | 81.6 to 81.7 | 84 | 63% | 75% to 78% | 45% | 78% |
| potion-base-32M, 512 | 1 | 8.1 | 17 | 99% to 100% | 100% to 101% | 9% | 15% to 16% |
|  | 8 | 50.3 to 51.0 | 103 to 104 | 77% to 78% | 85% | 56% to 57% | 96% to 98% |
|  | 10 | 54.1 to 55.2 | 111 to 113 | 67% | 75% | 60% to 61% | 104% to 106% |
| static-retrieval-mrl-en-v1, 256 | 1 | 17.0 to 17.1 | 17 | 96% to 98% | 98% to 101% | 9% | 16% |
|  | 8 | 92.2 to 92.4 | 94 to 95 | 65% to 66% | 77% to 78% | 51% | 88% |
|  | 10 | 97.1 to 99.9 | 99 to 102 | 56% | 68% to 69% | 53% to 55% | 93% to 95% |
| static-retrieval-mrl-en-v1, 1024 | 1 | 7.1 to 7.2 | 29 | 98% | 99% | 16% | 27% |
|  | 8 | 43.5 to 45.6 | 178 to 187 | 75% to 78% | 81% to 83% | 95% to 100% | 165% to 173% |
|  | 10 | 45.9 to 48.1 | 188 to 197 | 64% to 66% | 69% to 72% | 101% to 105% | 174% to 182% |
| static-retrieval-mrl-en-v1, 1024, int8 table | 1 | 7.5 | 8 | 98% | 100% to 101% | 5% | 9% |
|  | 8 | 47.8 to 48.0 | 49 | 78% | 88% to 97% | 31% to 32% | 54% to 55% |
|  | 10 | 49.8 to 50.1 | 51 | 65% | 76% to 85% | 33% | 57% |

In texts per second, 8 workers give about 1.0 million WikiText texts and 5.4 to 6.3 million MASSIVE texts at 256 dimensions, and 0.45 million and 3.0 to 3.1 million at 1024 dimensions.

The stages at one worker, in ns per token (second run; mrl-en-v1 is static-retrieval-mrl-en-v1):

| model, dims | corpus | join | scan | caller | rest of encode | tokenize | pool | normalize | rest of tasks | serial fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| potion-base-8M, 256 | WikiText | 2.7 | 0.1 | 0.1 | 0.5 | 19.2 | 21.0 | 1.6 | -0.0 | 0.072 |
| potion-base-32M, 512 | WikiText | 2.7 | 0.1 | 0.3 | 1.5 | 19.5 | 49.3 | 3.0 | 3.2 | 0.055 |
| mrl-en-v1, 256 | WikiText | 2.7 | 0.1 | 0.2 | 0.1 | 18.2 | 20.7 |  | 0.0 | 0.068 |
| mrl-en-v1, 1024 | WikiText | 2.7 | 0.1 | 0.4 | -1.9 | 18.5 | 114.5 |  | 1.2 | 0.007 |
| mrl-en-v1, 1024, int8 | WikiText | 2.7 | 0.1 | 0.3 | 0.7 | 18.5 | 83.1 |  | 1.1 | 0.033 |
| potion-base-8M, 256 | MASSIVE | 1.2 | 0.0 | 0.3 | 0.6 | 43.4 | 22.2 | 10.8 | 1.0 | 0.023 |
| potion-base-32M, 512 | MASSIVE | 1.3 | 0.1 | 0.5 | -0.5 | 47.9 | 47.0 | 21.3 | 5.3 | 0.007 |
| mrl-en-v1, 256 | MASSIVE | 1.2 | 0.0 | 0.2 | 0.1 | 34.0 | 21.7 |  | 0.6 | 0.022 |
| mrl-en-v1, 1024 | MASSIVE | 1.2 | 0.1 | 0.7 | -0.0 | 34.7 | 99.5 |  | 2.6 | 0.009 |
| mrl-en-v1, 1024, int8 | MASSIVE | 1.2 | 0.1 | 0.6 | 2.8 | 35.6 | 95.4 |  | 0.1 | 0.030 |

The rests are differences of the fastest times of separate stages, so they can come out slightly negative (at most 1.9 ns per token here, under 2 % of the total).

### What limits each case

- **One worker** is at 86 to 99 % of the per-core bound: the stages of the workers are almost all of the time. Its share of the Amdahl bound (92 to 103 %) is no independent check, because the rests are defined so that the stages add up to the measured encode; it only shows the noise between the fastest stage times and the median of the paired passes.
- **256 and 512 dimensions** (POTION, the Matryoshka cut to 256): tokenizing and pooling take about the same time on WikiText; on MASSIVE, with 13 to 15 tokens per text, tokenizing takes twice the pooling of 256 dimensions and normalize adds 11 to 21 ns per token. The bound is the work of the workers. Eight workers lose against it in two measured ways: each task takes 2 to 11 % more wall time and 2 to 10 % more CPU time than alone, and the workers are busy only 70 to 100 % of the time outside the serial part (an estimate, from the serial time at one worker: the last tasks of a call and the hand-over between tasks). Ten workers are no faster than eight on WikiText (the tasks take 20 to 24 % more wall time than alone, and the busy share falls to 63 to 80 %) and 4 to 12 % faster on MASSIVE.
- **1024 dimensions** (static-retrieval-mrl-en-v1): pooling, timed on its own over the prepared batches, takes 100 to 115 ns per token, 72 to 83 % of the time at one worker. With 8 or 10 workers fse reads 164 to 197 GB/s of rows. In the probe the cores read at most 115.6 GB/s from memory sequentially, and fse's kernel at most 88.6 GB/s on consecutive rows. That fse reads more points to reuse from the caches: the 10,141 (MASSIVE) to 19,871 (WikiText) distinct rows of a corpus take 40 to 78 MiB, but a few frequent tokens make up most reads. The probe gives lower estimates, so this is an indication, not a proof. Earlier reports read the 130 to 180 GB/s against Apple's 200 GB/s and concluded that this case is near the memory bandwidth limit; the measured read roofs do not support that conclusion, though they cannot rule out that the misses alone saturate memory. Pooling is the largest measured stage per core; its time includes its waits for memory, so the measurements do not tell whether its arithmetic or those waits limit it.
- **The serial part** (mostly the join of the strings, `_append_str`, 1.2 to 2.8 ns per token, the largest serial stage in most cells) limits only far beyond these core counts: its Amdahl limit is 230 million tokens per second or more.

### Profiles

`make profile-cpu` (samply) of `roofline.py _encode` for 12 s, hottest functions over all threads:

- potion-base-8M, MASSIVE, 1 worker: `_wp_piece` 27.6 % (35.1 % with its callees), `fse_loop_f32_neon` (pooling) 20.5 %, `fse_normalize_body` 8.8 %, `_append_str` 5.8 %, `TextLookup._sentence` 4.9 %. The process includes loading the model.
- static-retrieval-mrl-en-v1 1024, WikiText, 8 workers: `fse_loop_f32_neon` 67.3 %, `__psynch_cvwait` 17.3 % (threads waiting: for a task, a result or the GIL), `_wp_piece` 9.9 %; the calling thread spends 10.7 % of its samples in `_append_str` and 64.7 % waiting.
- potion-base-8M, WikiText, 8 workers: `fse_loop_f32_neon` 37.2 %, `__psynch_cvwait` 25.9 %, `_wp_piece` 21.6 %, `_append_str` 2.9 % (17.5 % of the calling thread).

Memory of the 1024-dimension WikiText run at 8 workers: peak RSS 362 MiB (`make profile-rss`). memray (`make profile-mem`) shows a peak of 1.32 GB of allocations, of which 1.08 GB is one allocation made while modules are imported; the RSS shows that most of it is never touched. Per pass the result (21 MiB) and the joined buffer are the largest allocations.

## Where the workers idle

`benchmarks/runtime_timeline.py` observes the executor from the benchmark process only (nothing in `src`): the start and end of every task, and the time of the calling thread in the task iterator, in `reserve`, waiting for the oldest future and in `commit`. The idle area of a run, `N * (end of the last task - first submit) - the wall time of all tasks`, is split exactly (integer nanoseconds, the parts add up to the area) into the causes below. Every moment of an idle worker slot gets the first cause that holds:

| cause | meaning |
|---|---|
| tail | every task has started: nothing is left to hand out |
| ramp | the slot has not started its first task (its thread is not running yet, or the run is too small to use it) |
| hand-over | a task is submitted and waits in the queue while the slot is idle, or the calling thread is between two tasks |
| head-of-line | the calling thread waits for the oldest task with a full window, a later task exists and the queue is empty |
| producer | the calling thread is in the task iterator |
| sink | the calling thread is in `reserve` or `commit` |

What happens before the first submit and after the last task ends is outside the area. The tool lists the join of the strings, the time from the entry of the executor to the first submit (`enter`) and the time from the last task to the exit of the executor (pool shutdown and the last commit, `after`); the planning inside `infer` and the building of the result are not separated. Per thread the tool can show the rate in rows per busy second; a difference does not name a core type.

Measured on the M1 Pro of this document, master 14dac29, Python 3.13, one process per cell, `Encoder`-equivalent calls (`StringSource` and `Pipeline.infer`), 30 calls per worker count (6 rounds of 5, the median of the calls), under the shared benchmark lock. The load was 4.5 to 5.7 when a cell started (the tool discards a block of calls with a load above 5 before or after it, so a cell can lose blocks: one worker of potion-base-8M on MASSIVE lost 5 and ten workers 14 blocks, the others none), 4.97 before and 4.52 after the whole series. The machine was not quiet: other work kept the load near 5.

**The observation costs nothing measurable.** Plain call A, a second plain call B (the control) and a traced call, the fastest of 3 calls per round, 6 rounds in rotating order, 8 workers: B over A has a median of 1.006, 0.991 and 1.004 (ranges 0.987 to 1.017, 0.972 to 1.006, 0.966 to 1.170) and traced over A 1.004, 0.989 and 1.019 (ranges 0.983 to 1.022, 0.985 to 1.006, 0.867 to 1.034) on MASSIVE twice and WikiText once. The noise between two equal calls is therefore 1 to 3 % on MASSIVE and up to 17 % on WikiText, where a call takes 6 ms; every ratio below is read against that.

Share of the idle area by cause, potion-base-8M (256 dimensions), median of the calls; "idle" is the median of the total, and every column is a median of its own, so the columns need not add up to it (busy is 100 % minus idle):

| corpus | workers | tasks | wall ms | idle | tail | ramp | hand-over | head-of-line | producer + sink |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WikiText | 1 | 3 | 24.1 | 0.3 % | 0 | 0 | 0.08 % | 0 | 0.2 % |
| WikiText | 4 | 8 | 7.7 | 5.3 % | 2.0 % | 3.2 % | 0.2 % | 0 | 0 |
| WikiText | 8 | 16 | 5.6 | 20.2 % | 8.9 % | 10.0 % | 0.3 % | 0 | 0 |
| WikiText | 10 | 20 | 5.7 | 30.2 % | 15.8 % | 12.7 % | 1.3 % | 0 | 0.02 % |
| MASSIVE | 1 | 14 | 164.9 | 0.3 % | 0 | 0 | 0.07 % | 0 | 0.2 % |
| MASSIVE | 4 | 14 | 48.3 | 8.3 % | 7.7 % | 0.6 % | 0.1 % | 0 | 0 |
| MASSIVE | 8 | 30 | 28.6 | 17.0 % | 14.9 % | 1.8 % | 0.1 % | 0 | 0 |
| MASSIVE | 10 | 36 | 26.0 | 16.3 % | 10.3 % | 3.5 % | 1.0 % | 0.2 % | 0.04 % |

static-retrieval-mrl-en-v1 at 1024 dimensions:

| corpus | workers | tasks | wall ms | idle | tail | ramp | hand-over | head-of-line | producer + sink |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WikiText | 1 | 3 | 72.5 | 0.1 % | 0 | 0 | 0.04 % | 0 | 0.1 % |
| WikiText | 4 | 8 | 20.6 | 3.1 % | 2.1 % | 1.0 % | 0.1 % | 0 | 0 |
| WikiText | 8 | 16 | 13.7 | 18.3 % | 14.3 % | 3.6 % | 0.1 % | 0 | 0 |
| WikiText | 10 | 20 | 13.5 | 25.9 % | 16.7 % | 6.5 % | 1.0 % | 0 | 0.01 % |
| MASSIVE | 1 | 57 | 312.6 | 0.5 % | 0 | 0 | 0.13 % | 0 | 0.4 % |
| MASSIVE | 4 | 57 | 85.9 | 2.7 % | 2.1 % | 0.3 % | 0.2 % | 0.01 % | 0.02 % |
| MASSIVE | 8 | 57 | 49.8 | 11.3 % | 9.9 % | 1.0 % | 0.2 % | 0 | 0 |
| MASSIVE | 10 | 57 | 49.3 | 18.5 % | 11.6 % | 2.2 % | 1.7 % | 1.5 % | 0.08 % |

Short runs (the first texts of a corpus, potion-base-8M; still above the inline limit of 64 KiB):

| texts | workers | tasks | wall ms | idle | tail | ramp | other causes |
|---|---:|---:|---:|---:|---:|---:|---:|
| WikiText, 500 | 4 | 3 | 1.10 | 42.7 % | 23.5 % | 19.4 % | 0 |
| WikiText, 500 | 8 | 3 | 1.06 | 70.7 % | 48.7 % | 22.3 % | 0 |
| WikiText, 500 | 10 | 3 | 1.06 | 76.3 % | 54.5 % | 21.9 % | 0 |
| MASSIVE, 5000 | 4 | 4 | 1.95 | 28.0 % | 16.2 % | 11.7 % | 0 |
| MASSIVE, 5000 | 8 | 4 | 1.92 | 63.9 % | 48.6 % | 15.0 % | 0 |
| MASSIVE, 5000 | 10 | 4 | 1.94 | 71.2 % | 55.1 % | 15.9 % | 0 |

What the tables show:

- **The tail is the largest cause**, 8.9 to 16.7 % of the area at 8 and 10 workers on full corpora and 49 to 55 % on the short runs, where there are fewer tasks than workers (4 tasks, 8 or 10 workers). The numbers of tasks differ from the guess in the plan: WikiText makes 16 and 20 tasks at 8 and 10 workers, MASSIVE with potion-base-8M 30 and 36 (not 19), with mrl-en-v1 57.
- **The ramp is second**: 10.0 and 12.7 % on WikiText with potion-base-8M, 3.6 and 6.5 % with mrl-en-v1 (a longer task hides the same start-up), 15 to 22 % on short runs, 0.3 to 3.5 % on MASSIVE. The `enter` time (call to first submit) is 0.01 to 0.02 ms; the starts of the threads happen inside the area.
- **Head-of-line blocking is not a cause**: 0 to 0.2 % in every cell, 1.5 % in the one with ten workers, 57 tasks and a window of 20 (mrl-en-v1, MASSIVE). Only cells with more tasks than the window can show it at all, which is MASSIVE with both models.
- **Producer and sink are not causes**: together 0.2 to 0.4 % at one worker (the commits of an inline run) and below 0.1 % with threads; the final commit after the last task (0.01 to 0.03 ms) is outside the area.
- **Hand-over** is 0.04 to 0.3 % up to 8 workers and 1.0 to 1.7 % at 10.
- **Threads at different rates**: not reported. The first version of the tool compared the rate of the n-th started thread across calls, which names different threads in each call and hides the spread; the tool now takes the slowest over the fastest rate within each call (`thread_spread`), and the figure has not been re-measured.
- The shutdown of the pool and the result after the last task take 0.16 to 0.35 ms with threads (the `after` column of the tool), 0.3 to 9 % of the call, most on the short ones.

### The ablations

All from the benchmark process, no change in `src`, variants alternating in rounds of 3 calls (4 rounds); the ratio is the median over rounds of a variant's wall time over the first variant's, with the range, so read it against the noise of 1 to 3 % (MASSIVE) and up to 17 % (WikiText) above.

- **A1, task size** (`chunk_bytes` for 1, 2, 3, 4, 8 tasks per worker; `chunk_bytes` does not force a count, the tool prints the real cuts). WikiText, potion-base-8M, 8 workers: 1/worker gives 8 tasks, 2/worker 16 (the default plan, the same cuts), 3/worker 24, 4/worker 32, 8/worker 64. Wall time over the default 0.983 (0.921 to 1.393), 1.014 (0.952 to 1.666), 1.056, 1.088 (0.886 to 1.094) and 1.146 (0.943 to 1.242). The tail falls from 8.9 % to 4.1 % at 64 tasks, but hand-over rises to 2.4 % and the run does not get faster. mrl-en-v1, MASSIVE, 8 workers: the settings 1 to 4 per worker all give the default 57 tasks (the row cap of the plan binds), 8/worker gives 65 tasks and an idle share of 5.7 % against 11.8 % and a ratio of 0.954 (0.873 to 1.059). Task size alone does not show a gain outside the noise.
- **A2, window** (experimental: the same task plan, only the window passed to the executor is 1, 2, 4 times or unbounded, which lifts the cap of WINDOW_BYTES on rows in flight). WikiText, 8 workers: 16 tasks fit every window, so nothing changes by construction (ratios 1.130, 0.936, 0.944, all within the noise); peak resident set per process after one call with the normal window: growth 0.2, 3.3, 0.2 and 0.1 MiB for x1, x2, x4 and unbounded. mrl-en-v1, MASSIVE: peak tasks in flight 16, 32, 57 and 57 (rows 44,531, 87,953, 151,674), head-of-line 0.00 % in all, ratios 0.996, 1.029, 1.030. A wider window does not remove idle time.
- **A3, pool** (a thread pool that outlives the call, its threads started, against one per call; the whole effect of reuse, not only of the start of the threads, and it breaks "no thread outlives a call"). WikiText, potion-base-8M, 8 workers: wall time 0.784 (0.737 to 1.113) of the per-call pool; idle 30.1 % to 13.1 %, tail 19.2 % to 5.7 %, ramp 10.2 % to 7.1 %, the time after the last task 0.27 to 0.04 ms. This is an upper bound, and with a range reaching 1.113 it is a single noisy cell (the cell of MASSIVE lost its blocks to the load).
- **A4, sink** (the same preallocated target array, `ArrayRows.commit` against a commit that does nothing): ratios 1.024 (0.826 to 1.157) and 1.001 (0.826 to 1.196) against the call with the sink of `infer`; sink idle 0.00 % in all three. Nothing to gain at the commit.

### Profiles

potion-base-8M, WikiText, 8 workers, the deciding cell of the ramp and tail (`make profile-cpu`, samply, 12 s, all threads, 75,747 samples): `fse_loop_f32_neon` 36.3 %, `__psynch_cvwait` 27.2 % (55.2 % of the samples of the calling thread, 16.3 % in `_append_str`), `_wp_piece` 21.5 %, `_append_str` 2.8 %, `fse_normalize_body` 2.7 %. `make profile-mem` (memray, native): peak 1.194 GB, of which one allocation of 1.074 GB is made while modules are imported (as in the profile above); the allocations of the run are the joined strings (`_strings.py:74`), the result rows (`_sinks.py:116`) and the pooled rows (`_cpu.py:116`). `make profile-rss`: peak 241 MiB at 8.5 s, 239 MiB at the end (8 workers, 106 samples). The profiles ran at a rising load (7.6 after) and are no timings.

### What this says for the next changes (DR-180)

- **Tail** (largest, 9 to 17 % of the area at 8 and 10 workers, 49 to 55 % on short runs): PR B (a piece count that is a multiple of the workers) can only move the cells whose task count is not one, such as MASSIVE potion-base-8M with 30 tasks at 8 workers (14.9 % tail); PR C (splitting the last tasks, factoring) targets all of it and is measured in "The tail split" below: it cuts the tail share by 40 to 60 % in the MASSIVE cells, and leaves the short runs (3 or 4 chunks near the minimum size) as they are. A1 shows that task size alone (C3, TASKS_PER_WORKER) does not give a gain outside the noise, so it needs its own measurement in PR B.
- **Ramp** (10 to 13 % on WikiText with potion-base-8M, 15 to 22 % on short runs): the ramp share also counts slots that no task could use and the wait for the first assignment, and it is a share of the worker area, not of the call time, so it is no bound on the saving. A3 gives 0.78 for full reuse in one noisy cell; whether an in-contract start of the threads (C5) can get part of it needs a maintainer decision and its own measurement.
- **Head-of-line blocking, producer, sink** (0 to 1.5 %): PR D (a window counted in rows) and any change at the sink are not supported by the numbers and should not be built.

## The tail split (DR-180)

The automatic plan (`chunk_bytes=None`) cuts each of the last `workers` chunks into two parts of about equal rows (`_Buffered.split_tail`), in the calling thread before the tasks run, when both parts keep the minimum chunk size (64 KiB). An explicit `chunk_bytes`, one worker, inline runs and streams keep their cuts.

Measured as in "Where the workers idle" (M1 Pro, Python 3.13, `runtime_timeline.py measure`, 8 rounds of 5 calls, the median wall time of the call), base (master 77e57e3) and branch alternating twice, each run under the shared benchmark lock at a load below 4 at its start (the load reached 4.4 at the end of two runs). Wall time in ms (the two runs of base, the two of the branch), tasks, and the tail share and the idle share of the means of the two runs:

| cell | workers | tasks | wall base | wall branch | branch / base (means) | tail share | idle share |
|---|---:|---:|---|---|---:|---|---|
| potion-base-8M, MASSIVE | 1 | 14 | 165.6, 165.4 | 164.6, 172.8 | 1.019 | 0 | 0.3 % |
| | 4 | 14 to 18 | 48.3, 48.3 | 46.0, 46.3 | 0.955 | 7.9 to 3.1 % | 8.6 to 3.8 % |
| | 8 | 30 to 38 | 28.5, 29.2 | 26.5, 30.9 | 0.994 | 13.9 to 5.7 % | 16.3 to 10.7 % |
| | 10 | 36 to 46 | 26.9, 29.2 | 26.3, 29.2 | 0.991 | 10.5 to 5.7 % | 20.8 to 19.5 % |
| mrl-en-v1 1024, MASSIVE | 8 | 57 to 61 | 49.8, 50.2 | 47.2, 47.5 | 0.947 | 10.1 to 3.8 % | 11.6 to 5.7 % |
| | 10 | 57 to 63 | 50.3, 49.0 | 48.0, 46.9 | 0.956 | 11.7 to 6.3 % | 18.3 to 14.4 % |
| potion-base-8M, WikiText | 4 | 8 to 12 | 7.86, 7.70 | 7.93, 7.92 | 1.019 | 2.3 to 3.5 % | 5.7 to 6.9 % |
| | 8 | 16 to 24 | 5.50, 5.48 | 5.60, 5.66 | 1.026 | 7.7 to 7.3 % | 18.1 to 18.0 % |
| | 10 | 20 | 5.82, 5.70 | 6.04, 5.77 | 1.025 | 15.5 % | 30.4 to 31.9 % |
| mrl-en-v1 1024, WikiText | 8 | 16 to 24 | 13.75, 14.39 | 13.86, 13.14 | 0.959 | 15.5 to 9.2 % | 19.8 to 14.1 % |
| | 10 | 20 | 13.02, 13.45 | 14.13, 13.50 | 1.044 | 16.9 to 16.4 % | 24.6 to 27.6 % |
| MASSIVE, 5000 texts | 4, 8, 10 | 4 | 1.92 to 2.03 | 1.92 to 2.10 | 1.02 to 1.07 | unchanged | 28 to 71 %, unchanged |
| WikiText, 500 texts | 4, 8, 10 | 3 | 1.03 to 1.28 | 1.06 to 1.28 | 0.98 to 1.13 | unchanged | 41 to 77 %, unchanged |

Where the plan has chunks of at least 128 KiB the tail share falls by 40 to 60 % in the MASSIVE cells and with mrl-en-v1 on WikiText at 8 workers (and by 5 % with potion-base-8M there), the idle share falls with it where the ramp does not rise, and MASSIVE gets 4.5 to 5.3 % faster at 4 workers and with mrl-en-v1 at 8 and 10 workers, in both pairs. The 8 workers of potion-base-8M on MASSIVE are inconclusive in this series (one branch run at a load of 4.4: 26.5 and 30.9 ms); five more runs of the branch in the tuning series (one halving, the same code) gave a median of 26.9 ms against 28.9 ms for six runs of the base. The WikiText cells with 20 tasks (10 workers) are not split at all, so their 2.5 and 4.4 % are the noise of two identical plans; in the other WikiText cells the idle time moves less than the noise (1 ms and 6 ms calls vary by 3 to 13 % between identical runs). The short runs have 3 or 4 chunks near 64 KiB, which the minimum size does not split: they are unchanged.

**Variants tried (the tuning series, same method):** halving the last chunks twice (two rounds) gave the same gain as once (potion-base-8M MASSIVE 4, 8, 10 workers: 46.0, 26.5, 26.2 ms against 46.2, 26.9, 25.5 for one round), with more tasks. Parts down to 16 KiB (a quarter of the minimum chunk size) split the short runs too: MASSIVE with 5000 texts fell from 1.93 to 1.52 ms at 8 workers (8 tasks, 20 % faster) and WikiText with 500 texts rose from 1.06 to 1.28 ms (9 tasks, ramp 21 to 37 %), because a thread start costs about as much as such a part: texts that cost less per byte lose what the others gain. The minimum chunk size stays.

**End to end** (`bench_runtime.py throughput --dim 256 --dtypes float32 --workers 1 4 8 10 --rounds 5`, 200,000 synthetic sentences, base and branch alternating in 4 pairs; ratio of the branch's sentences per second to the base's, median and range; the pairs ran at a load of 3.4 to 4.8 at their start and end, a fifth pair with a load of 5.5 at the end of one run was discarded):

| steps | workers | branch / base | range | spread of the base over the pairs |
|---|---:|---:|---|---:|
| plain | 1 | 1.009 | 0.972 to 1.340 | 1.34 |
| | 4 | 1.126 | 1.094 to 1.151 | 1.03 |
| | 8 | 0.975 | 0.813 to 1.204 | 1.22 |
| | 10 | 1.055 | 0.917 to 1.082 | 1.05 |
| normalize | 1 | 0.999 | 0.965 to 1.040 | 1.04 |
| | 4 | 1.119 | 1.049 to 1.137 | 1.01 |
| | 8 | 1.110 | 0.766 to 1.144 | 1.03 |
| | 10 | 1.005 | 0.807 to 1.147 | 1.14 |

One worker does not change (the plan is the same by construction); 4 workers gain 12 %, outside a base spread of 1 to 3 %; the 8 and 10 worker cells have a base spread of 3 to 22 %, so their medians (plain at 8 workers 0.975, the others 1.005 to 1.110) show no regression that the spread does not cover.

**Profile** (potion-base-8M, MASSIVE, 8 workers, 12 s, all threads; samples, no timings; the load was 3.9 at the start): `_wp_piece` 28.7 and 29.4 % (base, branch), `fse_loop_f32_neon` 20.6 and 20.9 %, `__psynch_cvwait` 14.8 and 13.3 %, `fse_normalize_body` 9.0 and 9.4 %, `_append_str` 6.0 and 5.9 %. Memory (`make profile-mem`, memray, native): peak 431.9 and 431.6 MB, the allocation sites the same (`_sinks.py:116`, the result rows: 42.1 and 44.9 GB over the whole run, because the branch made 6.6 % more passes in the same 8 s under memray, which is no timing). `make profile-rss` (8 workers, 8 s): peak 366 MiB against 368 MiB (+0.5 %).

## A pool reused across calls

Part of DR-180 (Leerlauf der Worker am Laufende messen und verringern). This experiment changes only the benchmark process. The production rule "No thread outlives a call" remains in force ([docs/architecture/runtime.md](docs/architecture/runtime.md), "Runtime"). Reuse measures startup avoidance, shutdown avoidance and retained thread state together, not startup alone.

Measured on the same M1 Pro (10 cores, 16 GiB), Python 3.13.16, source base `140e9e2`, with the model snapshots pinned in `roofline.py`. `bench_pool.py` runs base, a second base (A/A control) and reuse in fresh child processes, alternating their order. Every cell has three valid pairs, with load strictly below 5 before and after every side. Calls include `StringSource` construction and `Pipeline.infer`, without tracing. Full and short corpora use seven individual calls per side. The server cells time one block of 1000 sequential calls per side. Ratios compare each side's median time per call within each pair, then take the median over pairs. First warm and last timed outputs are checked against base with uint32 views, outside the clock. Warmup outputs are released before timing. Peak process RSS includes loading and bit checks as well as inference, and counts an old result still live while its replacement is built. It is not a steady-state footprint. [Raw kept and discarded rounds](benchmarks/baseline/thread_pool_20261031.json).

Rule fixed in the draft PR before timing: a gain requires median reuse/base at most 0.97. It is beyond noise only if every ratio is below min(1, minimum A/A, reciprocal maximum A/A) from the same cell. A regression is beyond noise only if every ratio exceeds the symmetric upper bound. Peak RSS may rise by at most 5 %. These classify evidence for the maintainer, without authorizing a lifetime change. In the tables, "gain/noise" means median gain of at least 3 % / beyond the observed noise, and RSS is base/control/reuse. Each range is over the three pairs.

**potion-base-8M (256 dimensions)**

| corpus and size | workers | base / reuse ms | reuse/base median (range) | A/A median (range) | gain/noise | peak RSS MiB | load range |
|---|---:|---|---|---|---|---|---|
| massive, full | 4 | 45.979 / 45.890 | 0.998 (0.994 to 0.998) | 1.009 (0.997 to 1.012) | no/no | 483.8/484.3/485.3 | 4.30 to 4.71 |
| massive, full | 10 | 25.690 / 25.897 | 1.015 (0.980 to 1.018) | 0.996 (0.990 to 1.000) | no/no | 486.9/487.3/487.3 | 4.25 to 4.36 |
| massive, 5000 texts | 4 | 1.967 / 1.849 | 0.945 (0.886 to 0.977) | 0.990 (0.969 to 1.053) | yes/no | 126.4/126.6/125.5 | 4.25 to 4.55 |
| massive, 5000 texts | 8 | 1.990 / 1.845 | 0.946 (0.918 to 0.956) | 1.003 (0.944 to 1.006) | yes/no | 126.2/126.1/128.0 | 4.38 to 4.58 |
| massive, 5000 texts | 10 | 1.998 / 1.829 | 0.937 (0.915 to 0.939) | 0.972 (0.971 to 1.059) | yes/yes | 125.8/127.6/128.1 | 4.43 to 4.55 |
| massive, 1000 server calls | 4 | 1.031 / 0.897 | 0.870 (0.864 to 0.878) | 0.974 (0.964 to 1.001) | yes/yes | 143.2/141.8/141.9 | 4.05 to 4.89 |
| massive, 1000 server calls | 10 | 1.009 / 0.917 | 0.913 (0.895 to 0.917) | 1.007 (0.997 to 1.027) | yes/yes | 139.8/141.8/139.2 | 4.24 to 4.87 |
| massive, 1000 inline calls | 4 | 0.064 / 0.063 | 0.980 (0.972 to 1.007) | 1.019 (1.000 to 1.064) | no/no | 120.3/119.9/119.1 | 4.46 to 4.58 |
| wikitext, full | 4 | 7.923 / 7.713 | 0.973 (0.961 to 0.992) | 1.006 (0.983 to 1.016) | no/no | 201.1/201.5/201.6 | 3.87 to 4.03 |
| wikitext, full | 8 | 5.910 / 5.653 | 0.954 (0.944 to 1.002) | 1.013 (0.991 to 1.022) | yes/no | 202.6/202.1/205.1 | 4.59 to 4.73 |
| wikitext, full | 10 | 5.783 / 5.331 | 0.928 (0.914 to 0.936) | 1.011 (0.981 to 1.018) | yes/yes | 204.3/202.1/204.2 | 3.80 to 3.87 |
| wikitext, 500 texts | 4 | 1.076 / 0.904 | 0.839 (0.817 to 0.878) | 0.986 (0.977 to 1.036) | yes/yes | 184.9/184.5/185.7 | 3.80 to 3.82 |
| wikitext, 500 texts | 10 | 1.104 / 0.979 | 0.898 (0.830 to 0.904) | 0.969 (0.953 to 0.975) | yes/yes | 184.4/187.2/184.5 | 3.82 to 4.07 |
| wikitext, 1000 server calls | 4 | 0.629 / 0.514 | 0.819 (0.815 to 0.864) | 0.992 (0.987 to 1.039) | yes/yes | 186.1/190.2/185.8 | 4.07 to 4.52 |
| wikitext, 1000 server calls | 8 | 0.614 / 0.506 | 0.822 (0.789 to 0.842) | 1.008 (0.987 to 1.032) | yes/yes | 186.0/189.1/187.7 | 4.55 to 4.59 |
| wikitext, 1000 server calls | 10 | 0.617 / 0.504 | 0.818 (0.719 to 0.927) | 0.998 (0.878 to 1.009) | yes/no | 187.6/186.3/184.4 | 4.52 to 4.88 |

**static-retrieval-mrl-en-v1 (1024 dimensions)**

| corpus and size | workers | base / reuse ms | reuse/base median (range) | A/A median (range) | gain/noise | peak RSS MiB | load range |
|---|---:|---|---|---|---|---|---|
| massive, full | 4 | 86.685 / 86.257 | 1.001 (0.983 to 1.001) | 0.996 (0.988 to 1.020) | no/no | 1588.8/1588.5/1591.6 | 4.17 to 4.70 |
| massive, full | 10 | 46.474 / 46.509 | 1.001 (0.971 to 1.035) | 1.047 (0.995 to 1.073) | no/no | 1590.1/1591.1/1592.1 | 4.06 to 4.98 |
| massive, 5000 texts | 4 | 3.807 / 3.651 | 0.956 (0.955 to 0.990) | 1.004 (0.983 to 1.013) | yes/no | 268.3/265.4/265.4 | 4.63 to 4.66 |
| massive, 5000 texts | 10 | 3.870 / 3.663 | 0.977 (0.896 to 0.995) | 0.990 (0.971 to 0.991) | no/no | 266.3/267.6/265.5 | 4.66 to 4.69 |
| massive, 1000 server calls | 4 | 1.697 / 1.577 | 0.929 (0.919 to 0.935) | 1.005 (0.995 to 1.008) | yes/yes | 308.3/308.1/308.1 | 4.31 to 4.93 |
| massive, 1000 server calls | 10 | 1.704 / 1.587 | 0.929 (0.908 to 0.940) | 0.995 (0.984 to 0.999) | yes/yes | 308.8/307.9/311.8 | 4.06 to 4.58 |
| wikitext, full | 4 | 21.109 / 20.971 | 0.996 (0.986 to 1.003) | 0.999 (0.994 to 1.008) | no/no | 342.2/343.1/343.7 | 4.68 to 4.74 |
| wikitext, full | 8 | 14.229 / 13.868 | 0.981 (0.957 to 0.995) | 1.011 (1.001 to 1.035) | no/no | 345.7/344.7/345.7 | 4.38 to 4.83 |
| wikitext, full | 10 | 13.586 / 12.763 | 0.939 (0.935 to 0.949) | 0.984 (0.966 to 1.010) | yes/yes | 344.4/344.7/343.5 | 4.50 to 4.68 |
| wikitext, 500 texts | 4 | 2.501 / 2.561 | 1.000 (0.919 to 1.149) | 1.044 (1.011 to 1.114) | no/no | 295.6/295.7/297.6 | 4.38 to 4.50 |
| wikitext, 500 texts | 8 | 2.675 / 2.352 | 0.867 (0.835 to 0.954) | 0.930 (0.888 to 1.032) | yes/no | 295.9/293.3/294.8 | 4.49 to 4.61 |
| wikitext, 500 texts | 10 | 2.635 / 2.344 | 0.898 (0.853 to 0.921) | 0.963 (0.932 to 1.016) | yes/yes | 296.8/295.5/299.9 | 4.27 to 4.38 |
| wikitext, 1000 server calls | 4 | 1.414 / 1.278 | 0.907 (0.904 to 0.909) | 0.996 (0.994 to 1.006) | yes/yes | 292.4/294.2/294.7 | 4.25 to 4.63 |
| wikitext, 1000 server calls | 10 | 1.409 / 1.266 | 0.898 (0.895 to 0.917) | 1.013 (0.994 to 1.026) | yes/yes | 293.8/292.8/293.8 | 4.49 to 4.69 |

Server requests accumulate consecutive whole texts until their joined UTF-8 bytes first exceed the normal 65,536-byte inline threshold. Actual request sizes (identical for both models):

| corpus | texts per server call | UTF-8 bytes per call | inline requests |
|---|---|---|---|
| wikitext | 76 to 260 | 65,537 to 67,315 | 0 of 1000 |
| massive | 452 to 2417 | 65,537 to 65,900 | 0 of 1000 |

The inline control uses 32 MASSIVE texts per call (609 to 4,834 bytes), below the threshold in all 1000 requests. It never uses a pool, so a ratio there is a control for noise, not evidence of reuse.

Eight workers are measured only where the 4- and 10-worker conclusions differ (gain beyond noise with acceptable RSS, regression beyond noise, or inconclusive). The remaining 8-worker cells are omitted to bound the experiment. Short prefixes use the tail split already on the base. The combination with a 16 KiB minimum chunk remains a follow-up.

Reuse passes both time criteria in 14 of 29 target cells. Full MASSIVE shows no gain beyond noise at either worker count, while full WikiText does at 10 workers with both models. Short and server results vary by cell, so the tables retain inconclusive rows. No cell regresses beyond the observed A/A range, and every peak RSS is within 5 % (largest ratio 1.019). The inline control shows no gain beyond noise.

Prestarting all workers to a barrier, outside timing, costs 4 workers: 0.378 ms (0.326 to 0.526), 8 workers: 0.598 ms (0.508 to 0.927), 10 workers: 0.714 ms (0.560 to 0.898). This is full-pool initialization, not the lazy start cost of one call, which may start fewer threads.

The corrected matrix took 325.7 active seconds including child startup and loading. Load pauses are separate. 9 rounds crossed the load limit and are retained as discarded data. The initial series was replaced in full after removing retained byte copies used for bit checks, which inflated RSS. It is not combined with this series.

**Timeline and profiles, deciding cell:** potion-base-8M, first 5000 MASSIVE texts, 10 workers, both variants. Three paired traced rounds of three calls at load 4.63, no discarded round. Every pass has an exact disjoint idle split; columns below are separate medians. [Raw timeline and profile summaries](benchmarks/baseline/thread_pool_profiles_20261031.json).

| variant | traced wall ms | idle share | tail share | ramp share | after last task ms |
|---|---:|---:|---:|---:|---:|
| per call | 1.966 | 70.86 % | 55.91 % | 15.03 % | 0.144 |
| reuse | 1.820 | 70.95 % | 61.66 % | 10.48 % | 0.057 |

Head-of-line, producer, sink and hand-over shares have zero medians. Four tasks leave six worker slots unused. The call becomes shorter while the idle share stays near 71 %. Time after the last task includes the final commit and pool shutdown.

CPU (`make profile-cpu`, samply, 6 s per variant): `_wp_piece` 17.7/11.1 %, `fse_loop_f32_neon` 17.3/8.0 %, `fse_normalize_body` 7.5/4.5 %, `__psynch_cvwait` 38.6/66.4 % (per call/reuse, self samples). The all-thread sample populations differ because reused workers remain alive. These shares are not speedup measurements.

Memory (`make profile-mem`, native memray, 3 s each): peak live allocations 74.58/77.20 MiB. The largest cumulative allocation sites are result rows (`_sinks._new`), joined strings (`StringSource.__init__`) and pooled rows (`text_rows`). Memray lacks interpreter debug symbols, limiting native stack attribution. Process-tree RSS (`make profile-rss`, 3 s each): peaks 146.89/151.91 MiB. The profile phase took 38.9 s under the lock, at load 4.63 before and 6.50 after. It supplies attribution and memory peaks, not wall-time decisions.

A production persistent pool would need a defined owner and lifetime, explicit close or atexit behavior, fork safety, cleanup after KeyboardInterrupt, concurrent and nested-call behavior, and the existing GIL-budget tests without weaker limits. The benchmark patch is process-global, sequential and limited to this experiment. None of those production changes is implemented.

## Store

### Sync costs (31.10.2026)

Benchmark-only ablations on an Apple M1 Pro (10 cores, 16 GiB), macOS 27.0.1, CPython 3.13.16, source `7ec7ba1`. `bench_store_sync.py` runs each side in a fresh process on a pre-initialised durable store. Store creation is counted separately. Write times include encoding, sink flush and close, plus the final sync point for `deferred` and `flush_model`. Verification and partition-independent tensor digests run outside timing. C3 warms the same side without verification before measuring. Process peak RSS includes preparation and warm-up, but excludes store verification.

Rule registered in the draft PR before timing: at least three valid pairs. Every run must have load strictly below 5 before and after it. Beyond noise means every side/full ratio is below min(min A/A, 1/max A/A, 1). Bulk needs median `flush_model/full` at most 0.90 in C2 default float32 or C3. Stream needs median `none/full` at most 0.50 at 10,000 sentences per batch. `deferred` is only an optimistic lower bound. Larger shards under `full` use the same bulk threshold and noise rule.

| Cell | Valid pairs | full median seconds | none/full | deferred/full | flush_model/full | A/A range | Run loads |
|---|---:|---:|---:|---:|---:|---|---|
| C1-1000 | 3 | 2.283 | 0.169 | 0.194 | 0.197 | 0.943 to 1.040 | 3.72 to 4.62 |
| C1-10000 | 3 | 0.279 | 0.387 | 0.415 | 0.446 | 0.867 to 1.036 | 3.82 to 4.33 |
| C1-65536 | 3 | 0.277 | 0.642 | 0.683 | 0.722 | 0.982 to 1.107 | 3.46 to 4.81 |
| C2-default | 3 | 0.105 | 0.710 | 1.370 | 0.987 | 0.857 to 1.082 | 4.12 to 4.47 |
| C2-small-f32 | 3 | 0.215 | 0.399 | 0.444 | 0.463 | 0.721 to 0.981 | 4.10 to 4.91 |
| C2-small-i8 | 3 | 0.207 | 0.255 | 0.289 | 0.350 | 0.917 to 1.094 | 3.47 to 4.10 |
| C3 | 3 | 0.613 | 0.572 | 0.908 | 1.345 | 0.681 to 1.284 | 3.92 to 4.88 |

C1 seals and publishes after every batch (200,000 sentences at 1,000 and 10,000 per batch, 524,288 at 65,536). C2 writes 250,000 synthetic sentences (float32 default shards, float32 and int8 at 16,384 rows). C3 uses the cached 20M-token corpus and a finished SIF-weighted table without component removal (about 1.33 GB of float32 vectors). All use 10 workers. `fsync` replaces full drive flushes with plain fsyncs. `flush_model` includes an initial marker sync, data and directory fsyncs, a full sync, marker replacement and a final full sync. Its hidden marker models costs only. `large4` uses four times the default shard size, `single` one shard, both under `full`.

C0, 50 samples per primitive and target, median/p90 milliseconds from a dirty file or directory after restoring a durable state:

| Primitive | File | Directory |
|---|---:|---:|
| fsync | 0.099/0.158 | 0.003/0.007 |
| full | 3.957/4.894 | 3.418/3.823 |
| barrier | 0.767/1.007 | 0.693/0.967 |

C0 load: 4.54 before, 4.33 after. F_BARRIERFSYNC uses macOS fcntl command 85.

The fixed rule selects **full + none in 2.0, flush in 2.1**. The 10,000-row stream saves 61.3 percent at median, beyond its A/A noise. Bulk `flush_model/full` medians are 0.987 (C2 default) and 1.345 (C3), so neither passes B. C3's A/A range is wide (0.681 to 1.284). A single full shard has median ratio 0.836 there, but does not pass the noise rule. Larger shards remain a measured alternative, without a qualifying gate result.

`pend1g` and `pend1g_n` have C3 paired median ratios 0.872 and 0.701. Neither is consistently beyond noise (the latter's pairs are 0.640, 0.701, 1.189). Their peak RSS ratios are 1.042 to 1.069 and 1.093 to 1.183, within 1.25. Keep both current queue constants. C0 qualifies the barrier by cost: median barrier/full ratios 0.194 on files and 0.203 on directories. The last `ShardWriter.finish` is 2.1 to 3.0 percent of C3 wall time. Encode flushes before returning, and close adds about 0.1 ms. Do not build an async final flush.

The shared timing/profile budget used 26.4 minutes including load waits. Five nonempty rounds (24 child runs) were discarded, with five individual runs crossing load 5. Another 51 empty attempts waited for load. One extra CPU profile crossed the limit and was discarded. All seven cells retain three complete valid pairs. All 156 retained child runs have matching tensor digests, expected sync counts, valid endpoints and load below 5. In particular `none` records zero write-phase syncs. Store creation records three full syncs separately.

CPU, native-memory and RSS profiles use `make profile-cpu`, `make profile-mem` and `make profile-rss` on the deciding configurations. Every accepted profile starts and ends below load 5 (3.47 to 4.98). Native peaks below are memray's reported live-allocation peaks in decimal MB. RSS is the sampled peak of the whole command's process tree. Profiles include preparation, C3 warm-up and post-timer verification, so their memory peaks do not isolate sink buffers.

| Configuration | Native peak MB | RSS peak MiB |
|---|---:|---:|
| C1-full | 508.7 | 922.6 |
| C1-none | 509.7 | 734.3 |
| C2-full | 755.2 | 1151.1 |
| C2-flush_model | 755.2 | 916.0 |
| C3-full | 2186.0 | 3320.3 |
| C3-flush_model | 2185.0 | 3465.5 |
| C3-pend1g_n | 2186.0 | 2541.8 |

CPU self samples: C1/full spends 18.3 percent in `__psynch_cvwait` and 5.2 in `__fcntl` (0 recorded write syncs under none). C2/full includes corpus preparation (`random_zipf` 6.5 percent), `XXH3_64bits_update` 4.0, `pwrite` 2.7 and `fse_loop_f32_neon` 2.0. C3/full shows `fse_loop_f32_neon` 17.7, `__psynch_cvwait` 16.3, `_platform_memmove` 15.3, `pwrite` 8.3 and hashing 5.0 percent. C3/pend1g_n shows pooling 41.5 percent. The populations include waits and verification, and these shares are not speedup measurements. Largest cumulative allocation sites include numpy memmaps, `NpyContainer.verify`, `ShardWriter.reserve` and model preparation (`_wordvectors_format._frozen`), with the synthetic corpus and table preparation prominent in C1/C2. Cumulative allocation totals are not simultaneous memory peaks.

[Raw kept and discarded runs, sync counts, endpoints and profile metadata](benchmarks/baseline/store_sync_20261031.json). These measurements concern this local filesystem and do not test power loss, Windows, bare-metal Linux or network filesystems.

### Implemented `durability="none"` (31.10.2026)

The real writer policy (`none_impl`) was measured against the full writer and the sync-ablation `none`, on the same M1 Pro, macOS 27.0.1 and CPython 3.13.16. Code head `c603984` includes merged base `2b75759`. Every kept timed child has load strictly below 5 before and after (3.36 to 4.984). The main series and fixed full-writer confirmation used 7.05 minutes of job time including waits and eight profiles, spanning 10.45 minutes with the intervening analysis. Three children crossing load 5 were discarded, with every result retained in the raw data. Store creation stays fully synced and is outside write timing. Digests compare tensor bytes and identical `store.json`; the expected durability tags are checked separately.

| Comparison | Pairs | Median paired ratio | Symmetric A/A band | Result |
|---|---:|---:|---|---|
| C1-10000 `none_impl/full` | 3 | 0.371916 | 0.961816 to 1.039700 (full control) | at most 0.50, every ratio below the noise floor |
| C1-10000 `none_impl/none` | 3 | 0.862463 | 0.859843 to 1.163004 (ablation control) | within A/A |
| Default full writer/base, fixed confirmation | 3 | 1.010353 | 0.842792 to 1.186533 (base control) | within A/A |

C1 uses 200,000 sentences, batches of 10,000, float32 and 10 workers. Independent median write times are 0.294475 seconds for full, 0.113770 for `none_impl` and 0.131913 for the ablation. The table uses medians of paired ratios, not ratios of those independent medians. Individual implementation/ablation ratios are 1.046946, 0.862463 and 0.810403, the last outside its A/A band; the registered median criterion passes. An additional initial full-writer/base median was 0.961028, just below its branch-full A/A lower bound 0.961816. This small speedup triggered three fixed confirmation pairs with an independent base/base control, without changing the criterion. Their individual writer ratios are 0.854499, 1.010353 and 1.078300. No speedup of the default writer is claimed.

The required `bench_runtime.py throughput` float32 base/branch gate also has three alternating pairs plus three base/base controls. All four median throughput ratios are inside their corresponding A/A bands:

| Normalization steps | Workers | Branch/base median | Symmetric A/A band |
|---:|---:|---:|---|
| 0 | 1 | 1.016897 | 0.981676 to 1.018666 |
| 0 | 10 | 1.031263 | 0.878878 to 1.137815 |
| 1 | 1 | 1.003363 | 0.987329 to 1.012834 |
| 1 | 10 | 1.019905 | 0.948581 to 1.054206 |

C2 and C3 each have one informative full/implementation pair, with verification outside the timer. C2 default: 0.217526/0.141617 seconds (full/none). C3: 0.776175/1.331381 seconds. **None is slower in that C3 pair; no bulk speedup is claimed.** C3 still warms each side before its timed write. These single pairs are not a bulk performance gate.

A separate C3 store was written with the real none policy: 1,105,490 rows, 300 dimensions, six tagged shards and 1,331,009,960 tensor bytes. Its tensor digest and store identity match the full fixture. Three alternating warm-open pairs, five opens per child, give median-of-child-medians 1.468 ms for full and 189.846 ms for none. Every none open reports all 1,331,009,960 bytes verified, full reports zero. Existing empty-store sink construction, three pairs with 20 opens per child, gives 0.539 ms for full and 0.508 ms for none; close is excluded and neither policy rewrites the identity. A full sink cannot open the tagged C3 store for append, so the constructor comparison uses the same empty store.

CPU, native-memory and RSS profiles cover C1 `none_impl` and C3 verified opens; the runtime gate also has CPU and native-memory profiles. Write profiles include preparation and warm-up but exclude post-write correctness verification. Reader profiles include import and warm-up. Sample shares include waiting threads and are not speedup measurements. Whole-process peaks therefore do not isolate sink buffers. Native peaks include tracked mappings and are decimal MB; RSS covers the sampled process tree in MiB.

| Configuration | Native peak MB | RSS peak MiB |
|---|---:|---:|
| Runtime throughput, float32, 10 workers | 1160.318 | not separately required |
| C1 none implementation, 10 workers | 413.477 | 564.984 |
| C3 verified open | 1348.303 | 1302.313 |

Runtime CPU self samples: `fse_loop_f32_neon` 18.8 percent, `__psynch_cvwait` 14.2, `fse_tokenize` 6.9 and `_platform_memmove` 6.1 (3,905 samples). C1 none: waits 43.2 percent, pooling 8.6, `pwrite` 4.6 and `fse_inverse_norms_body` 2.6 (5,397 samples). C3 reader: native `XXH3_64bits_update` symbol 70.4 percent, `FLOAT_isfinite` 11.8 and `__munmap` 3.1 (4,746 samples); artifact hashes remain xxh3-128. The single-thread reader CPU profile ended at load 5.405 and is used only for relative shares, under the permitted profile exception. Every timing remains below 5; memory/RSS profiles remain below 20. Cumulative allocation sites include the runtime reference and `MemorySink`, `ShardWriter.reserve`, corpus/table preparation, numpy memmaps and bounded `_recovery._vectors` checks. These totals are not simultaneous peaks.

[Raw kept/discarded runs, both control series, correctness digests, byte counts and profile metadata](benchmarks/baseline/store_none_20261031.json). These are local warm-cache measurements, not physical power-loss tests or Windows performance measurements.

## int8 tables at 1024 dimensions

Earlier measurements found no gain from int8 tables at 256 and 300 dimensions. At 1024 dimensions `roofline.py compare` ran static-retrieval-mrl-en-v1 with its table as float32 and as int8 (`WordVectors.quantize("int8")`) in one process, 5 paired rounds of alternating order, the best of 5 passes each, no round discarded. `Encoder` does not yet accept int8 tables for `wordpiece@1`, so both run the encoder's spec on a `Pipeline` of their own (`roofline_dtype.SpecEncoder`); the vectors of int8 differ from float32 by the quantization, which is not judged here.

| corpus | workers | float32, M tokens/s | int8, M tokens/s | float32 time / int8 time, median (range) |
|---|---:|---:|---:|---|
| WikiText | 1 | 7.3 | 9.4 | 1.30 (1.25 to 1.34) |
| WikiText | 8 | 43.3 | 54.9 | 1.27 (1.24 to 1.28) |
| WikiText | 10 | 40.5 | 50.8 | 1.27 (1.20 to 1.27) |
| MASSIVE | 1 | 7.2 | 7.5 | 1.04 (1.02 to 1.05) |
| MASSIVE | 8 | 45.5 | 47.6 | 1.05 (1.04 to 1.10) |
| MASSIVE | 10 | 48.5 | 48.9 | 1.02 (0.98 to 1.06) |

A run before PR 136 gave the same picture (WikiText 1.24 to 1.28, MASSIVE 1.04 to 1.05). int8 reads a quarter of the row bytes. Its pooling at one worker is 4 to 8 % faster per token on MASSIVE (93 to 95 against 100 to 101 ns) and 37 % faster on WikiText (83 to 84 against 114 to 115 ns). The two dtypes run different loops on NEON (float32 rows of up to 1024 values in register tiles, int8 rows one at a time, docs/architecture/numerical-contract.md, "Native kernel"), so the difference is not only the bytes. The distinct rows of WikiText take 78 MiB as float32 and 19 MiB as int8, those of MASSIVE 40 and 10 MiB; that int8 gains most where float32 has the larger working set fits a cache effect, but these measurements do not separate it from the different loops. In practice int8 tables pay off at 1024 dimensions on corpora with many distinct rows, by about a quarter, and barely on short texts with few.

## What PR 136 changes

PR 136 (2.0: stages 1 to 4 of a task in one nogil section on the cpu backend, Laufzeit 2) runs the stages of a task in one native call, so a task takes the GIL back far fewer times. The cells were measured before it too (separate runs, not paired with these): at 10 workers on WikiText the tasks took 1.46 to 1.96 times the wall time of one worker against 1.18 to 1.36 times the CPU time; with it they take 1.20 to 1.24 against 1.15 to 1.21 times, and potion-base-8M rose from 71.8 to 73.5 to 91.7 to 91.8 M tokens/s. The gap between wall and CPU time is almost gone; what remains is the CPU time growth of a task when all cores run and the idle time of workers. The per-core and memory bounds do not change.

## How to run

    uv run python benchmarks/roofline.py probe
    uv run python benchmarks/roofline.py measure --model potion-base-8M --corpus massive --workers 8 10 --json out.json
    uv run python benchmarks/roofline.py show out.json --hardware m4-pro-14 --workers 14
    uv run python benchmarks/roofline.py compare --model static-retrieval-mrl-en-v1 --dim 1024 --corpus wikitext --dtypes float32 int8 --workers 1 8
    uv run --with pyarrow python benchmarks/runtime_timeline.py measure --model potion-base-8M --corpus wikitext --workers 1 4 8 10
    uv run python benchmarks/runtime_timeline.py ablate all --model potion-base-8M --corpus massive --workers 8 --rss

The models and corpora come from the local Hugging Face cache; nothing is downloaded. A new machine gets a profile in `benchmarks/roofline_hardware.toml` with the sources of its numbers and the read roofs that `probe` measures on it.

## Sources

- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM 52(4), 2009, doi:10.1145/1498765.1498785. The roof is the sustained bandwidth a benchmark measures, not the vendor peak.
- G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", AFIPS 1967, doi:10.1145/1465482.1465560.
- M. D. Hill, M. R. Marty, "Amdahl's Law in the Multicore Era", IEEE Computer 41(7), 2008, doi:10.1109/MC.2008.209: on asymmetric cores the parallel part runs at the summed speed of the cores.
- J. D. McCalpin, STREAM, https://www.cs.virginia.edu/stream/: sustained memory bandwidth measured, not quoted.
- Apple Newsroom, "Introducing M1 Pro and M1 Max", 18.10.2021, https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/: "up to 200GB/s of memory bandwidth", "10-core CPU, including eight high-performance cores and two high-efficiency cores".
- Apple Newsroom, "Apple introduces M4 Pro and M4 Max", 30.10.2024, https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/: "273GB/s of memory bandwidth", "up to 14-core CPU consisting of up to 10 performance cores and four efficiency cores".


## Alternatives comparison (partial, 01.11.2026)

### Recorder correction for counterpart runtime attribution

The window5 fse rates, ranges and A/A controls remain valid. The recorded
Model2Vec package path and content fingerprint incorrectly identified fse.
Its measured rates remain retained, but their runtime attribution and final
same-model ratio claims are provisional until re-measurement. The same recorder
path affected Rust, FastEmbed and Sentence Transformers before correction.
No final comparison headline should use those counterpart identities.

Fix `02d387c` records each executing backend's own package version, install
origin and installed-file content hash, excluding mutable Python caches.
Rust uses the verified worker binary hash and locked build inputs. The recorder
checks the backend identity before and after measurement. Tests reject a
Model2Vec row carrying fse identity and verify Python, native and configuration
file changes alter the counterpart hash. Follow-up `d635a12` exposes the
attribution limitation in summaries without dropping valid fse timings.
Counterpart cells will be re-measured with fresh rotating fse A/A controls.


**Runtime attribution limitation:** the initial speed journals record checkout commits and package versions, but lack executed-runtime content fingerprints. They remain historical observations and need re-measurement before final throughput claims. New runs capture the loaded package location, native extension contents and verified Rust build identity.

DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen) has three initial complete speed cells and seven complete precision-matrix cells. The full eight-model, three-table-precision matrix and quality comparison remain open. The [decision-table draft and protocol](docs/fse-comparison.md) record the full scope.

These measurements use an Apple M1 Pro, macOS 27.0.1 arm64, Python 3.13.16, numpy 2.5.4, tokenizers 0.23.3 and model2vec 0.10.0. The measured source commit is `b7ffa8c`; the installed fse distribution identifies itself as `1.1.1.dev862+g82b6f5b12`. The runtime sources are unchanged by this benchmark PR. Both sides load `minishlab/potion-base-8M` at `bf8b056651a2c21b8d2565580b8569da283cab23`, with float32 tables and float32 outputs. The pinned source weight file contains 30,236,760 bytes.

Each cell has three valid rotating rounds of fse A, identical fse B and native model2vec in fresh processes. Each child records a first call and three warm calls. The table reports the median across rounds of each child's warm median. Ranges cover those three independent round rates. The ratio is the median of paired rate ratios, which can differ from the ratio of the displayed medians. WikiText has 5,352 nonempty validation/test texts; MASSIVE has 151,674 test texts. Ordered input digests and exact dataset revisions are in every operation record.

| Corpus | fse workers | model2vec tokenizer parallelism | fse A texts/s, median (range) | model2vec texts/s, median (range) | Paired fse/model2vec ratio | Symmetric A/A band | Repeatability |
|---|---:|---|---:|---:|---:|---:|---|
| WikiText-103 | 1 | off | 219,333 (199,745 to 221,304) | 8,906 (8,879 to 8,956) | 24.49 | 0.900 to 1.111 | noisy, re-measure in a quiet window |
| WikiText-103 | 10 | on | 909,270 (830,368 to 944,311) | 26,048 (22,894 to 26,880) | 36.25 | 0.865 to 1.156 | noisy, re-measure in a quiet window |
| MASSIVE | 1 | off | 878,260 (861,892 to 903,738) | 57,001 (56,620 to 57,191) | 15.51 | 0.969 to 1.031 | within 10% A/A band |

All retained endpoint loads are strictly below 5: 4.609 to 4.989 for one-worker WikiText, 4.355 to 4.609 for ten-worker WikiText, and 3.900 to 4.355 for MASSIVE. Model2vec uses its one-worker mode or its default automatic parallelism. WikiText is below its 10,000-text joblib threshold, so the default run uses tokenizer parallelism. The exact tokenizer thread count was not measured. MASSIVE's all-core cell crossed the load limit and remains incomplete. Its rejected observations are retained and excluded from this table.

Peak process RSS for fse A/model2vec is 206.1/223.6 MiB on one-worker WikiText, 207.8/233.1 MiB on ten-worker WikiText, and 452.4/602.6 MiB on one-worker MASSIVE. These peaks include imports, corpus preparation and encoding. Median model setup takes 0.204/0.150 s, 0.199/0.150 s and 0.216/0.159 s respectively. fse setup converts the pinned Hub files; model2vec loads its source format. Saved fse artifact startup will be measured separately with explicit table precision.

[Raw journals and retained/rejected rounds](benchmarks/baseline/results/) feed the [machine-readable summary](benchmarks/baseline/dr151_summary_20261101.json). Numerical parity, the full MTEB/NanoBEIR matrix, profiles, classifiers, search, lifecycle and remaining precision cells are still required before a complete recommendation.

The summary includes a reference-tariff calculation using the public CAX31 rate of EUR 0.0344 per server-hour, excluding VAT, observed on [Hetzner's pricing page](https://www.hetzner.com/cloud/cost-optimized/) on 01.11.2026. The [price evidence](benchmarks/baseline/dr151_cpu_price_20261101.json) records its eight Ampere vCPUs, 16 GiB memory and availability markup. It values measured local encode time with `1,000,000 / texts_per_second / 3,600 * hourly_price`. This scenario assumes the measured throughput at the reference tariff. Actual CAX31 throughput and deployment cost were not measured. Storage, transfer, startup, idle time, VAT and billing granularity are excluded.


A full-corpus numerical audit of the same 5,352 WikiText rows at source commit `945a640` uses exact cosine neighbors for every query, excluding itself. Native model2vec versus fse float32 has recall@10 1.0, minimum cosine 0.9999999999999971 and maximum absolute error 8.94e-8. fse float16 versus float32 has recall@10 0.999943946 and maximum centered-subspace angle 0.0236624 degrees. Int8 has recall@10 0.996337818 and maximum angle 1.3987229 degrees. All 5,352 queries are eligible and zero-vector disagreements are zero. This is numerical neighborhood stability; labeled task quality still requires MTEB. The audit ran at nice 10 with bounded CPU pools under the heavy-job lock. Its load ranged from 14.68 to 15.42 and its times are not used for speed comparisons. [Full diagnostics](benchmarks/baseline/results/dr151-numerics.jsonl) include every principal angle and provenance.


**Repeatability warning for the alternatives comparison:** the one-worker and
ten-worker WikiText POTION-8M cells above exceed the 10% symmetric A/A rule.
They are **noisy, re-measure in a quiet window**. Keep their raw observations,
but do not headline their throughput as a stable result. The one-worker MASSIVE
cell is within the 10% band. The [per-cell A/A audit](docs/fse-comparison-aa.md)
reports every initial and precision-matrix cell, including incomplete attempts.
The largest complete matrix band reaches 1.9543, so the endpoint load gate alone
did not ensure repeatability tonight. A/A is reported separately from the primary
fse-a medians. This label does not relax the load gate or discard any record.


### First completed non-deferred quality configuration (01.11.2026)

POTION-8M float32 completed all 40 non-deferred English tasks and all 13 NanoBEIR
tasks using unmodified MTEB 2.26.25. Its unweighted mean over the 40 English task
main scores is **53.9527 points**. This excludes the explicitly deferred
MindSmallReranking task and is not a complete 41-task MTEB mean. Its full
13-task NanoBEIR mean is **43.8485 points**. Other model/precision results remain
pending, so these numbers establish no quality ranking.

The run used two CPU workers at nice 10, source commit `265e70d`, and encoder
fingerprint `xxh3-128:7aed2ca9bf7f9073ffa6778672a7575f`, matching the pinned
POTION-8M artifact in the bound source manifest. Source revisions, dataset
metadata, package versions and every task main score are retained in the
[raw quality journal](benchmarks/baseline/results/dr151-quality-primary.jsonl)
and [coverage summary](benchmarks/baseline/dr151_quality_primary_20261101.json).
Quality-run encode times are not used for speed or cost claims.


### POTION-32M quality checkpoint (01.11.2026)

POTION-32M float32 completed the same 40 non-deferred English tasks and all
13 NanoBEIR tasks with unmodified MTEB 2.26.25. The English 40-task mean is
**55.2573 points**, excluding MIND. The complete 13-task NanoBEIR mean is
**46.1754 points**. The full 41-task English mean remains unavailable.

The model revision is `1e5a03f8eeb2c98b928fbbd846f22f816360919f`, and the
finished encoder fingerprint is `xxh3-128:99737b0f0d2dc7cfb4ee240faa2e278f`.
The run began from source `0de9c3d`. Its method and runtime fingerprints differ
from the first POTION-8M invocation because provenance checks and the installed
extension changed between runs. These checkpoints therefore do not establish
a controlled model ranking. A same-method POTION-8M confirmation remains queued.
[Raw task records](benchmarks/baseline/results/dr151-quality-potion32-float32.jsonl)
and the [complete coverage summary](benchmarks/baseline/dr151_quality_potion32_20261101.json)
were first pushed in `7bb78dd`.

The invocation took 3,064.107 seconds of elapsed wall time, including a manual
priority suspension, shared waits, setup and cooldowns. This is not model
computation time. The earlier POTION-8M invocation took 1,462.740 seconds with
different batch sizes and waits. At that checkpoint, applying those two elapsed rates uniformly
to the 32 remaining static configurations gave a planning scenario of
13.00 to 27.24 hours, plus MiniLM, classifiers and baseline confirmation.
This range is not a calibrated quiet-machine or cloud forecast.
[Invocation durations](benchmarks/baseline/dr151_configuration_times_20261101.json)
and [remaining-time assumptions](benchmarks/baseline/dr151_remaining_time_20261101.json)
retain the underlying observations.


### POTION-retrieval-32M quality checkpoint (01.11.2026)

POTION-retrieval-32M float32 completed all 53 non-deferred tasks with unmodified
MTEB 2.26.25. Its English 40-task mean is **54.5098 points** and its complete
13-task NanoBEIR mean is **50.3872 points**. MIND alone remains deferred, so the
full 41-task English mean is unavailable. The pinned revision is
`6fc8051fab2a1e0ee76689cf08c853792ac285e7`, and the encoder fingerprint is
`xxh3-128:6f78af26bab53a3d464e97d9189f5b20`.

This is the first completed configuration under the normalized result-writer
method and current runtime fingerprint. The raw measured checkout is
`300bde8`. Earlier POTION-8M and POTION-32M checkpoints use different method or
runtime identities. Both confirmations remain queued, so these checkpoints
still establish no controlled model ranking. [Raw task records](benchmarks/baseline/results/dr151-quality-retrieval32-float32.jsonl)
and [coverage summary](benchmarks/baseline/dr151_quality_retrieval32_20261101.json).

The invocation took **1,102.850 seconds** (18.38 minutes), including setup,
shared waits, cooldowns and the reserved store timing hold. Approximate state
samples are retained separately and are not encoding-throughput measurements.
Thirty-one static configurations remain, plus MiniLM and two same-method
confirmations. Applying the median observed invocation duration uniformly to
those 31 static configurations gives a **12.60-hour planning scenario**.
The shorter and longer observed durations give 9.50 and 26.39 hours.
These scenarios exclude MiniLM, classifiers and confirmations. Model dimensions,
precision and shared waits differ, so they do not calibrate a quiet-machine or
cloud forecast. [Invocation durations](benchmarks/baseline/dr151_configuration_times_20261101.json)
and [remaining-time assumptions](benchmarks/baseline/dr151_remaining_time_20261101.json).


### Quality on tasks with longer inputs (partial, 01.11.2026)

The document-task diagnostic joins full task scores to length statistics
packaged with unmodified MTEB 2.26.25. It selects every measured task with a
reported evaluation input stream averaging at least 1,000 characters.
Nineteen tasks qualify for the POTION-32M checkpoint. No evaluation rows are
sampled or removed, and no new aggregate score combines different metrics.
Where subset statistics exist, the report uses mapped English subsets.

Examples below use POTION-32M float32 from raw-source commit `7bb78dd`.
Scores are the complete task main scores multiplied by 100.

| Task | Upstream mean input characters | Main metric | Score (points) |
|---|---:|---|---:|
| ArguAna | 1,029.5 documents | nDCG@10 | 42.864 |
| BiorxivClusteringP2P.v2 | 1,664.0 texts | V-measure | 35.1222 |
| ImdbClassification | 1,293.8 texts | accuracy | 70.126 |
| SCIDOCS | 1,204.4 documents | nDCG@10 | 13.587 |
| SummEvalSummarization.v2 | 2,100.3 source texts | cosine Spearman | 24.7281 |
| Touche2020Retrieval.v3 | 2,096.4 documents | nDCG@10 | 48.228 |

These lengths are upstream descriptive metadata, not fresh measurements of
this run's inputs. They can precede the pinned task revision. They do not
measure model token lengths or truncation frequency. Scores use each model's
native input limits. The report retains the actual encoder description and
separate input streams. Other models remain pending, so this is no quality
ranking or evidence of full-document context retention.
[All 19 task breakouts and source-statistics digests](benchmarks/baseline/dr151_document_tasks_potion32_20261101.json).


### Comparison profiles (01.11.2026)

POTION-8M float32 on all 5,352 WikiText rows has four accepted instrumented
profiles. All endpoint loads are below five. fse CPU uses 100 full passes,
native model2vec CPU uses three, fse native-memory uses 30, and fse process-tree
RSS uses 100. Profiles include imports, input preparation, startup and encoding.
Their durations are not speed or cost evidence. The source commit is `0de9c3d`.

Across 6,612 fse CPU samples, `fse_loop_f32_neon` has 18.0% self samples,
`_wp_piece` 15.6%, and `fse_normalize_body` 1.2%. Across 4,034 native model2vec
samples, Unicode category lookup has 9.5% self samples and memory moves 8.7%.
Its BERT normalizer appears in 22.4% of sample stacks, including callees.
Self and stack shares measure different things.

The fse process-tree RSS peak is **231,424,000 bytes (220.7 MiB)**. Memray reports
a tracked allocation peak of 1.169 GB, including a 1.074 GB memory mapping
during imports. That allocation total is not resident RAM. Largest cumulative
allocation sites include imports, `StringSource` construction, `MemorySink._new`
and `CpuBackend.text_rows`. Cumulative totals over repeated calls are not
simultaneous memory peaks. [Profile metadata and load endpoints](benchmarks/baseline/dr151_profiles_20261101.json)
and [raw CPU, native-memory and RSS artifacts](benchmarks/baseline/results/dr151-profiles/).


### Additional quality checkpoint and runtime correction (01.11.2026)

POTION-multilingual-128M float32 completed all 53 non-deferred tasks with
unmodified MTEB 2.26.25. The 40-task English mean is 49.9565 points and the
complete 13-task NanoBEIR mean is 36.6528 points. MIND remains deferred because
it needs more than 16 GiB RAM on the M1. This is not a full 41-task English mean.
[Raw scores](benchmarks/baseline/results/dr151-quality-multilingual128-float32.jsonl)
and [coverage summary](benchmarks/baseline/dr151_quality_multilingual128_20261101.json)
retain the measured method and runtime. Earlier checkpoints use different
method cohorts, so these figures establish no controlled cross-model ranking.

The four completed invocations took 24.4, 51.1, 18.4 and 35.3 minutes, including
setup, shared waits and cooldowns. For 30 remaining static configurations,
a uniform elapsed-rate scenario is 14.93 hours at the observed median,
with endpoints of 9.19 and 25.53 hours. MiniLM, classifiers, search and necessary
same-method confirmations are additional. These observations do not calibrate
quiet-machine or cloud runtime. [Durations and corrected process-priority
meaning](benchmarks/baseline/dr151_configuration_times_20261101.json) retain the
two failed metadata jobs as incomplete configurations.

The counterpart recorder defect also affects attribution of historical
model2vec profiles above. Their observed samples remain diagnostic, but their
own installed-code identity was not captured. No final counterpart speed,
profile attribution or paired ratio headline is supported by those rows.
Commit `02d387c` adds each executing backend's package version, install path
and installed-file content hash. Rust additionally binds its verified binary.
Follow-up commits extend this identity to quality, classifiers and search.
Existing fse measurements remain valid. Counterpart measurements require
fresh runs with the corrected identity before final comparison claims.


Long-input breakouts are also available for the other completed checkpoints:
[POTION-8M](benchmarks/baseline/dr151_document_tasks_potion8_20261101.json),
[POTION-retrieval-32M](benchmarks/baseline/dr151_document_tasks_retrieval32_20261101.json)
and [POTION-multilingual-128M](benchmarks/baseline/dr151_document_tasks_multilingual128_20261101.json).
Each retains 19 full-task scores and the corresponding packaged length
statistics. Every score matches its raw completed task record. The same
limitations apply: these statistics describe upstream text streams, not
measured model token counts or truncation frequency. Separate method cohorts
remain separate. MiniLM and other counterparts are pending.


### Static-retrieval checkpoint and backend-attributed speed reruns

Static-retrieval-mrl-en-v1 float32 completed all 53 non-deferred tasks. Its
40-task English mean is 51.8893 points, explicitly not a full 41-task mean.
The complete 13-task NanoBEIR mean is 50.3072 points. MIND alone remains deferred
because it needs more than 16 GiB RAM on the M1. The invocation took 17.4 minutes,
including setup, evaluation, waits and cooldowns. Every executed runtime record
reports actual process priority 10. The parent's earlier startup probe observed
priority 0 before `nice` took effect and is retained separately.
[Raw quality records](benchmarks/baseline/results/dr151-quality-staticretr-float32.jsonl)
and [coverage summary](benchmarks/baseline/dr151_quality_staticretr_20261101.json).
Separate method cohorts remain separate and establish no controlled ranking.

Five unique quality configurations are complete. Twenty-nine static
configurations plus MiniLM remain. A uniform elapsed-rate scenario is 11.78
hours for those static configurations, with observed endpoint scenarios of
8.40 and 24.68 hours. MiniLM, classifiers, search and necessary same-method
confirmations are additional. This is not a calibrated quiet-machine or cloud
forecast. [Measured durations](benchmarks/baseline/dr151_configuration_times_20261101.json)
and [scenario assumptions](benchmarks/baseline/dr151_remaining_time_20261101.json).

Window6 retains fresh backend-specific speed records and every rejected or
incomplete attempt in the [matrix summary](benchmarks/baseline/dr151_matrix_summary_20261101.json).
Model2Vec, Rust, FastEmbed and Sentence Transformers now identify their own
executing package or binary. A/A noise remains independent of load acceptance.
The POTION-8M float32 Model2Vec tokenizer-off/on cells have symmetric A/A maxima
1.1197 and 1.1090, so neither supports a final same-model ratio headline.
FastEmbed with ten requested threads reaches 2.5565 and remains explicitly
noisy. Rust at one/ten requested threads and FastEmbed at one have A/A maxima
within 10%. No observation is silently removed, and no load rule is relaxed.

MiniLM and native multilingual POTION cache lookups failed before encoding
because unused repository exports or evaluation files were absent. The harness
now requests its pinned inference files. The existing required-file cache was
verified for all seven Hub models. Model revisions and complete input sets are
unchanged. This changes no MTEB package or loader.


Static-similarity-mrl-multilingual-v1 float32 also completed all 53 non-deferred
tasks. The English40 mean is 51.4292 points, not a full English41 mean. The
complete NanoBEIR13 mean is 43.5178 points. MIND alone is deferred.
The invocation took 17.59 minutes including setup and waits,
with actual process priority 10 throughout its runtime records.
[Raw records](benchmarks/baseline/results/dr151-quality-staticsim-float32.jsonl)
and [coverage summary](benchmarks/baseline/dr151_quality_staticsim_20261101.json).
Six unique configurations are complete. Twenty-eight static configurations
plus MiniLM remain. The uniform static scenario is now 9.98 hours,
with endpoints 8.11 and 23.83 hours.
MiniLM, classifiers, search and required same-method confirmations are additional.
These checkpoints use separate method cohorts and establish no controlled ranking.

### Classifier speed evidence and folding

Model2Vec 0.10.0 fine-tunes its embeddings and an MLP with one 512-unit ReLU hidden layer using PyTorch. The exported classifier runs in plain NumPy, with softmax or sigmoid for probabilities. Its label path selects from logits. fse's classifier head is linear. Accuracy and speed therefore compare a fine-tuned MLP against a frozen-embedding linear head. [Pinned MLP implementation](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/inference/mlp.py), [pinned training documentation](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/train/README.md).

No controlled classifier throughput cells have been published yet. The published [POTION-8M Model2Vec MLP subj reference](benchmarks/baseline/dr151_native_classifier_pinned_accuracy_20261101.jsonl) records accuracy 0.890000 and macro-F1 0.889887 on all 2,000 test rows. It follows the fixed 1,000-row training selection and has no local training or inference timing. Its comparison with the pod remains pending.

Pod 3 adds a controlled folding comparison on the same raw-mean linear model: fse `Classifier.predict` on its folded table, versus fse `Encoder.encode` followed by the identical float32 weight matrix and bias. Both worker settings, 1 and 32, receive the complete evaluation sets. A fine-tuned Model2Vec MLP running through its exported NumPy pipeline is a separate arm. Every arm records accuracy, size, runtime identity and rotating fse A/A controls. No rates are filled into a graphic before raw controlled results exist.

Raw-mean fse heads already fold internally. Saving standalone writes that projected table and bias; it is not a separate faster algorithm. The size comparison reports folded model bytes versus encoder plus linear-head bytes. The existing normalized-linear, LogReg, TF-IDF, fastText and Model2Vec MLP comparisons remain unchanged.

Pod 3 also compares Python Model2Vec with the **same frozen sklearn linear head**, including the identical float32 coefficients and bias. This isolates the encoding engines. Its fine-tuned Model2Vec MLP remains a separate trained-model comparison. AG News linear cells are published first, before MLP training, with raw and summary artifacts.

The fresh Linux Pod 3 cohort supplies both README graphics on one machine. It includes POTION-8M float32/float16/int8 tables and all same-model references at 1, 16 and 32 workers. Native sentence-transformers StaticEmbedding uses static-retrieval-mrl-en-v1 with its own fse baseline. Both models have measured float32 curve points at 1, 2, 4, 8, 16 and 32. The renderer retains one core and selects 32 only if its unrounded primary median is at least 1.20 times the 16-worker median. Otherwise it selects 16. The lower intermediate settings remain curve-only. All rates remain pending actual controlled pod artifacts. The Mac headline remains a separate machine-specific result.

The document-quality breakout now includes 19 complete tasks each for [static retrieval MRL](benchmarks/baseline/dr151_document_tasks_staticretr_20261101.json) and [static similarity MRL](benchmarks/baseline/dr151_document_tasks_staticsim_20261101.json), plus eleven currently completed [MiniLM document tasks](benchmarks/baseline/dr151_document_tasks_minilm_20261101.json). Each selected task has an upstream MTEB evaluation text stream averaging at least 1,000 characters. These are full task scores, without evaluation sampling, and preserve their native metrics. Length metadata is packaged MTEB descriptive evidence, not fresh input or truncation measurement. Model-specific limits and configuration cohorts remain explicit. No combined quality ranking or complete MiniLM score is claimed.

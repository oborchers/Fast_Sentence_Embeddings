# Performance: how close fse comes to the hardware limit

This document says how fast fse can be on a given machine, why, and how close it comes. It holds a small model of the limits (a roofline model), the assumptions behind it, and measurements on an Apple M1 Pro and an 8P+4E M4 Pro. `benchmarks/roofline.py` computes the same numbers for any encoder, corpus and hardware profile.

## In short

On an M1 Pro (8 performance and 2 efficiency cores), master of 30.10.2026 with PR 136:

- **One worker** runs at 86 to 99 % of the per-core bound: almost all of its time is the work of the stages that more cores share.
- **Eight workers** reach 69 to 97 % of the Amdahl bound and 44 to 78 % of the per-core bound. The gap has two measured parts: each task takes 2 to 11 % more CPU time than alone when all cores run, and the workers are busy only about 70 to 100 % of the parallel time (the last tasks of a call, and waits between tasks).
- **The measurements show no memory bandwidth limit**, also not at 1024 dimensions. static-retrieval-mrl-en-v1 at 1024 dimensions reads 164 to 197 GB/s of table rows (logical bytes) with 8 or 10 workers. That is 82 to 99 % of Apple's 200 GB/s, but more than any reader of the probe got from memory (115.6 GB/s at most). This points to many rows coming from the caches; it is no proof, because the probe gives lower estimates and hardware counters were not read. What the measurements do show: pooling is the largest stage, 72 to 83 % of the time of one worker when timed on its own; that time includes its waits for memory, which these measurements do not separate from its arithmetic.
- **int8 tables at 1024 dimensions** are 1.20 to 1.34 times as fast as float32 on WikiText and 0.98 to 1.10 times on MASSIVE, end to end.

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

The topology of the M1 Pro comes from `sysctl` (`hw.perflevel0.physicalcpu` 8, `hw.perflevel1.physicalcpu` 2, 12 MiB L2 per cluster of 4 performance cores). The `m4-pro-14` M4 Pro is an unmeasured specification profile, distinct from the measured 8P+4E machine below: `roofline.py show result.json --hardware m4-pro-14 --workers 14` gives its bounds with the stage times of the machine that measured them.

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

## MLX matrix on the M4 Pro (31 October 2026)

DR-147 (MLX-Gate entscheiden und auf dem M4 Pro nachmessen) measured all 60 requested real-model cells on an actual M4 Pro with 8 performance and 4 efficiency cores, 48 GiB, default 12 workers. CPU14 is oversubscribed. The m4-pro-14 roofline specification is not this machine, and actual-variant bandwidth is unverified. Work performed using Codex. Frozen base c754efd382f7b42504d140cabf5aab6e40ad51bb; 43 accepted cells retain measured driver d8a61a5, 17 retain 9fc6a93. Both have identical inference sources and native checksum, recorded in [compact evidence](benchmarks/evidence/dr147-m4/README.md).

Each cell has three paired alternating rounds, best of three passes on full WikiText (5352 texts) or two on MASSIVE (151674 texts), with CPU1/10/14 and MLX default plus A/A controls. Actual CPU-default plus A/A is included in the two deciding float32 WikiText cells. All model/dataset revisions are pinned; public Encoder.quantize supplies float16/int8 tables. All output rows are float32. All 60 full comparisons were bit-identical; worker/restart/GPU checks passed with no fallback or accepted discard. These results concern table dtype, not comparison of quantization error against original float32 tables.

Below are the float32 time ratios MLX/CPU10, median and range of three paired ratios. Values below 1 favor MLX. Full float16/int8 results, CPU1/14/default, all unrounded A/A controls and texts/s replay from [matrix.json](benchmarks/evidence/dr147-m4/matrix.json) using `python benchmarks/report_mlx_gate.py --json`, without MLX or measurements. L means all accepted pair observations below 3.89; E means all from 3.89 to strict 5.

| Model/dim | Corpus | Raw ratio (range) | Normalized ratio (range) | Raw/norm load |
|---|---|---|---|---|
| static-retrieval-mrl-en-v1/1024 | wikitext | 0.989344 (0.962579 to 1.440726) | 1.457620 (0.826732 to 1.677063) | L/L |
| static-retrieval-mrl-en-v1/1024 | massive | 1.800728 (1.742174 to 1.802737) | 2.699700 (2.601370 to 2.793482) | L/L |
| static-retrieval-mrl-en-v1/512 | wikitext | 1.996510 (1.930977 to 2.033243) | 2.019273 (1.901230 to 2.027938) | L/L |
| static-retrieval-mrl-en-v1/512 | massive | 1.846385 (1.810518 to 1.895829) | 2.489930 (2.341543 to 2.594380) | L/L |
| static-retrieval-mrl-en-v1/256 | wikitext | 2.206636 (2.099671 to 2.314705) | 2.247844 (2.074070 to 2.285782) | L/L |
| static-retrieval-mrl-en-v1/256 | massive | 1.802697 (1.723051 to 1.811691) | 2.307149 (2.226299 to 2.325002) | L/E |
| potion-base-8M/256 | wikitext | 2.125816 (1.750080 to 2.169504) | 2.121527 (2.098611 to 2.210607) | L/L |
| potion-base-8M/256 | massive | 1.695953 (1.656746 to 1.701466) | 2.093748 (2.029371 to 2.148079) | E/E |
| potion-base-32M/512 | wikitext | 1.892005 (1.840041 to 1.957113) | 1.862794 (1.799634 to 1.892968) | L/L |
| potion-base-32M/512 | massive | 1.731875 (1.692832 to 1.750620) | 2.464521 (2.408432 to 2.569616) | L/L |

Only the raw MRL1024/WikiText float32 cell has median MLX time below CPU10 among all 60 cells. Its range crosses parity, so the near tie is not a robust speed advantage. Actual CPU-default comparisons are:

| MRL1024/WikiText float32 | MLX/CPU-default median (range) | CPU-default / MLX median texts/s | MLX A/A range |
|---|---|---|---|
| raw | 1.183355 (1.088285 to 1.660089) | 708084.8 / 598370.5 | 0.975194 to 1.088690 |
| norm | 1.710866 (0.915435 to 1.882586) | 658326.2 / 400109.7 | 0.544806 to 1.844942 |

Normal baseline: the predeclared median of all eleven fixed uptime 1-minute readings is 2.39, giving min(2.39+1.5,5)=3.89. All raw timestamped values are retained in evidence. Oliver then explicitly authorized strict limit 5 and load-only automatic retries, received by coordinator at 2026-10-31T13:40:11+01:00. This is a decision-backed override, not a fabricated baseline. Previous stops retain their original limits. Defender remained active and unchanged.

Actual pair guards classify 150 accepted pairs/50 cells as low, 30 pairs/10 cells as elevated-only, no mixed/unknown. Accepted load range: 2.09033203125 to 4.6435546875. No accepted cell spans regimes, so within-cell load/spread dependence is not identifiable. Separate same-cell exploratory partial-pair comparisons show wider higher-load ranges in two tiny comparisons, but lack full historical restart/acceptance and share attempt/order confounds. A third n=1 group cannot establish spread. Cross-cell mixtures do not support load or Defender causality. Every valid higher-load cell is retained; partial or discarded rounds never supply acceptance.

### Completed deciding-cell profiles

All six requested make targets completed on profile commit 937860e99f1f6c373bb04d7fd2f8a5272e4cb29f: raw MRL1024/full WikiText/float32, CPU12 and MLX default12. Samply guards were CPU 2.6650390625 to 3.01171875 and MLX 2.7822265625 to 2.8505859375, strictly below 5. Memory/RSS-only limit 20 was explicit but never needed: every six-profile guard stayed below 5 (maximum 4.9619140625). All first/last bits matched accepted references; no fallback. Fresh guarded scratch Make overrides avoided deletion and checked every recorder/summary. Actual profile_top, native memray temporal/stats and RSS outputs remain local.

Offline mapping: sample Unix ms = meta.startTime + samples.time[i], contained exactly in [1000*encode_start,1000*encode_stop], no padding/rounding. Inclusive samples: CPU 30771, MLX 39125. Encode-only: CPU 28253, MLX 36706. The denominator below is non-null thread-stack samples summed over all threads in that scope, including waiting stacks. These are exclusive self sample shares, **not end-to-end wall percentages or summed thread-time fractions**. Loading/guard __open samples are excluded by filtering.

| Function/proxy | CPU inclusive / encode-only sample % | MLX inclusive / encode-only sample % |
|---|---|---|
| fse_loop_f32_neon | 42.3808 / 46.0447 | not resolved |
| host _wp_piece | 11.6993 / 12.7385 | 6.2696 / 6.6610 |
| __psynch_cvwait | 32.7939 / 35.5290 | 63.8722 / 67.7437 |
| __workq_kernreturn | 0.0130 / 0.0000 | 13.0888 / 13.6272 |
| IOKit+0x1ecc0 | not resolved | 4.3834 / 4.6314 |
| _platform_memmove | 0.3770 / 0.3752 | 0.5725 / 0.5530 |
| FLOAT_absolute (bounds proxy only) | not resolved | 0.3425 / 0.3623 |

All IOKit offsets together give 4.6341198% encode-only (1701 samples), versus the dominant symbol 4.6313954% (1700); one sample is IOKit+0x1ecbc. Neither measures GPU transfer wall cost. Copy samples occur on main and unlabelled native threads; worker/device roles and critical-path overlap are not recoverable from names alone.

| Exact peak measure | CPU bytes | MLX bytes | Scope |
|---|---:|---:|---|
| Native memray peak | 1308904403 | 1318022310 | FileReader.metadata.peak_memory, whole traced process including loading/checks |
| RSS target-tree sampled peak | 382844928 | 579715072 | tools/rss.py CSV maximum, separate recording |
| MLX allocator peak in mem recording | not applicable | 137620176 | get_peak_memory since process start |

These are different, overlapping scopes and must not be summed. Native cumulative allocation is not the peak and is not normalized per call. Instrumented record durations/call counts are not throughput comparisons.

| Desired deciding-cell end-to-end wall share | Result | Method limit | Smallest additional probe, not run |
|---|---|---|---|
| GPU transfer | unmeasured/not identifiable, no wall % | IOKit stacks do not isolate shared-memory conversion, device transfer or overlap. | Correlate input-conversion spans and Metal command-buffer timestamps with call/group IDs. |
| Critical-path copy, worker | unmeasured/not identifiable, no wall % | memmove samples have no reliable role/critical-path mapping. | Tag worker _use/np.copyto spans and native thread IDs; reconstruct overlap on the call DAG. |
| Critical-path copy, device thread | unmeasured/not identifiable, no wall % | No isolated device-copy span or counterfactual was recorded. | Tag device finish/copy spans separately, correlated with worker leases and group IDs. |
| Group-boundary wait | unmeasured/not identifiable, no wall % | cvwait combines multiple conditions and overlapping workers. | Timestamp group evaluation start/finish and boundary waits, with group/call IDs. |
| Queue wait | unmeasured/not identifiable, no wall % | Queue enqueue/take timestamps were not recorded. | Timestamp enqueue/take/done per job and correlate with the critical path. |
| Host WordPiece | unmeasured/not identifiable, no wall % | Parallel leaf sample shares are not exclusive wall duration. | Tag per-task WordPiece spans and overlap within each encode window. |
| Token/bucket checks | unmeasured/not identifiable, no wall % | No isolated check span or resolved exclusive symbol. | Tag host validation spans without removing safety checks. |
| _bounds | unmeasured/not identifiable, no wall % | FLOAT_absolute is a native proxy, not the complete Python bounds function. | Tag bounds subcalls on each worker and preserve overlap/critical-path correlation. |
| memmove | unmeasured/not identifiable, no wall % | Generic sampled copies include unrelated movement; role/overlap unknown. | Tag call-site copy spans and worker/device roles; do not sum overlapping thread times. |
| Fixed per-call costs | unmeasured/not identifiable, no wall % | A single workload provides no fixed-cost intercept. | Use a separately authorized small batch-size probe with unchanged outputs and validated overhead. |

The desired denominator is end-to-end encode wall time. Samples do not partition that critical path: overlapping worker times cannot be added, and cvwait cannot be split into queue/group/GPU causes. Unresolved costs are not zero. Any proposed instrumentation needs separate authorization and an unchanged-output probe validated within10% of base before ablation; none was run.

Compared with the M1 "Device thread, measured" table in docs/architecture/backends.md, the reported lease/copy ablation was on MRL1024/MASSIVE at 8 workers: moving copy to the device thread cost 1.179 (1.141 to 1.289), while WikiText was 1.007 (0.977 to 1.078). Its base profile reported 37% cvwait, 11 to 12% WordPiece and 8.8% memmove samples. Our M4 profile is WikiText at 12 workers with encode-window filtering, not a matching workload/scope. The M1 mechanism finding does not supply M4 wall shares or justify removing the lease; group-boundary benefit remains unproved here.

The small-dimension explanation is limited to the observed WikiText trend: MRL float32 raw MLX/CPU10 rises from 0.9893 at 1024 to 2.2066 at 256. Retained host WordPiece and waiting samples are compatible with less useful GPU work relative to host work as rows shrink, but this is inference, not a measured per-cause decomposition. It is not universal: MASSIVE raw is 1.8007 at 1024 and 1.8027 at 256; normalized is 2.6997 versus 2.3071. Larger dimensions do not consistently rescue MLX across corpora.

Recommendation for Oliver: retain CPU as default and MLX as opt-in; the measured near tie does not justify a default change, while host work/waits remain visible. This is a recommendation, not a policy decision or runtime change. MLX policy remains undecided and PR remains Draft by explicit instruction.

Whole six-profile recording/startup/model-loading/postprocess charge 60.617446750000454 seconds, reservation 360, no retries. Global 593.2869275000012/1800, matrix fixed 523.0340705840008/900. One own terminal lock was verified and Finder-released; immediate/delayed checks found no duplicates. Raw profiles/windows/symbols/traces/CSV and wrapper sources are preserved under dr147-profiles-3101e4a7fdd8 in task scratch. Compact profiles.json and profile_offline.py replay the existing evidence without a new recording. No extra measurement or ablation.

## Sources

- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM 52(4), 2009, doi:10.1145/1498765.1498785. The roof is the sustained bandwidth a benchmark measures, not the vendor peak.
- G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", AFIPS 1967, doi:10.1145/1465482.1465560.
- M. D. Hill, M. R. Marty, "Amdahl's Law in the Multicore Era", IEEE Computer 41(7), 2008, doi:10.1109/MC.2008.209: on asymmetric cores the parallel part runs at the summed speed of the cores.
- J. D. McCalpin, STREAM, https://www.cs.virginia.edu/stream/: sustained memory bandwidth measured, not quoted.
- Apple Newsroom, "Introducing M1 Pro and M1 Max", 18.10.2021, https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/: "up to 200GB/s of memory bandwidth", "10-core CPU, including eight high-performance cores and two high-efficiency cores".
- Apple Newsroom, "Apple introduces M4 Pro and M4 Max", 30.10.2024, https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/: "273GB/s of memory bandwidth", "up to 14-core CPU consisting of up to 10 performance cores and four efficiency cores".

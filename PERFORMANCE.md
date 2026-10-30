# Performance: how close fse comes to the hardware limit

This document says how fast fse can be on a given machine, why, and how close it comes. It holds a small model of the limits (a roofline model), the assumptions behind it, and measurements on an Apple M1 Pro. `benchmarks/roofline.py` computes the same numbers for any encoder, corpus and hardware profile.

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

Not in the model: waits of the tasks (they show as the gap between their wall and CPU time), imbalance at the end of a call, and the system level cache, whose size Apple does not publish.

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

A run before PR 136 gave the same picture (WikiText 1.24 to 1.28, MASSIVE 1.04 to 1.05). int8 reads a quarter of the row bytes. Its pooling at one worker is 4 to 8 % faster per token on MASSIVE (93 to 95 against 100 to 101 ns) and 37 % faster on WikiText (83 to 84 against 114 to 115 ns). The two dtypes run different loops on NEON (float32 rows of up to 1024 values in register tiles, int8 rows one at a time, ARCHITECTURE.md, "Native kernel"), so the difference is not only the bytes. The distinct rows of WikiText take 78 MiB as float32 and 19 MiB as int8, those of MASSIVE 40 and 10 MiB; that int8 gains most where float32 has the larger working set fits a cache effect, but these measurements do not separate it from the different loops. In practice int8 tables pay off at 1024 dimensions on corpora with many distinct rows, by about a quarter, and barely on short texts with few.

## What PR 136 changes

PR 136 (2.0: stages 1 to 4 of a task in one nogil section on the cpu backend, Laufzeit 2) runs the stages of a task in one native call, so a task takes the GIL back far fewer times. The cells were measured before it too (separate runs, not paired with these): at 10 workers on WikiText the tasks took 1.46 to 1.96 times the wall time of one worker against 1.18 to 1.36 times the CPU time; with it they take 1.20 to 1.24 against 1.15 to 1.21 times, and potion-base-8M rose from 71.8 to 73.5 to 91.7 to 91.8 M tokens/s. The gap between wall and CPU time is almost gone; what remains is the CPU time growth of a task when all cores run and the idle time of workers. The per-core and memory bounds do not change.

## How to run

    uv run python benchmarks/roofline.py probe
    uv run python benchmarks/roofline.py measure --model potion-base-8M --corpus massive --workers 8 10 --json out.json
    uv run python benchmarks/roofline.py show out.json --hardware m4-pro-14 --workers 14
    uv run python benchmarks/roofline.py compare --model static-retrieval-mrl-en-v1 --dim 1024 --corpus wikitext --dtypes float32 int8 --workers 1 8

The models and corpora come from the local Hugging Face cache; nothing is downloaded. A new machine gets a profile in `benchmarks/roofline_hardware.toml` with the sources of its numbers and the read roofs that `probe` measures on it.

## Sources

- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM 52(4), 2009, doi:10.1145/1498765.1498785. The roof is the sustained bandwidth a benchmark measures, not the vendor peak.
- G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", AFIPS 1967, doi:10.1145/1465482.1465560.
- M. D. Hill, M. R. Marty, "Amdahl's Law in the Multicore Era", IEEE Computer 41(7), 2008, doi:10.1109/MC.2008.209: on asymmetric cores the parallel part runs at the summed speed of the cores.
- J. D. McCalpin, STREAM, https://www.cs.virginia.edu/stream/: sustained memory bandwidth measured, not quoted.
- Apple Newsroom, "Introducing M1 Pro and M1 Max", 18.10.2021, https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/: "up to 200GB/s of memory bandwidth", "10-core CPU, including eight high-performance cores and two high-efficiency cores".
- Apple Newsroom, "Apple introduces M4 Pro and M4 Max", 30.10.2024, https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/: "273GB/s of memory bandwidth", "up to 14-core CPU consisting of up to 10 performance cores and four efficiency cores".

# Performance: how close fse comes to the hardware limit

This document says how fast fse can be on a given machine, why, and how close it comes. It holds a small model of the limits (a roofline model), the assumptions behind it, and measurements on an Apple M1 Pro. `benchmarks/roofline.py` computes the same numbers for any encoder, corpus and hardware profile.

## In short

On an M1 Pro (8 performance and 2 efficiency cores), master of 30.10.2026:

- **One worker** runs at 89 to 98 % of what the measured time per stage allows. The stages add up to the whole encode, so there is no hidden cost outside them.
- **Eight workers** reach 62 to 93 % of the Amdahl bound and 39 to 78 % of the per-core bound. Most of the gap comes from tasks that run slower when all cores run (5 to 48 % more wall time than alone, 3 to 22 % more CPU time) and from workers that wait at the end of a call.
- **Memory bandwidth is not the limit**, also not at 1024 dimensions. static-retrieval-mrl-en-v1 at 1024 dimensions reads 135 to 193 GB/s of table rows with 8 or 10 workers. That is 68 to 103 % of Apple's 200 GB/s, but more than the cores read from memory in a sequential test (115.6 GB/s at most), so most rows come from the caches. The limit there is the pooling work per core.
- **int8 tables at 1024 dimensions** are 1.18 to 1.32 times as fast as float32 on WikiText and 1.03 to 1.12 times on MASSIVE, end to end. They help where the rows a corpus uses do not fit the caches as float32.

## The model in plain words

fse turns a text into a vector in a few steps: the strings are joined into one buffer, the buffer is cut into tasks, each task is tokenized and its token rows are averaged (pooled), and the result is normalized. The work grows with the tokens, so the model counts **tokens per second**. A text costs its tokens: texts per second are tokens per second divided by the tokens per text.

Three things limit the rate. Each gives a bound, the largest rate that this limit allows.

| bound | what limits | in one sentence |
|---|---|---|
| per core | the work of the workers | N cores do N times the work of one core. |
| Amdahl | the serial part | Work in the calling thread does not get faster with more cores. |
| memory | reading the table | Each token reads one row of the table from memory, and memory has a bandwidth. |

The **combined bound** puts them together: the serial part, then the parallel part at the slower of its two limits (work or memory). Measured throughput divided by a bound is the **share**: how close fse comes to that limit.

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

Measured at one worker, in seconds per token: `t_ser`, the serial stages (join, scan, the calling thread's part of the run, the rest of `encode`), and `t_par`, the stages of the workers (tokenize and lookup, pool, normalize, the rest of the tasks). `C(N)` is the summed speed of the cores N workers use, counted in performance cores:

    C(N)          = min(N, P) + r_E * min(max(N - P, 0), E)        (Hill and Marty 2008)
    per core      = C(N) / t_par
    Amdahl        = 1 / (t_ser + t_par / C(N))                       (Amdahl 1967)
    memory        = BW / bytes_per_token                             (Williams et al. 2009)
    combined      = 1 / (t_ser + max(t_par / C(N), bytes_per_token / BW))

    bytes_per_token = row + weight (4) + id (4) [+ scale (4) for int8]
                      + 4 * dim / tokens_per_text  (the output row, written once)

P and E are the performance and efficiency cores, `r_E` the speed of an efficiency core relative to a performance core, BW a memory bandwidth in bytes per second.

## Assumptions

Each bound holds only under its assumptions. A share above 100 % (the tool flags it from 102 % on, the noise of the stage times) means an assumption failed, and the report says which.

1. **A core works as fast as it did alone.** The per-core and Amdahl bounds scale the time of one worker. With all cores busy a core can run slower (a lower clock, shared L2 caches, waits for the GIL). The tool measures this directly: the summed wall and CPU time of the tasks at N workers against one worker.
2. **Threads fill the performance cores first, and an efficiency core counts as a performance core** (`r_E = 1`) unless a profile or `--e-core-ratio` says otherwise. This gives the largest C(N), so the bound stays a ceiling. The real ratio on fse's work is not measured; it matters only for more workers than performance cores.
3. **The calling thread is not counted as a core** for the parallel part.
4. **Every byte comes from memory** in the memory bound. Repeated rows of frequent tokens hit the caches, so real traffic is lower, and a measured rate can exceed this bound. The report prints the rate of row bytes read (tokens per second times row bytes) next to it, and notes when the distinct rows of the corpus fit the L2 cache of one cluster.
5. **Work and memory overlap within a core** in the combined bound (`max`, not the sum).
6. **The scan at N workers is measured at N**, every other serial stage at one worker.

Not in the model: the cost of handing the GIL from thread to thread (it shows as the gap between wall and CPU time of the tasks), imbalance at the end of a call, and the system level cache, whose size Apple does not publish.

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

A second run gave the same values within 4 %. The CPU cores of this machine read at most about 116 GB/s from memory in this test, well below Apple's 200 GB/s, which the chip as a whole (with its GPU) is specified for. fse's kernel reads about 77 GB/s when no row is reused. Both are lower estimates: a better tuned reader may get more.

## Measured on an M1 Pro

Setup: master db387ff, Python 3.13.16, macOS on an M1 Pro (10 cores, 16 GB). The encoders are loaded with `Encoder.from_pretrained` at the snapshots pinned in `benchmarks/roofline.py`, with their own normalize (POTION normalizes, static-retrieval-mrl-en-v1 does not). The corpora are those of `benchmarks/bench_wordpiece.py`: WikiText-103 validation and test (5,352 texts, 454 bytes and about 95 tokens each) and MASSIVE intent test (151,674 texts, 55 bytes and 13 to 15 tokens each). Per cell, one process: the stages at one worker (the fastest of 15 rounds), then `encode` at 1, 8 and 10 workers in 5 paired rounds of alternating order (the best of 5 passes each, the median of the rounds). Every cell ran under the shared benchmark lock at a load below 5; no round of the cells shown was discarded (one earlier run of the 1024-dimension MASSIVE cell lost all its rounds to the load and was run again). The machine was not quiet: other work kept the load at 2.6 to 4.9. Each cell ran twice; the tables give the range of both runs.

WikiText:

| model, dims | workers | M tokens/s | rows read, GB/s | of per core | of Amdahl | of memory 200 GB/s (spec) | of read 115.6 GB/s (measured) |
|---|---:|---:|---:|---:|---:|---:|---:|
| potion-base-8M, 256 | 1 | 21.6 to 21.7 | 22 | 90% to 91% | 99% | 11% | 20% |
|  | 8 | 84.3 to 89.9 | 86 to 92 | 44% to 47% | 74% to 83% | 44% to 47% | 76% to 81% |
|  | 10 | 71.8 to 73.5 | 74 to 75 | 30% to 31% | 56% to 60% | 37% to 38% | 65% to 66% |
| potion-base-32M, 512 | 1 | 12.0 to 12.5 | 25 to 26 | 91% to 93% | 94% to 97% | 12% to 13% | 22% to 23% |
|  | 8 | 51.3 to 61.2 | 105 to 125 | 49% to 57% | 62% to 79% | 53% to 64% | 92% to 110% |
|  | 10 | 51.6 to 52.7 | 106 to 108 | 39% | 53% to 58% | 54% to 55% | 93% to 95% |
| static-retrieval-mrl-en-v1, 256 | 1 | 21.8 to 23.3 | 22 to 24 | 89% to 91% | 98% to 99% | 11% to 12% | 20% to 21% |
|  | 8 | 76.0 to 94.5 | 78 to 97 | 39% to 46% | 72% to 78% | 40% to 49% | 69% to 85% |
|  | 10 | 70.5 to 77.9 | 72 to 80 | 29% to 30% | 57% to 59% | 37% to 41% | 64% to 70% |
| static-retrieval-mrl-en-v1, 1024 | 1 | 7.1 to 7.3 | 29 to 30 | 98% | 100% | 15% | 25% to 26% |
|  | 8 | 32.9 to 40.8 | 135 to 167 | 57% to 69% | 67% to 77% | 68% to 85% | 118% to 146% |
|  | 10 | 34.9 to 37.2 | 143 to 153 | 49% to 50% | 58% to 59% | 72% to 77% | 125% to 134% |

MASSIVE:

| model, dims | workers | M tokens/s | rows read, GB/s | of per core | of Amdahl | of memory 200 GB/s (spec) | of read 115.6 GB/s (measured) |
|---|---:|---:|---:|---:|---:|---:|---:|
| potion-base-8M, 256 | 1 | 12.4 to 12.5 | 13 | 97% | 98% to 99% | 7% | 12% |
|  | 8 | 71.1 | 73 | 70% | 76% to 80% | 39% | 68% |
|  | 10 | 79.2 to 79.9 | 81 to 82 | 62% to 63% | 69% to 74% | 44% | 76% to 77% |
| potion-base-32M, 512 | 1 | 8.0 | 16 | 97% | 99% to 100% | 9% | 15% |
|  | 8 | 48.8 to 48.9 | 100 | 74% | 86% to 93% | 54% | 94% |
|  | 10 | 49.9 to 51.9 | 102 to 106 | 61% to 63% | 73% to 83% | 55% to 58% | 96% to 100% |
| static-retrieval-mrl-en-v1, 256 | 1 | 16.6 to 17.0 | 17 | 96% to 97% | 97% to 100% | 9% | 16% |
|  | 8 | 85.1 to 90.9 | 87 to 93 | 62% to 65% | 63% to 78% | 47% to 50% | 81% to 87% |
|  | 10 | 85.8 to 91.1 | 88 to 93 | 50% to 52% | 51% to 65% | 47% to 50% | 82% to 87% |
| static-retrieval-mrl-en-v1, 1024 | 1 | 7.0 to 7.1 | 29 | 98% to 99% | 99% to 101% | 15% to 16% | 27% |
|  | 8 | 42.3 to 44.6 | 173 to 183 | 75% to 77% | 82% to 88% | 93% to 98% | 161% to 169% |
|  | 10 | 44.7 to 47.2 | 183 to 193 | 63% to 65% | 71% to 78% | 98% to 103% | 169% to 179% |

In texts per second, 8 workers give about 0.8 to 1.0 million WikiText texts and 5.3 to 6.2 million MASSIVE texts at 256 dimensions, and 0.34 to 0.43 million and 2.9 to 3.0 million at 1024 dimensions.

The stages at one worker, in ns per token (second run):

| model, dims | corpus | join | scan | caller | rest of encode | tokenize | pool | normalize | rest of tasks | serial fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| potion-base-8M, 256 | WikiText | 2.7 | 0.1 | 0.1 | 0.7 | 19.0 | 21.2 | 1.6 | 0.2 | 0.079 |
| potion-base-32M, 512 | WikiText | 2.7 | 0.1 | 0.3 | -0.4 | 19.6 | 49.3 | 3.0 | 4.0 | 0.034 |
| mrl-en-v1, 256 | WikiText | 2.7 | 0.1 | 0.1 | 1.5 | 18.6 | 21.1 | | 1.0 | 0.097 |
| mrl-en-v1, 1024 | WikiText | 2.7 | 0.1 | 0.2 | -0.1 | 19.0 | 118.3 | | 1.7 | 0.021 |
| potion-base-8M, 256 | MASSIVE | 1.3 | 0.0 | 0.2 | -0.1 | 43.6 | 22.3 | 10.8 | 1.6 | 0.018 |
| potion-base-32M, 512 | MASSIVE | 1.3 | 0.1 | 0.5 | 2.0 | 47.9 | 47.7 | 21.4 | 4.5 | 0.031 |
| mrl-en-v1, 256 | MASSIVE | 1.2 | 0.0 | 0.2 | -1.3 | 34.4 | 21.8 | | 2.0 | 0.002 |
| mrl-en-v1, 1024 | MASSIVE | 1.2 | 0.1 | 0.6 | -0.6 | 35.1 | 100.4 | | 2.8 | 0.009 |

The two rests are differences of the fastest times of separate stages, so they can come out slightly negative (at most 1.3 ns per token here, under 2 % of the total).

### What limits each case

- **One worker** is at 89 to 98 % of the per-core bound and 94 to 101 % of the Amdahl bound. The second number checks the stage split: the stages account for the whole encode.
- **256 and 512 dimensions** (POTION, the Matryoshka cut to 256): tokenizing and pooling take about the same time on WikiText; on MASSIVE, with 13 to 15 tokens per text, tokenizing takes twice the pooling and normalize adds 10 to 21 ns per token. The bound is the work of the workers. Eight workers lose against it because each task runs slower when all run (task wall time 1.05 to 1.48 times that of one worker, CPU time 1.03 to 1.22 times) and because the workers are busy only 70 to 100 % of the time outside the serial part (the last tasks of a call; an estimate, from the serial time at one worker). Ten workers are no faster than eight on WikiText: their tasks take 1.4 to 2.0 times the wall time of one worker against 1.2 to 1.4 times the CPU time, so the extra wait is spent off the CPU, mostly for the GIL.
- **1024 dimensions** (static-retrieval-mrl-en-v1): pooling is 72 to 83 % of the time at one worker (100 to 118 ns per token). With 8 or 10 workers fse reads 135 to 193 GB/s of rows. The probe shows the cores read at most 115.6 GB/s from memory, and fse's kernel at most 88.6 GB/s even on consecutive rows, so these rows come mostly from the caches: the 10,141 (MASSIVE) to 19,871 (WikiText) distinct rows of a corpus take 40 to 78 MiB, but a few frequent tokens make up most reads. The limit is the pooling work per core, which also slows down when all cores run (task CPU time 1.06 to 1.27 times that of one worker). Earlier reports read the 130 to 180 GB/s against Apple's 200 GB/s and concluded that this case is near the memory bandwidth limit; the measured read roofs do not support that.
- **The serial part** (mostly the join of the strings, `_append_str`, 1.2 to 2.7 ns per token) limits only far beyond these core counts: its Amdahl limit is 229 million tokens per second or more.

### Profiles

`make profile-cpu` (samply) of `roofline.py _encode`, hottest functions over all threads:

- potion-base-8M, MASSIVE, 1 worker: `_wp_piece` 27.6 % (34.7 % with its callees), `fse_loop_f32_neon` (pooling) 20.0 %, `fse_normalize_body` 8.4 %, `_append_str` 5.2 %, `TextLookup._sentence` 4.9 %. The process includes loading the model.
- static-retrieval-mrl-en-v1 1024, WikiText, 8 workers: `fse_loop_f32_neon` 57.8 %, `__psynch_cvwait` 23.5 % (threads waiting: for a task, a result or the GIL), `_wp_piece` 10.1 %; the calling thread spends 9.7 % of its samples in `_append_str`.
- potion-base-8M, WikiText, 8 workers: `__psynch_cvwait` 39.0 %, `_wp_piece` 21.1 %, `fse_loop_f32_neon` 20.5 %, `_append_str` 2.2 % (13.6 % of the calling thread).

Memory of the 1024-dimension WikiText run at 8 workers: peak RSS 338 MiB (`make profile-rss`); memray (`make profile-mem`) shows a peak of 1.32 GB of allocations, of which 1.07 GB is one allocation made while modules are imported; the RSS of 338 MiB shows that most of it is never touched. Per pass the result (21 MiB) and the joined buffer are the largest allocations.

## int8 tables at 1024 dimensions

Earlier measurements found no gain from int8 tables at 256 and 300 dimensions. At 1024 dimensions `roofline.py compare` ran static-retrieval-mrl-en-v1 with its table as float32 and as int8 (`WordVectors.quantize("int8")`) in one process, 5 paired rounds of alternating order, the best of 5 passes each; rounds with a load above 5 were discarded and repeated (11 of them, all at one worker). `Encoder` does not yet accept int8 tables for `wordpiece@1`, so both run the encoder's spec on a `Pipeline` of their own (`roofline_dtype.SpecEncoder`); the vectors of int8 differ from float32 by the quantization, which is not judged here.

| corpus | workers | float32, M tokens/s | int8, M tokens/s | float32 time / int8 time, median (range) |
|---|---:|---:|---:|---|
| WikiText | 1 | 7.3 | 9.3 | 1.28 (1.24 to 1.32) |
| WikiText | 8 | 41.8 | 51.9 | 1.24 (1.20 to 1.24) |
| WikiText | 10 | 37.2 | 45.9 | 1.24 (1.18 to 1.28) |
| MASSIVE | 1 | 7.1 | 7.4 | 1.04 (1.03 to 1.05) |
| MASSIVE | 8 | 44.6 | 47.0 | 1.05 (1.05 to 1.06) |
| MASSIVE | 10 | 46.5 | 48.4 | 1.05 (1.03 to 1.12) |

int8 reads a quarter of the row bytes. Its pooling at one worker is 5 % faster per token on MASSIVE (95 against 100 ns) and 42 % faster on WikiText (84 against 118 ns). The kernel does the same work per value, and widening int8 costs about what the smaller reads save while the rows are in the caches. The distinct rows of WikiText take 78 MiB as float32 and 19 MiB as int8, those of MASSIVE 40 and 10 MiB, so more float32 rows miss the caches on WikiText, where int8 gains most. So int8 tables pay off at high dimensions when a corpus uses many distinct rows, by up to about a quarter, not by a factor.

## What PR 136 changes

PR 136 (2.0: stages 1 to 4 of a task in one nogil section on the cpu backend, Laufzeit 2) was open when this was measured, so these numbers are those of master without it. It runs the stages of a task in one native call, so a task hands the GIL back and forth fewer times. In this model that shrinks the gap between the wall and the CPU time of the tasks at many workers (WikiText at 10 workers: 1.4 to 2.0 times the wall time of one worker against 1.2 to 1.4 times the CPU time) and the per-task part of the serial work; it does not change the per-core and memory bounds. Measuring the bounds again after it merges shows how much of the gap it closes.

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

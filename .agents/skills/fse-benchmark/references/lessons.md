# Lessons and their evidence

These lessons come from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen), 1 and 2 November 2026. Later lessons name their own ticket.
Use the current [measurement guide](../../../../docs/benchmarks/measurement.md) for commands and policy.
The [comparison guide](../../../../docs/fse-comparison.md) links compact summaries and immutable historical evidence.
Historical snapshots are evidence, not current executables or permission to add raw files to Git.

## Adapter and output traps

| Observed trap | Required response | Source |
| :--- | :--- | :--- |
| The ST compatibility loader discarded UNK while native StaticEmbedding included its table row. | Use native loading and retain saved UNK examples in the sample. | [Paired encoding protocol](../../../../docs/fse-comparison.md#paired-encoding-protocol), `prepare_models` in [the adapters](../../../../benchmarks/_speed_pipeline.py). |
| Python Model2Vec 0.10.0 uses Joblib threads in its documented batch path, despite the `use_multiprocessing` name. | Distinguish allowed CPUs, requested workers and observed utilization. One process does not prove one thread. | Same paired-protocol section and `model2vec_encoder` in the adapters. |
| Earlier Rust wrappers did not set the pool used by encoding. | Set the global Rayon pool and record `actual_threads`. Use the full contiguous `encode_flat` output contract. | `RUST_SOURCE` and `rust_child` in the adapters. |
| FastEmbed's static backend differs from an ONNX transformer. | Name the NumPy StaticTextEmbedding backend and time output stacking. | Paired protocol, `fastembed_encoder` in the adapters. |
| The original ST 16-thread call did not complete. A later process-pool call succeeded. | Reuse `start_multi_process_pool`. Batch size 1024 is a starting choice, not a measured optimum. Limit native and Torch threads to one per process. | Paired protocol, `st_encoder` and `st_pool_worker` in the adapters. |
| The first ST pool failed with `received 0 items of ancdata` at soft `nofile` 1024. Identical code succeeded at 524288. | Check the hard limit, raise the soft limit before pool creation, and check `/dev/shm`. Descriptor pressure is supported, but no EMFILE syscall was captured. | [ST failure and file-descriptor limit](../../../../docs/fse-comparison.md#st-failure-and-file-descriptor-limit), `st_file_limits` in the adapters. |

Never attach fse's runtime fingerprint to another backend. `package_identity` in
[the recorder](../../../../benchmarks/_measurement.py) hashes the executing package.
The Rust receipt adds binary identity. Recheck identity after execution.

## Input, memory and timing traps

- Sentences and paragraphs have different costs. Record splitting rules and length statistics.
  Sampling with replacement requires seed, distinct-count definition and repetition factor.
  Source: [paired encoding protocol](../../../../docs/fse-comparison.md#paired-encoding-protocol) and
  [input preparation](../../../../benchmarks/_speed_cells.py).
- Roughly 5,000-text calls and all M1 speed measurements were withdrawn. Clean short-call controls do not restore validity.
  Source: [eleven measurement rules](../../../../docs/benchmarks/measurement.md#eleven-measurement-rules).
- Model2Vec, FastEmbed and ST can hold chunks alongside a concatenated output. fse writes into its output array.
  Freeing warmup does not eliminate real concatenation. The current shared plan conservatively reserves two output-sized buffers even for fse.
  Inspect actual paths before proposing another estimate.
  Source: `memory_plan` in input preparation and the adapter implementation linked above.
- A memory admission failure does not prove the OS killed a process. Report each component against the cgroup limit.
  Do not quietly reduce one arm. Source: `memory_plan` and [rehearsal contract](../../../../docs/benchmarks/measurement.md#rehearse-before-allocation).
- Close, fsync, check and delete owned output between writing rounds. Cleanup writeback must finish before the next timer.
  The 100M study observed 16 workers faster than 32. Sequential configurations do not isolate CPU scaling.
  Source: [native-file writing protocol](../../../../docs/fse-comparison.md#native-file-writing-protocol).
- Encoding, first-read-to-close, external fsync and classifier inference are distinct measurements.
  The historical collection has no eligible current classification inference-speed cells.
  fse's frozen linear head and Model2Vec's fine-tuned MLP are different trained models.
  Source: [classification](../../../../docs/fse-comparison.md#classification).

- Memray follows forks but not `exec`. The speed controller launches new interpreters, so its allocation profile misses their work.
  Instrument executing children directly and state coverage of spawned workers.
  Source: [subprocess launch](../../../../benchmarks/_speed_run.py),
  [Memray limitations](https://bloomberg.github.io/memray/supported_environments.html#known-issues-and-limitations).

## Setup and lifecycle traps

- uv's first-index behavior can hide the pinned PyPI version when the Torch index contains the same name.
  Setup uses `unsafe-best-match` with exact versions, `--no-deps`, `--require-hashes`, then `uv pip check`.
  Hashes constrain distribution files, not which index supplied identical bytes.
  Prefer explicit per-package sources when changing lock generation.
  Source: [setup implementation](../../../../benchmarks/_measurement_setup.py),
  [uv index documentation](https://docs.astral.sh/uv/concepts/indexes/#searching-across-multiple-indexes).
- An exiting bootstrap can restart and fail at an existing checkout before SSH is usable.
  Keep SSH first, a persistent restart marker and final `sleep infinity` on success and failure.
  Source: [bootstrap template](../../../../benchmarks/config/bootstrap.sh),
  [inspect and set up](../../../../docs/benchmarks/measurement.md#inspect-and-set-up).
- Missing files during setup are normal. Distinguish these from SSH failures. `DONE` can mean failure.
  Source: [monitor, collect and retain](../../../../docs/benchmarks/measurement.md#monitor-collect-and-retain).
- A tiny rehearsal cannot prove production descriptor or shared-memory capacity.
  Rehearse completion, failure, restart, SIGTERM and admission before final manifest/DONE generation.
  Source: [rehearsal implementation](../../../../benchmarks/_measurement_rehearse.py) and the measurement guide.
- Large caches and fit artifacts can exhaust a developer machine. Keep generated files external and check disk.
  Arrange targeted cleanup of owned intermediates. Shared caches require their owner's approval.
  Source: [common contract](../../../../docs/benchmarks/measurement.md#common-contract).
- zsh adds 5 to the inherited nice value of every background job (option BG_NICE, on by default), also in `zsh -fc` without job control or user startup files.
  A controller started with `cmd &` or `nohup cmd &` from a zsh at nice 0 therefore runs at nice 5, and its timed children inherit that value.
  `nice -n 0` adds nothing, and on macOS an ordinary unprivileged process cannot lower its nice value again.
  Start timing from a process at nice 0: in the foreground, through a harness background shell without `&`, or after `setopt NO_BG_NICE` in the same zsh command.
  `speed.py --diagnose` refuses a nonzero nice before it creates directories or starts the budget, except with `--diagnostic-rehearsal` or `--local-phase prepare`, where `os.getpriority` is available.
  Observed 5 November 2026 in DR-218 (Suche mit höchstens einem Treffer je Dokument anbieten); fixed in DR-234 (zsh-Falle BG_NICE bei lokalen Messungen abfangen und dokumentieren).
  Source: [zsh options, BG_NICE](https://zsh.sourceforge.io/Doc/Release/Options.html#Job-Control),
  [zsh 5.9 exec.c](https://github.com/zsh-users/zsh/blob/zsh-5.9/Src/exec.c#L2689-L2696),
  `require_timing_priority` in [the system module](../../../../benchmarks/_development_system.py).

## Quality traps

- Task names alone do not preserve language subsets or splits. Resolve the configured benchmark task.
  Keep evaluation rows complete. Current MTEB(eng, v2) coverage is explicitly 40 of 41 tasks.
  MindSmallReranking has no score after insufficient disk and a later status 137 on a 64 GB allocation.
  Status 137 alone is not proof of a captured OOM event.
  Source: [quality study data](../../../../docs/benchmarks/measurement.md#quality-study-data) and
  [quality comparison](../../../../docs/fse-comparison.md#quality).
- Historical WikiText SIF, wordfreq fits and published FineWeb SIF are different fitted states.
  Preserve corpus, frequencies, seed, tokenizer, components and exported artifact identities, including FastText OOV ngrams.
  Never attach an old quality row to a newly fitted artifact.
  Source: [published SIF-10 fit](../../../../docs/benchmarks/measurement.md#published-sif-10-fit).
- The two approved WikiText serializations feed identical split tokens despite whitespace differences.
  Accept only documented digests and record the actual file hash. Do not generalize this proof to arbitrary files.
  Source: [fit corpus equivalence](../../../../docs/benchmarks/measurement.md#fit-corpus-equivalence).
- Historical STS Pearson and MTEB Spearman differ. Corpus and frequencies also changed.
  No controlled ablation assigns the GloVe difference to one cause.
  Source: [quality comparison](../../../../docs/fse-comparison.md#quality).
- Keep published-fit reconstruction when its former caller was a retired experiment.
  FineWeb sampling remains part of reproducing released models.
  Source: [published SIF-10 fit](../../../../docs/benchmarks/measurement.md#published-sif-10-fit).

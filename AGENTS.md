# AGENTS.md

Guidance for coding agents working in this repository.

## Project

fse (Fast Sentence Embeddings) computes sentence vectors from word vectors, with a native kernel in `src/fse/_native_kernel.h` and `src/fse/_native.pyx`. master is the development branch of fse 2.0, a rewrite planned by the maintainer (docs/architecture/overview.md, "Principle and stages"): an inference engine for finished static embeddings, which trains nothing; the 1.x API was removed from master and is in maintenance mode on the branch `1.x`. No feature work on request.

- Python 3.11 to 3.15. gensim is optional (extra `fse[gensim]`) and does not build on 3.14 and newer yet (piskvorky/gensim#3640); there the tests of `fse.gensim` are skipped.
- Build backend is setuptools (pyproject.toml for metadata, setup.py only for the extension). Version comes from git tags (`v1.2.3`) via setuptools-scm.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design, the numerical contract and the rules that keep the native kernel and the reference consistent. The index and `docs/architecture/` are the single source of truth for those rules.

## Commands

Use uv and the Makefile; CI calls the same targets.

    make install       # uv sync --all-groups, builds the extension
    make check         # format-check + lint + typecheck + test
    make test          # tests without network, in parallel (pytest-xdist, PYTEST_N=auto)
    make test-network  # tests that download from the Hugging Face Hub
    make rebuild-ext   # after editing .pyx, .pxi, .pxd or .h
    make build         # sdist + wheel, twine check

`make test`, `make test-cov` and `make test-mlx` run the tests in worker processes of pytest-xdist; `make test PYTEST_N=0` runs them in one process (to debug a test, and as the baseline of a measurement), and a single file with `uv run pytest tests/test_x.py` is always serial. A test must not depend on the order of the tests, on another test's files or on running alone: each worker is a separate process with its own temporary directories. A test that waits for another thread proves it reached the blocking call (`tests/support/waiting.py`) instead of sleeping, because a loaded machine breaks a sleep.

Performance changes report before and after numbers: `uv run python benchmarks/bench_kernel.py --python-v1 <python of fse 1.1.0>` for the kernel and `benchmarks/bench_tokenize.py` from raw text; `benchmarks/baseline/` holds the 1.1.0 values 2.0 is compared with. Keep measurements proportionate:

- at most 5 to 10 minutes per comparison, and about 30 minutes for all measurements of a PR together; trim repetitions and variants before exceeding it; alternate the base commit and the branch in a few paired runs and note the load average (the scripts warn about a high one);
- when code only moves (the same machine code, shown by comparing the disassembly of the native functions of both builds with addresses normalized, or the same preprocessed header), that comparison replaces the timing;
- longer series only where code that computes or runs hot really changes.

Every PR that touches the hot path (the native kernels, the text sources, the runtime, the backends, the sinks) also measures the end-to-end float32 throughput against its base, `uv run python benchmarks/bench_runtime.py throughput` on both in paired runs, so a slowdown shows even where nobody expects one; the numbers go into the PR description. Every gate of a PR that touches the hot path or has a performance goal is profiled end to end, CPU and memory, not only when it misses; other PRs profile only when a reviewer asks. Keep profiling proportionate, as the measurements: one representative run per profile on the deciding configuration, a few minutes each, not a series: `make profile-cpu RUN="benchmarks/<script> ..."` (samply on macOS, which needs no root, with the hottest functions per thread in the terminal by `tools/profile_top.py`; `py-spy --native` on Linux; samply is a system tool, `cargo install --locked samply`), `make profile-mem RUN=...` (memray with native allocations, a flame graph over time) and `make profile-rss RUN=...` (the RSS of the process tree over time, `tools/rss.py`). The PR description names the functions that take the time and the memory peaks. When a gate misses, the profile decides what to optimize, and switching parts off one at a time (an ablation) checks the attribution before a redesign.

Before planning a hard question or technique (a new data structure, algorithm, numerical method or architecture pattern), research prior art briefly: has it been solved, and which tricks, tips and patterns do others use for exactly this problem. Cite primary sources (papers, official docs, source code), invent no numbers, and give the findings to the plan.

Never wait for pull request checks or CI in the foreground. Start the wait as a background job (a monitor that ends when the checks are done) and keep working; act on its result when it reports.

## Rules

- Validate strictly and never fail silently: raise a clear exception instead of coercing, ignoring or truncating input. Decorate new public entry points with `fse._validation.validate_arguments` unless they are called per element or receive bulk data; measure the overhead when in doubt. See ARCHITECTURE.md.
- 2.0 artifacts store no class paths. From the first release of 2.0 on, never rename or reuse their persistence names (identifiers such as `weighted_mean@1`, `fse.` metadata keys, tensor names, artifact kinds and format versions). Before that release, the maintainer may remove names that are no longer needed (not reserve them); artifacts of a pre-release that hold them are refused with `UnsupportedArtifactError`. See docs/architecture/persistence.md, "Persistence names replace class paths".
- New 2.0 code on the CPU follows the numerical contract in docs/architecture/numerical-contract.md, "Numerical contract (`transform_version` 2)" and matches the numpy reference in `fse._reference` bit for bit; compare bit patterns, not values. Any change of arithmetic bumps the version of the aggregation or step it changes.
- Device backends (MLX, later CUDA) follow the tolerance contract in docs/architecture/backends.md, "Tolerance contract of device backends" instead (maintainer's decision of 29.10.2026): every row within its bound of the cpu backend (an acceptance bound the conformance suites check; rows where it could fail are detected and computed by the host), deterministic per device, and the errors of the cpu backend. An operation they leave to their host is bit-identical to the cpu operation on the same inputs.
- Every backend passes the conformance suite (`tests/test_conformance.py` and its siblings): CPU backends bit for bit, device backends under the tolerance contract. Never weaken a comparison, the bound or the meta-tests; only the maintainer changes the bound.
- Do not hard-code vocabulary positions in tests. gensim versions order words with equal counts differently; derive expectations from `key_to_index`.
- The dependency lower bounds in pyproject.toml must match the pins of the minimum job in `.github/workflows/ci.yml`.
- pytest treats warnings as errors. Add a narrow, commented `filterwarnings` ignore only for warnings from third-party code.
- Log only through `fse._log.emit` with an event of `fse._log.EVENTS`; errors raise and are never also logged; no paths, error texts or input data in records; nothing per task, row or sentence, in native code or on the MLX device thread. See docs/architecture/logging.md, "Logging".
- Record user-visible changes in `CHANGELOG.md`.
- Test modules never import each other. Helpers several of them share live in `tests/support/` (on the pytest `pythonpath`); a long test module splits by topic into `test_<module>_<topic>.py`.

## Guardrails

`make lint` enforces, and CI and the pre-commit hooks run the same:

- at most 500 lines per source file under the `paths` of `[tool.fse.guardrails]` (src, tests, tools, benchmarks, media, setup.py; the generated Unicode tables excepted); a new top-level folder of code joins `paths`;
- the ruff complexity limits (C901 10, PLR0911 6, PLR0912 12, PLR0913 and PLR0917 15, PLR0915 50) and a cognitive complexity of at most 15 per Python function (complexipy; Cython and C functions are limited by their file length);
- the import layers and forbidden packages in `[tool.importlinter]`;
- that only `fse._text` (stages 1 and 2) and the cpu backend `fse._backends._cpu` (the kernels) import `fse._native`, and the extension imports only `fse._contract` from fse;
- that only `fse._container._hash` imports a hash library (hashlib, xxhash): every checksum and fingerprint is xxh3-128 through it;
- that only `fse._log` and `fse._log_config` import `logging` or `warnings` (the extension neither), and that the modules the MLX device thread runs (`log-free`) do not import `fse._log` (these import rules are `tools/guardrails_imports.py`);
- a test file for every module of fse (`fse/_a.py` is `tests/test_a.py`, `fse/_b/_c.py` is `tests/test_b_c.py`; only the generated `_version.py` has none, and the `.pxi` parts that `_native.pyx` includes are tested through `tests/test_native.py` and checked for imports like it);
- no unused code (vulture, with the names Python, pytest or a library use by name in `[tool.vulture]`) and no undeclared or unused dependency (deptry).

What exceeded a limit when the guardrails came is the ratchet in `[tool.fse.guardrails.ratchet]`, which only shrinks; CI also checks it against the base branch. Never add an entry, raise a number, add an inline suppression (`# noqa: C901`, `# complexipy: ignore`) or an `ignore_imports`. When code does not fit, split it: private modules may be split and renamed freely, persistence names never (from the first release on). When you improve an entry, lower or remove it (`tools/check_guardrails.py` says which). Place every new module in a layer of the import contract.

## Releases

A pushed `v*` tag on `master` runs the release job in CI. It waits for approval in the `release` environment and publishes to PyPI with trusted publishing. Never create tags or releases without the maintainer's explicit approval.

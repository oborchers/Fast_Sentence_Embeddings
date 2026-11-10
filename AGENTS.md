# AGENTS.md

Guidance for coding agents working in this repository.

## Project

fse (Fast Sentence Embeddings) runs finished static embedding models on the CPU and trains nothing. It computes sentence vectors from token vectors, with a native kernel in `src/fse/_native_src/kernel/_native_kernel.h` and `src/fse/_native.pyx`. fse 2.0 is released; master is the development branch of fse 2.1, planned by the maintainer (docs/architecture/overview.md, "Principle and stages"). fse is an inference engine for finished static embeddings, which trains nothing; the 1.x API was removed from master and is in maintenance mode on the branch `1.x`. No feature work on request.

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

`make test`, `make test-cov` run the tests in worker processes of pytest-xdist; `make test PYTEST_N=0` runs them in one process (to debug a test, and as the baseline of a measurement), and a single file with `uv run pytest tests/test_x.py` is always serial. A test must not depend on the order of the tests, on another test's files or on running alone: each worker is a separate process with its own temporary directories. A test that waits for another thread proves it reached the blocking call (`tests/support/waiting.py`) instead of sleeping, because a loaded machine breaks a sleep.

For measurement planning, rehearsal and audits, use the shared [fse-benchmark skill](.agents/skills/fse-benchmark/SKILL.md). Claude Code discovers the same source through `.claude/skills/fse-benchmark`.

Use the two central measurement entries, `benchmarks/quality.py` and `benchmarks/speed.py`, and follow [the measurement guide](docs/benchmarks/measurement.md). Studies, source pins and dependency locks live in `benchmarks/config/`. Add a data cell or native adapter instead of another runner. Do not add tests for measurement scripts, except for the cross-process coordination of the shared locks (`bench.lock`, `heavy.lock`). Existing 1.1.0 numerical and quality baselines stay in `benchmarks/baseline/`.

Before an experiment, declare its question, claim class, resulting decision, workload, relevant effect boundary, total wall-time budget, estimated cost and stop rule. Choose the least costly protocol that can support that claim:

| Class | Evidence and limit |
| :--- | :--- |
| correctness | Functional probes, bits, conformance and quality. No speed claim. |
| descriptive | One arm, fixed repetitions, absolute median and IQR. No comparison claim. |
| regression | Paired base/candidate with a declared relevant boundary. Short screens remain preliminary without sufficient uncertainty evidence. |
| optimization | Full local v2 controls, independent A/A validation and confirmation. A screen never decides keep. |
| publication | Existing Linux publication protocol and release guarantees. |

Budget includes preparation, warmups and load/lock waits across all selected cells. Stop at a safe operation boundary when exhausted, preserve raw data and report inconclusive. Never extend or retry automatically. Before long runs, use small adapter/reference and provenance probes. Stabilize APIs first. Review fixes use affected tests and conformance, with one full `make check` on the final state, repeated only for a new risk or failed gate. Profile representative changed paths and resource risks, verifying which children execute them.

A hot-path or performance change needs no Linux comparison by default. Choose its evidence by claim class. Local speed results on an Apple M1 or another development machine support the classes correctness, descriptive, regression and optimization as development indications and are never published as speed evidence (a recipe may show them, as the last rule of this paragraph allows). Local correctness and quality results keep their validity. The class publication (README or other published speed numbers, release guarantees) requires Linux publication evidence through the existing protocol. The only other case that requires a paired base/candidate end-to-end float32 comparison on one Linux machine is an explicit approval by the maintainer of a Linux machine for the ticket. Such an approved comparison can also serve class regression or optimization. Voluntary Linux comparisons remain allowed and follow the protocol of their class. A required Linux comparison (class publication or an approved machine) needs enough input for every measured fse inference call (encode or predict) to last at least two seconds, at least three alternating rounds, checked equivalent output, and CPU and memory profiling of a representative deciding configuration. Its PR names the machine, input, model, worker path, median, paired factor and profile attribution. Every PR description states which claim classes it used and its actual Linux comparison status. When no Linux comparison was run, it says so explicitly. Approval of a Linux machine does not approve paid resources: the next paragraph applies before every paid run. No Mac speed numbers in the README, in speed comparisons with other libraries or versions, in release guarantees or in publication evidence, and no millisecond-call measurements. The only exception is a recipe: it may show the numbers of the machine that ran it, with that machine named and without a claim of comparison. Other comparisons (for example of behavior or parsing) carry no speed numbers and are not affected. A reduced Linux rehearsal checks functionality only. When machine code merely moves, identical normalized disassembly or the same preprocessed header can replace timing. No local result replaces numerical conformance, bit identity with the numpy reference or persistence guarantees. Search time per request uses blocks of whole query-pool passes in a fixed order, not a two-second minimum for individual search calls. Calibrate each cell toward a block of at least two seconds (target three), freeze the maximum required pass count across all cells of a model, and use it for at least three rotating rounds. Report median and IQR across blocks. A production block below two seconds is invalid.

Before a paid run, agree on the workload, resources, soft runtime/cost limits and collection owner. Rehearse both the selected adapters and lifecycle controls in the pinned Linux image. Results and model caches stay outside the checkout. Do not commit model weights, matrices, raw data or reconstruction output, including on the evidence branch. No single file above 10 MB without the coordinator's approval. Inspect `git diff --stat` and file sizes before every push. The guide specifies continuous results, correctness checks, telemetry, safe restart and collection. It replaces the former advice to trim local comparisons to a few minutes.

Before planning a hard question or technique (a new data structure, algorithm, numerical method or architecture pattern), research prior art briefly: has it been solved, and which tricks, tips and patterns do others use for exactly this problem. Cite primary sources (papers, official docs, source code), invent no numbers, and give the findings to the plan.

Never wait for pull request checks or CI in the foreground. Start the wait as a background job (a monitor that ends when the checks are done) and keep working; act on its result when it reports.

## Rules

- **Validate before converting.** Validate a public input in its original form for both type and value before converting it with `np.asarray`, `list()`, `dict()` or another conversion. Reject booleans where a number is required, and reject sets and dictionaries where an ordered sequence is required. Conversion must never hide invalid input or change which values are accepted. Review every new or changed public input against this rule.
- Validate strictly and never fail silently: raise a clear exception instead of coercing, ignoring or truncating input. Integration packages import only public fse names and validate inputs with their own `_checks` modules. Decorate new public entry points with `fse._validation.validate_arguments` unless they are called per element or receive bulk data; measure the overhead when in doubt. See ARCHITECTURE.md.
- 2.0 artifacts store no class paths. From the first release of 2.0 on, never rename or reuse their persistence names (identifiers such as `weighted_mean@1`, `fse.` metadata keys, tensor names, artifact kinds and format versions). Before that release, the maintainer may remove names that are no longer needed (not reserve them); artifacts of a pre-release that hold them are refused with `UnsupportedArtifactError`. See docs/architecture/persistence.md, "Persistence names replace class paths".
- New 2.0 code on the CPU follows the numerical contract in docs/architecture/numerical-contract.md, "Numerical contract (`transform_version` 2)" and matches the numpy reference in `fse._reference` bit for bit; compare bit patterns, not values. Any change of arithmetic bumps the version of the aggregation or step it changes.
- Every backend passes the conformance suite (`tests/test_conformance.py` and its siblings): every backend bit for bit. Never weaken a comparison, the bound or the meta-tests; only the maintainer changes the bound.
- Do not hard-code vocabulary positions in tests. gensim versions order words with equal counts differently; derive expectations from `key_to_index`.
- The dependency lower bounds in pyproject.toml must match the pins of the minimum job in `.github/workflows/ci.yml`.
- pytest treats warnings as errors. Add a narrow, commented `filterwarnings` ignore only for warnings from third-party code.
- Log only through `fse._log.emit` with an event of `fse._log.EVENTS`; errors raise and are never also logged; no paths, error texts or input data in records; nothing per task, row or sentence or in native code. See docs/architecture/logging.md, "Logging".
- Record user-visible changes as `changelog.d/DR-<nnn>.<type>.md` fragments, following `changelog.d/README.md`. Only release preparation updates the root `CHANGELOG.md`. Integration packages maintain their own handwritten `CHANGELOG.md` files instead of root fragments.
- Test modules never import each other. Helpers several of them share live in `tests/support/` (on the pytest `pythonpath`); a long test module splits by topic into `test_<module>_<topic>.py`.

## Guardrails

`make lint` enforces, and CI and the pre-commit hooks run the same:

- at most 500 lines per source file under the `paths` of `[tool.fse.guardrails]` (see [the configuration](pyproject.toml) for the complete path list; the generated Unicode tables excepted); a new top-level folder of code joins `paths`;
- the ruff complexity limits (C901 10, PLR0911 6, PLR0912 12, PLR0913 and PLR0917 15, PLR0915 50) and a cognitive complexity of at most 15 per Python function (complexipy; Cython and C functions are limited by their file length);
- the import layers and forbidden packages in `[tool.importlinter]`;
- that only `fse._text` (stages 1 and 2) and the cpu backend `fse._backends._cpu` (the kernels) import `fse._native`, and the extension imports only `fse._contract` from fse;
- that only `fse._container._hash` imports a hash library (hashlib, xxhash): every checksum and fingerprint is xxh3-128 through it;
- that only `fse._log` and `fse._log_config` import `logging` or `warnings` (the extension neither) (these import rules are `tools/guardrails_imports.py`);
- a test file for every module of fse (`fse/_a.py` is `tests/test_a.py`, `fse/_b/_c.py` is `tests/test_b_c.py`; only the generated `_version.py` has none, and the `.pxi` parts that `_native.pyx` includes are tested through `tests/test_native.py` and checked for imports like it);
- no unused code (vulture, with the names Python, pytest or a library use by name in `[tool.vulture]`) and no undeclared or unused dependency (deptry).

What exceeded a limit when the guardrails came is the ratchet in `[tool.fse.guardrails.ratchet]`, which only shrinks; CI also checks it against the base branch. Never add an entry, raise a number, add an inline suppression (`# noqa: C901`, `# complexipy: ignore`) or an `ignore_imports`. When code does not fit, split it: private modules may be split and renamed freely, persistence names never (from the first release on). When you improve an entry, lower or remove it (`tools/check_guardrails.py` says which). Place every new module in a layer of the import contract.

## Releases

A pushed `v*` tag on `master` runs the release job in CI. It waits for approval in the `release` environment and publishes to PyPI with trusted publishing. The GitHub release belongs to the release: the job `github-release` creates it from the `CHANGELOG.md` section of the tag, and the steps are in [the release skill](.agents/skills/release/SKILL.md). Never create tags or releases without the maintainer's explicit approval.

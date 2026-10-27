# AGENTS.md

Guidance for coding agents working in this repository.

## Project

fse (Fast Sentence Embeddings) computes sentence vectors from word vectors, with a native kernel in `src/fse/_native_kernel.h` and `src/fse/_native.pyx`. master is the development branch of fse 2.0, a rewrite planned by the maintainer (ARCHITECTURE.md, section 2.0 design); the 1.x API was removed from master and is in maintenance mode on the branch `1.x`. No feature work on request.

- Python 3.11 to 3.15. gensim is optional (extra `fse[gensim]`) and does not build on 3.14 and newer yet (piskvorky/gensim#3640); there the tests of `fse.gensim` are skipped.
- Build backend is setuptools (pyproject.toml for metadata, setup.py only for the extension). Version comes from git tags (`v1.2.3`) via setuptools-scm.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design, the numerical contract and the rules that keep the native kernel and the reference consistent. It is the single source of truth for those rules.

## Commands

Use uv and the Makefile; CI calls the same targets.

    make install       # uv sync --all-groups, builds the extension
    make check         # format-check + lint + typecheck + test
    make test          # tests without network
    make test-network  # tests that download from the Hugging Face Hub
    make rebuild-ext   # after editing .pyx, .pxd or .h
    make build         # sdist + wheel, twine check

Performance changes report before and after numbers on an idle machine (the scripts warn about a high load average): `uv run python benchmarks/bench_kernel.py --python-v1 <python of fse 1.1.0>` for the kernel and `benchmarks/bench_tokenize.py` from raw text; `benchmarks/baseline/` holds the 1.1.0 values 2.0 is compared with.

## Rules

- Validate strictly and never fail silently: raise a clear exception instead of coercing, ignoring or truncating input. Decorate new public entry points with `fse._validation.validate_arguments` unless they are called per element or receive bulk data; measure the overhead when in doubt. See ARCHITECTURE.md.
- 2.0 artifacts store no class paths; never rename or reuse their persistence names (identifiers such as `weighted_mean@1`, `fse.` metadata keys, tensor names). See ARCHITECTURE.md, section 2.0 design.
- New 2.0 code follows the numerical contract in ARCHITECTURE.md and matches the numpy reference in `fse._reference` bit for bit; compare bit patterns, not values. Any change of arithmetic bumps the version of the aggregation or step it changes.
- Every backend passes `tests/test_conformance.py`; never weaken its comparison or its meta-tests.
- Do not hard-code vocabulary positions in tests. gensim versions order words with equal counts differently; derive expectations from `key_to_index`.
- The dependency lower bounds in pyproject.toml must match the pins of the minimum job in `.github/workflows/ci.yml`.
- pytest treats warnings as errors. Add a narrow, commented `filterwarnings` ignore only for warnings from third-party code.
- Record user-visible changes in `CHANGELOG.md`.

## Guardrails

`make lint` enforces, and CI and pre-commit run the same:

- at most 500 lines per hand-written source file (the generated Unicode tables excepted);
- the ruff complexity limits (C901 10, PLR0911 6, PLR0912 12, PLR0913 and PLR0917 5, PLR0915 50) and a cognitive complexity of at most 15 per function (complexipy);
- the import layers and forbidden packages in `[tool.importlinter]`;
- that only `fse._text` imports `fse._native` and the extension imports only `fse._contract` from fse;
- a test file for every module of fse (`fse/_a.py` is `tests/test_a.py`, `fse/_b/_c.py` is `tests/test_b_c.py`; only the generated `_version.py` has none);
- no unused code (vulture, with the names Python, pytest or a library use by name in `[tool.vulture]`) and no undeclared or unused dependency (deptry).

What exceeded a limit when the guardrails came is the ratchet in `[tool.fse.guardrails.ratchet]`, which only shrinks; CI also checks it against the base branch. Never add an entry, raise a number, add an inline suppression (`# noqa: C901`, `# complexipy: ignore`) or an `ignore_imports`. When code does not fit, split it: private modules may be split and renamed freely, persistence names never. When you improve an entry, lower or remove it (`tools/check_guardrails.py` says which). Place every new module in a layer of the import contract.

## Releases

A pushed `v*` tag on `master` runs the release job in CI. It waits for approval in the `release` environment and publishes to PyPI with trusted publishing. Never create tags or releases without the maintainer's explicit approval.

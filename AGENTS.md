# AGENTS.md

Guidance for coding agents working in this repository.

## Project

fse (Fast Sentence Embeddings) computes sentence vectors on top of gensim word vectors, with a Cython core in `src/fse/models/average_inner.pyx`. master is the development branch of fse 2.0, a rewrite planned by the maintainer (ARCHITECTURE.md, section 2.0); 1.x is in maintenance mode on the branch `1.x`. No feature work on request.

- Python 3.11 to 3.13. Python 3.14 is blocked until gensim builds there (piskvorky/gensim#3640); CI runs it as an informational job.
- Build backend is setuptools (pyproject.toml for metadata, setup.py only for the extension). Version comes from git tags (`v1.2.3`) via setuptools-scm.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and the rules that keep the Cython and numpy paths consistent. It is the single source of truth for those rules.

## Commands

Use uv and the Makefile; CI calls the same targets.

    make install       # uv sync --all-groups, builds the extension
    make check         # format-check + lint + typecheck + test
    make test          # tests without network
    make test-network  # tests that download from the Hugging Face Hub
    make rebuild-ext   # after editing .pyx or .pxd
    make build         # sdist + wheel, twine check

Performance changes report before and after numbers from `uv run python benchmarks/bench_train.py` on an idle machine (the script warns about a high load average).

## Rules

- Validate strictly and never fail silently: raise a clear exception instead of coercing, ignoring or truncating input. Decorate new public entry points with `fse._validation.validate_arguments` unless they are called per element or receive bulk data; measure the overhead when in doubt. See ARCHITECTURE.md.
- Never rename modules under `fse.models` while master still ships them: saved 1.x models pickle class paths such as `fse.models.average.Average`. 2.0 artifacts store no class paths; never rename or reuse their persistence names (identifiers such as `weighted_mean@1`, `fse.` metadata keys, tensor names). See ARCHITECTURE.md, section 2.0.
- New 2.0 code follows the numerical contract in ARCHITECTURE.md and matches the numpy reference in `fse._reference` bit for bit; compare bit patterns, not values. Any change of arithmetic bumps the version of the aggregation or step it changes.
- Keep the numpy fallback in `average.py` and the Cython routines in `average_inner.pyx` numerically identical; tests compare them.
- Do not hard-code vocabulary positions in tests. gensim versions order words with equal counts differently; derive expectations from `key_to_index`.
- The dependency lower bounds in pyproject.toml must match the pins of the minimum job in `.github/workflows/ci.yml`.
- pytest treats warnings as errors. Add a narrow, commented `filterwarnings` ignore only for warnings from third-party code.
- Record user-visible changes in `CHANGELOG.md`.

## Releases

A pushed `v*` tag on `master` runs the release job in CI. It waits for approval in the `release` environment and publishes to PyPI with trusted publishing. Never create tags or releases without the maintainer's explicit approval.

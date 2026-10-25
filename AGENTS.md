# AGENTS.md

Guidance for coding agents working in this repository.

## Project

fse (Fast Sentence Embeddings) computes sentence vectors on top of gensim word vectors, with a Cython core in `src/fse/models/average_inner.pyx`. The project is in maintenance mode: compatibility fixes only, no new features.

- Python 3.10 to 3.13. Python 3.14 is blocked until gensim builds there (piskvorky/gensim#3640); CI runs it as an informational job.
- Build backend is setuptools (pyproject.toml for metadata, setup.py only for the extension). Version comes from git tags (`v1.2.3`) via setuptools-scm.

## Commands

Use uv and the Makefile; CI calls the same targets.

    make install       # uv sync --all-groups, builds the extension
    make check         # format-check + lint + test
    make test          # tests without network
    make test-network  # tests that download from the Hugging Face Hub
    make rebuild-ext   # after editing .pyx or .pxd
    make build         # sdist + wheel, twine check

## Rules

- Never rename modules under `fse.models`. Saved models pickle class paths such as `fse.models.average.Average`.
- Keep the numpy fallback in `average.py` and the Cython routines in `average_inner.pyx` numerically identical; tests compare them.
- Do not hard-code vocabulary positions in tests. gensim versions order words with equal counts differently; derive expectations from `key_to_index`.
- The dependency lower bounds in pyproject.toml must match the pins of the minimum job in `.github/workflows/ci.yml`.
- pytest treats warnings as errors. Add a narrow, commented `filterwarnings` ignore only for warnings from third-party code.
- Record user-visible changes in `CHANGELOG.md`.

## Releases

A pushed `v*` tag on `master` runs the release job in CI. It waits for approval in the `release` environment and publishes to PyPI with trusted publishing. Never create tags or releases without the maintainer's explicit approval.

# Contributing

fse is in maintenance mode. Pull requests that keep it working with current versions of Python, numpy, scipy, gensim or huggingface-hub are welcome. New features are out of scope and will be closed.

## Setup

You need [uv](https://docs.astral.sh/uv/) and a C compiler.

    make init      # virtual environment, dependencies, compiled extension, pre-commit hooks
    make check     # format check, lint, type check and tests

Run `make help` for all targets.

## Notes

- Read [ARCHITECTURE.md](ARCHITECTURE.md) before changing the training loop or the Cython kernel.
- After editing `src/fse/models/average_inner.pyx` or `.pxd`, run `make rebuild-ext`. The tests check that the compiled extension is loaded and fail on the slow numpy fallback.
- For performance changes, compare `uv run python benchmarks/bench_train.py` before and after on an idle machine and put both tables in the pull request. `--help` lists the inputs, models, worker counts and dimensions it covers.
- Tests that download from the Hugging Face Hub are marked `network`. `make test` skips them, `make test-network` runs them.
- Do not rename modules under `fse.models`. Saved models store their class paths, such as `fse.models.average.Average`, and would no longer load.
- Add an entry to `CHANGELOG.md` for every user-visible change.

## Releases

Maintainers release by pushing a tag such as `v1.1.0` on `master`. CI builds the sdist and wheels, and after approval in the `release` environment publishes them to PyPI with trusted publishing.

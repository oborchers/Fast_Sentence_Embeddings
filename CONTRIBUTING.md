# Contributing

master is the development branch of fse 2.0, which the maintainer plans and writes; 1.x is in maintenance mode on the branch `1.x`. Pull requests that keep 1.x working with current versions of Python, numpy, scipy, gensim or huggingface-hub are welcome there. Feature requests and feature pull requests are out of scope and will be closed.

## Setup

You need [uv](https://docs.astral.sh/uv/) and a C compiler.

    make init      # virtual environment, dependencies, compiled extension, pre-commit hooks
    make check     # format check, lint, type check and tests

Run `make help` for all targets.

## Notes

- Read [ARCHITECTURE.md](ARCHITECTURE.md), the index of [docs/architecture/](docs/architecture/), before changing the contracts or the native kernel.
- `make lint` checks file length, function complexity and the import layers against a ratchet that only shrinks; see the Guardrails section of [AGENTS.md](AGENTS.md). Split code that does not fit instead of adding exceptions.
- After editing `src/fse/_native.pyx`, its `.pxi` parts, `.pxd` or a `_native*.h` header, run `make rebuild-ext`.
- For performance changes, compare `uv run python benchmarks/bench_kernel.py --python-v1 <python of fse 1.1.0>` before and after on an idle machine and put both tables in the pull request. `--help` lists the dtypes, dimensions and vocabulary sizes it covers.
- Tests that download from the Hugging Face Hub are marked `network`. `make test` skips them, `make test-network` runs them.
- Do not rename or reuse the persistence names of 2.0 artifacts (identifiers, `fse.` metadata keys, tensor names).
- Add an entry to `CHANGELOG.md` for every user-visible change.

## Releases

Maintainers release by pushing a tag such as `v1.1.0` on `master`. CI builds the sdist and wheels, and after approval in the `release` environment publishes them to PyPI with trusted publishing.

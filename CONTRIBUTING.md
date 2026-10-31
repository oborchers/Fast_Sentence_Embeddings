# Contributing

fse 2.0 is an inference engine for finished static embeddings. It trains nothing.
The maintainer plans the 2.0 rewrite on `master`. Feature requests and unsolicited feature PRs are out of scope.
Bug reports and fixes are welcome. The `1.x` branch accepts bug and compatibility fixes for the released API.

## AI assistance is welcome

You may use AI tools to write code, tests, documentation and measurements.
You remain responsible for every line and every claim. Name the tools and model versions in your PR.
Explain what they did and what you checked yourself. Never submit private data or credentials as evidence.

The review criteria below apply whether or not you used AI.
We may defer a PR with missing evidence, or close it with a link to these requirements.
You can return with the missing evidence.

## Set up and check

You need standard CPython 3.11 to 3.15, [uv](https://docs.astral.sh/uv/), Make and a C compiler.

Free-threaded Python builds are unsupported.

```sh
UV_PYTHON=3.13 make init
make check PYTEST_N=4
```

`make init` creates the environment, installs dependencies, builds the extension and installs pre-commit hooks.
`make check` checks formatting, lint, import contracts, guardrails, types and tests without network downloads.
Use `make test-network` for Hub download tests and `make test-st` for sentence-transformers comparisons.
Run `make help` for all targets.
After editing Cython or native headers, run `make rebuild-ext`.

Read [ARCHITECTURE.md](ARCHITECTURE.md) and [AGENTS.md](AGENTS.md) before changing a contract or native code.
Keep conformance tests and bit identity unchanged. Bit identity means equal output bits, not just close numbers.
Do not add guardrail ratchet entries or suppressions. Split code that exceeds a limit.
Record user-visible changes in [CHANGELOG.md](CHANGELOG.md).

## Make the result reproducible

Include the base and head commits, exact commands, data sources, data revisions and random seeds.
Record CPU, instruction set, operating system, Python, NumPy, compiler and relevant package versions.
Link raw results and explain any skipped check. A screenshot or a claimed percentage alone is not enough.
For documentation-only changes, identify the source of each factual claim. Mark performance measurements as not applicable when appropriate.

## Measure performance changes

Follow the measurement rules in [AGENTS.md](AGENTS.md#commands). In particular:

- Alternate base and change in several paired rounds. Report the median, range and machine load.
- Run an A/A control, comparing the base with itself, to estimate noise.
- Use the same data, settings and environment on both sides. Keep measurements within the repository's time budget.
- For a hot-path change, compare end-to-end float32 throughput with `uv run python benchmarks/bench_runtime.py throughput` on both commits.
- For performance changes, include the required kernel comparison against 1.1.0 and any relevant tokenizer measurements.
- Profile hot-path or performance work end to end for CPU, allocation memory and process-tree resident memory.
  Name the time-consuming functions and memory peaks. Use `make profile-cpu`, `make profile-mem` and `make profile-rss`.
- If code only moves, use the repository's accepted disassembly or preprocessed-header comparison where it replaces timing.

Measure in the environment the change targets. An ARM NEON result does not prove an x86 AVX2 speedup.
Report which of x86 AVX2, ARM NEON, Linux and Windows you checked, and which remain untested.
Do not claim results for an untested platform. Maintainers may defer review until the relevant environment is covered.

See [PR #132 (perf(text): skip suffix checks -- 18.7% less reference time)](https://github.com/oborchers/Fast_Sentence_Embeddings/pull/132)
for an example with scoped measurements, commands, AI disclosure and explicit MIT agreement.
Its reference-tokenizer gain was not presented as a native-encoder gain.

## License agreement

In your PR, state: "I agree that my contribution is licensed under the MIT license as part of fse 2.0."
Do not contribute code or data you cannot license this way. Identify third-party material and its license.
The `1.x` branch retains its own license and contribution instructions.

## Releases

Maintainers release from `master` with a `v*` tag.
CI builds the source archive and wheels, then requires approval in the `release` environment before publishing to PyPI.
Opening a PR does not authorize a release.

# Contributing

fse 2 is an inference engine for finished static embeddings. It trains nothing.
The maintainer plans the next release, 2.1, on `master`. Feature requests and unsolicited feature PRs are out of scope.
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
After changing README Python examples, `make check` runs `tests/test_init.py::test_the_readme_shows_only_names_fse_has`, which checks that the README shows only names that fse has; it does not run the examples. Run them unchanged with real Hub models as described in [Example verification](docs/benchmarks/readme-evidence.md#example-verification). Keep executable Python fences unindented at the top level, so the name check finds them.
Run `make help` for all targets.
After editing Cython or native headers, run `make rebuild-ext`.

Read [ARCHITECTURE.md](ARCHITECTURE.md) and [AGENTS.md](AGENTS.md) before changing a contract or native code.
Keep conformance tests and bit identity unchanged. Bit identity means equal output bits, not just close numbers.
Do not add guardrail ratchet entries or suppressions. Split code that exceeds a limit.
Record user-visible changes as [changelog fragments](changelog.d/README.md).
Only release preparation updates CHANGELOG.md.

## Make the result reproducible

Include the base and head commits, exact commands, data sources, data revisions and random seeds.
Record CPU, instruction set, operating system, Python, NumPy, compiler and relevant package versions.
Link raw results and explain any skipped check. A screenshot or a claimed percentage alone is not enough.
For documentation-only changes, identify the source of each factual claim. Mark performance measurements as not applicable when appropriate.

## Measure performance changes

Follow the measurement rules in [AGENTS.md](AGENTS.md#commands). In particular:

- Choose the claim class first (correctness, descriptive, regression, optimization, publication, see [AGENTS.md](AGENTS.md#commands)) and provide the evidence it requires. Class descriptive measures the change alone with fixed repetitions and reports the absolute median and IQR.
- For classes regression, optimization and publication, alternate base and change in paired rounds. Report the median, range and machine load.
- Run an A/A control, comparing the base with itself, where the class requires one. Class optimization always needs its full controls, including independent A/A validation and confirmation.
- Use the same data, settings and environment on both sides. Keep measurements within the repository's time budget.
- A paired Linux end-to-end float32 comparison of both commits, following the [measurement guide](docs/benchmarks/measurement.md), is required only for published speed numbers or release guarantees (class publication), or when the maintainer has explicitly approved a Linux machine for the issue. Voluntary Linux comparisons are welcome. Report the actual Linux comparison status, and state explicitly when none was run.
- For performance changes, include the kernel comparison against 1.1.0 and any relevant tokenizer measurements when the chosen claim class and protocol require them.
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

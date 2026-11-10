# Package Layout

Historical M1 timing observations in this document are withdrawn and must not support speed claims. Numerical contracts and implementation descriptions remain valid. Current evidence is in the [comparison guide](../fse-comparison.md).


Scope: outside the former "2.0 design" section.

```
src/fse/
  __init__.py            # public API, __all__, version, the hint for removed 1.x names
  typing.py              # public annotation types: Encoding, LengthLevel, ScoreHead, SinkRows, Tokenizer
  _contract.py           # Batch, identifiers, the step list, errors
  _tokenizer_spec.py     # TokenizerSpec: a tokenizer with its parameters (wordpiece@1)
  _tokenizer_unigram.py  # UnigramConfig: the parameters of unigram@1, its scores and charsmap
  _reference.py          # the reference: numpy code that defines every result
  _reference_head.py     # the reference of linear heads: linear@1, exp@1 and the links
  _reference_prefilter.py # the reference of the int8 prefilter of exact search: its bound and its rule of survivors
  _reference_bits.py     # the reference of the sign bits and the Hamming distance
  _codes.py              # float16 and int8 storage arithmetic in blocks
  _ids.py                # strict signed-int64 input, shared by document writers and search filters
  _partition.py          # the partition rule of a source cut into parts (shard=(i, n))
  _container/            # the containers of artifacts (safetensors, npy, foreign files) and the one hash of the artifacts
  _wordvectors.py        # WordVectors
  _wordvectors_format.py # the format rules of word_vectors 1: keys, checks, vocabulary encoding
  _tokenizer_json.py     # a BERT tokenizer.json of Hugging Face as the parameters of wordpiece@1
  _tokenizer_json_common.py # what the readers of a tokenizer.json share: keys, added tokens, post-processor, truncation
  _tokenizer_json_unigram.py # a Unigram tokenizer.json as the parameters of unigram@1
  _cpus.py               # usable_cpus: the default number of workers
  _log.py                # the loggers and events of fse, emit with its error boundary
  _log_config.py         # enable_logging and disable_logging
  _backends/             # the backends behind one contract (cpu, numpy) and the checks every backend shares
  _runtime/              # the runtime of encoding: tasks, executor, pool, sinks, heads and length reports
  _encoder/              # the encoder: its files, the Hub loaders, encode and encode_batches, document IDs, positions and resume
  _static_model/         # fse.StaticModel, the API of model2vec.StaticModel on Encoders
  _store/                # stores of sentence vectors on a shared shard base: write, open, search, rescore, views, refresh, bits and the prefilter
  _scores/               # scores of a group of heads: ScoreSink, ScoreStore, ScoreBatch
  _head/                 # linear heads (linear_head 1) and their runs: Classifier, Regressor, Heads, imports from scikit-learn and Model2Vec
  _validation.py         # validate_arguments: strict pydantic checks of entry points
  _native.pyx            # the extension: directives, imports and its parts, in order
  _native.pxd, .pyi      # shared C types of the extension, and its type stubs
  _native_src/          # native sources, no Python package
    _native_buffer.pxi     #   growable buffers,
    _native_task.pxi       #   pool_text: stages 1 to 4 of a task in one section without the GIL
    _native_threads.h     # thread-local storage and thread identifiers
    kernel/
      _native_kernel.pxi     #   the kernel wrappers, ISA dispatch and FP environment,
      _native_steps.pxi      #   dot16, normalize@1 and score,
      _native_store.pxi      #   storage of finished rows (quantize_rows@1),
      _native_search.pxi     #   inverse norms, cosines and scores of many vectors,
      _native_prefilter.pxi  #   the int8 prefilter: slack of a query and one pass over a block,
      _native_head.pxi       #   the logits, exp@1, the links and the labels of linear heads
      _native_kernel.h       # the C kernel: its contract, ISA detection and dispatch, and
      _native_kernel_base.h  #   build checks, macros and shared types,
      _native_kernel_mean.h  #   the weighted mean loops per ISA and dtype,
      _native_kernel_tiles.h #   the register-blocked weighted mean loop (NEON float32),
      _native_kernel_steps.h #   FP environment, dot16, normalize@1 and score,
      _native_kernel_store.h #   the storage of rows per ISA,
      _native_kernel_search.h #  inverse norms, cosines and scores,
      _native_kernel_int8.h  #   the exact int32 sums of the prefilter: portable, NEON SDOT, AVX2,
      _native_kernel_prefilter.h # the slack of a query and a pass over a block of int8 rows,
      _native_kernel_head.h  #   the bias of folded logits, exp@1, the links and the labels
    text/
      _native_hash.pxi       # its parts: FastText ngram hashing and chunk limits,
      _native_vocab.pxi      #   word hashing and the vocabulary table,
      _native_scan.pxi       #   UTF-8 classes and the scan of sentence spans,
      _native_lists.pxi      #   strings and token lists joined into one UTF-8 buffer,
      _native_text.pxi       #   the tokenizers and the text lookup,
      _native_wordpiece.pxi  #   wordpiece@1: its state, the greedy match, a sentence
      _native_unigram.pxi    #   unigram@1: its state and its checks,
      _native_unigram_normalize.pxi # normalizer passes and Metaspace,
      _native_unigram_steps.pxi # the candidates, the lattice, a sentence
      _native_arrow.pxi      #   Arrow string columns: the C structs, their checks and errors,
      _native_arrow_column.pxi #  the ArrowColumn: chunks taken over, scanned, encoded and released
      _native_text.h         # the native tokenizers fse_words@1 and external
      _native_added.h        # the scan for added tokens, UTF-8 validation and the character cut
      _native_wordpiece.h    # wordpiece@1: normalization and pre-tokenization
      _native_unigram.h      # unigram@1: graphemes, the charsmap search, the float64 Viterbi
      _native_wordpiece_trie.h # the shared vocabulary byte trie of wordpiece@1 and unigram@1
      _native_arrow.h        # the Arrow C Data and C Stream structs, and one row of utf8, large_utf8 or utf8_view
    tables/
      _native_unicode.h      # generated: the simple lowercase table for the native code
      _native_unicode_bert.h # generated: the composed normalization table of wordpiece@1
      _native_unicode_graphemes.h # generated: the same, one byte per code point, for the native code
  _unicode.py            # generated: the same table for the reference
  _unicode_bert.py       # generated: the Unicode data of wordpiece@1 for the reference
  _unicode_graphemes.py  # generated: the grapheme data of unigram@1 (Unicode 17.0.0) for the reference
  _charsmap.py           # the precompiled charsmap of SentencePiece normalizers: its checks and search
  _reference_text.py     # the reference of stages 1 and 2: tokenizers and lookup
  _reference_wordpiece.py # the reference of wordpiece@1
  _reference_unigram.py  # the reference of unigram@1
  _text/                 # text sources to batches (stages 1 and 2), the only importer of the extension besides the cpu backend
  _text_file.py          # FileSource: the lines of a memory-mapped text file
  _hub.py                # a Hub repository at one commit, its files checked (imports huggingface-hub)
  _hub_push.py           # a model folder pushed to the Hub in one commit, checked at that commit
  _stream.py             # StreamSource: sentences from an iterator, read once
  gensim.py              # adapter from gensim word vectors (extra fse[gensim])
  _version.py            # generated by setuptools-scm: __version__
```

The tree names the packages and the top-level modules. The modules inside a package, and the layer
each one belongs to, are listed in full by the import contracts under `[tool.importlinter]` in
`pyproject.toml`, which the lint step checks; this page does not repeat them.


# Build

Scope: outside the former "2.0 design" section.

The extension is compiled from `_native.pyx`, the `.pxi` parts it includes (one translation unit, so `cdef inline` functions inline across parts) and the headers in `_native_src/` at build time with Cython 3 against numpy 2 headers; generated C is never committed. Wheels built this way also run with numpy 1.x. `make rebuild-ext` rebuilds it in the development environment after edits to `.pyx`, `.pxi`, `.pxd` or `.h` files.

## Test scheduling and release checks

Automatic pull requests do not allocate macOS runners, including wheel jobs.
Their Linux, Windows, minimum-dependency and packaging selection is unchanged.
Documentation-only PRs run lint, including the offline local-reference check, and still skip tests. Optional macOS evidence uses the explicit
`pr-macos` manual scope described below, never a comment bot or `pull_request_target`.

Pushes to master retain the full macOS test matrix (CPython 3.11 to 3.15),
including the macOS sentence-transformers comparison and arm64 wheels when selected.
Release tags retain one macOS test on Python 3.13 plus arm64 wheels, and Linux-only
sentence-transformers jobs. Python 3.13 is the newest interpreter where gensim runs.
Scheduled runs and manual `scope=full` runs retain their existing full selection.

Create a release tag only on a commit whose full master CI has already passed.
Before downloading or publishing artifacts, the release job queries the GitHub
Actions API for push runs of `ci.yml` on master with the exact release commit SHA.
The newest matching run must be completed and successful. A missing, running,
failed or cancelled run blocks publication with an explicit error.
The gate does not accept PR, manual, scheduled or tag runs as substitutes.

The required branch check is `check`. It requires every selected job among
`changes`, `lint`, `test`, `test-st`, `coverage`, `sdist` and `wheels` to succeed.
Only jobs intentionally omitted by the declared selection may be skipped. The
summary also inspects individual matrix jobs through the Actions API. Failure,
cancellation, `abandoned`, unknown results and unfinished work fail closed, including
optional jobs that actually started. Failed-only reruns retain earlier successful jobs,
while the newest attempt of each exact job name supplies its current outcome.
A runner outage is missing evidence, never success.
All existing tag prerequisites and the release gate remain required.

### Optional macOS evidence for a PR

The existing `ci.yml` manual entry accepts `scope=full` (the default) or `scope=pr-macos`.
The latter requires `pr_number` and the complete lowercase 40-character `head_sha`.
It runs only a macOS Python 3.13 test, a macOS Python 3.13 sentence-transformers
comparison and macOS arm64 wheels with all supported wheel interpreters tested.
Linux jobs resolve the source and summarize the results. It starts no new full matrix,
Linux/Windows builds, minimum job, coverage merge, release or publication.

The controller checks that the PR is open against this repository's master and that
its current head exactly equals the supplied SHA. Each executing macOS job checks
out that immutable commit from the PR's source repository, verifies `git rev-parse HEAD`
and builds/tests that checkout. Credentials are not persisted, job tokens remain read-only,
and the targeted Python jobs do not save caches into the controller branch.
No repository or branch-protection setting changes are involved.

To dispatch the targeted run, from the PR's checkout:

```sh
ci_pr="$(gh pr view --json number --jq .number)"
ci_sha="$(gh pr view "$ci_pr" --json headRefOid --jq .headRefOid)"
gh workflow run ci.yml --ref master \
  -f scope=pr-macos -f pr_number="$ci_pr" -f head_sha="$ci_sha"
```

Do not omit `scope=pr-macos`: the default is the full manual matrix.

The run's controller SHA/ref may differ from the tested PR SHA. Executing job names,
source summaries and wheel artifact names identify the complete tested SHA. Targeted
wheel artifacts use `dist-wheels-macos-latest-manylinux-pr-NUMBER-SHA`. The final
summary rechecks the current PR head. A changed or closed PR makes it fail, even if
all tests of the older SHA passed. Do not transfer a successful result to a newer head.
These manual results remain separate optional evidence, not an automatic status check
attached to a different commit. Record the run URL and tested SHA when reviewing them.

The ordinary full manual command remains `gh workflow run ci.yml --ref BRANCH -f scope=full`.
It is a different operation from targeted PR verification.

## Wheel verification

CI uses cibuildwheel 4.3.5, the action pinned by commit in `.github/workflows/ci.yml`
(the local tool of the dev group and the Linux images below are pinned to 4.3.4). It selects
CPython 3.11 to 3.15, without free-threaded variants. Tags and full `workflow_dispatch` runs
build and test all 30 wheels (manylinux 2.28 and musllinux 1.2 on x86_64 and aarch64, macOS
arm64, and Windows x86_64), with one difference on Windows (DR-344): the full suite runs on
the CPython 3.15 wheel only, on every event, and the other four Windows wheels run a short
check (`tools/wheel_smoke.py`, selected in `pyproject.toml` under
`[[tool.cibuildwheel.overrides]]`): it imports `fse` and its extension and encodes a few
texts for float32, float16 and int8 tables, with the kernel and with the numpy reference, and
requires the same bits. Unix keeps the only full suite over all versions; Windows adds the
real Windows filesystem and process semantics once. The newest Python is where the
Windows-specific risks (native extension, files, processes) show first; the optional gensim
tests are pure Python, skipped on 3.14 and newer, and run on Linux for 3.11 to 3.13.
Every Windows wheel is built and uploaded, and the release job publishes all of them. On musl Linux, atomic artifact publication calls the Linux `renameat2` syscall
with `RENAME_NOREPLACE` when libc has no wrapper. It never falls back to a rename
that could overwrite an existing directory. Unsupported syscall architectures fail
explicitly. Linux images are pinned by digest to
[cibuildwheel's 2026.10.21-1 images](https://github.com/pypa/cibuildwheel/blob/v4.3.4/cibuildwheel/resources/pinned_docker_images.cfg).
A manual run on a branch verifies the release matrix without publishing. Publication
requires a tag push and approval in the release environment. Automatic PRs keep the three
Linux/Windows runner jobs, without musllinux, and test only CPython 3.15 on Linux
(Windows: the full suite on 3.15 and the short check on the other four wheels, on every
event). macOS wheel builds explicitly select arm64.
The path filter still omits wheels when an unrelated file changes. Store/container source changes and their tests trigger the wheel jobs so the store suite also runs on real Windows.

[cibuildwheel's selectors](https://cibuildwheel.pypa.io/en/stable/options/#build-skip)
include musllinux for both architectures. [Version 4.2.0](https://cibuildwheel.pypa.io/en/stable/changelog/#v420)
enabled CPython 3.15 by default at its ABI-stable release candidate. Version 4.3.1
uses the final release. No prerelease option is needed with our pin.
For beta interpreters, `enable = ["cpython-prerelease"]` opts in, but their ABI may
change. [The option's documentation](https://cibuildwheel.pypa.io/en/stable/options/#enable)
says to upload such wheels to PyPI only for package prereleases. We recommend keeping
that option off and publishing wheels built against stable CPython releases.

musllinux runs the offline suite with Arrow producers, tokenizers and Model2Vec.
gensim and scikit-learn lack musllinux wheels. The musllinux test command excludes
`test_gensim.py` and the native random-Unicode comparison that imports gensim.
The native hash tests against pinned gensim values still run. Scikit-learn adapter
tests skip through `pytest.importorskip`.
PyArrow, pandas, Polars and datasets remain installed on every target interpreter that runs
the full suite (not on the four Windows wheels that run the short check).
PyYAML, required by huggingface-hub and datasets, builds from source on CPython 3.15
musllinux because its current release has no wheel for that interpreter.
On every platform, gensim tests skip on Python 3.14 and newer. The wheel logs show
the test summary for each interpreter, and artifacts retain the built filenames.

The minimum-dependency job uses numpy 1.26.4. Its Arrow producer tests skip because
pyarrow 26 requires numpy 2. This is a coverage gap in that job, not in the latest
or wheel jobs, which exercise the Arrow producers with numpy 2.

# Tests

Scope: outside the former "2.0 design" section.

- Tests mirror `src/`: `tests/test_wordvectors.py` tests `src/fse/_wordvectors.py`, `tests/test_container_npy.py` tests `src/fse/_container/_npy.py`. A long test module splits by topic into `test_<module>_<topic>.py`, such as `tests/test_conformance_steps.py`.
- Test modules never import each other. What several share (the conformance backends, word vectors and edits of saved files, container tensors) lives in `tests/support/`, which pytest puts on the path (`pythonpath`).
- Tests write files only into `tmp_path`; `tests/test_data` is read-only input.
- Expected values never depend on vocabulary positions, because gensim versions order words with equal counts differently. Derive them from `key_to_index`.
- Tests that download from the Hugging Face Hub are marked `network`.
- pytest treats warnings as errors. Narrow, commented ignores are allowed only for third-party warnings.
- gensim does not build on Python 3.14 and newer yet; the tests of `fse.gensim` are skipped there, everything else runs.

## Layout

Binding for every 2.0 module (formerly under "2.0 design").

New 2.0 code goes into private modules of `fse`: `_contract` and `_reference` (contracts and reference), `_codes` (storage arithmetic in blocks), `_container`, `_wordvectors` and `_wordvectors_format`, the gensim adapter `fse.gensim` (extra `fse[gensim]`), then the runtime and backends, the encoder, sinks and store. gensim is not a dependency of fse: `import fse` and every 2.0 class work without it (tests check it in a subprocess that blocks gensim and scipy).

The modules form layers. A module imports only from layers below it. The import contracts
in [pyproject.toml](../../pyproject.toml), under `[tool.importlinter]`, define the complete
order and the independent modules within each layer. They also define the layers inside
packages. `make lint` enforces these contracts. Place every new module in a layer there.
Only the generated `_version` is exempt. This document keeps no second layer list.

The paths checked for file length and complexity are defined by
`[tool.fse.guardrails].paths` in the same configuration. Add new top-level code folders there.

The native extension `_native` is invisible to import-linter. `tools/check_guardrails.py` checks that only `_text` (stages 1 and 2) and the cpu backend `_backends._cpu` (the kernels) import it and that it imports only `_contract` from fse. It also checks (in `tools/guardrails_imports.py`) that only `_container._hash` imports a hash library (hashlib or xxhash), that only `_log` and `_log_config` import `logging` or `warnings`. A further contract forbids gensim, scipy, smart_open, psutil, threadpoolctl, wordfreq, torch and sklearn in all of fse, except gensim in `fse.gensim`; the checker applies the same list to the extension.

## Baseline

Binding for every 2.0 module (formerly under "2.0 design").

`benchmarks/baseline/` records the values of fse 1.1.0 from PyPI that 2.0 is compared with: STS, an MTEB selection, training throughput, memory and a lifecycle benchmark. Its README holds the tables and the commands to reproduce them.

The [comparison guide](../fse-comparison.md) records current publication eligibility, compact results and immutable raw evidence. Historical M1 timing is withdrawn.

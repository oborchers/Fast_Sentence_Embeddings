# Package Layout

Scope: outside the former "2.0 design" section.

```
src/fse/
  __init__.py            # public API, __all__, version, the hint for removed 1.x names
  _contract.py           # Batch, identifiers, the step list, errors
  _tokenizer_spec.py     # TokenizerSpec: a tokenizer with its parameters (wordpiece@1)
  _tokenizer_unigram.py  # UnigramConfig: the parameters of unigram@1, its scores and charsmap
  _reference.py          # the reference: numpy code that defines every result
  _reference_head.py     # the reference of linear heads: linear@1, exp@1 and the links
  _codes.py              # float16 and int8 storage arithmetic in blocks
  _container/            # the containers of artifacts, a package of its own:
    __init__.py          #   their names and CONTAINERS,
    _common.py           #   dtype codes, limits, integrity keys, fingerprint, strict JSON,
    _hash.py             #   the one hash of the artifacts (xxh3-128), its values and classes,
    _safetensors.py      #   SafetensorsContainer (models),
    _npy.py              #   NpyContainer (output),
    _npy_inplace.py      #   NpyContainer artifacts written in place: rows at their offsets, exclusive publish,
    _foreign.py          #   safetensors files of other writers (Model2Vec), read strictly
  _wordvectors.py        # WordVectors
  _wordvectors_format.py # the format rules of word_vectors 1: keys, checks, vocabulary encoding
  _tokenizer_json.py     # a BERT tokenizer.json of Hugging Face as the parameters of wordpiece@1
  _tokenizer_json_common.py # what the readers of a tokenizer.json share: keys, added tokens, post-processor, truncation
  _tokenizer_json_unigram.py # a Unigram tokenizer.json as the parameters of unigram@1
  _cpus.py               # usable_cpus: the default number of workers
  _log.py                # the loggers and events of fse, emit with its error boundary
  _log_config.py         # enable_logging and disable_logging
  _backends/             # the backends behind one contract:
    __init__.py          #   BACKEND_NAMES, get_backend,
    _base.py             #   Capabilities, PoolInput, Prepared, the Backend protocol and checks,
    _cpu.py              #   the cpu backend on the native kernel,
    _numpy.py            #   the numpy backend on fse._reference (the oracle),
    _mlx/                #   the mlx backend (fse[mlx]): the weighted mean on Apple GPUs:
      _load.py           #     the only importer of MLX, on first use, with the reason it cannot run,
      _sources.py        #     the Metal source of the pooling kernel and its subnormal flags,
      _kernels.py        #     the device rows, the arrays of a task, the launch, the canary,
      _device.py         #     the device thread that evaluates waiting calls together,
      _backend.py        #     MlxBackend on its injected cpu host
  _runtime/              # the runtime of encoding:
    _executor.py         #   run: inline or threads, ordered commits, a window, errors in order,
    _tasks.py            #   the tasks of a source and the inline rule,
    _sinks.py            #   Sink, MemorySink, ArrayRows, GrowingRows,
    _spec.py             #   EncodeSpec: what an encoder hands to the runtime,
    _pipeline.py         #   Pipeline: an encoder on one backend, infer,
    _head.py             #   HeadSpec, the fold, HeadPipeline: a linear head in the workers
  _encoder/              # the encoder:
    _encoder.py          #   Encoder: construction, encode, save and load, from_pretrained,
    _pretrained.py       #   what a local path or Hub repository holds,
    _persist.py          #   an encoder as an artifact and back,
    _folder.py           #   Hugging Face folders: config.json, the atomic write, save and push,
    _card.py             #   the README.md model card of a folder,
    _model2vec.py        #   Model2Vec models as the parts of an encoder,
    _model2vec_rules.py  #   a model read under model2vec's own rules (fse.StaticModel),
    _sentence_transformers.py # sentence-transformers StaticEmbedding models, the same,
    _files.py            #   what both readers share: files, configs, the one table,
    _batches.py          #   encode_batches and encoding into a sink,
    _format.py           #   the format rules of encoder 1,
    _format_tokenizer.py #   its keys of a tokenizer with parameters and the table it needs,
    _sources.py          #   what encode takes as texts, and its source,
    _inputs.py           #   the rules of texts in memory
  _static_model/         # fse.StaticModel, the API of model2vec.StaticModel on Encoders:
    _model.py            #   the class, its properties and encode,
    _load.py             #   where from_pretrained finds a model and under which rules,
    _variants.py         #   the encoders of one model per (max_length, normalize), cached,
    _ids.py              #   the token ids of texts per length rule, from the text sources,
    _tokenizer.py        #   the read-only tokenizer facade,
    _profile.py          #   the fse encoders that fit model2vec's rules,
    _arguments.py        #   the arguments that change no row, and the refusals
  _store/                # stores of sentence vectors (sentence_vectors 1):
    _layout.py           #   store.json, shard names and metadata, the shards in order,
    _shard.py            #   one shard being written and published,
    _sink.py             #   ShardSink,
    _search.py           #   search and score by blocks, the top-k merge,
    _vectorstore.py      #   VectorStore
  _head/                 # linear heads (linear_head 1):
    _content.py          #   what a head is: A, b, labels, link, flag, and their checks,
    _format.py           #   the format rules of linear_head 1 and the adapter to encoder 1,
    _sklearn.py          #   the heads of fitted scikit-learn estimators, read by attribute,
    _outputs.py          #   PredictedBatch and the identity of outputs,
    _run.py              #   texts or a store to outputs, in arrays, batches or sinks,
    _model.py            #   what both classes share: construction, properties, save, load,
    _classifier.py       #   fse.Classifier,
    _regressor.py        #   fse.Regressor
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
      _native_head.pxi       #   the logits, exp@1, the links and the labels of linear heads
      _native_kernel.h       # the C kernel: its contract, ISA detection and dispatch, and
      _native_kernel_base.h  #   build checks, macros and shared types,
      _native_kernel_mean.h  #   the weighted mean loops per ISA and dtype,
      _native_kernel_tiles.h #   the register-blocked weighted mean loop (NEON float32),
      _native_kernel_steps.h #   FP environment, dot16, normalize@1 and score,
      _native_kernel_store.h #   the storage of rows per ISA,
      _native_kernel_search.h #  inverse norms, cosines and scores,
      _native_kernel_head.h  #   the bias of folded logits, exp@1, the links and the labels
    text/
      _native_hash.pxi       # its parts: FastText ngram hashing and chunk limits,
      _native_vocab.pxi      #   word hashing and the vocabulary table,
      _native_scan.pxi       #   UTF-8 classes and the scan of sentence spans,
      _native_lists.pxi      #   strings and token lists joined into one UTF-8 buffer,
      _native_text.pxi       #   the tokenizers and the text lookup,
      _native_wordpiece.pxi  #   wordpiece@1: its state, the greedy match, a sentence
      _native_unigram.pxi    #   unigram@1: its state and its checks,
      _native_unigram_steps.pxi # the normalizer steps, the candidates, the lattice, a sentence
      _native_arrow.pxi      #   Arrow string columns: the C structs, their checks and errors,
      _native_arrow_column.pxi #  the ArrowColumn: chunks taken over, scanned, encoded and released
      _native_text.h         # the native tokenizers fse_words@1 and external
      _native_added.h        # the scan for added tokens, UTF-8 validation and the character cut
      _native_wordpiece.h    # wordpiece@1: normalization and pre-tokenization
      _native_unigram.h      # unigram@1: graphemes, the charsmap search, the float64 Viterbi
      _native_wordpiece_trie.h # the byte trie of a vocabulary for the greedy match of wordpiece@1
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
  _text/                 # text sources to batches (stages 1 and 2), the only importer of the extension besides the cpu backend:
    _scan.py             #   Chunk, Scan and the scan of one UTF-8 buffer (_Buffered),
    _strings.py          #   StringSource and TokenListSource, joined natively,
    _lookup.py           #   the lookup of stage 2, tokenize and the Vocabulary
  _text_file.py          # FileSource: the lines of a memory-mapped text file
  _hub.py                # a Hub repository at one commit, its files checked (imports huggingface-hub)
  _hub_push.py           # a model folder pushed to the Hub in one commit, checked at that commit
  _stream.py             # StreamSource: sentences from an iterator, read once
  gensim.py              # adapter from gensim word vectors (extra fse[gensim])
```

# Build

Scope: outside the former "2.0 design" section.

The extension is compiled from `_native.pyx`, the `.pxi` parts it includes (one translation unit, so `cdef inline` functions inline across parts) and the headers in `_native_src/` at build time with Cython 3 against numpy 2 headers; generated C is never committed. Wheels built this way also run with numpy 1.x. `make rebuild-ext` rebuilds it in the development environment after edits to `.pyx`, `.pxi`, `.pxd` or `.h` files.

## Wheel verification

CI uses cibuildwheel 4.3.4, pinned by commit in `.github/workflows/ci.yml`. It selects
CPython 3.11 to 3.15, without free-threaded variants. Tags and `workflow_dispatch`
build and test all 35 wheels: manylinux 2.28 and musllinux 1.2 on x86_64 and aarch64,
macOS arm64 and x86_64, and Windows x86_64. On musl Linux, atomic artifact publication calls the Linux `renameat2` syscall
with `RENAME_NOREPLACE` when libc has no wrapper. It never falls back to a rename
that could overwrite an existing directory. Unsupported syscall architectures fail
explicitly. Linux images are pinned by digest to
[cibuildwheel's 2026.10.21-1 images](https://github.com/pypa/cibuildwheel/blob/v4.3.4/cibuildwheel/resources/pinned_docker_images.cfg).
A manual run on a branch verifies the release matrix without publishing. Publication
requires a tag push and approval in the release environment. PRs keep the four
existing runner jobs, without musllinux or macOS Intel, and test only CPython 3.15.
The path filter still omits wheels when an unrelated file changes.

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
PyArrow, pandas, Polars and datasets remain installed on every target interpreter.
PyYAML, required by huggingface-hub and datasets, builds from source on CPython 3.15
musllinux because its current release has no wheel for that interpreter.
MLX requires Apple silicon and Metal and is tested separately by `test-mlx`.
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

The modules form layers, and a module imports only from layers below it. The import contracts in `pyproject.toml` (`[tool.importlinter]`, run by `make lint`) enforce this, and every new module must take a layer there:

1. `_head` (linear heads; they read stores in store mode)
2. `gensim` (the adapter, the only module that imports gensim), `_store` (the shard sink and the vector store, which take an encoder) and `_static_model` (the API of model2vec on encoders), independent of each other
3. `_encoder` (the encoder; `fse` loads its names on first access, because import-linter counts every import of the native extension as an import of `fse` itself, so a static import in `fse/__init__.py` would read as a cycle through every layer)
4. `_hub_push` (a model folder pushed to the Hub, on `_hub`)
5. `_runtime`
6. `_backends`, `_stream`, `_text_file`, `_hub`, independent of each other
7. `_text`
8. `_wordvectors`
9. `_wordvectors_format`, `_tokenizer_json`
10. `_reference_head` (the reference of linear heads, on the reference of the encoder) and `_log_config` (`enable_logging`, on `_validation`)
11. `_container`, `_codes`, `_reference`, `_reference_text`, `_reference_wordpiece`, `_tokenizer_spec`, `_validation`, independent of each other: the references never share code with what they check
12. `_contract`, `_cpus`, `_unicode`, `_unicode_bert`
13. `_log` (the records of fse, which every module may emit)

Inside a package, an import contract keeps its parts apart: `_npy` and `_safetensors` share only `_common`, which sits on `_hash`, each backend shares only `_base`, the runtime's pipeline sits on its tasks, sinks, spec and executor, in the encoder the encoder sits on `_pretrained` and `_persist` (an encoder as an artifact and back), `_pretrained` on `_folder` (Hugging Face folders), that on `_card` (their model card), that on the readers `_model2vec` and `_sentence_transformers`, those on `_files`, that on its batches and format and those on its inputs, in the containers the in-place writer `_npy_inplace` sits on `_npy` and the reader of foreign files `_foreign` on `_safetensors`, in the store the sink and the vector store sit on the shard writer and the search, those on the layout, the runtime's head sits on its pipeline, and in the heads the classes sit on the model, that on its runs, those on the scikit-learn reader, the format and the outputs, and those on the content.

The native extension `_native` is invisible to import-linter. `tools/check_guardrails.py` checks that only `_text` (stages 1 and 2) and the cpu backend `_backends._cpu` (the kernels) import it and that it imports only `_contract` from fse. It also checks (in `tools/guardrails_imports.py`) that only `_container._hash` imports a hash library (hashlib or xxhash), that only `_log` and `_log_config` import `logging` or `warnings` and the modules the MLX device thread runs do not import `_log`, and that the library of each device backend (`device-libraries` in `[tool.fse.guardrails]`: mlx, later CUDA's) is imported, by an import statement or `importlib`, only by that backend's `_backends._<name>._load`; `get_backend` imports that backend's package only when it makes the backend, so that `import fse` and the other backends never load it. A further contract forbids gensim, scipy, smart_open, psutil, threadpoolctl, wordfreq, torch and sklearn in all of fse, except gensim in `fse.gensim`; the checker applies the same list, with the device libraries, to the extension. Inside the mlx backend the backend sits on the device thread, that on the kernels, those on the Metal source and the loader.

## Baseline

Binding for every 2.0 module (formerly under "2.0 design").

`benchmarks/baseline/` records the values of fse 1.1.0 from PyPI that 2.0 is compared with: STS, an MTEB selection, training throughput, memory and a lifecycle benchmark. Its README holds the tables and the commands to reproduce them.

PERFORMANCE.md holds the model of the throughput bounds of a machine (per core, Amdahl, memory), its assumptions and the measured shares; `benchmarks/roofline.py` computes them from a hardware profile in `benchmarks/roofline_hardware.toml`, whose numbers each name their source.

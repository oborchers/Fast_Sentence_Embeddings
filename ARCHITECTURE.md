# Architecture

This document describes how fse is put together and the rules that keep it correct. It is the reference for changes to the contracts, the native kernel and the tests. When a change alters one of these rules, update this file in the same pull request.

master is the development branch of 2.0, a rewrite around a frozen encoder. The 1.x API (`Average`, `SIF`, `uSIF`, the inputs and `SentenceVectors`) was removed from master; it lives on the branch `1.x` and in fse 1.1.0 on PyPI, and accessing one of its names on `fse` raises an `AttributeError` that says so. Items marked 2.1 are additions that must fit without changing anything below.

## Package Layout

```
src/fse/
  __init__.py            # public API, __all__, version, the hint for removed 1.x names
  _contract.py           # Batch, identifiers, step lists, errors
  _reference.py          # the reference: numpy code that defines every result
  _codes.py              # float16 and int8 storage arithmetic in blocks
  _container/            # the containers of artifacts, a package the 1.x export copies:
    __init__.py          #   their names and CONTAINERS,
    _common.py           #   dtype codes, limits, integrity keys, fingerprint, strict JSON,
    _hash.py             #   the one hash of the artifacts (xxh3-128), its values and classes,
    _safetensors.py      #   SafetensorsContainer (models),
    _npy.py              #   NpyContainer (output),
    _npy_inplace.py      #   NpyContainer artifacts written in place: rows at their offsets, exclusive publish
  _wordvectors.py        # WordVectors
  _wordvectors_format.py # the format rules of word_vectors 1: keys, checks, vocabulary encoding
  _gram.py               # Gram matrix helpers for fit
  _cpus.py               # usable_cpus: the default number of workers
  _backends/             # the backends behind one contract:
    __init__.py          #   BACKEND_NAMES, get_backend,
    _base.py             #   Capabilities, PoolInput, Prepared, the Backend protocol and checks,
    _cpu.py              #   the cpu backend on the native kernel,
    _numpy.py            #   the numpy backend on fse._reference (the oracle)
  _runtime/              # the runtime of train and infer:
    _executor.py         #   run: inline or threads, ordered commits, a window, errors in order,
    _tasks.py            #   the tasks of a source and the inline rule,
    _sinks.py            #   Sink, MemorySink, ArrayRows, GrowingRows, GramSink,
    _spec.py             #   EncodeSpec: what an encoder hands to the runtime,
    _pipeline.py         #   Pipeline: an encoder on one backend, infer
  _encoder/              # the encoder and its recipes:
    _recipes.py          #   Average, SIF, USIF: parameters and fit,
    _fit.py              #   the passes of fit, the learned steps,
    _encoder.py          #   Encoder: encode, save and load,
    _batches.py          #   encode_batches and encoding into a sink,
    _fold.py             #   folding linear steps into word vectors,
    _format.py           #   the format rules of encoder 1,
    _weights.py          #   SIF and uSIF word weights,
    _sources.py          #   what encode and fit take as texts, and its source,
    _corpus.py           #   Corpus, and the rules of texts in memory
  _store/                # stores of sentence vectors (sentence_vectors 1):
    _layout.py           #   store.json, shard names and metadata, the shards in order,
    _shard.py            #   one shard being written and published,
    _sink.py             #   ShardSink,
    _search.py           #   search and score by blocks, the top-k merge,
    _vectorstore.py      #   VectorStore
  _validation.py         # validate_arguments: strict pydantic checks of entry points
  _native.pyx            # the extension: directives, imports and its parts, in order
  _native_hash.pxi       # its parts: FastText ngram hashing and chunk limits,
  _native_buffer.pxi     #   growable buffers,
  _native_vocab.pxi      #   word hashing and the vocabulary table,
  _native_scan.pxi       #   UTF-8 classes and the scan of sentence spans,
  _native_lists.pxi      #   token lists joined into one UTF-8 buffer,
  _native_kernel.pxi     #   the kernel wrappers, ISA dispatch and FP environment,
  _native_steps.pxi      #   dot16, steps and score,
  _native_store.pxi      #   storage of finished rows (quantize_rows@1),
  _native_search.pxi     #   inverse norms, cosines and scores of many vectors,
  _native_text.pxi       #   the tokenizers and the text lookup
  _native.pxd, .pyi      # shared C types of the extension, and its type stubs
  _native_kernel.h       # the C kernel: its contract, ISA detection and dispatch, and
  _native_kernel_base.h  #   build checks, macros and shared types,
  _native_kernel_mean.h  #   the weighted mean loops per ISA and dtype,
  _native_kernel_steps.h #   FP environment, dot16, steps and score,
  _native_kernel_store.h #   the storage of rows per ISA,
  _native_kernel_search.h #  inverse norms, cosines and scores
  _native_text.h         # the native tokenizers fse_words@1 and external
  _native_unicode.h      # generated: the simple lowercase table for the native code
  _unicode.py            # generated: the same table for the reference
  _reference_text.py     # the reference of stages 1 and 2: tokenizers and lookup
  _text.py               # text sources (strings, token lists) to batches, the Vocabulary
  _text_file.py          # FileSource: the lines of a memory-mapped text file
  _stream.py             # StreamSource: sentences from an iterator, read once
  gensim.py              # adapter from gensim word vectors (extra fse[gensim])
```

## Native kernel

`_native_kernel.h` and the headers it includes implement `weighted_mean@1` (pool and accumulate), `dot16`, the steps `remove_components@1`, `project@1` and `normalize@1`, and the linear score of the numerical contract below; storage quantization, inverse norms and cosine search follow with their PRs. Each is one C loop, instantiated per ISA: portable (`fmaf`), AVX2 with FMA and F16C on x86 (selected at run time by cpuid and xgetbv, so one wheel serves every x86 machine) and NEON on arm64. `_native.pyx` validates all arrays before it releases the GIL, then calls the loop of the chosen ISA (`available_isas`, `default_isa`; `isa=` selects one for tests). The rules that make it bit-identical to `fse._reference`:

- Every multiply-add is an explicit fused multiply-add in token order; the compiler must not contract or reorder anything else. The extension is built with `-ffp-contract=off -fno-fast-math` (`/fp:precise` on MSVC), see `setup.py`. MinGW is refused, because its runtime computes `fmaf` with two roundings.
- The floating-point environment is fixed for the duration of every call: round to nearest, no flush-to-zero or denormals-are-zero (MXCSR on x86, FPCR on arm64). The previous state is restored afterwards, also after an error. Each Python entry point sets it for the whole call, so the numpy conversions of its arguments (such as `bias`) run in it too, and each C entry point sets it again and calls its arithmetic only through functions marked never to be inlined, so no compiler can move the arithmetic before it.
- Dimensions are limited to `MAX_DIM` (2**30), because the loops index a row with `int`. Batches are read-only views; the kernel checks their offsets and the shapes of step arrays again before it reads through raw pointers, and `out` must not overlap the rows, weights or the batch.
- Reductions over the dimension use `dot16`: 16 lanes, element i in lane i % 16, then a tree 8/4/2/1. Lanes are not padded with zeros, because a lane can hold -0.0.
- Word rows are prefetched 8 tokens ahead, also across sentence boundaries; prefetching changes no result.
- A row that is not finite raises with its row number, as the reference does.

`tests/test_conformance.py` and `tests/test_conformance_steps.py` compare every backend in `BACKENDS` (`tests/support/conformance.py`) with the reference bit for bit for every dtype, step list, empty, long and unknown-word sentences and subnormal values. Its meta-tests check that the suite fails for a kernel without fma, with `acc / n`, with an EPS start or with zero-padded `dot16`, so a weakened comparison cannot pass unnoticed.

## Text sources

`fse._text` turns text into batches (stages 1 and 2). A source holds its sentences as one UTF-8 buffer: strings in memory with their offsets, or the memory-mapped lines of a file. `fse.Corpus` (`fse._encoder._corpus`) is the public face of the sources in memory: a list, tuple or one-dimensional numpy array either of str or of lists or tuples of str, the first row deciding which, a row of the other kind raising with both rows named; its content errors come when it is made, and a corpus of token lists is checked against the tokenizer of every call it is passed to. `fit` and `encode` build a plain list through the same rules, checking the tokenizer before any row; a path and a stream are passed to them directly. Token lists (the external tokenizer) are joined into that buffer, the tokens of a sentence separated by one space, in one native pass (`join_token_lists`): it reads exact lists, tuples and str through borrowed pointers, checks every token (non-empty, no code point of W, no lone surrogate) and writes its UTF-8 at once, in parts of at least 4,096 sentences that up to 8 native threads run while the calling thread holds the GIL and waits. No Python code can run during the pass (no calls, no references dropped, no Python object allocated, and no other Python thread gets the GIL), and every str object, scattered in memory, is read once: that read is what the pass costs. 85.7 million WikiText tokens take 0.65 s on an M1 Pro (2.1 s in one thread, 11 s in Python); a second pass to size the buffer first doubled the time, prefetching the next tokens gained nothing measurable. Other types (subclasses, a numpy array of sentences) and every invalid token go through the Python path, which raises the errors in their order: the types and tokens of all sentences first, then the UTF-8 of every sentence. `scan` validates the UTF-8 of every sentence in threads without the GIL and cuts the source into chunks of about 64 KiB to 1 MiB (at most `MAX_CHUNK_ROWS` sentences); `encode` turns one chunk into a Batch with a `TextLookup` (the tokenizer, the `VocabTable` built from the vocabulary bytes of the word vectors, and the FastText buckets) and a `Workspace`, the buffers one worker reuses. `make_vocabulary` builds the `VocabTable` of word vectors once as a `Vocabulary`; every lookup of the same words in the same order (checked by `WordVectors.vocabulary_fingerprint`, the fingerprint of `vocab.bytes` and `vocab.offsets` alone) can share it, so an encoder builds it once for all its backends and passes. `fse._text_file.FileSource` reads the lines of a file. Every sentence is one row, empty ones included. Tokens are matched to the vocabulary by their exact UTF-8 bytes, so with `fse_words@1` a word with uppercase letters or whitespace (such as `New_York` in GoogleNews) is never matched. `tools/make_unicode_tables.py` generates the lowercase tables from `UnicodeData.txt`; both generated files are checked in and compared with each other in the tests.

`fse._stream.StreamSource` takes sentences from any iterable (a queue consumer, a crawler, log lines) and is read once. The runtime reads it in the calling thread, a collection of at most `chunk_rows` sentences (65,536) or about `chunk_bytes` (8 MiB) at a time, and only when its window of tasks has room, so a slow sink slows the source and memory stays bounded by one collection and the window. A sentence that is no str or has no UTF-8 form, and an error of the iterator, are raised after the rows before them were committed, with the row number counted from the start of the stream. The iterator is never read from another thread. One thread reads a stream, so a stream gives at most what that thread can collect: about 2.8 million sentences per second on a 16-vCPU EPYC (`bench_runtime.py stream`), the same as a StringSource up to about 8 workers and 0.8 to 0.9 of it at 16; a native reader may lift this later.

## Rules

### Validate strictly, never fail silently

Invalid input raises a clear exception at the API boundary. Nothing is coerced, ignored or truncated without an error, and an unexpected keyword argument is an error (`pydantic.ValidationError` at validated entry points). A slower but correct fallback is logged as a warning.

Public entry points whose own work dwarfs a few microseconds use `fse._validation.validate_arguments`, a strict pydantic `validate_call`, with constraints such as `PositiveInt` in the signatures. Methods called per element in loops, and parameters that hold bulk data, check explicitly instead, because validation would add measurable overhead or copy the data; such parameters stay `Any` in validated signatures. Integer constraints accept numpy integer scalars, float constraints reject infinity and NaN.

### float32 throughout

Word vectors, weights and sentence vectors are computed in float32. Word vectors may be stored as float16 or int8 and are widened exactly; every sum runs in float32. The kernel takes raw pointers, so arrays must have the expected dtype and be C-contiguous, which `_native.pyx` checks before the GIL is released.

## Build

The extension is compiled from `_native.pyx`, the `.pxi` parts it includes (one translation unit, so `cdef inline` functions inline across parts) and the `_native*.h` headers at build time with Cython 3 against numpy 2 headers; generated C is never committed. Wheels built this way also run with numpy 1.x. `make rebuild-ext` rebuilds it in the development environment after edits to `.pyx`, `.pxi`, `.pxd` or `.h` files.

## Tests

- Tests mirror `src/`: `tests/test_wordvectors.py` tests `src/fse/_wordvectors.py`, `tests/test_container_npy.py` tests `src/fse/_container/_npy.py`. A long test module splits by topic into `test_<module>_<topic>.py`, such as `tests/test_conformance_steps.py`.
- Test modules never import each other. What several share (the conformance backends, word vectors and edits of saved files, container tensors) lives in `tests/support/`, which pytest puts on the path (`pythonpath`).
- Tests write files only into `tmp_path`; `tests/test_data` is read-only input.
- Expected values never depend on vocabulary positions, because gensim versions order words with equal counts differently. Derive them from `key_to_index`.
- Tests that download from the Hugging Face Hub are marked `network`.
- pytest treats warnings as errors. Narrow, commented ignores are allowed only for third-party warnings.
- gensim does not build on Python 3.14 and newer yet; the tests of `fse.gensim` are skipped there, everything else runs.

## 2.0 design

This section is binding for every 2.0 module.

### Principle and stages

A frozen encoder is a pure function from one sentence to one finished row. Every row is computed in five stages, all inside the worker that owns the sentence:

| Stage | What | Where | Owned by |
|---|---|---|---|
| 1 tokenize | split the text into token spans | CPU | the Source |
| 2 token rows and factors | look up vocabulary ids, hash FastText ngrams; the factor of each row comes from the weights of the fit | CPU for ids and ngrams, the backend kernel applies the factor | Source (ids), Encoder (weights) |
| 3 aggregation | combine the weighted rows of a sentence into one row | backend | an `Aggregation` |
| 4 postprocessing | versioned steps on the row, such as removing components | backend | the Encoder's steps |
| 5 store | quantize per row and write | backend `store`, in the worker; then the Sink | the Sink |

The tokenizer is part of the encoder contract: the manifest names it with an identifier like the aggregation (`fse.tokenizer`), and reading an encoder with an unknown tokenizer raises. `external` means the tokens were given by the caller, who then owns reproducing them; `fse_words@1` is the native tokenizer specified under *Tokenizers*. A text Source runs stages 1 and 2 in one native pass without the GIL.

There are no Python hooks per row: they would hold the GIL, could not run on a device, could not be saved and would break bit-identity. Two escape hatches remain: own tokenization in Python inside the Source (stage 1, once per sentence, slower), and own processing of the finished output arrays. Models in which a row depends on more than its own sentence (attention over neighbors, sentence pairs) are out of scope.

Three rules keep new models possible without a new layer:

1. **An aggregation is not a sum.** It is defined by an initial accumulator, a combine step `combine(acc, f, x)` per weighted token row, a finish step `finish(acc, n)`, the result of an empty sentence and its output dimension. There is no group mode (many sentences to one row, as `CIndexedList` with repeated indices in 1.x): a document is encoded as one text, a mean over all its tokens, and a mean of equally weighted sentences is a reduction of their rows (`np.add.reduceat` over the group starts, then normalize), also over the batches of a stream; the migration guide shows both. It only adds, so it can come in a later 2.x if it is asked for.
2. **Weights (stage 2) are separate from the aggregation (stage 3).** Stage 2 supplies a factor and a stored row per contribution; the aggregation never sees weights or scales. A later max over SIF-weighted rows therefore needs no own kernel.
3. **The output dimension belongs to the encoder.** The aggregation gives the dimension of its rows (`Aggregation.output_dim(d)`: d for the weighted mean; a concatenation such as SWEM would give 2d), and a step may reduce it (`project`). The result is written into the manifest, and Sinks and Store read it from there, never from the word vectors.

**Linear steps are folded into the word vectors.** The weighted mean is linear, and so are `remove_components` and `project`: applying them to every word and ngram row before averaging gives the same rows, within float32 rounding, as applying them to every averaged row. `fit` of SIF and uSIF therefore folds its learned steps into the word vectors by default, and encoding only averages, as fast as Average; `normalize` is not linear and stays a step per row. A folded encoder owns its word vectors (they are tensors of the encoder, with their own fingerprint) and records the folded steps and the fingerprint of the original vectors as provenance; an unfolded encoder references the original word vectors and applies its steps per row. Folded word vectors in float16 or int8 are quantized after folding. Before `normalize`, folded and unfolded rows agree within float32 rounding relative to the size of the input rows (and within the quantization error for float16 and int8); `normalize` scales a row that the removal nearly cancels up to unit length, so for such rows the rounding or quantization residues decide the direction, and folded and unfolded rows can point anywhere relative to each other. Average learns nothing and needs no corpus; its encoder references the word vectors and holds no tensors of its own, unless `fit` learns a projection.

### Numerical contract (`transform_version` 2)

`fse._contract` holds the identifiers and versions, `fse._reference` the numpy reference that computes every rule below bit for bit; it is the oracle for every compiled and device backend, and tests compare by bit pattern (`view(np.uint32)`), because `array_equal` treats -0.0 and +0.0 as equal.

- **Arithmetic.** IEEE-754 float32 with round to nearest even and subnormals kept (no flush to zero). `fma(a, b, c)` rounds once. Native code calls `fmaf` or FMA intrinsics explicitly and is built without other contractions (`-ffp-contract=off`). Division and square root are correctly rounded; no reciprocal approximations or fast math.
- **Stage 2.** A known word w contributes the stored row `x` (float32 as is, float16 widened exactly, int8 codes converted exactly) with the factor `f = weight[w]`, or `f = weight[w] * scale[w]` with one rounding for int8. An unknown FastText word with m >= 1 ngram buckets (in the order of the hashing, capped at `max_ngrams`) contributes each ngram row with `g = (1/m) * oov_weight`, each product rounded, times the ngram scale for int8; with m = 0 it counts for the length and adds nothing. Without FastText, unknown words are no tokens. The scale of int8 rows always goes into the factor, never into the row.
- **`weighted_mean@1`.** The accumulator starts at +0.0 per value; every contribution does `acc[i] = fma(f, x[i], acc[i])` in token order (ngrams in bucket order); `n` counts the tokens. The row is `(1/n) * acc` with `1/n` rounded first, not `acc / n`. An empty sentence is the zero row; there is no EPS.
- **`dot16`**, the dot product of every row reduction: 16 partial sums start at +0.0; index i goes into lane `i % 16` by fma in index order; the lanes are then added 8, 4, 2 and 1 lanes apart, and lane 0 is the result. It vectorizes on NEON, AVX2, AVX-512 and Metal. A lane can hold -0.0 (a product that rounds to zero keeps its sign), so implementations must not pad the reduction with zeros: a tail shorter than a vector updates only its own lanes.
- **`remove_components@1`** with k unit components `u_j` (1 <= k <= d) and optional weights `w_j`: `c_j = dot16(row, u_j)`, times `w_j` if given; `p` starts at +0.0 and takes `p[i] = fma(c_j, u_j[i], p[i])` for j ascending; the result is `row - p`. All coefficients use the unmodified row, as 1.x did.
- **`project@1`** with a basis of k orthonormal rows `b_j` (1 <= k <= d): the new row has the k values `dot16(row, b_j)`; it reduces the dimension from d to k. `fit` takes the basis from the rows after `remove_components`: with `M = I - Σ_j w_j u_j u_jᵀ` (w_j = 1 without weights) their Gram matrix is `Mᵀ G M`, computed in float64 from the Gram matrix G that `fit` already has, without another pass over the rows. Components removed with weight 1 are gone from every row, so the eigenproblem is solved on their orthogonal complement (`Qᵀ Mᵀ G M Q`, Q an orthonormal basis of it) and k is at most d minus their number; `fit` raises if one of the k axes has a variance below the rounding bound of the Gram matrix, instead of picking a direction from noise.
- **`normalize@1`**: `s = dot16(row, row)`; an all-zero row stays zero; a nonzero row whose `s` underflows to 0 or overflows raises; otherwise every value is multiplied by `1 / sqrt(s)`.
- **Step lists.** Any subset of `remove_components`, `project` and `normalize`, in this order and each at most once; anything else is rejected when an encoder is built or loaded. In a folded encoder only `normalize` remains. After stage 4 every row must be finite, otherwise the row id is reported.
- **Stage 5, storage.** float32 as is; float16 rounds to nearest even and raises beyond +-65504; int8 is symmetric per row: `s = absmax / 127` in float32, stepped down once with `nextafter` if `127 * s` overflows, `code = clip(rint(x / s), -127, 127)`; a row whose scale is 0 (a zero row, or one whose largest value divided by 127 underflows) has codes 0. Reading gives `code * s` with one rounding. These are the 1.x rules (`fse.models._quantize` in 1.1.0), which `fse._codes` and the reference reproduce bit for bit.
- **Reading stored rows.** The inverse norm of a stored row is `float32(1 / sqrt(s))`, where s is the sum of the squares of the stored values (int8 codes without their scale) in float64 and index order; a zero row has 0. A linear `score(w, bias)` of a row is `dot16(stored, w) * scale + bias` (scale 1 for float32 and float16). A cosine search normalizes the query by `normalize@1` (a zero query, or one whose norm underflows or overflows, raises) and scores `dot16(stored, q) * inverse norm`; the scale cancels and is not used. A score, inverse norm or cosine beyond float32 raises; top-k ties go to the smaller row id.

What is bit-identical across what:

| Across | Guarantee |
|---|---|
| worker counts, job and batch sizes, input classes with the same tokens, shards, restarts | bit-identical: every row is a pure function of its sentence |
| CPU instruction sets and operating systems (portable, NEON, AVX2) | bit-identical, through explicit fma and `dot16` |
| the numpy reference and every backend | bit-identical for every profile the conformance suite checks |
| CPU and a device backend (MLX, later CUDA) | bit-identical for the profiles the conformance suite checks |
| `fit` (weights from counts, Gram matrix, eigendecomposition) | deterministic per machine and BLAS only; the encoder freezes its result, so encoding stays portable |
| 1.x and 2.0 | not identical (EPS, `saxpy`, BLAS in the removal); compared with tolerances, STS and MTEB |

The Gram matrix of `fit` is summed from float32 block products in float64 in block order; blocks are fixed by absolute row number (multiples of 16,384 rows), so the shards of 2.1 start at such multiples.

### Identifiers and versions

Aggregations, steps and tokenizers are written `"<name>@<version>"` (name `[a-z][a-z0-9_]*`, version a decimal integer without leading zeros), for example `weighted_mean@1`, `remove_components@1`, `normalize@1`, `fse_words@1`. `transform_version` versions the shared rules above, the identifier versions the rules of one operation. Any change of arithmetic bumps the matching version; a released identifier is never removed or given another meaning within 2.x. Reading an artifact with an unknown name, version or `transform_version`, or a disallowed step list, raises `UnsupportedArtifactError` (a `ValueError`) that names the supported ones; nothing is skipped. Data exported from 1.x carries `transform_version` 1: it can be read and searched, but not extended with a 2.0 encoder.

### Batch

A `Batch` is the input of an aggregation: `first_row` (the row of its first sentence), `offsets` (int64, n + 1 entries, from 0 to the number of tokens), `tokens` (uint32 vocabulary ids, or `OOV_TOKEN` for an unknown FastText word) and, for FastText, `ngram_offsets` and `ngrams` (uint32 buckets, one range per unknown word, in order). Sentence s is row `first_row + s`. Weights are not part of a batch, so a Source does not depend on the encoder; 2.1 adds token ids from Arrow and `TokenIds` as further Sources of the same batches.

### Tokenizers

The manifest key `fse.tokenizer` holds `external` or the identifier of a native tokenizer; reading anything else raises `UnsupportedArtifactError` naming the supported ones (`fse._contract.check_tokenizer`). A tokenizer is part of the encoder contract like its aggregation: any change of its rules, its tables or its Unicode version is a new identifier. `fse._reference_text` defines the tokens of every tokenizer, and the native code gives the same bytes.

- `external`: the caller gives the tokens. Text input is split at the whitespace W below, as `str.split()` does, and the tokens are not changed.
- `fse_words@1`: the native tokenizer specified here.

**Input.** One sentence is a `str` that UTF-8 can encode (a lone surrogate raises), or one line of a UTF-8 file: a line ends at the byte `\n`, which is not part of it; the last line needs no `\n`, a final `\n` starts no empty line, and an empty file has no lines. A file must not start with a UTF-8 byte order mark, which would otherwise stick to the first token as U+FEFF; it raises. Bytes are decoded strictly, as CPython's UTF-8 decoder does, and invalid UTF-8 raises. Sentences are not split further and not normalized: NFC and NFD spellings stay different.

**Output.** A list of non-empty tokens, matched to the vocabulary by the exact equality of their UTF-8 bytes. `fse_words@1` computes it in five steps.

1. **Lowercase.** Every code point is replaced by its simple lowercase mapping (field 13 of `UnicodeData.txt` of Unicode 15.1.0, from the checked-in table `fse._unicode`); code points without a mapping stay. There is no SpecialCasing and no context: U+0130 becomes `i`, and Σ always becomes σ.
2. **Classes** of the lowercased code points:
   - whitespace W: U+0009 to U+000D, U+001C to U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000;
   - separators S: ``! " # $ % & ( ) * ; < > ? @ [ ] ` { }`` and U+00AB, U+00BB, U+2012 to U+2015, U+2018, U+2019, U+201C, U+201D, U+201E;
   - token characters: every other code point, subject to step 3;
   - word characters: `a` to `z`, `0` to `9`, `_`, and every code point from U+0080 that is in neither W nor S;
   - digits: `0` to `9`.
3. **Context.** Applied from left to right; t(i) says whether s[i] is a token character.
   - A `.` directly before or after another `.` is not a token character; the same holds for `-`.
   - A `,` or `:` is a token character exactly if the next code point is a digit.
   - A `'` directly before or after another `'` is not a token character. Otherwise it is not one if i > 0, t(i-1) holds, s[i-1] is not a word character, and s[i+1] exists and is a word character.
4. **Spans.** Every maximal run of token characters is a candidate u.
5. **Per candidate u.** Strip `.` and `:` from both ends. Split u into parts by the first rule that applies:
   - the table: `cannot` to can, not; `gimme` to gim, me; `gonna` to gon, na; `gotta` to got, ta; `lemme` to lem, me; `wanna` to wan, na; `more'n` to more, n; `d'ye` to d, ye; `'tis` to t, is; `'twas` to t, was;
   - u is `n't`: the part `not`;
   - u ends with `n't` after a non-empty rest that does not end with `'`: the rest and `not`;
   - u ends with the first matching one of `'s`, `'m`, `'d`, `'ll`, `'re`, `'ve` after a non-empty rest that does not end with `'`: the rest and the clitic without `'`;
   - otherwise u itself.

   From every part, strip `.`, `:` and `'` from both ends; drop it if it holds no `a` to `z` and no `0` to `9`; otherwise split it at every `-` and emit the non-empty pieces in order.

Apart from W, S, the lowercase table and the word characters from U+0080, every rule refers to ASCII only, so an implementation may apply them to UTF-8 bytes.

**Stage 2**, for every tokenizer: a token whose bytes are a vocabulary word becomes its id. Otherwise, with FastText it becomes `OOV_TOKEN` with the first `max_ngrams` `fasttext@1` buckets of `<token>` (possibly none, which still counts for the length); without FastText it is no token.

| Input | `fse_words@1` |
|---|---|
| `I don't know.` | i, do, not, know |
| `Can't, won't` | ca, not, wo, not |
| `'Tis` | t, is |
| `O'Reilly's` | o'reilly, s |
| `James' book` | james, book |
| `rock'n'roll` / `rock’n’roll` | rock'n'roll / rock, n, roll |
| `x='abc'` | x=, abc |
| `U.S. e.g.` | u.s, e.g |
| `Wait... what?` | wait, what |
| `1,000 and 10:30` / `a,b c:d` | 1,000, and, 10:30 / a, b, c, d |
| `http://example.com/a-b?x=1` | http, //example.com/a, b, x=1 |
| `well-known --- a--b -x` | well, known, a, b, x |
| `café 中国 Paris` | café, paris |
| `İstanbul` / `ΣΟΦΟΣ 2Σ` | istanbul / 2σ |
| `snake_case foo/bar a+b` | snake_case, foo/bar, a+b |
| `...`, an empty or blank sentence | none |

The rules reproduce the preprocessing of the published 1.x tables on all STS sentences; on WikiText 99.98 % of the paragraphs agree, the rest differ only by the simple lowercase of İ and final sigma.

### Backends

A backend implements one contract (`fse._backends`) and is chosen by name from a fixed set (`"cpu"`, `"numpy"`, later `"mlx"` and `"cuda"`); there is no plugin registry and no `"auto"`:

- `prepare(pool, steps)`: once per encoder and backend, what it needs on that backend (the kernel's parameters, uploaded rows); `pool` holds the word rows, scales, weights and FastText rows, `steps` the postprocessing steps.
- `rows(batch, prepared, out)`: stages 2 to 4 of one batch into `out`, the aggregation and then the postprocessing steps on the rows of that batch, so every worker applies the steps to the finished rows of its own task. The cpu backend runs them one after the other: without a `project` it pools into `out` and applies the steps there in place (`postprocess_pooled`: neither the copy nor the finiteness check of `postprocess`, as pooling raises for every row that is not finite, and rows that share memory with an array of a step are refused); with a `project`, even one that keeps the dimension, the new array it gives is copied once. An error leaves part of `out` written: a task writes only its own target, and the runtime commits it only after the task succeeded, so a sink that keeps its targets to itself never publishes such rows. A device backend may fuse them.
- `gram_block(rows)`: the float32 product of one block for `fit`; `GramSink` sums the blocks in float64 in block order, as `fse._gram.gram_matrix` does.
- `score(stored, scales, weights, bias)`: the scores of stored rows.
- `inverse_norms(stored)`, `cosines(stored, inverse, queries)` and `scores(stored, scales, weights, bias)`: the search of the vector store on stored float32, float16 or int8 rows without a float32 copy: the inverse norm of every row, and the cosine with many queries or the linear score with many weight vectors as a (rows, vectors) array, column by column the rule of one vector. Every query is checked before any is used; errors name the query or weight vector and the row, in that order. The cpu backend computes them in the kernel (`_native_kernel_search.h`) in its own floating-point environment; the inverse norms of 8 rows at a time, each with its own float64 sum in index order, so the bits are those of one row at a time while the chains of dependent additions overlap (2.4 times as fast for float32, 3.2 times for int8 on an M1 Pro).
- `store(rows, codes, scales, first_row)`: stage 5 of finished float32 rows into float16 or int8 codes (and one scale per row for int8) by `quantize_rows@1`, in the worker that made them, so a compact storage dtype costs no work in the calling thread. It runs on the host; a device backend may later fuse it with `rows` (PR 9). The cpu backend stores each row in one pass in the kernel (`_native.store_rows`, in its own floating-point environment); within one call the first row that is not finite raises, else for float16 the first row beyond +-65504.
- `capabilities()`: the word dtypes, aggregations, steps, FastText, the largest dimension, and whether several threads may call `rows` at once (`concurrent`; otherwise the runtime runs one task at a time while encoding stays parallel).

Arguments and outputs are host numpy arrays, checked the same way by every backend (shape, dtype, C order, writability, no overlap with the inputs). The kernels and the cpu backend set the floating-point environment of their thread themselves; the numpy backend is the oracle of the tests, not a production backend, and like `fse._reference` computes in the environment of its thread, as do the Gram block products and their float64 sum, which numpy computes for every backend. Before reading any input, the runtime checks that the backend can run the encoder and raises otherwise, naming the backend and what it cannot run; it never falls back silently. The conformance suites run every backend through its contract and compare it with the reference bit for bit for every dtype, step list, FastText, source and worker count, including empty, long and unknown-word sentences and subnormal values; a new backend is done when it passes. accumulate and dot16 are kernel primitives, not contract operations, and are tested on the kernels directly.

### Runtime

`fse._runtime` runs train and infer on one executor (`run`):

- **Tasks.** A source is scanned into chunks, one task each: about `chunk_bytes` of text (64 KiB to 1 MiB by default, scan ranges never smaller than a chunk) and at most 16 MiB of finished rows (`TASK_ROW_BYTES`; a task has a fixed cost of about 0.2 ms for its checks, so smaller tasks cost throughput). A stream is read in collections of up to 65,536 sentences, each scanned into tasks with rows counted from the start of the stream.
- **Inline.** A source of at most 64 KiB, and `workers=1`, run in the calling thread without any thread; with the default workers such inputs are within 8 % of `workers=1` (`benchmarks/bench_runtime.py latency`).
- **Threads.** Otherwise a pool of `workers` threads, created per call, with one workspace per thread; the native routines release the GIL and set the floating-point environment of their thread themselves, so a worker that inherited flush-to-zero gives the same bits. The tasks are taken, and each task's output rows reserved (`Sink.target`), in the calling thread; results are committed in task order in the calling thread, so a sink needs no locks.
- **Window.** At most `2 * workers` tasks are in flight or waiting to be committed, and their rows take at most 512 MiB (`WINDOW_BYTES`; with many workers a task holds fewer rows; every row counts its target, the float32 rows before a float16 or int8 target is stored, and with steps the pooled rows, their copy and the output rows): a slow sink slows the source, and memory stays bounded by the window and one collection of a stream.
- **Storage.** A sink with a `dtype` of float16 or int8 reserves codes (and int8 scales); the worker of a task pools into a float32 buffer of its thread and stores the rows with the backend's `store`, so the calling thread only commits. On an M1 Pro float16 and int8 output run at 0.91 to 1.03 of float32 for 1 to 10 workers (`bench_runtime.py store`); in 1.1, int8 output ran at about 0.33. The window counts the stored target of a row besides its float32 rows; it bounds the rows in flight, not a result that joins all rows.
- **Sinks.** A sink (`fse.Sink`) takes the rows of a run: `target` and `commit` in the calling thread, in row order, each row once; a target is valid only until its commit. Optional parts: `dtype`, `inverse_norms` (the target ends with a float32 vector the worker fills with the inverse norm of every stored row, so a store computes none in the calling thread; the window counts it), `encoder_fingerprint` (the fingerprint of the only encoder it takes), `flush()` after the last commit of a successful run, `publish()`, which `encode_batches` calls after the rows of every batch are committed and before it reads `durable_rows`, and `durable_rows`, the leading rows of the run that are durable. A sink takes one run; a second run into it starts again at row 0, which it refuses. `Pipeline.infer` numbers the rows of a sink from `first_row`, so the batches of a stream continue the rows of a run. `Encoder.encode(texts, sink=)` commits to a sink and returns nothing; `encode_batches(batches)` encodes a stream batch by batch, each a finite list (as a queue consumer polls them), and gives one `EncodedBatch` per batch once its rows are finished: its rows without a sink, else the sink's `durable_rows`, so a consumer confirms its input only up to there (at least once: after a crash it reads again from the last confirmed offset). That count is a snapshot after the sink's `publish`: the sink's `flush` runs after the last batch, so after a successful full iteration the sink's own `durable_rows` says what is durable (for a ShardSink every row), and the last batch may report fewer. Stopping early does not flush; the owner of the sink closes it. `MemorySink` keeps the rows of a run in memory and has no `durable_rows`.
- **Errors.** For a given sequence of tasks, the error raised is the one of the first failing task in order, after every task before it was committed, as with one worker; tasks after it are skipped where they have not started. Once a task has failed, nothing more is read from the source. Every argument of `infer` is checked before the source is read. A bad sentence or an error of a stream's iterator is raised after the rows before it. An error of the sink (reserving or committing rows) or an interrupt cancels what has not started, waits for running tasks and is raised. No thread outlives a call.

A `Pipeline` holds an encoder on one backend (prepared once, before any input is read) and is immutable, so several threads may call `infer` at once; a server that encodes many small requests in parallel sets `workers=1` per request. An `Encoder` builds its `EncodeSpec` once, the `VocabTable` of its vocabulary once (`Vocabulary`, shared by all its pipelines and passes), and one `Pipeline` per backend on first use.

`fit` plans its passes before it reads the corpus: a count of the tokens (`count_tokens`) for the sentence length of uSIF unless it is given, then one pass of the weighted rows into a `GramSink` for the components and the projection; `encode` after `fit` reads the corpus again. A corpus read once (a stream) that would need two passes raises before it is read. A simple alternative that kept the float32 rows in a memmap and applied the steps afterwards in one thread was slower with several workers (0.25 against 0.14 s for 200,000 sentences at 10 workers on an M1 Pro, `bench_encoder.py fit`); a sink that writes the memmap during the Gram pass with the steps in the workers was not measured. The Gram pass takes 1.2 to 1.4 times a plain pass of the rows, because the block products run in the calling thread as the rows are committed; tasks aligned to the blocks of the Gram matrix would let the workers compute them.

### Artifacts

Word vectors, encoders and stored sentence vectors are sets of named tensors with a flat string map of metadata, written through a container:

```python
class TensorContainer(Protocol):
    def write(self, path, tensors: Mapping[str, ndarray], metadata: Mapping[str, str]) -> None: ...
    def open(self, path, *, mmap: bool) -> tuple[Mapping[str, ndarray], Mapping[str, str]]: ...
    def verify(self, path) -> None: ...
```

- **Tensors** follow the rules of safetensors: numeric dtypes only (F32, F16, I8, U8, U32, I64), little-endian, C order, every tensor aligned to its type and the data of a file starting at a multiple of 64 bytes, names matching `[a-z0-9_]+(\.[a-z0-9_]+)*` except `__metadata__`, which safetensors reserves; dimensions are int64 and the product of the non-zero ones times the item size stays within int64, also for tensors with a zero dimension; writers check the same rule. Writers first take a snapshot of the mapping of tensors, of every array's shape and dtype (as a view) and of the metadata, so a caller that changes them during a write does not change the header or the fingerprint. The data are not copied as a whole: every block is copied once when it is written and that copy is hashed, so the checksums always describe the written bytes, and `expected` checksums detect data that changed since they were checked. Strings are UTF-8 bytes (U8) plus int64 offsets, so the vocabulary is `vocab.bytes` and `vocab.offsets`. Every floating-point number that affects the result, such as `oov_weight`, is a tensor, never a decimal string. Integer parameters (dimensions, `min_n`, `max_n`, `bucket`, `max_ngrams`) are metadata, written as decimal strings without sign or leading zeros, or as canonical JSON for a group of them.
- **Names.** Word vectors: `vectors`, `scales` (int8), `counts` (optional), `vocab.bytes`, `vocab.offsets`, and for FastText `ngrams` and `ngram_scales`. Encoder: `weights`, `oov_weight`, `step.<i>.components`, `step.<i>.weights`, `step.<i>.basis`, and for a folded encoder its word vectors under the names above (without `counts`) and the folded steps as `folded.<i>.components`, `folded.<i>.weights`, `folded.<i>.basis`. Stored sentence vectors and shards: `vectors`, `scales`, `inv_norms`; `row_ids` is reserved for 2.1.
- **Metadata** (`fse.` keys, mapped one to one to the `__metadata__` of safetensors): `fse.kind`, `fse.format_version`, `fse.created_by` and a `fse.checksum.<tensor>` (xxh3-128, see Identities) per tensor for all; `fse.dim`, `fse.dtype`, `fse.fasttext` (min_n, max_n, bucket) and the quantization rule for word vectors; `fse.transform_version`, `fse.tokenizer`, `fse.aggregation`, `fse.steps`, `fse.input_dim`, `fse.output_dim`, `fse.max_ngrams`, the fingerprints of the word vectors and the vocabulary, and the recipe and its parameters (provenance only) for an encoder; the encoder fingerprint, `fse.transform_version`, the storage dtype, `fse.output_dim` and the range of rows for stored vectors.
- **Containers by purpose.** Every kind of artifact has exactly one container, enforced when it is read:

  | Artifact | Container | Path |
  |---|---|---|
  | word vectors, encoders (models, the hub) | safetensors 1 | one `.safetensors` file |
  | stored sentence vectors and their shards (output) | NpyContainer 1 | a directory of `.npy` files and `manifest.json` |

  Models are shared and read: one standard file per model and dtype, which the hub shows and other tools read. Output is a large matrix written in parallel and read by numpy-based tools: every shard tensor is a plain `.npy` that `np.load(mmap_mode="r")` opens directly. Both containers record the same checksums and give the same fingerprint for the same content; a golden test writes the same tensors through both and checks content and fingerprint.

### safetensors 1

`fse._container.SafetensorsContainer`, in the package of both containers, whose modules import only the standard library, numpy, xxhash (in `_hash` alone), `UnsupportedArtifactError` and each other (relatively), so the 1.x export can copy it; `_npy` and `_safetensors` share only `_common`, and all of them hash through `_hash`.

- The checksums describe themselves, and they alone tell the artifacts of a pre-release of 2.0 apart, which had the same layout and versions but sha256 checksums (64 hex digits) and `sha256:` fingerprints; no version number does. Both containers reject an artifact whose checksums and fingerprint are all sha256 values with an `UnsupportedArtifactError` that names sha256 and xxh3-128, values of another algorithm (`<name>:<hex>`) with an `UnsupportedArtifactError` too, and a malformed value or a mix of algorithms with a `CorruptArtifactError`, before the content is used. There is no migration: such artifacts are written again.

- The file is safetensors: an 8-byte little-endian header length, a UTF-8 JSON header `{"__metadata__": {...}, name: {"dtype", "shape", "data_offsets"}, ...}`, then the data. The tensors lie by descending item size, then name, without gaps, so each is aligned to its type; the header is padded with spaces so that the data start at a multiple of 64 bytes. The reference implementation reads these files.
- Reading is strict: a header of at most 16 MiB and within the file, a JSON object padded with spaces only and data starting at a multiple of 8 bytes, no duplicate keys or NaN constants, exactly `dtype`, `shape` and `data_offsets` per tensor, only the dtypes above (others raise `UnsupportedArtifactError`), offsets that fit the shape (computed with Python integers), tensors aligned to their type and covering the data without gaps or overlaps up to the end of the file, and `__metadata__` mapping strings to strings with the checksums and the fingerprint of the content. Files from other writers lack these keys and are rejected.
- A normal open reads the data into one immutable `bytes` and checks every checksum; `open(mmap=True)` maps the data once, read-only, and the tensors are views of that map; `verify` hashes the tensors on disk. A normal open and `verify` raise if the size or modification time of the file changed while they read it. A mapped file must not be changed or truncated while it is mapped: a change is caught by the checksums the next time the data are checked, but a truncated file can end the process with SIGBUS on POSIX, which no reader can catch; copy files that other processes may change. The file is opened as described for the NpyContainer: a regular file, not followed if it is a link, checked against its path.
- Writing goes to a temporary sibling created exclusively (`tempfile.mkstemp`, `.<name>.writing-...`): the header is reserved (its length does not depend on the checksums), every block of data is copied once into immutable bytes, and that copy is hashed and written (and compared with `expected` checksums if given), so the file holds exactly what its checksums describe even if the source changes meanwhile; then the header is written and the file synced. It is published without ever replacing a file: `os.link` on POSIX (then the directory is synced), `MoveFileExW` with `MOVEFILE_WRITE_THROUGH` and without `MOVEFILE_REPLACE_EXISTING` on Windows. A failed write removes only its own temporary file. A path without `.safetensors` raises.

### NpyContainer 1

`fse._container.NpyContainer`, for stored sentence vectors.

- An artifact is a directory with `manifest.json` and one `<tensor name>.npy` per tensor. The manifest holds exactly `{"container": "npy", "container_version": 1, "tensors": {name: {"file", "dtype", "shape"}}, "metadata": {...}}`: `file` is the name plus `.npy`, `dtype` a code of the table above, `shape` a list of non-negative integers, metadata values strings. Duplicate keys, NaN constants, unknown or missing keys raise `CorruptArtifactError` (a `ValueError`); another container or version raises `UnsupportedArtifactError`; a pre-release artifact with sha256 checksums is rejected by its checksums (see safetensors 1). Unlisted files in the directory are ignored.
- The manifest and every `.npy` must be regular files, not symbolic links; each is opened once (without following links and without blocking), checked against the path it was found at, and read or mapped through that one handle; numpy parses one bounded copy of each header. The manifest may hold at most 16 MiB, a `.npy` header at most 64 KiB (checked before numpy reads it), a tensor at most 32 dimensions; sizes are computed with Python integers, so a shape cannot overflow.
- Every `.npy` is version 1.0 or 2.0 with the exact type description of its code (`<f4`, `<f2`, `|i1`, `|u1`, `<u4`, `<i8`), C order, the shape of the manifest, data at a multiple of 64 bytes and nothing after the data; pickled objects are never read.
- The writer enforces the limits of the reader, and with `expected` checksums it hashes every tensor while writing and commits only if they match. The container writes `fse.checksum.<name>` (the xxh3-128 checksum of the data, see Identities) for every tensor and `fse.fingerprint`, and rejects them from the caller. A normal open reads every tensor into immutable `bytes` and checks its checksum (hashed in 64 MiB chunks); `open(mmap=True)` checks the manifest, the headers, the sizes and the fingerprint against the recorded checksums, and maps the data read-only; `verify` hashes the data on disk. Tensors with no elements are empty arrays, as nothing can be mapped.
- Writing creates the target directory exclusively (an existing one raises `FileExistsError`, also when another writer created it first; artifacts are never overwritten, which also avoids replacing mapped files on Windows), writes and syncs every tensor, syncs the directory, and commits by renaming the manifest from a temporary name into place; a directory without a manifest is no artifact. Any failure removes the directory.

### word_vectors 1

`fse.WordVectors` (`fse._wordvectors`), one `.safetensors` file, created by `from_arrays`, `fse.gensim.from_gensim` (extra `fse[gensim]`; gensim is imported only there) or `load`. Immutable: the arrays it returns are new read-only headers on its data, and a normal load reads every tensor into immutable `bytes`; making an internal array writable again through `.base` is outside the contract. Read as float32, never pickled. `verify` checks the data the object holds against the checksums it was loaded with, then the files on disk. `save` of loaded vectors compares every tensor with those checksums while writing it, and `quantize` reads every block once and hashes and quantizes that same copy, so damaged or changed data never gets new checksums; for mapped vectors these checks are never skipped, as the file can change underneath the map. Quantized vectors own all their arrays.

- Tensors: `vectors` (n x dim, F32, F16 or I8), `scales` (F32, n) exactly for int8, `counts` (I64, n, optional), `vocab.bytes` (U8), `vocab.offsets` (I64, n + 1, from 0 to the number of bytes), and for FastText `ngrams` (bucket x dim, the dtype of the vectors) with `ngram_scales` exactly for int8. 1 <= n < 2**32 - 2 (word ids stay below the reserved token ids), dim >= 1.
- Metadata (closed: any other key raises): `fse.kind=word_vectors`, `fse.format_version=1`, `fse.created_by`, `fse.dim`, `fse.dtype` (`float32`, `float16`, `int8`), `fse.quantization=quantize_rows@1` exactly for float16 and int8 (the storage rules of the numerical contract), optional `fse.quantized_from` (the fingerprint of the float32 vectors, only for float16 and int8), and for FastText `fse.fasttext` (canonical JSON of `bucket`, `max_n` and `min_n`, 1 <= min_n <= max_n, 1 <= bucket <= 2**32 - 1) with `fse.ngram_hash=fasttext@1`.
- The vocabulary is UTF-8; a word is found by the exact match of its bytes (NFC and NFD spellings are different words, a string without a UTF-8 form is no word). Every open checks the structure, the offsets and the UTF-8 of the vocabulary in chunks; `verify` and every writer also check that words are unique, floats finite, int8 codes within -127 to 127, scales finite and not negative, counts not negative, and every int8 row times its scale finite.
- For FastText, the row of a known word is the word's own row (gensim's `vectors_vocab`), not gensim's average of it with its ngrams; combining ngrams for unknown words belongs to the encoder.
- `quantize` makes float16 or int8 vectors from float32 ones in blocks, without a float32 copy; quantized vectors are never quantized again. Loading int8 vectors allocates about their codes; a test checks it with `tracemalloc`.
- `tests/test_data/v2/` holds golden float32 and int8 vectors (`.safetensors`) with pinned fingerprints, written by `make_golden.py` (and golden encoders by `make_encoder_golden.py`, see encoder 1); the pinned files (`word_vectors_float32.safetensors`, `word_vectors_int8.safetensors`) are never rewritten, and the 1.2.0 export must give the same fingerprints. `tests/test_data/v2/sha256/` holds the same vectors with the sha256 checksums of a pre-release (the same tensors and keys; only the hashes differ), kept byte for byte only as files that must be rejected, which their checksums do.

### encoder 1

An encoder (`fse._encoder`), one `.safetensors` file. The keys and tensors are a closed set derived from its content (`fse._encoder._format`); anything else raises.

- Metadata: `fse.kind=encoder`, `fse.format_version=1`, `fse.created_by`; `fse.transform_version=2`; `fse.tokenizer`; `fse.aggregation=weighted_mean@1`; `fse.steps`, the canonical JSON list of the identifiers of the steps applied per row; `fse.input_dim`, the dimension of the word vectors the encoder pools, and `fse.output_dim`, which the steps must give; `fse.max_ngrams` exactly for FastText vectors; `fse.word_vectors` and `fse.vocabulary`, the fingerprints of the word vectors the encoder pools and of their vocabulary (`WordVectors.vocabulary_fingerprint`); `fse.recipe`, canonical JSON of the recipe and its parameters, provenance only, never read to compute anything.
- Tensors: `weights` (F32, one per word; absent means all ones, as for Average), `oov_weight` (F32, a scalar, exactly with `fse.max_ngrams`), and per step `i` of `fse.steps` its arrays: `step.<i>.components` and optionally `step.<i>.weights` for `remove_components`, `step.<i>.basis` for `project`, none for `normalize`.
- An **unfolded** encoder holds no word vectors; loading it takes the word vectors whose fingerprint is `fse.word_vectors`, any other vectors raise.
- A **folded** encoder holds its word vectors: their tensors under the names of word_vectors 1 without `counts`, and their keys `fse.dim` (equal to `fse.input_dim`), `fse.dtype`, `fse.quantization`, `fse.quantized_from`, `fse.fasttext` and `fse.ngram_hash`; `fse.word_vectors` is their fingerprint as word_vectors 1. `fse.folded` lists the linear steps folded into them (a non-empty canonical JSON list of `remove_components` and `project`) and `folded.<i>.<field>` holds their arrays, as provenance; the folded steps take rows of the original dimension (the dimension of their arrays) and give `fse.input_dim`; `fse.folded_from` is the fingerprint of the original word vectors. A folded encoder keeps at most `normalize` as a step per row.
- Golden files: `tests/test_data/v2/encoder_sif_unfolded.safetensors` and `encoder_usif_folded_int8.safetensors` with pinned fingerprints and the pinned bits of the rows they give; these pinned files are never rewritten. `tests/test_data/v2/sha256/` holds the same encoders with the sha256 checksums of a pre-release, fitted once, kept byte for byte only as files that must be rejected. As fit depends on BLAS, `make_encoder_golden.py` made the pinned files from them by transcoding, never by fitting again: every checksum and fingerprint of a sha256 file is checked, every tensor is taken byte for byte, and every fingerprint they name was renamed only after its sha256 value was recomputed from the same content and matched (`fse.quantized_from` of the folded encoder by folding the int8 vectors again, which gave the stored rows). The pinned rows did not change.

### Identities

- A **row id** is the int64 position of a sentence in its source, from 0.
- A **shard** is a contiguous range of rows; 2.1 splits sources with `shard=(i, n)` at multiples of 16,384 rows.
- A **checksum** is `xxh3-128:` and the 32 lowercase hex digits of XXH3-128 with seed 0 (in the canonical big-endian form, as `xxhash.xxh3_128().hexdigest()` gives it) of bytes fed in order; a tensor's checksum is that of its little-endian C-order bytes, and a tensor without elements has the checksum of no bytes, `xxh3-128:99aa06d3014798d86001c324468d497f`. `fse._container._hash` is the only module that computes one and the only one that imports a hash library (`hash-importers` of the guardrails); it streams (`update` releases the GIL for large inputs), takes bytes, C-contiguous memory and C-contiguous little-endian arrays only, and the containers feed it in chunks. Values are compared as whole strings, the algorithm included.
- A **fingerprint** is the checksum of the UTF-8 bytes of the canonical JSON (keys sorted, separators `,` and `:`, no whitespace, non-ASCII characters unescaped) of `{"fingerprint": 1, "metadata": {...}, "tensors": [[name, dtype, shape, checksum], ...]}`: the metadata without `fse.created_by`, `fse.fingerprint` and the `fse.checksum.` keys; the tensors sorted by name, with the dtype code (such as `"F32"`), the shape as a list of integers and the checksum of the tensor. It describes the content, not the container, so every container gives the same fingerprint. A pre-release of 2.0 hashed the same document with sha256; its values are not read. The test vector (`tests/test_container_common.py`): tensors `vectors` F32 `[[1, -0.0], [0.5, 2]]`, `vocab.bytes` U8 of `"aé"`, `vocab.offsets` I64 `[0, 1, 3]` with metadata `fse.kind=word_vectors`, `fse.format_version=1`, `fse.dim=2`, `fse.dtype=float32` give `xxh3-128:8011e82efda8fabff21969cbaad4fae3`, and with `fse.note=é` added `xxh3-128:9e1f9e8737c4b9c190d5600af20953e0`.
- **What the hashes guarantee.** Checksums and fingerprints identify content against accidents: damaged, truncated or changed files and data that changed since they were checked (with 128 bits, a chance collision needs about 2**64 artifacts). They are no authentication: xxh3 promises neither collision resistance nor second-preimage resistance, so someone who can write a file can make different content with the same checksums and fingerprint, and no security-relevant binding may rest on them, such as an encoder's `fse.word_vectors`, a Vocabulary's fingerprint or the `expected` checksums of `save`, `quantize` and `fold`. Anyone who can write a file could always recompute its values anyway, as no key is involved; the files are free of pickle and read strictly, so a forged file gives wrong data, not code execution. Authenticity needs a trusted reference: a cryptographic hash of the whole file from a trusted source, or a pinned revision; a hash computed from the file itself authenticates nothing. The Hub loader (a later PR) takes this from the Hub: registry artifacts must be LFS files, whose sha256 (`lfs.sha256`) a commit pins; a missing `lfs.sha256` is rejected, and a cached file is checked against it at its first trusted use.
- An unfolded encoder references its word vectors and vocabulary by fingerprint, a folded one holds them and names the original fingerprint as provenance; stored vectors reference their encoder; opening them with a mismatch raises. 2.1 adds the fingerprint of the source for resuming.

### Persistence names replace class paths

Saved 2.0 artifacts contain no Python class paths. Their persistence API is the set of strings above: `fse.kind` values, identifiers, dtype codes, tensor and metadata names. These are never renamed or reused with another meaning. The public Python API is `fse.__all__`; private modules (with a leading underscore) may move freely.

### Migration

2.0 has no pickle loader. A last 1.x release, 1.2.0 on the branch `1.x`, adds `export_v2`, which writes the word vectors of a model and its encoder (weights, `oov_weight`, and the learned components as a `remove_components` step) into the 2.0 format, and optionally its sentence vectors as data with `transform_version` 1. uSIF models saved before 1.1 keep their weighting by lambda squared, because the export takes the weights the model applies. The hub word vectors are converted the same way. The hub also gets pretrained, frozen encoders: Average for every vector set, SIF and uSIF for a selection, fitted once on WikiText-103 with the native tokenizer, so `from_pretrained` encodes at once and gives the values of the published tables. A golden manifest and fingerprint of a tiny model, written by 1.2.0 and loaded by 2.0 in the tests, catch drift between the two writers.

### sentence_vectors 1

A store of sentence vectors (`fse._store`), a directory written by `fse.ShardSink` and read by `fse.VectorStore`.

- **`store.json`**, written once when the store is created and never changed: strict JSON, exactly `{"store": "fse_shards", "store_version": 1, "metadata": {...}}` with `fse.kind=sentence_vectors`, `fse.format_version=1`, `fse.transform_version=2`, `fse.encoder` (the fingerprint of the encoder of every row), `fse.dtype` (float32, float16 or int8; one per store) and `fse.output_dim`. It is created in a temporary directory beside the store, synced, and renamed into place without replacing anything, then the parent directory is synced; a store another writer created first is kept and checked. Another store or version raises `UnsupportedArtifactError`.
- **Shards**, each an NpyContainer 1 artifact `shard-<first row>` (the row as at least 16 decimal digits): `vectors` (rows of the dtype), `inv_norms` (F32, from the stored values, see the numerical contract) and for int8 `scales` (F32); `row_ids` is reserved for 2.1. Their metadata repeat those of `store.json` and add `fse.created_by`, `fse.quantization=quantize_rows@1` for float16 and int8, `fse.first_row` and `fse.rows`. Shards are immutable; an append adds shards after the last row, nothing is rewritten.
- **Reading** follows the shards from row 0 by looking up the name of the next one (`shard-<rows so far>`), so a shard published while a reader lists the store is either seen whole or not at all, and the rows always run without a gap; every shard must match `store.json` and hold exactly its tensors (the vectors in the dtype of the store, the inverse norms and scales as F32) and exactly its metadata keys; a float16 or int8 shard names `fse.quantization=quantize_rows@1` (another rule of that family raises `UnsupportedArtifactError`). `store.json` is read whole (at most 64 KiB) with exactly its keys and the integer version 1. Entries starting with a dot (shards being written, or left by a crash) and other names are ignored; remove leftovers only while no writer runs. `VectorStore.open` maps the shards published at that time (a snapshot).
- **Writing.** The ShardSink writes a shard in `.shard-<first row>.writing-<hex>`: `vectors.npy` gets the header of the full shard (by default the largest multiple of 16,384 rows within 256 MiB of vectors, at most 2**20 rows). The worker of a task writes its rows into a buffer of the task, which the window of the runtime bounds; the commit writes them into the file at their offset (`pwrite`, so a full disk raises an OSError in the calling thread) and feeds the checksum of the vectors in row order; a task that crosses the end of a shard is split over the shards at its commit. Writing through a mapping of the file from the workers instead cost about half the throughput (page faults on fresh file pages and a mapping per task: 1.04 against 1.96 million int8 rows per second at 10 workers on an M1 Pro, `bench_store.py`), and needed a zero-filled preallocation so a full disk could not end the process with SIGBUS. A shard whose rows are all committed is finished at once: a shard that ends early is shrunk to the header of its rows at the same data offset, the file is synced, the inverse norms, scales and manifest are written and synced, and the directory is renamed into place without ever replacing one, not even an empty one (`renameat2` with `RENAME_NOREPLACE` on Linux, `renamex_np` with `RENAME_EXCL` on macOS, `MoveFileExW` without replacing and with `MOVEFILE_WRITE_THROUGH` on Windows), then the store directory is synced (Windows has no directory sync, so its rename writes through). Only then `durable_rows` grows. On macOS `fsync` reaches only the drive cache, and `F_FULLFSYNC` does an `fsync` and then asks the drive to flush all its buffered data (fcntl(2)), which costs milliseconds. So a shard makes two: the files, the manifest and the first sync of the temporary directory are plain `fsync`s, the sync of that directory after the manifest is in place is the full one that makes all of them durable before the rename, and the sync of the store after the rename is the second (`write_manifest(..., staged=True)`; before, a shard made six or seven). Elsewhere every sync is the `fsync` it was. `seal` ends the current shard early, so a stream can make its rows durable before a shard is full; `publish` publishes what is finished; `flush` does both and needs every reserved row committed.
- **Errors and concurrency.** Leaving a ShardSink (`close` or its context) publishes the committed rows and drops the rest, also after an error (maintainer decision). Two writers that append at the same time race for the name of the next shard: exactly one rename succeeds, the other raises a FileExistsError naming the rows it already published, and removes its temporary data. The crash model is the local filesystem: a process that dies leaves the published shards as a valid prefix and its temporary directories, which readers ignore; durability against power loss rests on the syncs above; network filesystems are not covered.
- **Search.** `VectorStore.search(queries, k)` runs block by block in worker threads on the stored rows, through the inverse norms, without a normalized float32 copy, for blocks of rows and of queries that hold at most about 4 MiB of scores. `mode="exact"` uses the backend's `cosines` (the contract) and gives the ids and bits of the reference. `mode="blas"` decodes each block to float32 (at most 16 MiB per task) and multiplies with BLAS: close to the contract, not bit-defined, so ties and nearly equal scores may order differently; it is faster for many queries, since BLAS uses the matrix units of the CPU (Accelerate on Apple silicon) and the contract fixes the order of every sum. For a few queries the kernel is as fast or faster, since every row is read once and need not be decoded. `mode="auto"`, the default, takes the contract for at most 8 queries and BLAS above (equal at 8 to 12 on 859,346 rows), so its bits depend on how many queries are asked at once; reproducible results need `mode="exact"` (maintainer decisions). Each block keeps the candidates of every query at or above its floor, the k-th best score merged so far, by the order of `top_k` (the larger score first, equal scores, -0.0 equal to 0.0, by the smaller row), which is total, so merging the blocks in any order gives the same ids. Every query is checked before any block is searched. `score(weights, bias)` gives the linear score with one or many weight vectors, with the same modes. `verify` hashes every shard and checks its values (finite scales and inverse norms of at least 0, finite float rows, int8 codes within +-127). A store cannot be closed while it is searched.

### Layout

New 2.0 code goes into private modules of `fse`: `_contract` and `_reference` (contracts and reference), `_codes` (storage arithmetic in blocks), `_container`, `_wordvectors` and `_wordvectors_format`, the gensim adapter `fse.gensim` (extra `fse[gensim]`), then the runtime and backends, the encoder and recipes, sinks and store, and the thin `train`/`infer` facade of the release. gensim is not a dependency of fse: `import fse` and every 2.0 class work without it (tests check it in a subprocess that blocks gensim and scipy).

The modules form layers, and a module imports only from layers below it. The import contracts in `pyproject.toml` (`[tool.importlinter]`, run by `make lint`) enforce this, and every new module must take a layer there:

1. `gensim` (the adapter, the only module that imports gensim) and `_store` (the shard sink and the vector store, which take an encoder), independent of each other
2. `_encoder` (the encoder and its recipes; `fse` loads their names on first access, because import-linter counts every import of the native extension as an import of `fse` itself, so a static import in `fse/__init__.py` would read as a cycle through every layer)
3. `_runtime`
4. `_backends`, `_stream`, `_text_file`, independent of each other
5. `_text`
6. `_wordvectors`
7. `_wordvectors_format`
8. `_container`, `_codes`, `_gram`, `_reference`, `_reference_text`, `_validation`, independent of each other: the references never share code with what they check
9. `_contract`, `_cpus`, `_unicode`

Inside a package, an import contract keeps its parts apart: `_npy` and `_safetensors` share only `_common`, which sits on `_hash`, each backend shares only `_base`, the runtime's pipeline (and its token count) sits on its tasks, sinks, spec and executor, in the encoder the recipes sit on fit, fit on the encoder, the encoder on its batches, format, folding and weights and those on its inputs, in the containers the in-place writer `_npy_inplace` sits on `_npy`, and in the store the sink and the vector store sit on the shard writer and the search, those on the layout.

The native extension `_native` is invisible to import-linter. `tools/check_guardrails.py` checks that only `_text` (stages 1 and 2) and the cpu backend `_backends._cpu` (the kernels) import it and that it imports only `_contract` from fse. It also checks that only `_container._hash` imports a hash library (hashlib or xxhash), and that the library of each device backend (`device-libraries` in `[tool.fse.guardrails]`: mlx, later CUDA's) is imported, by an import statement or `importlib`, only by that backend's `_backends._<name>._load`, inside a function, so that `import fse` never loads it. A further contract forbids gensim, scipy, smart_open, psutil, threadpoolctl, wordfreq, torch and sklearn in all of fse, except gensim in `fse.gensim`; the checker applies the same list, with the device libraries, to the extension.

### Defaults

Defaults are part of the API and chosen for speed: `workers` is the number of usable CPUs (`fse._cpus.usable_cpus`: an explicit `PYTHON_CPU_COUNT` or `-X cpu_count`, else the CPUs of the process affinity limited by the CPU quota of its Linux cgroup and every visible ancestor, rounded up (the quota is read once per process); `PYTHON_CPU_COUNT` thus also sets the default workers; small inputs run inline in the calling thread), and new encoders, also the pretrained ones on the hub, end with `normalize`, so the cosine of two rows is their dot product. A server that encodes many small requests in parallel sets `workers=1`.

### Baseline

`benchmarks/baseline/` records the values of fse 1.1.0 from PyPI that 2.0 is compared with: STS, an MTEB selection, training throughput, memory and a lifecycle benchmark. Its README holds the tables and the commands to reproduce them.

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
    _safetensors.py      #   SafetensorsContainer (models),
    _npy.py              #   NpyContainer (output)
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
    _sinks.py            #   Sink, ArrayRows, GrowingRows,
    _spec.py             #   EncodeSpec: what an encoder hands to the runtime,
    _pipeline.py         #   Pipeline: an encoder on one backend, infer
  _validation.py         # validate_arguments: strict pydantic checks of entry points
  _native.pyx            # the extension: directives, imports and its parts, in order
  _native_hash.pxi       # its parts: FastText ngram hashing and chunk limits,
  _native_buffer.pxi     #   growable buffers,
  _native_vocab.pxi      #   UTF-8 classes, word hashing and the vocabulary table,
  _native_scan.pxi       #   the UTF-8 scan of sentence spans,
  _native_kernel.pxi     #   the kernel wrappers, ISA dispatch and FP environment,
  _native_steps.pxi      #   dot16, steps and score,
  _native_text.pxi       #   the tokenizers and the text lookup
  _native.pxd, .pyi      # shared C types of the extension, and its type stubs
  _native_kernel.h       # the C kernel: its contract, ISA detection and dispatch, and
  _native_kernel_base.h  #   build checks, macros and shared types,
  _native_kernel_mean.h  #   the weighted mean loops per ISA and dtype,
  _native_kernel_steps.h #   FP environment, dot16, steps and score
  _native_text.h         # the native tokenizers fse_words@1 and external
  _native_unicode.h      # generated: the simple lowercase table for the native code
  _unicode.py            # generated: the same table for the reference
  _reference_text.py     # the reference of stages 1 and 2: tokenizers and lookup
  _text.py               # text sources (strings, files, token lists) to batches
  _stream.py             # StreamSource: sentences from an iterator, read once
  gensim.py              # adapter from gensim word vectors (extra fse[gensim])
```

## Native kernel

`_native_kernel.h` and the headers it includes implement `weighted_mean@1` (pool and accumulate), `dot16`, the steps `remove_components@1`, `project@1` and `normalize@1`, and the linear score of the numerical contract below; group mode, storage quantization, inverse norms and cosine search follow with their PRs. Each is one C loop, instantiated per ISA: portable (`fmaf`), AVX2 with FMA and F16C on x86 (selected at run time by cpuid and xgetbv, so one wheel serves every x86 machine) and NEON on arm64. `_native.pyx` validates all arrays before it releases the GIL, then calls the loop of the chosen ISA (`available_isas`, `default_isa`; `isa=` selects one for tests). The rules that make it bit-identical to `fse._reference`:

- Every multiply-add is an explicit fused multiply-add in token order; the compiler must not contract or reorder anything else. The extension is built with `-ffp-contract=off -fno-fast-math` (`/fp:precise` on MSVC), see `setup.py`. MinGW is refused, because its runtime computes `fmaf` with two roundings.
- The floating-point environment is fixed for the duration of every call: round to nearest, no flush-to-zero or denormals-are-zero (MXCSR on x86, FPCR on arm64). The previous state is restored afterwards, also after an error. Each Python entry point sets it for the whole call, so the numpy conversions of its arguments (such as `bias`) run in it too, and each C entry point sets it again and calls its arithmetic only through functions marked never to be inlined, so no compiler can move the arithmetic before it.
- Dimensions are limited to `MAX_DIM` (2**30), because the loops index a row with `int`. Batches are read-only views; the kernel checks their offsets and the shapes of step arrays again before it reads through raw pointers, and `out` must not overlap the rows, weights or the batch.
- Reductions over the dimension use `dot16`: 16 lanes, element i in lane i % 16, then a tree 8/4/2/1. Lanes are not padded with zeros, because a lane can hold -0.0.
- Word rows are prefetched 8 tokens ahead, also across sentence boundaries; prefetching changes no result.
- A row that is not finite raises with its row number, as the reference does.

`tests/test_conformance.py` and `tests/test_conformance_steps.py` compare every backend in `BACKENDS` (`tests/support/conformance.py`) with the reference bit for bit for every dtype, step list, empty, long and unknown-word sentences and subnormal values. Its meta-tests check that the suite fails for a kernel without fma, with `acc / n`, with an EPS start or with zero-padded `dot16`, so a weakened comparison cannot pass unnoticed.

## Text sources

`fse._text` turns text into batches (stages 1 and 2). A source holds its sentences as one UTF-8 buffer: strings in memory with their offsets, or the memory-mapped lines of a file. `scan` validates the UTF-8 of every sentence in threads without the GIL and cuts the source into chunks of about 64 KiB to 1 MiB (at most `MAX_CHUNK_ROWS` sentences); `encode` turns one chunk into a Batch with a `TextLookup` (the tokenizer, the `VocabTable` built from the vocabulary bytes of the word vectors, and the FastText buckets) and a `Workspace`, the buffers one worker reuses. Every sentence is one row, empty ones included. Tokens are matched to the vocabulary by their exact UTF-8 bytes, so with `fse_words@1` a word with uppercase letters or whitespace (such as `New_York` in GoogleNews) is never matched. `tools/make_unicode_tables.py` generates the lowercase tables from `UnicodeData.txt`; both generated files are checked in and compared with each other in the tests.

`fse._stream.StreamSource` takes sentences from any iterable (a queue consumer, a crawler, log lines) and is read once. The runtime reads it in the calling thread, a collection of at most `chunk_rows` sentences (65,536) or about `chunk_bytes` (8 MiB) at a time, and only when its window of tasks has room, so a slow sink slows the source and memory stays bounded by one collection and the window. A sentence that is no str or has no UTF-8 form, and an error of the iterator, are raised after the rows before them were committed, with the row number counted from the start of the stream. The iterator is never read from another thread.

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
| 5 store | quantize per row and write | Sink | the Sink |

The tokenizer is part of the encoder contract: the manifest names it with an identifier like the aggregation (`fse.tokenizer`), and reading an encoder with an unknown tokenizer raises. `external` means the tokens were given by the caller, who then owns reproducing them; `fse_words@1` is the native tokenizer specified under *Tokenizers*. A text Source runs stages 1 and 2 in one native pass without the GIL.

There are no Python hooks per row: they would hold the GIL, could not run on a device, could not be saved and would break bit-identity. Two escape hatches remain: own tokenization in Python inside the Source (stage 1, once per sentence, slower), and own processing of the finished output arrays. Models in which a row depends on more than its own sentence (attention over neighbors, sentence pairs) are out of scope.

Three rules keep new models possible without a new layer:

1. **An aggregation is not a sum.** It is defined by an initial accumulator, a combine step `combine(acc, f, x)` per weighted token row, a finish step `finish(acc, n)`, the result of an empty sentence and its output dimension. Group mode (many sentences to one row, CPU only) combines the sentences of a group by the rule of the aggregation, defined below for the weighted mean: a mean sums the sentence means, as 1.x did; a later max would take the maximum.
2. **Weights (stage 2) are separate from the aggregation (stage 3).** Stage 2 supplies a factor and a stored row per contribution; the aggregation never sees weights or scales. A later max over SIF-weighted rows therefore needs no own kernel.
3. **The output dimension belongs to the encoder.** The aggregation gives the dimension of its rows (`Aggregation.output_dim(d)`: d for the weighted mean; a concatenation such as SWEM would give 2d), and a step may reduce it (`project`). The result is written into the manifest, and Sinks and Store read it from there, never from the word vectors.

**Linear steps are folded into the word vectors.** The weighted mean is linear, and so are `remove_components` and `project`: applying them to every word and ngram row before averaging gives the same rows, within float32 rounding, as applying them to every averaged row. `fit` of SIF and uSIF therefore folds its learned steps into the word vectors by default, and encoding only averages, as fast as Average; `normalize` is not linear and stays a step per row. A folded encoder owns its word vectors (they are tensors of the encoder, with their own fingerprint) and records the folded steps and the fingerprint of the original vectors as provenance; an unfolded encoder references the original word vectors and applies its steps per row. Folded word vectors in float16 or int8 are quantized after folding. Average learns nothing and needs no corpus; its encoder references the word vectors and holds no tensors of its own, unless `fit` learns a projection.

### Numerical contract (`transform_version` 2)

`fse._contract` holds the identifiers and versions, `fse._reference` the numpy reference that computes every rule below bit for bit; it is the oracle for every compiled and device backend, and tests compare by bit pattern (`view(np.uint32)`), because `array_equal` treats -0.0 and +0.0 as equal.

- **Arithmetic.** IEEE-754 float32 with round to nearest even and subnormals kept (no flush to zero). `fma(a, b, c)` rounds once. Native code calls `fmaf` or FMA intrinsics explicitly and is built without other contractions (`-ffp-contract=off`). Division and square root are correctly rounded; no reciprocal approximations or fast math.
- **Stage 2.** A known word w contributes the stored row `x` (float32 as is, float16 widened exactly, int8 codes converted exactly) with the factor `f = weight[w]`, or `f = weight[w] * scale[w]` with one rounding for int8. An unknown FastText word with m >= 1 ngram buckets (in the order of the hashing, capped at `max_ngrams`) contributes each ngram row with `g = (1/m) * oov_weight`, each product rounded, times the ngram scale for int8; with m = 0 it counts for the length and adds nothing. Without FastText, unknown words are no tokens. The scale of int8 rows always goes into the factor, never into the row.
- **`weighted_mean@1`.** The accumulator starts at +0.0 per value; every contribution does `acc[i] = fma(f, x[i], acc[i])` in token order (ngrams in bucket order); `n` counts the tokens. The row is `(1/n) * acc` with `1/n` rounded first, not `acc / n`. An empty sentence is the zero row; there is no EPS. In group mode, the first non-empty sentence of a group gives `g = (1/n) * acc` as above, and every further non-empty sentence s adds `g[i] = fma(1/n_s, acc_s[i], g[i])` in source order (the mean is not rounded before it is added); a group of one sentence therefore equals that sentence's row bit for bit, and a group of empty sentences is the zero row. Stages 4 and 5 then run once per group on this row (normalizing the group's mean, not summing normalized sentences). The sentences of a group are consecutive in the source, and a worker always gets whole groups, so a group never spans batches or workers. Group mode cannot be combined with shards (2.1 raises for `shard=` with groups), so the fixed shard boundaries never cut a group.
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

A `Batch` is the input of an aggregation: `first_row` (the row of its first sentence), `offsets` (int64, n + 1 entries, from 0 to the number of tokens), `tokens` (uint32 vocabulary ids, or `OOV_TOKEN` for an unknown FastText word) and, for FastText, `ngram_offsets` and `ngrams` (uint32 buckets, one range per unknown word, in order). Sentence s is row `first_row + s`. Weights are not part of a batch, so a Source does not depend on the encoder; 2.1 adds token ids from Arrow and `TokenIds` as further Sources of the same batches, and group mode adds a target per sentence.

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

A backend implements four operations behind a device-neutral interface and is chosen by name from a fixed set (`"cpu"`, `"numpy"`, later `"mlx"` and `"cuda"`); there is no plugin registry and no `"auto"`:

- `pool(batch, params, out)`: stages 2 to 4 (and 5 if it can quantize) for one batch into the rows of a sink buffer. `params` holds what an encoder needs on that backend (vocabulary table, weights, uploaded rows) and is built once per encoder and backend.
- `gram_block(rows)`: the float32 product of one block for `fit`; the runtime sums the blocks in float64 in block order.
- `score(block, weights, bias)`: the scores of stored rows.
- `capabilities()`: the word dtypes, aggregations, steps and storage dtypes it supports, and whether it supports group mode.

Before reading any input, the runtime checks that the backend can run the encoder and raises otherwise; it never falls back silently. The first error of any worker stops the others and is raised. A conformance suite runs over every available backend and compares each with the reference bit for bit for every dtype, aggregation and step, including empty, long and unknown-word sentences and subnormal values; a new backend is done when it passes.

### Artifacts

Word vectors, encoders and stored sentence vectors are sets of named tensors with a flat string map of metadata, written through a container:

```python
class TensorContainer(Protocol):
    def write(self, path, tensors: Mapping[str, ndarray], metadata: Mapping[str, str]) -> None: ...
    def open(self, path, *, mmap: bool) -> tuple[Mapping[str, ndarray], Mapping[str, str]]: ...
    def verify(self, path) -> None: ...
```

- **Tensors** follow the rules of safetensors: numeric dtypes only (F32, F16, I8, U8, U32, I64), little-endian, C order, every tensor aligned to its type and the data of a file starting at a multiple of 64 bytes, names matching `[a-z0-9_]+(\.[a-z0-9_]+)*` except `__metadata__`, which safetensors reserves; dimensions are int64 and the product of the non-zero ones times the item size stays within int64, also for tensors with a zero dimension; writers check the same rule. Writers first take a snapshot of the mapping of tensors, of every array's shape and dtype (as a view) and of the metadata, so a caller that changes them during a write does not change the header or the fingerprint. The data are not copied as a whole: every block is copied once when it is written and that copy is hashed, so the checksums always describe the written bytes, and `expected` checksums detect data that changed since they were checked. Strings are UTF-8 bytes (U8) plus int64 offsets, so the vocabulary is `vocab.bytes` and `vocab.offsets`. Every floating-point number that affects the result, such as `oov_weight`, is a tensor, never a decimal string. Integer parameters (dimensions, `min_n`, `max_n`, `bucket`, `max_ngrams`) are metadata, written as decimal strings without sign or leading zeros, or as canonical JSON for a group of them.
- **Names.** Word vectors: `vectors`, `scales` (int8), `counts` (optional), `vocab.bytes`, `vocab.offsets`, and for FastText `ngrams` and `ngram_scales`. Encoder: `weights`, `oov_weight`, `step.<i>.components`, `step.<i>.weights`, `step.<i>.basis`, and for a folded encoder its word vectors under the names above. Stored sentence vectors and shards: `vectors`, `scales`, `inv_norms`; `row_ids` is reserved for 2.1.
- **Metadata** (`fse.` keys, mapped one to one to the `__metadata__` of safetensors): `fse.kind`, `fse.format_version`, `fse.created_by` and a `fse.checksum.<tensor>` (sha256) per tensor for all; `fse.dim`, `fse.dtype`, `fse.fasttext` (min_n, max_n, bucket) and the quantization rule for word vectors; `fse.transform_version`, `fse.tokenizer`, `fse.aggregation`, `fse.steps`, `fse.input_dim`, `fse.output_dim`, `fse.max_ngrams`, the fingerprints of the word vectors and the vocabulary, and the recipe and its parameters (provenance only) for an encoder; the encoder fingerprint, `fse.transform_version`, the storage dtype, `fse.output_dim` and the range of rows for stored vectors.
- **Containers by purpose.** Every kind of artifact has exactly one container, enforced when it is read:

  | Artifact | Container | Path |
  |---|---|---|
  | word vectors, encoders (models, the hub) | safetensors 1 | one `.safetensors` file |
  | stored sentence vectors and their shards (output) | NpyContainer 1 | a directory of `.npy` files and `manifest.json` |

  Models are shared and read: one standard file per model and dtype, which the hub shows and other tools read. Output is a large matrix written in parallel and read by numpy-based tools: every shard tensor is a plain `.npy` that `np.load(mmap_mode="r")` opens directly. Both containers record the same checksums and give the same fingerprint for the same content; a golden test writes the same tensors through both and checks content and fingerprint.

### safetensors 1

`fse._container.SafetensorsContainer`, in the package of both containers, whose modules import only the standard library, numpy, `UnsupportedArtifactError` and each other (relatively), so the 1.x export can copy it; `_npy` and `_safetensors` share only `_common`.

- The file is safetensors: an 8-byte little-endian header length, a UTF-8 JSON header `{"__metadata__": {...}, name: {"dtype", "shape", "data_offsets"}, ...}`, then the data. The tensors lie by descending item size, then name, without gaps, so each is aligned to its type; the header is padded with spaces so that the data start at a multiple of 64 bytes. The reference implementation reads these files.
- Reading is strict: a header of at most 16 MiB and within the file, a JSON object padded with spaces only and data starting at a multiple of 8 bytes, no duplicate keys or NaN constants, exactly `dtype`, `shape` and `data_offsets` per tensor, only the dtypes above (others raise `UnsupportedArtifactError`), offsets that fit the shape (computed with Python integers), tensors aligned to their type and covering the data without gaps or overlaps up to the end of the file, and `__metadata__` mapping strings to strings with the checksums and the fingerprint of the content. Files from other writers lack these keys and are rejected.
- A normal open reads the data into one immutable `bytes` and checks every checksum; `open(mmap=True)` maps the data once, read-only, and the tensors are views of that map; `verify` hashes the tensors on disk. A normal open and `verify` raise if the size or modification time of the file changed while they read it. A mapped file must not be changed or truncated while it is mapped: a change is caught by the checksums the next time the data are checked, but a truncated file can end the process with SIGBUS on POSIX, which no reader can catch; copy files that other processes may change. The file is opened as described for the NpyContainer: a regular file, not followed if it is a link, checked against its path.
- Writing goes to a temporary sibling created exclusively (`tempfile.mkstemp`, `.<name>.writing-...`): the header is reserved (its length does not depend on the checksums), every block of data is copied once into immutable bytes, and that copy is hashed and written (and compared with `expected` checksums if given), so the file holds exactly what its checksums describe even if the source changes meanwhile; then the header is written and the file synced. It is published without ever replacing a file: `os.link` on POSIX (then the directory is synced), `MoveFileExW` with `MOVEFILE_WRITE_THROUGH` and without `MOVEFILE_REPLACE_EXISTING` on Windows. A failed write removes only its own temporary file. A path without `.safetensors` raises.

### NpyContainer 1

`fse._container.NpyContainer`, for stored sentence vectors.

- An artifact is a directory with `manifest.json` and one `<tensor name>.npy` per tensor. The manifest holds exactly `{"container": "npy", "container_version": 1, "tensors": {name: {"file", "dtype", "shape"}}, "metadata": {...}}`: `file` is the name plus `.npy`, `dtype` a code of the table above, `shape` a list of non-negative integers, metadata values strings. Duplicate keys, NaN constants, unknown or missing keys raise `CorruptArtifactError` (a `ValueError`); another container or version raises `UnsupportedArtifactError`. Unlisted files in the directory are ignored.
- The manifest and every `.npy` must be regular files, not symbolic links; each is opened once (without following links and without blocking), checked against the path it was found at, and read or mapped through that one handle; numpy parses one bounded copy of each header. The manifest may hold at most 16 MiB, a `.npy` header at most 64 KiB (checked before numpy reads it), a tensor at most 32 dimensions; sizes are computed with Python integers, so a shape cannot overflow.
- Every `.npy` is version 1.0 or 2.0 with the exact type description of its code (`<f4`, `<f2`, `|i1`, `|u1`, `<u4`, `<i8`), C order, the shape of the manifest, data at a multiple of 64 bytes and nothing after the data; pickled objects are never read.
- The writer enforces the limits of the reader, and with `expected` checksums it hashes every tensor while writing and commits only if they match. The container writes `fse.checksum.<name>` (the lowercase hex sha256 of the data) for every tensor and `fse.fingerprint`, and rejects them from the caller. A normal open reads every tensor into immutable `bytes` and checks its checksum (hashed in 64 MiB chunks); `open(mmap=True)` checks the manifest, the headers, the sizes and the fingerprint against the recorded checksums, and maps the data read-only; `verify` hashes the data on disk. Tensors with no elements are empty arrays, as nothing can be mapped.
- Writing creates the target directory exclusively (an existing one raises `FileExistsError`, also when another writer created it first; artifacts are never overwritten, which also avoids replacing mapped files on Windows), writes and syncs every tensor, syncs the directory, and commits by renaming the manifest from a temporary name into place; a directory without a manifest is no artifact. Any failure removes the directory.
- The fingerprint test vector (`tests/test_container_common.py`): tensors `vectors` F32 `[[1, -0.0], [0.5, 2]]`, `vocab.bytes` U8 of `"aé"`, `vocab.offsets` I64 `[0, 1, 3]` with metadata `fse.kind=word_vectors`, `fse.format_version=1`, `fse.dim=2`, `fse.dtype=float32` give `sha256:98a86ce422b1d61979da4b088934ebbb5c4f128463ca20123f7508d0f74f32db`.

### word_vectors 1

`fse.WordVectors` (`fse._wordvectors`), one `.safetensors` file, created by `from_arrays`, `fse.gensim.from_gensim` (extra `fse[gensim]`; gensim is imported only there) or `load`. Immutable: the arrays it returns are new read-only headers on its data, and a normal load reads every tensor into immutable `bytes`; making an internal array writable again through `.base` is outside the contract. Read as float32, never pickled. `verify` checks the data the object holds against the checksums it was loaded with, then the files on disk. `save` of loaded vectors compares every tensor with those checksums while writing it, and `quantize` reads every block once and hashes and quantizes that same copy, so damaged or changed data never gets new checksums; for mapped vectors these checks are never skipped, as the file can change underneath the map. Quantized vectors own all their arrays.

- Tensors: `vectors` (n x dim, F32, F16 or I8), `scales` (F32, n) exactly for int8, `counts` (I64, n, optional), `vocab.bytes` (U8), `vocab.offsets` (I64, n + 1, from 0 to the number of bytes), and for FastText `ngrams` (bucket x dim, the dtype of the vectors) with `ngram_scales` exactly for int8. 1 <= n < 2**32 - 2 (word ids stay below the reserved token ids), dim >= 1.
- Metadata (closed: any other key raises): `fse.kind=word_vectors`, `fse.format_version=1`, `fse.created_by`, `fse.dim`, `fse.dtype` (`float32`, `float16`, `int8`), `fse.quantization=quantize_rows@1` exactly for float16 and int8 (the storage rules of the numerical contract), optional `fse.quantized_from` (the fingerprint of the float32 vectors, only for float16 and int8), and for FastText `fse.fasttext` (canonical JSON of `bucket`, `max_n` and `min_n`, 1 <= min_n <= max_n, 1 <= bucket <= 2**32 - 1) with `fse.ngram_hash=fasttext@1`.
- The vocabulary is UTF-8; a word is found by the exact match of its bytes (NFC and NFD spellings are different words, a string without a UTF-8 form is no word). Every open checks the structure, the offsets and the UTF-8 of the vocabulary in chunks; `verify` and every writer also check that words are unique, floats finite, int8 codes within -127 to 127, scales finite and not negative, counts not negative, and every int8 row times its scale finite.
- For FastText, the row of a known word is the word's own row (gensim's `vectors_vocab`), not gensim's average of it with its ngrams; combining ngrams for unknown words belongs to the encoder.
- `quantize` makes float16 or int8 vectors from float32 ones in blocks, without a float32 copy; quantized vectors are never quantized again. Loading int8 vectors allocates about their codes; a test checks it with `tracemalloc`.
- `tests/test_data/v2/` holds golden float32 and int8 vectors (`.safetensors`) with pinned fingerprints, written by `make_golden.py`; they are never rewritten, and the 1.2.0 export must give the same fingerprints. The fingerprints were pinned with the NpyContainer and did not change with the move to safetensors.

### Identities

- A **row id** is the int64 position of a sentence in its source, from 0.
- A **shard** is a contiguous range of rows; 2.1 splits sources with `shard=(i, n)` at multiples of 16,384 rows.
- A **fingerprint** is `"sha256:"` and the lowercase hex sha256 of the UTF-8 bytes of the canonical JSON (keys sorted, separators `,` and `:`, no whitespace, non-ASCII characters unescaped) of `{"fingerprint": 1, "metadata": {...}, "tensors": [[name, dtype, shape, sha256], ...]}`: the metadata without `fse.created_by`, `fse.fingerprint` and the `fse.checksum.` keys; the tensors sorted by name, with the dtype code (such as `"F32"`), the shape as a list of integers and the lowercase hex sha256 of the little-endian C-order bytes. It describes the content, not the container, so every container gives the same fingerprint; the tests of the container pin a test vector with its expected fingerprint.
- An unfolded encoder references its word vectors and vocabulary by fingerprint, a folded one holds them and names the original fingerprint as provenance; stored vectors reference their encoder; opening them with a mismatch raises. 2.1 adds the fingerprint of the source for resuming.

### Persistence names replace class paths

Saved 2.0 artifacts contain no Python class paths. Their persistence API is the set of strings above: `fse.kind` values, identifiers, dtype codes, tensor and metadata names. These are never renamed or reused with another meaning. The public Python API is `fse.__all__`; private modules (with a leading underscore) may move freely.

### Migration

2.0 has no pickle loader. A last 1.x release, 1.2.0 on the branch `1.x`, adds `export_v2`, which writes the word vectors of a model and its encoder (weights, `oov_weight`, and the learned components as a `remove_components` step) into the 2.0 format, and optionally its sentence vectors as data with `transform_version` 1. uSIF models saved before 1.1 keep their weighting by lambda squared, because the export takes the weights the model applies. The hub word vectors are converted the same way. The hub also gets pretrained, frozen encoders: Average for every vector set, SIF and uSIF for a selection, fitted once on WikiText-103 with the native tokenizer, so `from_pretrained` encodes at once and gives the values of the published tables. A golden manifest and fingerprint of a tiny model, written by 1.2.0 and loaded by 2.0 in the tests, catch drift between the two writers.

### Layout

New 2.0 code goes into private modules of `fse`: `_contract` and `_reference` (contracts and reference), `_codes` (storage arithmetic in blocks), `_container`, `_wordvectors` and `_wordvectors_format`, the gensim adapter `fse.gensim` (extra `fse[gensim]`), then the runtime and backends, the encoder and recipes, sinks and store, and the thin `train`/`infer` facade of the release. gensim is not a dependency of fse: `import fse` and every 2.0 class work without it (tests check it in a subprocess that blocks gensim and scipy).

The modules form layers, and a module imports only from layers below it. The import contracts in `pyproject.toml` (`[tool.importlinter]`, run by `make lint`) enforce this, and every new module must take a layer there:

1. `gensim` (the adapter, the only module that imports gensim)
2. `_runtime`
3. `_backends`, `_stream`, independent of each other
4. `_text`
5. `_wordvectors`
6. `_wordvectors_format`
7. `_container`, `_codes`, `_gram`, `_reference`, `_reference_text`, `_validation`, independent of each other: the references never share code with what they check
8. `_contract`, `_cpus`, `_unicode`

Inside a package, an import contract keeps its parts apart: `_npy` and `_safetensors` share only `_common`, each backend shares only `_base`, and the runtime's pipeline sits on its tasks, sinks, spec and executor.

The native extension `_native` is invisible to import-linter. `tools/check_guardrails.py` checks that only `_text` (stages 1 and 2) and the cpu backend `_backends._cpu` (the kernels) import it and that it imports only `_contract` from fse. A further contract forbids gensim, scipy, smart_open, psutil, threadpoolctl, wordfreq, mlx, torch and sklearn in all of fse, except gensim in `fse.gensim`; the checker applies the same list to the extension. The runtime, backends, encoder, sinks and store of the coming PRs add their layers above `_text`.

### Defaults

Defaults are part of the API and chosen for speed: `workers` is the number of usable CPUs (`fse._cpus.usable_cpus`: an explicit `PYTHON_CPU_COUNT` or `-X cpu_count`, else the CPUs of the process affinity limited by the CPU quota of its Linux cgroup and every visible ancestor, rounded up; small inputs run inline in the calling thread), and new encoders, also the pretrained ones on the hub, end with `normalize`, so the cosine of two rows is their dot product. A server that encodes many small requests in parallel sets `workers=1`.

### Baseline

`benchmarks/baseline/` records the values of fse 1.1.0 from PyPI that 2.0 is compared with: STS, an MTEB selection, training throughput, memory and a lifecycle benchmark. Its README holds the tables and the commands to reproduce them.

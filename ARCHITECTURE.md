# Architecture

This document describes how fse is put together and the rules that keep it correct. It is the reference for changes to the training loop, the Cython kernel and the tests. When a change alters one of these rules, update this file in the same pull request.

## Package Layout

```
src/fse/
  __init__.py            # public API, __all__, version, gensim>=4 guard
  inputs.py              # input containers that yield (tokens, index) tuples
  vectors.py             # Vectors / FTVectors: pretrained vectors from the Hugging Face Hub
  models/
    base_s2v.py          # BaseSentence2VecModel: scan, sanity checks, threads, save/load
    average.py           # Average model and the numpy fallback train_average_np
    average_inner.pyx    # Cython kernel train_average_cy (with .pxd and .pyi stub)
    sif.py, usif.py      # SIF and uSIF: word weights and principal component removal
    sentencevectors.py   # SentenceVectors: storage and similarity queries
    utils.py             # SVD helpers
    _single_pass.py      # private: state of the single pass during train
    _index_tracker.py    # private: bitmap of seen indices for the scan
    _quantize.py         # private: float16 and int8 storage arithmetic
    _windowed.py         # private: training into a ring of float32 windows
```

**Module paths under `fse.models` are part of the API.** Models are saved with gensim's `SaveLoad`, which pickles class paths such as `fse.models.average.Average`. Renaming or moving a module breaks loading of every saved model. New internal code goes into new private modules; existing ones stay where they are.

## Data Flow

`train(sentences)` and `infer(sentences)` take a re-iterable of `(tokens, index)` tuples. The `inputs` classes produce exactly that shape. `index` is the row of the output matrix the sentence contributes to.

1. **Scan:** `scan_sentences` makes one pass over the input. It counts sentences and words, finds the largest index, validates every item and reports whether all indices are unique and whether every index is the position of its sentence (`identity_indices`). Indices that are not the positions of their sentences are tracked in a bitmap (`IndexTracker` in the private module `fse.models._index_tracker`), which grows by doubling up to `max(BITMAP_MIN_BYTES, one byte per sentence)`; indices beyond it wait as int64 until it may double, and are sorted at the end if it never may cover them. With the extension the loop runs in Cython (`scan_indexed_sentences`) with the same checks, messages and tracker rules; a subclass that overrides `_check_indexed_sent_valid` keeps the Python loop.
2. **Sanity and memory:** `_check_pre_training_sanity` checks the vectors and weights; `estimate_memory` warns when the result will not fit into RAM.
3. **Prepare:** `BaseSentence2VecPreparer` allocates the sentence vectors (`max_index` rows, initialized to `EPS`), in RAM or as a memmap. With identity indices and the kernel, the rows are not initialized here: every job sets its own contiguous rows to `EPS` right before it averages them (see *Single pass*). With `train(update=True)` it appends the rows instead, and every sentence index is shifted by the previous number of rows. Rows are initialized in blocks. An in-memory update copies all existing rows into a new array (`vstack`); a memmap grows in place.
4. **Pre-train hook:** `_pre_train_calls` computes the word weights. All weights are 1 for Average; SIF and uSIF derive them from word frequencies.
5. **Jobs:** `_train_manager` starts one producer thread and `workers` worker threads. The producer batches the input into jobs of up to `batch_words` words and `batch_words` sentences; a single longer sentence forms a job of its own. With the extension, `_kernel_context` builds a `KernelContext` once per call, and the producer reads, batches and encodes the input in one pass with a `JobEncoder` while it holds the GIL. Each job copies its data out of the encoder's reusable buffers: word indices, sentence offsets, target rows, and the UTF-8 bytes of unknown FastText words. Without the extension, or when a subclass or instance replaces `_do_train_job`, the workers call `_do_train_job` on lists of tuples instead (`train_average_np` without the extension).
6. **Kernel:** `run_job` averages an encoded job without the GIL: it hashes the ngrams of unknown FastText words in C, sums the weighted word vectors of each sentence with a row kernel (see *Row kernels*) and adds the average into its target row with BLAS `saxpy`. `infer` and `train_average_cy` use the same encoder and kernel in the calling thread. The producer holds the GIL while it reads and encodes the input, so it bounds the throughput once two to four workers are busy; native inputs and list chunks (below) avoid it.
   For `CorpusFile` and `TextList`, scan and training work on bytes instead (see *Native inputs*).
7. **Post-train hook:** `_post_train_calls` removes principal components for SIF and uSIF. The components are computed exactly from the Gram matrix `X^T X` of all sentence vectors, summed in float64 over blocks of rows and decomposed with `eigh`; the removal works in blocks of rows as well, removed by `workers` threads in parallel. Both read the vectors once from front to back, so neither needs memory proportional to the number of sentences, and a memmap streams from disk. With identity indices and the kernel, training has summed the Gram matrix already (see *Single pass*), so only the removal reads the vectors again. After `train(update=True)` it reuses the components of the first training for the appended rows only, as `infer` does.

`infer(sentences)` takes a shorter path. It scans the input, allocates a zero-filled output matrix, runs the jobs in the calling thread with the weights learned in `train`, and finishes with `_post_inference_calls`, which removes the stored principal components for SIF and uSIF. It has no preparer and no pre-train hook, and it averages in the calling thread; a native scan and the removal of components use `workers` threads.

### Single pass

For a memmap larger than the RAM, every pass over the sentence vectors goes to disk. With identity indices (index i for the i-th sentence, as for every input class except the `C*` lists with custom indices) and the kernel, training saves up to two separate full passes. The state of one such call is a `SinglePass` in the private module `fse.models._single_pass`, held by the model only during `train`, like the row locks.

- **Prefill** (every model, also for `update=True`): `KernelContext.prefill_rows` makes `run_job` and `run_chunk` set the rows of their job to `EPS` before averaging, instead of `_fill_eps` writing all new rows in advance. The arithmetic is the same, so the vectors are bit-identical. The producer checks before a worker writes that every job starts at the next row and covers contiguous rows, so jobs neither overlap nor leave gaps, and training checks at the end that the jobs covered every new row. `run_chunk` checks that a chunk holds exactly the announced number of sentences. If allocation succeeded but the pre-train hook, the threads or a job fail, training sets the new rows to `EPS` and keeps the existing rows, so no uninitialized row remains; an error in the post-train hook, for example a non-finite Gram matrix, leaves the trained rows as they are.
- **Gram matrix** (SIF and uSIF with `components > 0`, without `update=True`, and without a replaced `_post_train_calls`): after each job the worker reports its rows to an `OrderedGram`; the worker that completes a block of `REMOVAL_BLOCK_ROWS` rows computes the block product while the rows are cached, and the products are summed in block order. Blocks, products and summation order are those of `gram_matrix`, so the components are bit-identical to a separate pass for any number of workers. Products that wait for a missing predecessor are bounded by `PENDING_BUDGET_BYTES`, reserved before they are computed; beyond it, a finished block is recomputed from its rows when it is summed. One thread at a time sums, outside the lock, so other workers keep reporting. `_post_train_calls` receives the finished matrix as `gram`.

Other indices, the numpy fallback and a replaced `_do_train_job` take the separate passes. An update that computes new components from old and new rows prefills the new rows but reads all rows for the Gram matrix.

### Storage formats

`SentenceVectors.dtype` is `"float32"` (the default), `"float16"` or `"int8"`. Sentence vectors are always computed in float32 and stored in the format; every read returns float32. int8 rows are symmetric codes with one float32 scale per row in `scales` (`codes * scale` restores the row), so the cosine of two code rows equals the cosine of the rows they represent. With a mapfile, codes live in `<path>.vectors` and scales in `<path>.scales`. All format arithmetic is in `fse.models._quantize`; `quantize_into` raises for non-finite rows and float16 overflow instead of storing infinity.

For float16 and int8, training takes one of two plans:

- **Windows** (index i for the i-th sentence, the compiled kernel, built-in `_do_train_job` and `_post_train_calls`): `KernelContext.use_ring` makes the kernel write row r into row `r % ring_rows` of a ring of float32 windows of about `WINDOW_BYTES` (a multiple of `REMOVAL_BLOCK_ROWS`). A `WindowedWriter` counts finished rows per window; the worker that completes the next window in row order finalizes it outside the lock, one thread at a time, and frees its slot. The producer waits in `SinglePass.check_job` until the rows of a job have a free slot, and stops waiting once training fails. The ring has enough slots for the longest job; jobs hold at most `batch_words` sentences, native chunks at most `MAX_CHUNK_ROWS` (65,536) and list chunks at most 8,192, so with windows of at least that many rows the ring has two slots. `estimate_memory` uses the same ring planning (`plan_ring`). Average and updates that reuse learned components take one pass, which removes known components from each window and stores it. SIF and uSIF learning components take two passes: the first sums the Gram blocks of each window in block order, as `gram_matrix` does, the second removes the components and stores. Windows and removal blocks coincide with those of a float32 training, so the stored rows are bit-identical to a float32 training stored afterwards. `_set_components` and `_remove_components` hold the component logic both plans and inference share.
- **Staging** (every other case, logged with the reason): `SentenceVectors._stage` replaces `sv.vectors` by a float32 matrix of the rows of this call, which training and the hooks use as for float32 vectors, with `offset` 0; afterwards its rows are stored in the format. During the hooks, `sv.vectors` therefore holds only the rows of this call, and `sv.dtype` is float32, so reads, searches and `score` in a hook work as for float32 vectors.

Both plans store through `_RowWriter`: a new training writes `.writing` files that replace the old ones only when it succeeds (the old files move aside and come back if any step fails, including opening the new ones), an update grows a memmap in place and cuts it back on failure, and a failure leaves the stored vectors and learned components as they were. A failure before the commit restores the learned components before any file is cleaned up; once the new rows are committed (`SentenceVectors.commits` counts that), a later failure keeps them and their components together. The staging file is removed before the commit, so its removal cannot fail after it. If another map of a file prevents cutting it back (Windows), the file stays larger; loading accepts sentence vector mapfiles larger than their recorded shape for that reason. Before files move, every map of them is released and cyclic garbage is collected, because Windows refuses to replace, cut or delete a mapped file; `tests/conftest.py` emulates that rule on every platform. An update that would compute new components from stored float16 or int8 rows raises before training.

Reading: `most_similar`, `similar_by_*` and `score` call `score_rows` in the extension, which multiplies the stored values (float16 through a 65,536-entry table) with float32 weights, scaled per row, without the GIL; `_StoredScores` runs row ranges in a thread pool. For float16 and int8, `init_sims` keeps only one inverse norm per row of the stored values instead of a normalized copy; for int8 that is the norm of the codes, as the scale does not change the direction, so even tiny vectors have a finite inverse norm. Without the extension, numpy converts each block to float32; both give the same results within float32 rounding.

### Word vector formats and row kernels

`wv_dtype` is `"float32"` (the default), `"float16"` or `"int8"`. For float16 and int8 the model stores codes of the word vectors (`vectors`, or `vectors_vocab` and `vectors_ngrams` for FastText) in `wv_codes`/`wv_scales` and `ngram_codes`/`ngram_scales`, quantized once at construction with `quantize_into`, the same arithmetic as for sentence vectors. The passed `KeyedVectors` stay float32 and unchanged. `_word_arrays` returns the rows the kernel and the numpy fallback read: the float32 vectors themselves, or the codes. With `wv_mapfile_path`, codes live in `<path>_wv.<dtype>` (and `<path>_ngrams.<dtype>`), written under a `.writing` name and renamed when complete; an existing file is reused only if it holds exactly these codes. `save` leaves mapped codes out of the pickle and `load` reopens them; the scales stay in the pickle. Class attributes default models saved before `wv_dtype` to float32.

The kernel adds each stored row with a routine of type `add_row_ptr`, `mem += factor * row`; int8 rows fold their scale into the factor. The routines are chosen once at import (`fse_select_kernels`), not per call:

- float32 rows, on every platform: BLAS `saxpy`, which dispatches by CPU itself. End to end, an own loop was no faster on Apple M1 or AMD Zen 5, and `saxpy` keeps float32 results bit-identical to 1.0; prefetching does not change them.
- x86-64 with GCC or Clang and a CPU with AVX2, FMA and F16C: float16 and int8 rows through intrinsics in functions with `__attribute__((target(...)))`, so the build flags stay the baseline of the wheel. AVX2 and FMA are detected with `__builtin_cpu_supports`, F16C with CPUID (Clang knows `"f16c"` only since version 18). `_kernel_info()` returns `"avx2"`.
- arm64: float16 rows convert with the native `__fp16` type, and the compiler vectorizes them and the int8 loop with NEON. `"neon"`.
- Everything else, including Windows (MSVC) and x86 CPUs without AVX2: float16 through a 65,536-entry table and int8 with a portable loop. `"portable"`. `_use_portable_kernels(True)` selects this set on every platform, for tests.

While a row is added, the kernel prefetches every cache line of the row `PREFETCH_DISTANCE` (8) tokens ahead (one prefetch per 64 bytes and one at its last byte), and inside the ngrams of an unknown word the ngram row 8 ahead. 8 was the best or within noise of the best distance on Apple M1 and AMD Zen 4 for every format and vocabulary size measured. Every routine adds the values of a row in index order and a sentence's rows in token order, so the rule below (the same result for any number of workers) holds for every format. For float16 and int8 word vectors, different routines may round differently in the last bits, so their sentence vectors can differ in the last bits between CPUs.

### Native inputs

`CorpusFile` (a memory-mapped local file), `TextList` (strings copied once into a UTF-8 buffer with offsets) and a `SplitIndexedList` whose items are a `list` of `str` or numpy `str_` (copied into such a buffer by every scan, released when the call ends; a list changed between the scan and training raises; only in `train`, as `infer` of a few sentences would build a `VocabTable` of the whole vocabulary) skip the Python producer. A `SplitIndexedList` with other items, an `ndarray`, `__getitem__` or `__iter__` overridden, or strings that cannot be encoded as UTF-8 (lone surrogates) keeps the Python path, whose sentences the native path reproduces:

1. **Scan:** `scan_sentences` calls their `_scan_native`. The source is split into about `4 × workers` ranges that start at sentence starts, so every sentence belongs to exactly one range; threads validate and count the ranges with `scan_spans` without the GIL and record chunks of 64 KB to 1 MB and at most 65,536 sentences. Prefix sums turn range-local sentence numbers into indices. The chunks and the file's size and modification time are kept on the input until training.
2. **Training:** `Average._job_encoder` returns a `_ChunkFeeder` that hands out the chunks. Each worker runs `run_chunk`: it splits its chunk, looks up every word in a `VocabTable` (open addressing over the UTF-8 bytes of `key_to_index`, built once per call; each 32-byte slot holds the hash, the index and words of up to 16 bytes inline, so a lookup usually reads one cache line) and averages each sentence with `_average_sentence`, the routine encoded jobs use too, all without the GIL. Every sentence has its own row, so no locks are needed.

The input contract is the one of `IndexedLineDocument` and `SplitIndexedList`, and tests compare the vectors bit for bit: a line ends after `\n` and the last needs none; text is decoded strictly as UTF-8; words are split at every code point with `str.isspace()`. Invalid UTF-8 raises the `UnicodeDecodeError` Python raises for that sentence, with its number and byte offset added. The fingerprint of a `CorpusFile` (device, inode, size, modification time) is read from the descriptor that was mapped and compared again before training and after every chunk, so a change that alters one of them raises; a change that keeps all four is not detected. The Python path of `CorpusFile` checks the same fingerprint before and after every iteration. A file truncated while it is mapped can still end the process with SIGBUS on Linux; only a private copy would prevent that. `scan_spans` and `run_chunk` check every range and offset before they release the GIL. A validator replaced on the instance or in a subclass forces the Python scan, and training then reads tuples as well. Without the extension, both classes are iterated in Python like their counterparts.

### List chunks

With two or more workers, `train` lets the workers read an `IndexedList` or `CIndexedList` themselves (`Average._list_plan_for`): the class or a subclass that keeps its `__getitem__` and `__iter__`, `items` a `list`, and a scan that saw only exact `list` sentences of exact `str` (`scan_indexed_sentences` reports it). The producer hands out `_ListChunk` row ranges of about 65,536 tokens (256 to 8,192 sentences) and reads nothing itself. Each worker runs `run_list_chunk`: holding the GIL, it copies the UTF-8 bytes of the tokens of its chunk into private buffers (a custom index is read first, as it may run Python code; the sentence is held by a strong reference and its length read again for every token, because a failed encoding may run the garbage collector); without the GIL it looks them up in the `VocabTable` and averages with `_average_sentence`. Empty tokens are looked up like any word. Words without a UTF-8 form (lone surrogates) cannot be in the `VocabTable` and are unknown words, as for the `JobEncoder`; FastText raises for them on both paths, and any other error of the encoding is raised. The vectors equal the `JobEncoder`'s; tests compare them. One worker, `infer`, other types, `ndarray` items, custom indices other than an exact list or a one-dimensional integer array, a vocabulary without a UTF-8 form for every word, and subclasses that change how items are read keep the `JobEncoder`, which is faster for a single thread. A list that changed after the scan (another type, another length, an index out of range or, with prefill, not at its position) raises instead of being read differently; the plan (`_ListPlan`, with the `VocabTable`) exists only during `train`, is not used by an `infer` that a hook calls meanwhile, and is never saved. A vocabulary word without a UTF-8 form makes native inputs read their sentences through their iterator with the `JobEncoder` as well.

### BLAS threads

Workers call BLAS for the Gram block products, and SIF and uSIF remove components with BLAS after training and `infer`. `train` and the removal after `infer` run inside `utils.blas_threads(workers)`, which, with more than one worker, limits BLAS (threadpoolctl) to `usable_cpus() // workers` threads, at least one; without it, OpenBLAS started its own threads in every worker. The limit holds for the whole process: blocks that overlap, also from calls in other threads, apply the smallest active limit, and the last one to end restores the limits from before the first. `remove_principal_components(..., workers=n)` removes its blocks of `REMOVAL_BLOCK_ROWS` rows in a thread pool; every block is computed as before, so the result does not depend on `n`.

## Rules

### The numpy fallback matches the Cython kernel

`train_average_np` and `train_average_cy` must produce the same vectors, within float tolerance, for every input and every `wv_dtype`, including FastText out-of-vocabulary words and many-to-one index mappings. The fallback restores the rows it reads to float32 with `dequantize`, multiplying int8 codes by `weight * scale` in the kernel's order, so tiny scales underflow alike. Tests compare both paths. A change to one path needs the same change and a parity test for the other.

### One kernel, the same result for any number of workers

`train` and `infer` encode with `JobEncoder`, `train_average_cy` with its wrapper `encode_job`, and all three average with `run_job`; `run_chunk` and `run_list_chunk` only find the word indices differently and average with the same `_average_sentence`, so there is no second averaging code path to keep in sync. Every sentence is summed in the same order whatever the job size or worker count, so with one row per sentence the vectors are bit-identical for 1 to n workers, which tests check. The kernel has no length limits: `batch_words` and `batch_ngrams` only size jobs and cap the ngrams of unknown FastText words, as in the numpy fallback.

### Many-to-one mappings accumulate

Several sentences may share one index. Each sentence average is added into the row; a row is not overwritten.

`scan_sentences` reports whether every index is unique. If it is not and more than one worker trains, the Cython kernel adds into rows under striped locks (`RowLocks`, 256 stripes), which exist only during `train` and are never pickled. Without the extension, such inputs train with one worker. Unique indices, the common case, take no lock.

### The passed word vectors stay unchanged

A model never changes the `KeyedVectors` it was given. Where it replaces attributes (the counts for `lang_freq`, the vector arrays for `wv_mapfile_path`), it first makes a shallow copy (`_own_word_vectors`): the vector arrays are shared, while the vocabulary containers, the arrays of per-word attributes (`expandos`) and the replaced attributes are its own, so a later `add_vectors` on the passed model cannot give the copy word indices beyond its vectors. The kernel context checks that the vocabulary has no more words than there are word vectors. An existing word mapfile is reused only if its size and content equal the vectors.

### Appended rows never change existing rows

`train(update=True)` writes only into the rows it appends. Existing sentence vectors, including their removed components, stay bit-identical.

### Hooks keep their signatures

Subclasses override `_check_parameter_sanity`, `_pre_train_calls`, `_post_train_calls`, `_post_inference_calls` and `_check_dtype_santiy`, all of which accept `**kwargs`, and `_post_load_calls`, which migrates models saved by earlier versions. Tests replace `_do_train_job` on instances, so the job path must keep calling it; `_kernel_context` returns `None` for such instances and subclasses, which disables encoded jobs.

### Validate strictly, never fail silently

Invalid input raises a clear exception at the API boundary. Nothing is coerced, ignored or truncated without an error, and an unexpected keyword argument is an error (`pydantic.ValidationError` at validated entry points). A slower but correct fallback, such as training with one worker, is logged as a warning.

Entry points whose own work dwarfs a few microseconds use `fse._validation.validate_arguments`, a strict pydantic `validate_call`: constructors, `train`, `infer`, `from_pretrained` and the similarity searches. Constraints such as `PositiveInt` live in the signatures. Two places check explicitly instead: methods called per element in loops (`similarity`, `get_vector`, `__getitem__`), where validation would add measurable overhead, and the training data, which pydantic would iterate or copy. Parameters that hold data stay `Any` in validated signatures for the same reason; the scan checks them. Integer constraints accept numpy integer scalars, float constraints reject infinity and NaN.

### Reads return float32

Whatever the storage format, `sv[i]`, `get_vector`, `get_vectors`, `iter_blocks`, `similarity`, the searches and `score` return float32. Only `sv.vectors` (and `sv.scales`) expose the stored values.

### float32 throughout

Word vectors, weights and sentence vectors are computed in `float32` (`REAL`); word and sentence vectors may be stored as float16 or int8 (see *Storage formats* and *Word vector formats and row kernels*), and every sum runs in float32. The kernel takes raw pointers, so arrays must have the expected dtype and be C-contiguous, which `_check_inputs` checks before the GIL is released.

## Build

The extension is compiled from `average_inner.pyx` at build time with Cython 3 against numpy 2 headers; generated C is never committed. Wheels built this way also run with numpy 1.x. `make rebuild-ext` rebuilds it in the development environment after edits to `.pyx` or `.pxd` files.

## Tests

- Tests mirror `src/`: `tests/models/test_average.py` tests `src/fse/models/average.py`.
- Tests use pytest fixtures from `tests/conftest.py` and write files only into `tmp_path`; `tests/test_data` is read-only input.
- Expected values never depend on vocabulary positions, because gensim versions order words with equal counts differently. Derive them from `key_to_index`.
- Tests that download from the Hugging Face Hub are marked `network`.
- pytest treats warnings as errors. Narrow, commented ignores are allowed only for third-party warnings.

## 2.0 (in development)

master is the development branch of 2.0, a rewrite around a frozen encoder; 1.x lives on the branch `1.x`. The sections above describe the 1.x code, which master keeps running until each part is replaced. This section is binding for every new 2.0 module. Items marked 2.1 are additions that must fit without changing anything below.

### Principle and stages

A frozen encoder is a pure function from one sentence to one finished row. Every row is computed in five stages, all inside the worker that owns the sentence:

| Stage | What | Where | Owned by |
|---|---|---|---|
| 1 tokenize | split the text into token spans | CPU | the Source |
| 2 token rows and factors | look up vocabulary ids, hash FastText ngrams; the factor of each row comes from the weights of the fit | CPU for ids and ngrams, the backend kernel applies the factor | Source (ids), Encoder (weights) |
| 3 aggregation | combine the weighted rows of a sentence into one row | backend | an `Aggregation` |
| 4 postprocessing | versioned steps on the row, such as removing components | backend | the Encoder's steps |
| 5 store | quantize per row and write | Sink | the Sink |

The tokenizer is part of the encoder contract: the manifest names it with an identifier like the aggregation (`fse.tokenizer`), and reading an encoder with an unknown tokenizer raises. `external` means the tokens were given by the caller, who then owns reproducing them; a native, specified tokenizer can be added as a further identifier without changing anything else.

There are no Python hooks per row: they would hold the GIL, could not run on a device, could not be saved and would break bit-identity. Two escape hatches remain: own tokenization in Python inside the Source (stage 1, once per sentence, like `CSplitIndexedList` today, slower), and own processing of the finished output arrays. Models in which a row depends on more than its own sentence (attention over neighbors, sentence pairs) are out of scope.

Three rules keep new models possible without a new layer:

1. **An aggregation is not a sum.** It is defined by an initial accumulator, a combine step `combine(acc, f, x)` per weighted token row, a finish step `finish(acc, n)`, the result of an empty sentence and its output dimension. Group mode (many sentences to one row, CPU only) combines the sentences of a group by the rule of the aggregation, defined below for the weighted mean: a mean sums the sentence means, as 1.x does; a later max would take the maximum.
2. **Weights (stage 2) are separate from the aggregation (stage 3).** Stage 2 supplies a factor and a stored row per contribution; the aggregation never sees weights or scales. A later max over SIF-weighted rows therefore needs no own kernel.
3. **The output dimension belongs to the encoder.** The aggregation gives the dimension of its rows (`Aggregation.output_dim(d)`: d for the weighted mean; a concatenation such as SWEM would give 2d), and a step may reduce it (`project`). The result is written into the manifest, and Sinks and Store read it from there, never from the word vectors.

**Linear steps are folded into the word vectors.** The weighted mean is linear, and so are `remove_components` and `project`: applying them to every word and ngram row before averaging gives the same rows, within float32 rounding, as applying them to every averaged row. `fit` of SIF and uSIF therefore folds its learned steps into the word vectors by default, and encoding only averages, as fast as Average; `normalize` is not linear and stays a step per row. A folded encoder owns its word vectors (they are tensors of the encoder, with their own fingerprint) and records the folded steps and the fingerprint of the original vectors as provenance; an unfolded encoder references the original word vectors and applies its steps per row. Folded word vectors in float16 or int8 are quantized after folding. Average learns nothing and needs no corpus; its encoder references the word vectors and holds no tensors of its own, unless `fit` learns a projection.

### Numerical contract (`transform_version` 2)

`fse._contract` holds the identifiers and versions, `fse._reference` the numpy reference that computes every rule below bit for bit; it is the oracle for every compiled and device backend, and tests compare by bit pattern (`view(np.uint32)`), because `array_equal` treats -0.0 and +0.0 as equal.

- **Arithmetic.** IEEE-754 float32 with round to nearest even and subnormals kept (no flush to zero). `fma(a, b, c)` rounds once. Native code calls `fmaf` or FMA intrinsics explicitly and is built without other contractions (`-ffp-contract=off`). Division and square root are correctly rounded; no reciprocal approximations or fast math.
- **Stage 2.** A known word w contributes the stored row `x` (float32 as is, float16 widened exactly, int8 codes converted exactly) with the factor `f = weight[w]`, or `f = weight[w] * scale[w]` with one rounding for int8. An unknown FastText word with m >= 1 ngram buckets (in the order of the hashing, capped at `max_ngrams`) contributes each ngram row with `g = (1/m) * oov_weight`, each product rounded, times the ngram scale for int8; with m = 0 it counts for the length and adds nothing. Without FastText, unknown words are no tokens. The scale of int8 rows always goes into the factor, never into the row.
- **`weighted_mean@1`.** The accumulator starts at +0.0 per value; every contribution does `acc[i] = fma(f, x[i], acc[i])` in token order (ngrams in bucket order); `n` counts the tokens. The row is `(1/n) * acc` with `1/n` rounded first, not `acc / n`. An empty sentence is the zero row; there is no EPS. In group mode, the first non-empty sentence of a group gives `g = (1/n) * acc` as above, and every further non-empty sentence s adds `g[i] = fma(1/n_s, acc_s[i], g[i])` in source order (the mean is not rounded before it is added); a group of one sentence therefore equals that sentence's row bit for bit, and a group of empty sentences is the zero row. Stages 4 and 5 then run once per group on this row (normalizing the group's mean, not summing normalized sentences). The sentences of a group are consecutive in the source, and a worker always gets whole groups, so a group never spans batches or workers. Group mode cannot be combined with shards (2.1 raises for `shard=` with groups), so the fixed shard boundaries never cut a group.
- **`dot16`**, the dot product of every row reduction: 16 partial sums start at +0.0; index i goes into lane `i % 16` by fma in index order; the lanes are then added 8, 4, 2 and 1 lanes apart, and lane 0 is the result. It vectorizes on NEON, AVX2, AVX-512 and Metal. A lane can hold -0.0 (a product that rounds to zero keeps its sign), so implementations must not pad the reduction with zeros: a tail shorter than a vector updates only its own lanes.
- **`remove_components@1`** with k unit components `u_j` (1 <= k <= d) and optional weights `w_j`: `c_j = dot16(row, u_j)`, times `w_j` if given; `p` starts at +0.0 and takes `p[i] = fma(c_j, u_j[i], p[i])` for j ascending; the result is `row - p`. All coefficients use the unmodified row, as 1.x does.
- **`project@1`** with a basis of k orthonormal rows `b_j` (1 <= k <= d): the new row has the k values `dot16(row, b_j)`; it reduces the dimension from d to k. `fit` takes the basis from the rows after `remove_components`: with `M = I - Σ_j w_j u_j u_jᵀ` (w_j = 1 without weights) their Gram matrix is `Mᵀ G M`, computed in float64 from the Gram matrix G that `fit` already has, without another pass over the rows. Components removed with weight 1 are gone from every row, so the eigenproblem is solved on their orthogonal complement (`Qᵀ Mᵀ G M Q`, Q an orthonormal basis of it) and k is at most d minus their number; `fit` raises if one of the k axes has a variance below the rounding bound of the Gram matrix, instead of picking a direction from noise.
- **`normalize@1`**: `s = dot16(row, row)`; an all-zero row stays zero; a nonzero row whose `s` underflows to 0 or overflows raises; otherwise every value is multiplied by `1 / sqrt(s)`.
- **Step lists.** Any subset of `remove_components`, `project` and `normalize`, in this order and each at most once; anything else is rejected when an encoder is built or loaded. In a folded encoder only `normalize` remains. After stage 4 every row must be finite, otherwise the row id is reported.
- **Stage 5, storage.** float32 as is; float16 rounds to nearest even and raises beyond +-65504; int8 is symmetric per row: `s = absmax / 127` in float32, stepped down once with `nextafter` if `127 * s` overflows, `code = clip(rint(x / s), -127, 127)`; a row whose scale is 0 (a zero row, or one whose largest value divided by 127 underflows) has codes 0. Reading gives `code * s` with one rounding. These are the 1.x rules of `fse.models._quantize`, which the reference reproduces bit for bit.
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

Aggregations and steps are written `"<name>@<version>"` (name `[a-z][a-z0-9_]*`, version a decimal integer without leading zeros), for example `weighted_mean@1`, `remove_components@1`, `normalize@1`. `transform_version` versions the shared rules above, the identifier versions the rules of one operation. Any change of arithmetic bumps the matching version; a released identifier is never removed or given another meaning within 2.x. Reading an artifact with an unknown name, version or `transform_version`, or a disallowed step list, raises `UnsupportedArtifactError` (a `ValueError`) that names the supported ones; nothing is skipped. Data exported from 1.x carries `transform_version` 1: it can be read and searched, but not extended with a 2.0 encoder.

### Batch

A `Batch` is the input of an aggregation: `first_row` (the row of its first sentence), `offsets` (int64, n + 1 entries, from 0 to the number of tokens), `tokens` (uint32 vocabulary ids, or `OOV_TOKEN` for an unknown FastText word) and, for FastText, `ngram_offsets` and `ngrams` (uint32 buckets, one range per unknown word, in order). Sentence s is row `first_row + s`. Weights are not part of a batch, so a Source does not depend on the encoder; 2.1 adds token ids from Arrow and `TokenIds` as further Sources of the same batches, and group mode adds a target per sentence.

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

`fse._container.SafetensorsContainer`, in the module of both containers, which imports only the standard library, numpy and `UnsupportedArtifactError`, so the 1.x export can copy it.

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
- The fingerprint test vector (`tests/test_container.py`): tensors `vectors` F32 `[[1, -0.0], [0.5, 2]]`, `vocab.bytes` U8 of `"aé"`, `vocab.offsets` I64 `[0, 1, 3]` with metadata `fse.kind=word_vectors`, `fse.format_version=1`, `fse.dim=2`, `fse.dtype=float32` give `sha256:98a86ce422b1d61979da4b088934ebbb5c4f128463ca20123f7508d0f74f32db`.

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

Saved 2.0 artifacts contain no Python class paths. Their persistence API is the set of strings above: `fse.kind` values, identifiers, dtype codes, tensor and metadata names. These are never renamed or reused with another meaning. The public Python API is `fse.__all__`; private modules (with a leading underscore) may move freely. The rule not to rename modules under `fse.models` holds only while master still ships the 1.x models.

### Migration

2.0 has no pickle loader. A last 1.x release, 1.2.0 on the branch `1.x`, adds `export_v2`, which writes the word vectors of a model and its encoder (weights, `oov_weight`, and the learned components as a `remove_components` step) into the 2.0 format, and optionally its sentence vectors as data with `transform_version` 1. uSIF models saved before 1.1 keep their weighting by lambda squared, because the export takes the weights the model applies. The hub word vectors are converted the same way. The hub also gets pretrained, frozen encoders: Average for every vector set, SIF and uSIF for a selection, fitted once on WikiText-103 with the native tokenizer, so `from_pretrained` encodes at once and gives the values of the published tables. A golden manifest and fingerprint of a tiny model, written by 1.2.0 and loaded by 2.0 in the tests, catch drift between the two writers.

### Layout

New 2.0 code goes into private modules of `fse`: `_contract` and `_reference` (contracts and reference), `_codes` (storage arithmetic in blocks), `_container` and `_wordvectors`, the gensim adapter `fse.gensim` (extra `fse[gensim]`), then the runtime and backends, the encoder and recipes, sinks and store, and the thin `train`/`infer` facade of the release. The 1.x modules under `fse.models` are deleted once their replacement is in. gensim is not a dependency of fse: `import fse` and every 2.0 class work without it (tests check it, and a static check keeps the 2.0 modules free of gensim, scipy and smart_open); the 1.x classes are loaded from `fse` on first access and need gensim, which the extra and the test group provide.

### Defaults

Defaults are part of the API and chosen for speed: `workers` is the number of usable CPUs (small inputs run inline in the calling thread), and new encoders, also the pretrained ones on the hub, end with `normalize`, so the cosine of two rows is their dot product. A server that encodes many small requests in parallel sets `workers=1`.

### Baseline

`benchmarks/baseline/` records the values of fse 1.1.0 from PyPI that 2.0 is compared with: STS, an MTEB selection, training throughput, memory and a lifecycle benchmark. Its README holds the tables and the commands to reproduce them.

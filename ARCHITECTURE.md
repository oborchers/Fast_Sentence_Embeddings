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
7. **Post-train hook:** `_post_train_calls` removes principal components for SIF and uSIF. The components are computed exactly from the Gram matrix `X^T X` of all sentence vectors, summed in float64 over blocks of rows and decomposed with `eigh`; the removal works in blocks of rows as well. Both read the vectors once from front to back, so neither needs memory proportional to the number of sentences, and a memmap streams from disk. With identity indices and the kernel, training has summed the Gram matrix already (see *Single pass*), so only the removal reads the vectors again. After `train(update=True)` it reuses the components of the first training for the appended rows only, as `infer` does.

`infer(sentences)` takes a shorter path. It scans the input, allocates a zero-filled output matrix, runs the jobs in the calling thread with the weights learned in `train`, and finishes with `_post_inference_calls`, which removes the stored principal components for SIF and uSIF. It has no preparer, no pre-train hook and no threads.

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

`CorpusFile` (a memory-mapped local file), `TextList` (strings copied once into a UTF-8 buffer with offsets) and a `SplitIndexedList` of `str` or numpy `str_` (copied into such a buffer by every scan, released when the call ends) skip the Python producer. A `SplitIndexedList` with other items, with `__getitem__` or `__iter__` overridden, or with strings that cannot be encoded as UTF-8 (lone surrogates) keeps the Python path, whose sentences the native path reproduces:

1. **Scan:** `scan_sentences` calls their `_scan_native`. The source is split into about `4 × workers` ranges that start at sentence starts, so every sentence belongs to exactly one range; threads validate and count the ranges with `scan_spans` without the GIL and record chunks of 64 KB to 1 MB and at most 65,536 sentences. Prefix sums turn range-local sentence numbers into indices. The chunks and the file's size and modification time are kept on the input until training.
2. **Training:** `Average._job_encoder` returns a `_ChunkFeeder` that hands out the chunks. Each worker runs `run_chunk`: it splits its chunk, looks up every word in a `VocabTable` (open addressing over the UTF-8 bytes of `key_to_index`, built once per call; each 32-byte slot holds the hash, the index and words of up to 16 bytes inline, so a lookup usually reads one cache line) and averages each sentence with `_average_sentence`, the routine encoded jobs use too, all without the GIL. Every sentence has its own row, so no locks are needed.

The input contract is the one of `IndexedLineDocument` and `SplitIndexedList`, and tests compare the vectors bit for bit: a line ends after `\n` and the last needs none; text is decoded strictly as UTF-8; words are split at every code point with `str.isspace()`. Invalid UTF-8 raises the `UnicodeDecodeError` Python raises for that sentence, with its number and byte offset added. The fingerprint of a `CorpusFile` (device, inode, size, modification time) is read from the descriptor that was mapped and compared again before training and after every chunk, so a change that alters one of them raises; a change that keeps all four is not detected. The Python path of `CorpusFile` checks the same fingerprint before and after every iteration. A file truncated while it is mapped can still end the process with SIGBUS on Linux; only a private copy would prevent that. `scan_spans` and `run_chunk` check every range and offset before they release the GIL. A validator replaced on the instance or in a subclass forces the Python scan, and training then reads tuples as well. Without the extension, both classes are iterated in Python like their counterparts.

### List chunks

With two or more workers, `train` lets the workers read an `IndexedList` or `CIndexedList` themselves (`Average._list_chunk_rows`): the class or a subclass that keeps its `__getitem__` and `__iter__`, `items` a `list`, and a scan that saw only exact `list` sentences of exact `str` (`scan_indexed_sentences` reports it). The producer hands out `_ListChunk` row ranges of about 65,536 tokens (256 to 8,192 sentences) and reads nothing itself. Each worker runs `run_list_chunk`: holding the GIL, it copies the UTF-8 bytes of the tokens of its chunk into private buffers (borrowed references, no Python code runs meanwhile, except reading a custom index first); without the GIL it looks them up in the `VocabTable` and averages with `_average_sentence`. The vectors equal the `JobEncoder`'s; tests compare them. One worker, `infer`, other types, `ndarray` items and subclasses that change how items are read keep the `JobEncoder`, which is faster for a single thread. A list that changed after the scan (another type, a shorter list, an index out of range or, with prefill, not at its position) raises instead of being read differently.

### BLAS threads

Workers call BLAS for the Gram block products, and SIF and uSIF remove components with BLAS after training and `infer`. `train` and the removal after `infer` run inside `utils.blas_threads(workers)`, which limits BLAS (threadpoolctl) to `usable_cpus() // workers` threads, at least one, and restores the previous limits afterwards; without it, OpenBLAS started its own threads in every worker. The limit holds for the whole process during the call. `remove_principal_components(..., workers=n)` removes its blocks of `REMOVAL_BLOCK_ROWS` rows in a thread pool; every block is computed as before, so the result does not depend on `n`.

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

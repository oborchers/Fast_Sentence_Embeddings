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
```

**Module paths under `fse.models` are part of the API.** Models are saved with gensim's `SaveLoad`, which pickles class paths such as `fse.models.average.Average`. Renaming or moving a module breaks loading of every saved model. New internal code goes into new private modules; existing ones stay where they are.

## Data Flow

`train(sentences)` and `infer(sentences)` take a re-iterable of `(tokens, index)` tuples. The `inputs` classes produce exactly that shape. `index` is the row of the output matrix the sentence contributes to.

1. **Scan:** `scan_sentences` makes one pass over the input. It counts sentences and words, finds the largest index, validates every item and reports whether all indices are unique and whether every index is the position of its sentence (`identity_indices`). Indices that are not the positions of their sentences are tracked in a bitmap (`IndexTracker` in the private module `fse.models._index_tracker`), which grows by doubling up to `max(BITMAP_MIN_BYTES, one byte per sentence)`; indices beyond it wait as int64 until it may double, and are sorted at the end if it never may cover them. With the extension the loop runs in Cython (`scan_indexed_sentences`) with the same checks, messages and tracker rules; a subclass that overrides `_check_indexed_sent_valid` keeps the Python loop.
2. **Sanity and memory:** `_check_pre_training_sanity` checks the vectors and weights; `estimate_memory` warns when the result will not fit into RAM.
3. **Prepare:** `BaseSentence2VecPreparer` allocates the sentence vectors (`max_index` rows, initialized to `EPS`), in RAM or as a memmap. With identity indices and the kernel, the rows are not initialized here: every job sets its own contiguous rows to `EPS` right before it averages them (see *Single pass*). With `train(update=True)` it appends the rows instead, and every sentence index is shifted by the previous number of rows. Rows are initialized in blocks. An in-memory update copies all existing rows into a new array (`vstack`); a memmap grows in place.
4. **Pre-train hook:** `_pre_train_calls` computes the word weights. All weights are 1 for Average; SIF and uSIF derive them from word frequencies.
5. **Jobs:** `_train_manager` starts one producer thread and `workers` worker threads. The producer batches the input into jobs of up to `batch_words` words and `batch_words` sentences; a single longer sentence forms a job of its own. With the extension, `_kernel_context` builds a `KernelContext` once per call, and the producer reads, batches and encodes the input in one pass with a `JobEncoder` while it holds the GIL. Each job copies its data out of the encoder's reusable buffers: word indices, sentence offsets, target rows, and the UTF-8 bytes of unknown FastText words. Without the extension, or when a subclass or instance replaces `_do_train_job`, the workers call `_do_train_job` on lists of tuples instead (`train_average_np` without the extension).
6. **Kernel:** `run_job` averages an encoded job without the GIL: it hashes the ngrams of unknown FastText words in C, sums the weighted word vectors of each sentence and adds the average into its target row with BLAS `saxpy`. `infer` and `train_average_cy` use the same encoder and kernel in the calling thread. The producer still holds the GIL while it reads and encodes the input, so it bounds the throughput once two to four workers are busy.
   For `CorpusFile` and `TextList`, scan and training work on bytes instead (see *Native inputs*).
7. **Post-train hook:** `_post_train_calls` removes principal components for SIF and uSIF. The components are computed exactly from the Gram matrix `X^T X` of all sentence vectors, summed in float64 over blocks of rows and decomposed with `eigh`; the removal works in blocks of rows as well. Both read the vectors once from front to back, so neither needs memory proportional to the number of sentences, and a memmap streams from disk. With identity indices and the kernel, training has summed the Gram matrix already (see *Single pass*), so only the removal reads the vectors again. After `train(update=True)` it reuses the components of the first training for the appended rows only, as `infer` does.

`infer(sentences)` takes a shorter path. It scans the input, allocates a zero-filled output matrix, runs the jobs in the calling thread with the weights learned in `train`, and finishes with `_post_inference_calls`, which removes the stored principal components for SIF and uSIF. It has no preparer, no pre-train hook and no threads.

### Single pass

For a memmap larger than the RAM, every pass over the sentence vectors goes to disk. With identity indices (index i for the i-th sentence, as for every input class except the `C*` lists with custom indices) and the kernel, training saves up to two separate full passes. The state of one such call is a `SinglePass` in the private module `fse.models._single_pass`, held by the model only during `train`, like the row locks.

- **Prefill** (every model, also for `update=True`): `KernelContext.prefill_rows` makes `run_job` and `run_chunk` set the rows of their job to `EPS` before averaging, instead of `_fill_eps` writing all new rows in advance. The arithmetic is the same, so the vectors are bit-identical. The producer checks before a worker writes that every job starts at the next row and covers contiguous rows, so jobs neither overlap nor leave gaps, and training checks at the end that the jobs covered every new row. `run_chunk` checks that a chunk holds exactly the announced number of sentences. If allocation succeeded but the pre-train hook, the threads or a job fail, training sets the new rows to `EPS` and keeps the existing rows, so no uninitialized row remains; an error in the post-train hook, for example a non-finite Gram matrix, leaves the trained rows as they are.
- **Gram matrix** (SIF and uSIF with `components > 0`, without `update=True`, and without a replaced `_post_train_calls`): after each job the worker reports its rows to an `OrderedGram`; the worker that completes a block of `REMOVAL_BLOCK_ROWS` rows computes the block product while the rows are cached, and the products are summed in block order. Blocks, products and summation order are those of `gram_matrix`, so the components are bit-identical to a separate pass for any number of workers. Products that wait for a missing predecessor are bounded by `PENDING_BUDGET_BYTES`, reserved before they are computed; beyond it, a finished block is recomputed from its rows when it is summed. One thread at a time sums, outside the lock, so other workers keep reporting. `_post_train_calls` receives the finished matrix as `gram`.

Other indices, the numpy fallback and a replaced `_do_train_job` take the separate passes. An update that computes new components from old and new rows prefills the new rows but reads all rows for the Gram matrix.

### Native inputs

`CorpusFile` (a memory-mapped local file) and `TextList` (strings copied once into a UTF-8 buffer with offsets) skip the Python producer:

1. **Scan:** `scan_sentences` calls their `_scan_native`. The source is split into about `4 × workers` ranges that start at sentence starts, so every sentence belongs to exactly one range; threads validate and count the ranges with `scan_spans` without the GIL and record chunks of 64 KB to 1 MB. Prefix sums turn range-local sentence numbers into indices. The chunks and the file's size and modification time are kept on the input until training.
2. **Training:** `Average._job_encoder` returns a `_ChunkFeeder` that hands out the chunks. Each worker runs `run_chunk`: it splits its chunk, looks up every word in a `VocabTable` (open addressing over the UTF-8 bytes of `key_to_index`, built once per call) and averages each sentence with `_average_sentence`, the routine encoded jobs use too, all without the GIL. Every sentence has its own row, so no locks are needed.

The input contract is the one of `IndexedLineDocument` and `SplitIndexedList`, and tests compare the vectors bit for bit: a line ends after `\n` and the last needs none; text is decoded strictly as UTF-8; words are split at every code point with `str.isspace()`. Invalid UTF-8 raises the `UnicodeDecodeError` Python raises for that sentence, with its number and byte offset added. The fingerprint of a `CorpusFile` (device, inode, size, modification time) is read from the descriptor that was mapped and compared again before training and after every chunk, so a change that alters one of them raises; a change that keeps all four is not detected. The Python path of `CorpusFile` checks the same fingerprint before and after every iteration. A file truncated while it is mapped can still end the process with SIGBUS on Linux; only a private copy would prevent that. `scan_spans` and `run_chunk` check every range and offset before they release the GIL. A validator replaced on the instance or in a subclass forces the Python scan, and training then reads tuples as well. Without the extension, both classes are iterated in Python like their counterparts.

## Rules

### The numpy fallback matches the Cython kernel

`train_average_np` and `train_average_cy` must produce the same vectors, within float tolerance, for every input, including FastText out-of-vocabulary words and many-to-one index mappings. Tests compare both paths. A change to one path needs the same change and a parity test for the other.

### One kernel, the same result for any number of workers

`train` and `infer` encode with `JobEncoder`, `train_average_cy` with its wrapper `encode_job`, and all three average with `run_job`; there is no second Cython code path to keep in sync. Every sentence is summed in the same order whatever the job size or worker count, so with one row per sentence the vectors are bit-identical for 1 to n workers, which tests check. The kernel has no length limits: `batch_words` and `batch_ngrams` only size jobs and cap the ngrams of unknown FastText words, as in the numpy fallback.

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

### float32 throughout

Word vectors, weights and sentence vectors are `float32` (`REAL`). The kernel takes raw pointers, so arrays must be `float32` and C-contiguous.

## Build

The extension is compiled from `average_inner.pyx` at build time with Cython 3 against numpy 2 headers; generated C is never committed. Wheels built this way also run with numpy 1.x. `make rebuild-ext` rebuilds it in the development environment after edits to `.pyx` or `.pxd` files.

## Tests

- Tests mirror `src/`: `tests/models/test_average.py` tests `src/fse/models/average.py`.
- Tests use pytest fixtures from `tests/conftest.py` and write files only into `tmp_path`; `tests/test_data` is read-only input.
- Expected values never depend on vocabulary positions, because gensim versions order words with equal counts differently. Derive them from `key_to_index`.
- Tests that download from the Hugging Face Hub are marked `network`.
- pytest treats warnings as errors. Narrow, commented ignores are allowed only for third-party warnings.

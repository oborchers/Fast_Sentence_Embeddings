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
```

**Module paths under `fse.models` are part of the API.** Models are saved with gensim's `SaveLoad`, which pickles class paths such as `fse.models.average.Average`. Renaming or moving a module breaks loading of every saved model. New internal code goes into new private modules; existing ones stay where they are.

## Data Flow

`train(sentences)` and `infer(sentences)` take a re-iterable of `(tokens, index)` tuples. The `inputs` classes produce exactly that shape. `index` is the row of the output matrix the sentence contributes to.

1. **Scan:** `scan_sentences` makes one pass over the input. It counts sentences and words, finds the largest index and validates the first item.
2. **Sanity and memory:** `_check_pre_training_sanity` checks the vectors and weights; `estimate_memory` warns when the result will not fit into RAM.
3. **Prepare:** `BaseSentence2VecPreparer` allocates the sentence vectors (`max_index` rows, initialized to `EPS`), in RAM or as a memmap.
4. **Pre-train hook:** `_pre_train_calls` computes the word weights. All weights are 1 for Average; SIF and uSIF derive them from word frequencies.
5. **Jobs:** `_train_manager` starts one producer thread and `workers` worker threads. The producer batches the input into jobs of at most `batch_words` words. Each worker calls `_do_train_job`, which runs `train_average_cy` (or `train_average_np` when the extension is missing).
6. **Kernel:** `populate_*_config` turns tokens into word indices while holding the GIL. The averaging runs without the GIL and adds each sentence average into its target row with BLAS `saxpy`.
7. **Post-train hook:** `_post_train_calls` removes principal components for SIF and uSIF.

## Rules

### The numpy fallback matches the Cython kernel

`train_average_np` and `train_average_cy` must produce the same vectors, within float tolerance, for every input, including FastText out-of-vocabulary words and many-to-one index mappings. Tests compare both paths. A change to one path needs the same change and a parity test for the other.

### Many-to-one mappings accumulate

Several sentences may share one index. Each sentence average is added into the row; a row is not overwritten.

### Hooks keep their signatures

Subclasses override `_check_parameter_sanity`, `_pre_train_calls`, `_post_train_calls`, `_post_inference_calls` and `_check_dtype_santiy`, all of which accept `**kwargs`. Tests replace `_do_train_job` on instances, so the job path must keep calling it.

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

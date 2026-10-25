# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/).

## [1.1.0] - Unreleased

fse is now in maintenance mode: compatibility fixes only, no new features.

### Added

- Pre-built wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x86_64) on Python 3.11 to 3.13.
- Support for numpy 2, Cython 3, current scipy, scikit-learn, gensim 4.4 and huggingface-hub 2.x.
- A clear `ImportError` when gensim < 4 is installed.
- `__all__` in `fse` and `fse.models`.

### Changed

- Arguments of the public entry points are validated strictly with pydantic: the model, `SentenceVectors` and `IndexedLineDocument` constructors, `train`, `infer`, `most_similar`, the `similar_by_*` methods and `from_pretrained`. Wrong types and out-of-range values such as `workers=0` or `alpha=-1` raise a `pydantic.ValidationError` (a `ValueError`) at the call, and nothing is coerced, so `workers="2"` is an error. Mapfile paths accept `str` and `pathlib.Path`.
- New dependency: `pydantic>=2.5`.
- Unknown keyword arguments to `Average` raise a `TypeError` instead of being silently ignored, as they already did for `SIF` and `uSIF`.
- The input classes reject sets with a `TypeError`. A set has no order, so the sentence indices would be arbitrary; pass `sorted(...)` instead.
- Every sentence is checked to be a list of `str`, not only the first one.
- `set_madvise_for_mmap` is no longer called on every model construction and load; it never applied any advice.
- Requires Python 3.11 or newer. Python 3.10 reached its end of life in October 2026; installations on 3.10 and older keep fse 1.0.0. Python 3.14 is not supported yet, because gensim does not build there.
- Minimum versions: numpy 1.26.4, scipy 1.11.4, scikit-learn 1.3.2, gensim 4.3.3, smart_open 6.4.0, wordfreq 3.1.0, huggingface-hub 0.23.0, psutil 5.9.0.
- Packaging moved to `pyproject.toml`; the version now comes from git tags.
- The warning about a missing C extension is a `RuntimeWarning`.

### Fixed

- **uSIF results change.** uSIF removed each principal component weighted by λ² instead of λ, unlike the reference implementation ([kawine/usif](https://github.com/kawine/usif)). Embeddings from uSIF models trained with earlier versions differ from 1.1.0.
- An exception in a worker thread or in the input iterable made `train()` hang forever. It is now raised in the calling thread after all threads have stopped.
- Only the first input item was validated. A negative or non-integer index later in the input could hang training or silently write into the wrong row. Every item is now checked, and `numpy` integers are accepted as indices, so `CIndexedList` works with an ndarray `custom_index` as documented.
- Generators and other one-shot iterators returned all-zero vectors from `infer` and failed with a misleading error in `train`, because the input is read twice. They are now rejected with a clear `TypeError`.
- `save()` with `wv_mapfile_path` set the word vectors of the live model to `None`, so later `infer` calls read invalid memory, and a second save and load cycle failed. The vectors are now only left out of the pickle.
- The Cython kernel now checks dtype, memory layout, writeability and shapes of the arrays it accesses through raw pointers, and the index of every sentence against the target rows, instead of reading or writing invalid memory.
- The FastText ngram buffer of `BaseSentence2VecModel` could be smaller than the kernel expects when `batch_words` or `batch_ngrams` were set below the defaults.
- `most_similar` and `init_sims` failed with a `TypeError` for sentence vectors stored on disk (`sv_mapfile_path`). The normalized vectors are now written into the mapfile in blocks.
- `init_sims(replace=True)` did not replace the vectors, and `get_vector(use_norm=True)` failed before `init_sims` was called.
- Normalizing a sentence without known words returned NaN; it now stays a zero vector.
- `train(update=True)` wrote the new sentences into the existing rows and left the appended rows empty. New sentences keep their indices starting at 0 and are stored after the existing rows. SIF and uSIF remove the components of the first training from the appended rows only, instead of removing them from all rows again.
- SIF and uSIF with `components=0` could not be trained a second time.
- In the Cython path, a FastText out-of-vocabulary token without any ngrams (for example a token shorter than `min_n - 2` characters) added the vector of the first vocabulary word. It now adds nothing, as in the numpy fallback.
- Saving a model with `sv_mapfile_path` wrote the sentence vectors a second time into `.npy` files, and loading returned them in RAM instead of reopening the mapfile.
- `x in sentence_vectors` raised a `KeyError` for non-integer values and accepted negative indices; it now returns `False` for both. The `KeyError` for a missing index names the index.
- `str(model)` reported the number of sentences as `vector_size`.
- Scanning failed for an empty input and for corpora with less than one word per sentence on average.
- The numpy fallback counted empty sentences as effective sentences, unlike the Cython path.
- The SVD sample for SIF and uSIF is seeded, so repeated runs give the same components.
- Interrupting training, for example with Ctrl+C, could leave worker threads blocked on a full queue.
- With several workers, sentences mapped to the same row (for example `CIndexedList` with repeated indices) could lose updates when two workers added into the row at the same time. Rows shared by several sentences are now updated under locks; unique indices stay lock-free.
- `pip install fse` failed on Python 3.11 and newer, because the shipped C source was generated by an old Cython version.
- Building under pip or poetry crashed with `AttributeError: 'dict' object has no attribute '__NUMPY_SETUP__'`.
- `import fse` failed in a clean environment, because `requests` was imported but not declared.
- `Vectors.from_pretrained` and `FTVectors.from_pretrained` raise `ValueError` again for a model that does not exist on the hub.
- The SVD for SIF and uSIF sampled a fixed one million vectors and ignored `cache_size_gb`, which crashed for corpora between the sample size and one million sentences.
- The numpy fallback now accumulates many-to-one sentence mappings like the Cython routine instead of overwriting them.
- Training crashed with `ZeroDivisionError` when a run finished faster than the timer resolution, for example on Windows.
- Wheels and source distributions no longer install a top-level `test` package.
- `fse.__version__` reported 0.2.0.

### Removed

- Support for Python 3.6 to 3.10.
- The `requests` and `dataclasses` dependencies.

## [1.0.0] - 2022-04-10

- Added support for gensim>=4. This library is no longer compatible with gensim<4. For migration, see the [gensim migration guide](https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4).
- `size` argument is now `vector_size`.

## [0.2.0] - 2021-12-03

- Added `Vectors` and `FTVectors` class and hub support by `from_pretrained`.
- Extended benchmark.
- Fixed zero division bug for uSIF.
- Moved tests out of the main folder.
- Moved sts out of the main folder.

## [0.1.17] - 2021-12-01

- Fixed dependency issue where you cannot install fse properly.
- Updated readme.
- Updated travis python versions (3.6, 3.9).

## [0.1.15]

Changes from 0.1.11:

- Fixed major FT Ngram computation bug.
- Rewrote the input class. Turns out NamedTuple was pretty slow.
- Added further unittests.
- Added documentation.
- Major speed improvements.
- Fixed division by zero for empty sentences.
- Fixed overflow when infer method is used with too many sentences.
- Fixed similar_by_sentence bug.

[1.1.0]: https://github.com/oborchers/Fast_Sentence_Embeddings/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/oborchers/Fast_Sentence_Embeddings/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/oborchers/Fast_Sentence_Embeddings/compare/v0.1.17...v0.2.0
[0.1.17]: https://github.com/oborchers/Fast_Sentence_Embeddings/releases/tag/v0.1.17

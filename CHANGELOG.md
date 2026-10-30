# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

fse 2.0 is in development on master; it changes the API and the file format (see ARCHITECTURE.md, section 2.0). Entries collect here until the release.

### Added

- `fse.WordVectors`: word vectors saved as one `.safetensors` file with checksums and a fingerprint, without pickle, memory-mappable, stored as float32, float16 or int8 and read as float32, with FastText ngrams. Loading int8 vectors needs no float32 copy, and other safetensors readers can open the files.
- `fse.gensim.from_gensim` converts gensim `KeyedVectors`, `Word2Vec` and `FastText` into `WordVectors`.
- The native tokenizer `fse_words@1`: lowercase, split at whitespace and punctuation, and expand contractions like the preprocessing of the published 1.x tables, specified in ARCHITECTURE.md and fixed for 2.x. It gives the tokens of that preprocessing on every STS Benchmark sentence and on 99.98 % of the WikiText paragraphs, and runs without the GIL from raw text to token ids.
- `Encoder.from_pretrained` loads an fse encoder, or converts a Model2Vec model (such as `minishlab/potion-base-8M`), from a local path or the Hugging Face Hub (extra `fse[hub]`, huggingface-hub 1.0 or newer). The vectors equal those of model2vec 0.10.0 within float32 rounding, with its truncation by default (`truncate=False` reads texts whole). Files of the Hub are checked against their commit (`lfs.sha256` for every safetensors file, the git blob id for the rest), links of the cache are followed only into its blobs, and only float32 models are read; vocabulary-quantized models, heads and models that are no static embeddings are refused with the reason.
- `Encoder.from_pretrained` converts sentence-transformers models of one StaticEmbedding (such as `sentence-transformers/static-retrieval-mrl-en-v1` and `sentence-transformers/static-similarity-mrl-multilingual-v1`) with the rules of sentence-transformers 6.2.0: no special tokens, every id pooled (`[UNK]` kept), texts read whole or by the plain truncation of the tokenizer.json, and vectors equal to its `encode` within float32 rounding. `truncate_dim=` keeps the first dimensions (Matryoshka), recorded in `fse.source` as `dims`; `normalize=True` adds fse's `normalize@1` after the cut. Models with a Normalize module (sentence-transformers divides by `max(norm, 1e-12)`), a Dense module, other modules, prompts or non-float32 tables are refused with the reason. Repositories without a `config.json` load with `local_files_only` too, files in folders of a repository are read, and no folder of a model or cache may be a link.
- The native tokenizer `wordpiece@1`: the WordPiece tokenizer of BERT vocabularies (POTION/Model2Vec, the static models of sentence-transformers) with the ids Hugging Face tokenizers 0.23.3 gives without special tokens, its added tokens matched on the raw text, its normalization with the Unicode versions of tokenizers (categories of 8.0.0, decomposition of 9.0.0, lowercase of 17.0.0), specified in ARCHITECTURE.md and fixed for 2.x. It runs in the native text sources without the GIL; its parameters are data of the model, in the new keys `fse.tokenizer_config`, `fse.unknown` (drop or keep the unknown id) and the optional length rule `fse.max_tokens` and `fse.max_chars` of encoder 1, with the provenance keys `fse.source` and `fse.license` (format version 1 unchanged). Models that use it are converted by the Model2Vec reader, which follows.
- The tokenizer `unigram@1`, specified in ARCHITECTURE.md: the Unigram tokenizer of SentencePiece vocabularies with the ids Hugging Face tokenizers 0.23.3 gives without special tokens, for `minishlab/potion-multilingual-128M`: its Precompiled charsmap searched per grapheme of Unicode 17.0.0 as tokenizers does, the Replace and Strip steps, Metaspace, and the Viterbi over float64 scores. So far its reference and the reading of a Unigram tokenizer.json, checked against tokenizers on every code point and 884,948 texts (MASSIVE in 51 locales, WikiText, STS22, random strings); the native tokenizer and Model2Vec models with it follow, so `from_pretrained` still refuses such a model.
- `fse.Encoder`, a finished model: a table of word rows (with FastText ngrams, as float32, float16 or int8), its tokenizer and whether every row is normalized, saved as one self-contained `.safetensors` file (`save`, `Encoder.load(path, mmap=)`). `Encoder.from_word_vectors(wv, tokenizer=, normalize=True, max_ngrams=, oov_weight=)` and `Encoder.from_arrays(vectors, words=, ...)` make one; every word weighs 1, so weights, removed components or projections of a model are folded into its rows before. `Encoder.encode` takes lists of sentences, a path of a text file, token lists (read in one native pass) or any iterable, in parallel, and returns float32 or float16 rows, or `fse.QuantizedRows` (int8 codes with a scale per row) computed in the workers at about the speed of float32.
- Stores of sentence vectors. `fse.ShardSink` writes the rows of an encoder straight into a store of immutable shards while it encodes (`encoder.encode(texts, sink=sink)`): a directory of plain `.npy` files that `np.load(mmap_mode="r")` opens, as float32, float16 or int8, with appends as new shards. A shard is published whole once its rows are written and synced, and `durable_rows` says how many rows are safe, so a stream confirms its input only up to there; after an error the finished rows are kept. `Encoder.encode_batches` encodes an endless stream batch by batch (as a queue consumer polls it). `fse.VectorStore` opens a store and searches it on the stored float16 or int8 rows, without a float32 copy, and scores it with linear weights: with `mode="exact"` the cosines, scores and order of the reference bit for bit, with `mode="blas"` faster for many queries, and by default (`"auto"`) the faster of both for the number of queries. `fse.Sink` documents the interface for other sinks, such as a vector database; `fse.MemorySink` keeps the rows in memory.
- A collection passed to `encode` that mixes str and token lists raises naming both rows.
- `fse.MemorySink.result()` gives int8 rows as `fse.QuantizedRows`, as `encode` does.
- `fse.FastTextParams` (for `WordVectors.from_arrays` of FastText vectors), `fse.CorruptArtifactError` and `fse.UnsupportedArtifactError` are public. Stored rows are `WordVectors.stored_vectors` and `stored_ngrams` (codes for float16 and int8), decoded rows `get_vectors` (`VectorStore.get_vectors(start=, stop=)` too), and a store or sink names the encoder it takes by `encoder_fingerprint`: a custom sink names it so for its check to apply. `WordVectors` are made only with `from_arrays` and `load` (the constructor raises), and the internal state of `VectorStore` (`info`, `shards`) and `ShardSink` (`info`, `reserved_rows`, `committed_rows`) is not public.
- The optional backend `"mlx"` (`pip install "fse[mlx]"`, macOS 14 or newer on Apple silicon): `encode` and `encode_batches` with `backend="mlx"` average the word vectors in a Metal kernel within the tolerance contract of device backends (ARCHITECTURE.md), including for float16 and int8 word vectors and FastText. The kernel keeps the CPU's order of the sums and its fused multiply-adds, so rows without subnormal intermediate values have the bits of the cpu backend. With `normalize` (the only step) the kernel also normalizes (up to 1,024 dimensions when they are a multiple of four, else up to 256; wider rows are normalized on the CPU); the storage dtypes and the store run on the CPU. An encoder with more than 65,536 dimensions, or whose weights, scales or word vectors are or may give subnormal factors or values, which the GPU flushes to zero, runs on the CPU. A row that is not finite, a sentence of 2**24 or more tokens and buckets, and a row whose error bound cannot exclude another result (sums near the float32 overflow, a row zero or tiny before `normalize`, values near the float16 range, whatever the storage dtype) come from the CPU, which computes the task again and raises the CPU's error if there is one, while the other rows keep the GPU's result. The first such fallback logs a warning. Without MLX or Metal the backend raises an error that says why.
- Linear heads: `fse.Classifier` and `fse.Regressor` hold an encoder and one linear head `z = A x + b` trained elsewhere; fse trains none. `from_sklearn` reads a fitted scikit-learn estimator by its attributes, without importing scikit-learn (LogisticRegression and LogisticRegressionCV with the probability rule of the installed version, SGDClassifier, LinearSVC, the ridge classifiers, Perceptron, the linear regressors, and a Pipeline with an optional `StandardScaler`; a `Normalizer` is refused, as its rule for tiny rows is not `normalize@1`), `from_arrays` takes A, b, the labels and the link. They predict as scikit-learn names it (`predict`, `predict_proba`, `decision_function`, `classes_`) from texts, files, streams or the rows of a `fse.VectorStore`, and `predict_batches` predicts a stream batch by batch, optionally with the encoder's own vectors from the same pooling (`return_vectors=True`, which a `fse.ShardSink` can store). A head trained on raw means (an encoder with `normalize=False`) folds into the table on first use, so it predicts faster than `encode`; one trained on normalized vectors runs on the full mean in the worker. `save` writes the complete model or, for a head on raw means, a standalone file of the folded table and the bias with the same predictions (file kind `linear_head`, format 1). The logits and probabilities are bit-identical across CPU backends, worker counts and input classes, with the new identifiers `linear@1`, `exp@1` (an exp defined by its operations, within 0.94 ulp of exp on every float32 input it takes), `sigmoid@1`, `softmax@1` and `ovr@1`; the folded path and the vectors' path of a head on raw means agree within rounding, not bit for bit. The mlx backend does not run heads yet and says so.
- Python 3.14 and 3.15, with pre-built wheels. On 3.14 and newer, `fse.gensim` needs gensim, which does not build there yet; everything else works.

### Changed

- The native `wordpiece@1` is faster with the same ids: the scan for added tokens jumps to the bytes they start with, ASCII text is normalized and pre-tokenized through tables derived from the Unicode table (8 bytes at a time where it can), the length rule counts 8 ASCII bytes at a time, and a word that is no token is split by a byte trie of the vocabulary (built at load, never stored: about 14 ms and 0.7 MiB for potion-base-8M, 32 ms and 1.6 MiB for potion-base-32M) instead of one lookup per shorter candidate. On an M1 Pro the tokenizer alone takes WikiText-103 2.16 and MASSIVE 1.71 times as fast at one thread (1.33 and 1.44 at ten), and `Encoder.encode` of potion-base-8M 1.64 and 1.37 times as fast at one worker (1.12 and 1.28 at ten). Every id is unchanged; nothing new is persisted.
- `fse_words@1` tokenizes plain words (only a-z and 0-9) without the rules of step 5, unless their length and first letter could make them one of the contractions without an apostrophe (`cannot`, `gimme`, `gonna`, `gotta`, `lemme`, `wanna`): text to batches is about 20 % faster on one thread (WikiText-103 paragraphs 182 to 145 ms, STS Benchmark sentences 106 to 86 ms on an M1 Pro); the tokens are unchanged.
- On macOS a `ShardSink` publishes a shard with two full syncs (`F_FULLFSYNC`, which flushes the drive cache) instead of six or seven, with the same durability: the files and the manifest are sent to the drive with `fsync`, one full sync of the shard's directory makes them durable before its rename, and one of the store the rename. Linux and Windows sync as before.
- Encoding with normalization (every encoder by default) is about 10 % faster: it runs in place on the pooled rows, without two copies. The inverse norms a store keeps of every row are 2.4 (float32) to 3.2 times (int8) as fast to compute. Every vector is unchanged bit for bit.
- A list or tuple of str passed to `encode` is converted to UTF-8 in parallel native threads instead of one `str.encode` per sentence: 400,000 sentences of 20 tokens take 3 ms instead of 70 ms, and `Encoder.encode` of them on the cpu takes 0.64 of its time at 8 workers and 0.95 at 1, of sentences of 100 tokens 0.83 at 8 workers (M1 Pro). Subclasses of str, numpy arrays and invalid input (a row that is no str, a lone surrogate) take the former path, so the errors of invalid input are raised as before; every vector is unchanged. On macOS the join of token lists no longer keeps about the size of the joined text resident after every call.
- Pooling float32 word vectors on arm64 (NEON) keeps the sums of 32 values at a time in registers over up to 64 words instead of reading and writing them for every word: the kernel is 1.12 to 1.25 times as fast on one thread for 100 to 1024 dimensions (1.21 at 300), the runtime on sentences of about 12 words 1.12 times at 1 worker and 1.08 at 8, and `Encoder.encode` of WikiText paragraphs and documents about 1.11 times at 1 worker (M1 Pro). Every vector is unchanged bit for bit; float16 and int8 vectors, x86 and more than 1024 dimensions keep the previous loop.
- gensim is no longer a dependency: install `fse[gensim]` for `fse.gensim`.
- Building from source needs MSVC on Windows, or GCC or Clang elsewhere; MinGW is refused, because its `fmaf` rounds twice.
- Checksums and fingerprints of every artifact are xxh3-128 (`xxh3-128:` and 32 hex digits) instead of sha256, computed by one harness. Artifacts written by a pre-release of 2.0 with sha256 checksums raise `UnsupportedArtifactError` and must be written again; there is no migration. The values identify content against accidents, not against an attacker (ARCHITECTURE.md, "Identities").
- The only dependencies are numpy, pydantic and xxhash (`xxhash>=3.2.0`). scipy, smart_open, wordfreq, huggingface-hub, psutil and threadpoolctl are no longer needed; the Hugging Face Hub returns with the 2.0 loader.

### Removed

- The 1.x API: the 1.x models `Average`, `SIF` and `uSIF`, `SentenceVectors`, the input classes, `Vectors` and `FTVectors`, and the modules `fse.models`, `fse.inputs` and `fse.vectors`. Accessing one of the removed 1.x names on `fse` raises an `AttributeError` that points to fse 1.1.0 on PyPI and the branch `1.x`.
- Training. fse 2.0 is an inference engine for finished static embeddings and learns nothing: the 2.0 recipes `Average`, `SIF` and `USIF` with `fit` (word weights from counts, the Gram matrix and its components, projections, folding), `fse.Corpus` (its reason was the passes of fit), stored recipes and learned components, word counts in `WordVectors` (`counts=`, `.counts`, and `from_gensim(counts=)`), and encoders that reference a separate table (`Encoder.load(path, word_vectors)`, `Encoder.folded`, `Encoder.recipe`, `Encoder.input_dim`): every encoder holds its own table, so two encoders of the same word vectors hold two copies. The postprocessing steps `remove_components@1` and `project@1` are gone from the contract and the kernel; `normalize@1` remains. SIF and uSIF tables are made offline with the 1.1.0 recipe, in a notebook that follows before the release. Encoders and word vectors written by a development version of 2.0 before this change (with recipes, weights, learned or folded steps, counts, or without their own table) raise `UnsupportedArtifactError`: rebuild them.
- The 1.x notebooks; they stay on the branch `1.x`.

### Fixed

- An interrupt (Ctrl+C) while a `ShardSink` closes no longer leaves threads running and temporary shard directories behind. When `close` runs in the main thread with Python's default SIGINT handler, the work still waiting is dropped, the piece each thread runs is finished (a shard being published is published if all its rows were written), further interrupts are deferred, and one KeyboardInterrupt is raised once every thread has ended and the temporary directories are removed; a piece that blocks, such as a write to a drive that hangs, still keeps `close` waiting. With another handler, or from another thread, nothing is deferred.

## [1.1.0] - 2026-10-27

1.1.0 is the last feature release of the 1.x line; 1.x receives bug and compatibility fixes from here on, while fse 2.0 is developed as a rewrite.

**Upgrading from 1.0:** SIF and uSIF give different vectors than 1.0, because their principal components are computed exactly and uSIF weights the removed components like the reference implementation; retrain such models for comparable vectors (see Changed and Fixed). Arguments are validated strictly and raise a `pydantic.ValidationError`. fse requires Python 3.11 or newer and adds the dependencies pydantic and threadpoolctl.

### Added

- **Smaller sentence vectors:** `sv_dtype="float16"` or `"int8"` on `Average`, `SIF` and `uSIF` (`dtype` on `SentenceVectors`) stores the sentence vectors in half or a quarter of the memory. They are computed in float32, and every read returns float32. Training, `train(update=True)`, `infer`, the similarity searches, `save`/`load` and `sv_mapfile_path` support both formats. int8 keeps one float32 scale per vector. On the STS Benchmark over the 15 vector sets of the README with CBOW, SIF-10 and uSIF, float16 changes the scores by at most 0.01 points and int8 by at most 0.02 points; float16 changes at most 0.4 % and int8 up to 7 % of the ten nearest neighbours of a sentence.
  - With index i for the i-th sentence, training writes into a ring of float32 windows and stores each finished window, so no float32 matrix of all sentences exists; SIF and uSIF read the input twice to learn their components first. The stored vectors are bit-identical to a float32 training stored afterwards. Other inputs (custom or shared indices, no compiled extension, a replaced `_do_train_job` or `_post_train_calls`) train into a float32 matrix of the new rows first, and the log names the reason.
  - float16 and int8 vectors are searched and scored by a compiled kernel directly on the stored values, in parallel, instead of being converted to float32 block by block.
  - `SentenceVectors.convert(dtype)` changes the format of existing vectors, also of loaded and memory-mapped ones; `get_vectors` and `iter_blocks` read float32 rows, the latter in blocks for any number of vectors.
- **Smaller word vectors:** `wv_dtype="float16"` or `"int8"` on `Average`, `SIF` and `uSIF` keeps the word vectors (and FastText ngram vectors) the model averages in half or a quarter of the memory. The model quantizes a copy once; the passed word vectors stay unchanged. With `wv_mapfile_path` the codes are written to a file and memory-mapped, and `save`/`load` keep them. On the STS Benchmark over the 15 vector sets with CBOW, SIF-10 and uSIF, float16 changes the scores by at most 0.01 points and int8 by at most 0.12 points.
- **Prefetching and row kernels:** the kernel prefetches the rows of the next words while it adds one, which makes float32 training up to about 1.2 times as fast end to end without changing its results. float16 and int8 rows are added by routines chosen at import: AVX2, FMA and F16C on x86-64 Linux and macOS when the CPU has them, NEON on arm64, portable loops otherwise; float32 rows keep using BLAS `saxpy`.
- `SentenceVectors.score(weights, bias)` computes `X @ weights + bias` over all sentence vectors in blocks, for example the logits of a logistic regression, in any storage format and optionally into a memmap.
- `CorpusFile(path)` for a local UTF-8 text file with one sentence per line, and `TextList(strings)` for strings in memory. They give exactly the sentences of `IndexedLineDocument` and `SplitIndexedList`, but every worker reads, splits and looks up its part without the GIL, so training scales with `workers`.
- `min_similarity` for `most_similar` and the `similar_by_*` methods returns only results at or above a cosine similarity (#34).
- A `NOTICE` file documents the parts adapted from gensim (LGPL 2.1, used under the GPL 3) and fastText (MIT), with their original copyright notices.
- Pre-built wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x86_64) on Python 3.11 to 3.13.
- Support for numpy 2, Cython 3, current scipy, gensim 4.4 and huggingface-hub 2.x.
- A clear `ImportError` when gensim < 4 is installed.
- `__all__` in `fse` and `fse.models`.
- Type information: the package ships a `py.typed` marker and a stub for the Cython extension, so type checkers see the annotations of fse.

### Changed

- Arguments of the public entry points are validated strictly with pydantic: the model, `SentenceVectors` and `IndexedLineDocument` constructors, `train`, `infer`, `most_similar`, the `similar_by_*` methods and `from_pretrained`. Wrong types and out-of-range values such as `workers=0` or `alpha=-1` raise a `pydantic.ValidationError` (a `ValueError`) at the call, and nothing is coerced, so `workers="2"` is an error. Integer arguments accept numpy integer scalars, float arguments must be finite, and mapfile paths accept `str` and `pathlib.Path`.
- New dependency: `pydantic>=2.5`.
- **Python inputs scale with workers.** With two or more workers, an `IndexedList` or `CIndexedList` of lists of str is no longer encoded by the single producer thread: every worker copies the tokens of its own chunk while holding the GIL and looks them up and averages them without it. A `SplitIndexedList` whose items are a list of str is trained natively like a `TextList`. Word lookups in the native paths read one 32-byte slot per word. Results are bit-identical; one worker, `infer`, subclasses that change how items are read and other item types keep the previous path. A list that changes between the scan and training now raises. Measurements are in #88.
- **SIF and uSIF no longer slow down with more workers.** OpenBLAS started its own threads in every worker for the Gram matrix and the removal of components, so 16 workers could run 256 BLAS threads on 16 cores. During `train`, and while `infer` removes components, BLAS now gets `cores // workers` threads (at least one; the limit holds for the whole process during the call), and the components are removed by all workers in parallel. Results are unchanged.
- New dependency: `threadpoolctl>=3.1.0`, to limit the BLAS threads.
- At DEBUG level, every training logs how much of its time the producer and the workers spend reading, averaging and waiting, and which side limits it.
- A new logo, with a light and a dark variant that the README picks by color scheme. The sources and their generator are in `media/logo/`; `media/fse.png` stays for the READMEs of earlier releases.
- Unknown keyword arguments to `Average` and `BaseSentence2VecModel` raise a `pydantic.ValidationError` instead of being silently ignored, like at every other validated entry point.
- Sentence tuples must have exactly two elements, `(words, index)`; longer tuples raise a `ValueError` instead of losing the extra element. `restrict_size=(lo, hi)` with `hi < lo` raises a `ValueError` instead of returning nothing.
- The input classes reject sets with a `TypeError`. A set has no order, so the sentence indices would be arbitrary; pass `sorted(...)` instead.
- Every sentence is checked to be a list of `str`, not only the first one.
- `set_madvise_for_mmap` is no longer called on every model construction and load; it never applied any advice.
- Requires Python 3.11 or newer. Python 3.10 reached its end of life in October 2026; installations on 3.10 and older keep fse 1.0.0. Python 3.14 is not supported yet, because gensim does not build there.
- Minimum versions: numpy 1.26.4, scipy 1.11.4, gensim 4.3.3, smart_open 6.4.0, wordfreq 3.1.0, huggingface-hub 0.23.0, psutil 5.9.0.
- Packaging moved to `pyproject.toml`; the version now comes from git tags.
- The warning about a missing C extension is a `RuntimeWarning`.
- Faster preparation and input handling, with identical results: sentence vectors are initialized in blocks instead of one row at a time, the list input classes iterate directly instead of through `__getitem__`, and SIF and uSIF read all word counts at once.
- Removing principal components for SIF and uSIF works in blocks of rows. It used to allocate a second copy of the whole sentence matrix in RAM, also for vectors on disk (`sv_mapfile_path`).
- **Training writes the sentence vectors in one pass.** With index i for the i-th sentence (every input class except the `C*` lists with custom indices) and the compiled extension, each job sets its own rows to EPS instead of a separate pass over all rows, and SIF and uSIF sum the Gram matrix of the finished rows during training instead of reading them again. Results are bit-identical. For SIF and uSIF trained from scratch on a memmap larger than the RAM, this removes two separate full passes over the file (about 28 percent less time and 43 GB less read for 15 million sentences of dimension 300); measurements are in #83.
- **SIF and uSIF compute their principal components exactly, over all sentence vectors.** They used a randomized SVD from scikit-learn on a sample of at most `cache_size_gb` (one million vectors in 1.0). Now the Gram matrix of all vectors is summed over blocks in float64 and decomposed, without a random sample: every vector is read once, memory stays at about one block, and repeated runs give the same components. The components change, so SIF and uSIF embeddings differ from earlier versions; on the STS Benchmark, SIF-10 scores move by at most 0.04 points and uSIF scores not at all, and the README table is remeasured. Computing the components is about 6 times faster for one million vectors of dimension 300 in memory, and SIF and uSIF train 2.5 to 4.5 times faster end to end on that corpus; measurements are in #82. When a training computes new components, SIF and uSIF raise a `ValueError` before it starts, leaving the model unchanged, if `components` exceeds the number of sentence vectors or their dimension; `train(update=True)` with learned components reuses them and is not checked; `compute_principal_components` raises a `ValueError` for more components than vectors or dimensions and for NaN or infinite values, and a `TypeError` for vectors that do not hold real numbers.
- **More workers now help (#61).** The producer thread encodes each job and the workers average it without holding the GIL, so training no longer stalls at two workers; with the input in memory, the producer is the limit from two to four workers on. The input scan runs in Cython. With one row per sentence, results are bit-identical for any number of workers; when several sentences share a row, the order of additions can vary within float32 rounding. Before and after measurements are in the pull request; the throughput of the current release is in the README.
- `batch_ngrams` above 2^24 (`fse.models.average_inner.MAX_NGRAMS_LIMIT`) raises a `ValueError` in the Cython routines; no word has that many ngrams, and the bound keeps buffer sizes in range.
- The Cython routine averaged at most 10,000 words per sentence and always up to 40 ngrams per unknown FastText word; it now averages every word and uses `batch_ngrams`, like the numpy fallback.
- FastText ngrams of out-of-vocabulary words are hashed in C instead of Python, with the same buckets as gensim. For a FastText model without ngram buckets (`bucket=0`), both averaging routines give unknown words no vector instead of the `ZeroDivisionError` of the numpy fallback; `train` still requires ngram vectors.
- Jobs hold at most `batch_words` sentences as well as `batch_words` words, so long runs of empty sentences no longer form one large job.
- The training log reports the end-to-end time, including the scan and component removal, next to the time of the training phase.
- `IndexedLineDocument` stores its line offsets as an int64 array, 8 bytes per line instead of about 40 in a Python list, and finds them with numpy in blocks instead of line by line in Python.
- The scan tracks indices that are not the positions of their sentences in a bitmap instead of a Python set (60 to 70 bytes per sentence). For a permutation, the bitmap ends with an eighth to a quarter byte per sentence; on the way, up to about one byte per sentence of indices waits for it to grow, and merging them takes a few temporary arrays of that size. Indices above 2⁶³ - 2 raise a `ValueError`, since their output could not be allocated anyway.
- `most_similar` keeps a running top-N over blocks of rows instead of computing and sorting the similarities of all sentences at once, so with a positive `topn` the search needs memory for one block and the candidates instead of a similarity per sentence. The normalized vectors from `init_sims` are unchanged. Equally similar sentences are ordered by index; the order used to be arbitrary.

### Fixed

- `lang_freq` overwrote the word counts of the passed model, and `wv_mapfile_path` replaced its vectors with read-only memmaps, so the caller and every further model built from it saw the change. Models now work on a shallow copy of the word vectors in both cases, with their own vocabulary containers, so words added to the passed model later do not reach the copy; the vector arrays themselves are still shared, not copied. The compiled routines also check that the vocabulary has no more words than there are word vectors.
- An existing word mapfile (`wv_mapfile_path`) was reused without any check: a larger file was read as a prefix, and a file of the same size with other vectors gave wrong sentence vectors without an error. The file must now hold exactly the vectors of the model, checked by size and content, otherwise a `ValueError` asks to delete it or use another path; loading a model checks the sizes.
- `Average` failed with `KeyError: 'count'` for word vectors without frequency counts, such as `KeyedVectors` built with `add_vectors`, although averaging needs no frequencies. SIF and uSIF now raise a clear `ValueError` when counts are missing or all zero.
- FastText models without ngram buckets (`bucket=0`), which gensim supports, were rejected by `train`. Unknown words of such models add no vector.
- **uSIF results change.** uSIF removed each principal component weighted by λ² instead of λ, unlike the reference implementation ([kawine/usif](https://github.com/kawine/usif)). Embeddings from uSIF models trained with earlier versions differ from 1.1.0. On the STS Benchmark the fix raises uSIF scores on all 15 published vector sets, by 0.18 to 1.30 points (mean 0.75) when both formulas run in the same environment; the README table is remeasured. The previous table is in the [1.0.0 README](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/v1.0.0/README.md#results).
  uSIF models saved by earlier versions keep the old weighting when loaded, so `infer` and `train(update=True)` produce vectors that match their stored ones; a warning recommends retraining.
- An exception in a worker thread or in the input iterable made `train()` hang forever. It is now raised in the calling thread after all threads have stopped.
- Only the first input item was validated. A negative or non-integer index later in the input could hang training or silently write into the wrong row. Every item is now checked, and `numpy` integers are accepted as indices, so `CIndexedList` works with an ndarray `custom_index` as documented.
- Generators and other one-shot iterators returned all-zero vectors from `infer` and failed with a misleading error in `train`, because the input is read twice. They are now rejected with a clear `TypeError`.
- `save()` with `wv_mapfile_path` set the word vectors of the live model to `None`, so later `infer` calls read invalid memory, and a second save and load cycle failed. The vectors are now only left out of the pickle.
- The Cython kernel now checks dtype, memory layout, writeability and shapes of the arrays it accesses through raw pointers, and the index of every sentence against the target rows, instead of reading or writing invalid memory.
- The FastText ngram buffer of `BaseSentence2VecModel` could be smaller than the kernel expects when `batch_words` or `batch_ngrams` were set below the defaults.
- `most_similar` and `init_sims` failed with a `TypeError` for sentence vectors stored on disk (`sv_mapfile_path`). The normalized vectors are now written into the mapfile in blocks.
- `init_sims(replace=True)` did not replace the vectors, and `get_vector(use_norm=True)` failed before `init_sims` was called.
- Normalizing a sentence without known words returned NaN; it now stays a zero vector.
- The Cython routine computed the position of a row as row times dimension in 32 bits. From about 14.3 million sentences at 300 dimensions (43 million at 100) on, sentence vectors were silently added to wrong rows near the start of the matrix. Offsets are now 64 bit, and more than 2³² - 1 rows or words raise a `ValueError`.
- `train(update=True)` wrote the new sentences into the existing rows and left the appended rows empty. New sentences keep their indices starting at 0 and are stored after the existing rows. SIF and uSIF remove the components of the first training from the appended rows only, instead of removing them from all rows again.
- SIF and uSIF with `components=0` could not be trained a second time.
- In the Cython path, a FastText out-of-vocabulary token without any ngrams (for example a token shorter than `min_n - 2` characters) added the vector of the first vocabulary word. It now adds nothing, as in the numpy fallback.
- Saving a model with `sv_mapfile_path` wrote the sentence vectors a second time into `.npy` files, and loading returned them in RAM instead of reopening the mapfile.
- `x in sentence_vectors` raised a `KeyError` for non-integer values and accepted negative indices; it now returns `False` for both. The `KeyError` for a missing index names the index.
- `str(model)` reported the number of sentences as `vector_size`.
- Scanning failed for an empty input and for corpora with less than one word per sentence on average.
- The numpy fallback counted empty sentences as effective sentences, unlike the Cython path.
- Interrupting training, for example with Ctrl+C, could leave worker threads blocked on a full queue.
- With several workers, sentences mapped to the same row (for example `CIndexedList` with repeated indices) could lose updates when two workers added into the row at the same time. Rows shared by several sentences are now updated under locks; unique indices stay lock-free.
- `pip install fse` failed on Python 3.11 and newer, because the shipped C source was generated by an old Cython version.
- Building under pip or poetry crashed with `AttributeError: 'dict' object has no attribute '__NUMPY_SETUP__'`.
- `import fse` failed in a clean environment, because `requests` was imported but not declared.
- `Vectors.from_pretrained` and `FTVectors.from_pretrained` raise `ValueError` again for a model that does not exist on the hub.
- The numpy fallback now accumulates many-to-one sentence mappings like the Cython routine instead of overwriting them.
- Training crashed with `ZeroDivisionError` when a run finished faster than the timer resolution, for example on Windows.
- Wheels and source distributions no longer install a top-level `test` package.
- `fse.__version__` reported 0.2.0.

### Deprecated

- `cache_size_gb` of `SIF`, `uSIF` and `fse.models.utils.compute_principal_components` has no effect, because the components are computed over all vectors. Passing it raises a `DeprecationWarning`; it will be removed in 2.0.

### Removed

- Support for Python 3.6 to 3.10.
- The `requests`, `dataclasses` and `scikit-learn` dependencies.

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

[Unreleased]: https://github.com/oborchers/Fast_Sentence_Embeddings/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/oborchers/Fast_Sentence_Embeddings/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/oborchers/Fast_Sentence_Embeddings/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/oborchers/Fast_Sentence_Embeddings/compare/v0.1.17...v0.2.0
[0.1.17]: https://github.com/oborchers/Fast_Sentence_Embeddings/releases/tag/v0.1.17

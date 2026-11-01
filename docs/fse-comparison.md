# Choosing fse and comparing alternatives

This is the decision-table draft for DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen).
Fresh measurements are still pending. This table describes capabilities, not a measured speed or quality ranking.

| Need | Candidate | Reason |
|---|---|---|
| Run a finished, supported static embedding table with persistent rows and optional linear heads | fse | Its runtime combines tokenization, aggregation, quantization, heads and sinks. |
| Train or distill a new static embedding model | model2vec | fse 2.0 performs inference and does not fit embeddings. |
| Context-dependent token representations | A transformer such as all-MiniLM-L6-v2 | A static table gives each token one stored vector. |
| Approximate nearest-neighbor search over a large collection | FAISS or hnswlib | fse searches its stored rows exactly or with BLAS and has no ANN index. |
| A lexical classification or retrieval baseline | scikit-learn TF-IDF | It provides a vocabulary-based baseline to test against embeddings. |

## Scope and evidence requirements

The comparison includes model2vec/POTION, model2vec-rs, FastEmbed, Sentence Transformers StaticEmbedding and all-MiniLM-L6-v2, TF-IDF, a fastText filter classifier, FAISS and another local search library. It includes one-worker and all-core CPU encoding, startup, memory, long texts, whole-lifecycle latency, numerical subspaces and recall, and estimated compute cost.

Quality uses all 41 tasks of `MTEB(eng, v2)` and all 13 NanoBEIR tasks. The six static Hub models are potion-base-8M, potion-base-32M, potion-retrieval-32M, potion-multilingual-128M, static-retrieval-mrl-en-v1 and static-similarity-mrl-multilingual-v1. GloVe-300 and word2vec-300 use explicit mean-pooling recipes. Each fse model has float32, float16 and int8 variants. The contextual comparator is all-MiniLM-L6-v2. These English tasks do not establish multilingual quality.

Native model2vec and the MTEB Sentence Transformers loader are separate comparators. Their character cuts, unknown-token handling and normalization can differ. Native model2vec is the conversion comparator for fse's POTION import. A quality change between loaders must not be attributed solely to quantization.

## Reproduce the current harness

Use Python 3.13. Install quality dependencies from `benchmarks/quality/requirements.txt`, which includes the complete resolved lock file. FastEmbed uses a separate environment from `benchmarks/comparison-requirements.lock.txt`: its published Hub dependency conflicts with the quality environment. Install fse from the measured checkout into each environment.

`benchmarks/quality/prepare_models.py` converts the six pinned Hub models. `prepare_word_models.py` converts the two pinned word-vector tables without fitting. Both write a source manifest next to the resulting artifacts and refuse existing output files.

`benchmarks/bench_compare.py run` rotates fse A, identical fse B and a counterpart. Supply `--lock`, `--results`, `--threads 1` or the usable CPU count, and at least three rounds. `--against` selects model2vec, st, rust or fastembed. `--other-python` selects the isolated counterpart environment. Build the Rust worker with its committed Cargo lock file before selecting rust.

Every operation and round carries a unique series ID. Only a completed round with matching ordered-input digests and strict endpoint loads below 5 qualifies. Completed, incomplete and rejected rounds remain in the raw journal. Pass percentiles describe repeated calls inside one process. Independent round ratios and the A/A control assess repeatability.

The corpus reader selects one immutable dataset revision. It records explicit splits and a length-delimited XXH3-128 digest of every ordered input text. The journal records the model revision, source commit, complete Python package environment, CPU, OS and Python. Model setup includes the backend import in a fresh child; corpus IO, provenance and result serialization are outside encode timing. RSS scopes distinguish Python and native Rust processes.

Quality runs on the CPU with bounded torch, tokenizer and BLAS pools. Use `--coordination` for shared heavy-job locking and `nice -n 10`. The runner saves each task immediately. It yields at a task boundary when a timing window appears, releases after a bounded task batch, and waits at least 120 seconds before reacquiring the heavy lock.

`benchmarks/compare_numerics.py` compares all rows of two ordered matrices. It computes principal angles between centered column spaces using SciPy's implementation, and exact cosine neighborhoods for every eligible query. It excludes self matches, resolves ties by document ID and reports zero-vector disagreements. Neighborhood recall measures numerical stability, not labeled retrieval relevance. No ANN or row sampling enters that calculation.

## Published reference targets

MinishLab's [results at commit 79c6559](https://github.com/MinishLab/model2vec/blob/79c655983241d68f8754dda48246b30bc34b897b/results/README.md) report 17,925 samples/s for model2vec plus logistic regression. This is the historical target named in the brief. Its [October 2026 revision](https://github.com/MinishLab/model2vec/blob/528a84fe92387009a5b0e1be679c7380dc005026/results/README.md) instead reports 32,869 samples/s. These are published reference figures, not measurements of this checkout or this machine. A reproduction must identify the dataset protocol and machine before comparing rates.

The native Rust worker pins [model2vec-rs source](https://github.com/MinishLab/model2vec-rs/tree/19f58bef6ce145f2bc08de7016498f28d521feea). The subspace method follows [SciPy's primary documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html), which cites Knyazev and Argentati (2002). The exact FAISS reference is [IndexFlatIP](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatIP.html), applied to normalized vectors when comparing cosine search.

## Still required

Raw valid timing pairs, full quality results, both classifier targets, search and lifecycle comparisons, CPU and memory profiles, a sourced CPU-price estimate, and final review remain open. No new comparative speed, quality, memory or cost conclusion is established yet.

# Choosing fse and comparing alternatives

This is the decision-table draft for DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen).
Measurements are partial. This table describes capabilities, not a measured speed or quality ranking.

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

Three initial speed cells and seven precision-matrix cells have complete paired evidence. Every cell reports its A/A spread, with noisy cells explicitly marked. POTION-8M and POTION-32M float32 each completed all 53 non-deferred quality tasks. Their method/runtime fingerprints differ, so these checkpoints establish no controlled model ranking. Four accepted profiles and a sourced tariff scenario are published in [PERFORMANCE.md](../PERFORMANCE.md). Missing or noisy speed cells, the remaining quality configurations, both classifier targets, search/lifecycle measurements and final review remain open.

### Common-vector local search protocol

`bench_search_compare.py` compares fse with FAISS `IndexFlatIP` and hnswlib
HNSW on the same POTION-8M float32 WikiText vectors. Every corpus row is a
query, including its stored self. Query batches contain either one or 64 rows.
Both one-worker and ten-worker configurations are explicit. All contenders
receive one shared unit-normalized float32 matrix, recorded by its digest.
FAISS computes inner products of that matrix. fse and hnswlib compute cosine
similarity, so rounding differences can change close neighbors.

The oracle computes float64 cosine over all rows. Strict recall uses lower-ID
ties. Boundary-equivalent recall also accepts another ID with exactly the
float64 score at the kth boundary. Both scores are reported. Zero queries are
included in throughput and excluded, with their count, from numerical recall.
These scores measure index fidelity, not relevance on a labeled retrieval task.

Each fresh child records encoder load, encode, index construction, durable
save, reopen, first search, and three complete warm search passes. The empirical
p50 and p95 of batch calls are separate from whole-pass statistics. fse writes
its durable store during construction. FAISS and hnswlib construct in memory
then serialize. Compare construction plus save, not construction alone. All
saved files receive fsync and their parent directories receive F_FULLFSYNC on
macOS (fsync on other POSIX systems), matching fse full durability. The peak
RSS includes the encoder, index and untimed full-corpus recall audit.

HNSW uses M=16, ef_construction=200, ef=100 and random_seed=100. Parallel
construction can still change graph ordering. Three rotating paired rounds
include a separate fse A/A control. Each operation requires the shared timing
lock and load strictly below five at both endpoints. Results are pending.

Sources: [FAISS metrics](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances),
[hnswlib parameters](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md).

### Classifier reconstruction protocol

The historical MinishLab source at `79c655983241d68f8754dda48246b30bc34b897b`
states 14 classification datasets, but its table has 15 dataset rows. The local
reconstruction uses all 15 displayed rows. The [pinned SetFit manifest](../benchmarks/baseline/dr151_setfit_sources_20261101.json)
records exact revisions and the two ambiguous dataset-name mappings. Original
hardware, training selection, CV settings and speed averaging were not disclosed
in that source. The stated 17,925 samples/s therefore remains a historical
reference, not an exact replication target with a fully known protocol.

`quality/prepare_setfit.py` selects exactly 1,000 stratified training examples
with seed zero and evaluates the full test set. It fits LogisticRegressionCV
with five folds, ten Cs and max_iter=1000, without a validation-set search.
Both frozen POTION-8M and POTION-32M are tested through fse and native model2vec.
Each source library gets its own fitted head. fse float16 and int8 reuse the
head fitted on float32, so their task-score deltas isolate table quantization.
TF-IDF uses the default scikit-learn vectorizer fitted on those same training
examples, with a complete vectorizer/classifier pipeline cross-validated over the same ten C values and five folds. Accuracy and macro-F1 are
reported separately. No speed claim comes from classifier preparation.

`quality/prepare_fineweb.py` uses all 467,424 pinned FineWeb-Edu annotations.
A seed-zero permutation holds out 46,743 pages. Every arm receives the same
whitespace-cleaned pages and the binary label score >=3. Frozen POTION uses its
pretrained character/token limits, whereas fastText reads the full page. This
is a full-page filtering-task comparison with different model input limits.
The fse heads use LogisticRegression(max_iter=1000, random_state=0). fastText
uses wordNgrams=2 and explicitly recorded supervised defaults, with two training
threads. Parallel fastText training is not bit deterministic. All predictions
are evaluated on the full held-out split. Results are pending.

### Authorized MIND deferral

MindSmallReranking is **deferred: needs more than 16 GiB RAM on the M1**.
The first unmodified MTEB 2.26.25 invocation reached a sampled physical footprint
of 15.4 GiB while loading the 97,006,943 test relevance rows. It was stopped,
retaining 19 completed tasks. The coordinator explicitly deferred this one task
to a maintainer decision about a larger machine. The loader and evaluator remain
unmodified. Every other selected task and model remains in scope. No sampling
is used. Run with `--defer-mind` to write an explicit deferred record for each
model. The expected coverage is 40/41 English tasks and 13/13 NanoBEIR tasks.
A complete 41-task English mean must not be reported from this coverage.

`bench_search_compare.py --encoder-start source` measures pinned source-weight
IO and finished-table conversion together as `table_load_build_s`. This is the
public import API's combined stage. `--encoder-start saved` loads the already
finished table instead. The combined stage is not added a second time to startup.
Both modes then encode, build and durably save the index, reopen it, and perform
first and warm queries. All search arms use the same encoder startup mode.

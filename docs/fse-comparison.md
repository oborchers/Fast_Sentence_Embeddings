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

Float32 quality uses all 41 tasks of `MTEB(eng, v2)` plus all 13 NanoBEIR tasks. MIND runs separately on the larger pod. The six static Hub models are potion-base-8M, potion-base-32M, potion-retrieval-32M, potion-multilingual-128M, static-retrieval-mrl-en-v1 and static-similarity-mrl-multilingual-v1. The four classics are paranmt-300 with plain mean, and GloVe-300, word2vec-300 and crawl FastText-300 with SIF-10 folded using only the pinned WikiText-103 training fit corpus. The contextual float32 reference is all-MiniLM-L6-v2. Float16 and int8 of all ten static models use all STS tasks, all NanoBEIR tasks and numerical recall against matched float32. No other task is sampled. These English tasks do not establish multilingual quality. FastText benchmark inclusion makes no fse 2.0 release decision.

Native model2vec and the MTEB Sentence Transformers loader are separate comparators. Their character cuts, unknown-token handling and normalization can differ. Native model2vec is the conversion comparator for fse's POTION import. A quality change between loaders must not be attributed solely to quantization.

## Reproduce the current harness

Use Python 3.13. Install quality dependencies from `benchmarks/quality/requirements.txt`, which includes the complete resolved lock file. FastEmbed uses a separate environment from `benchmarks/comparison-requirements.lock.txt`: its published Hub dependency conflicts with the quality environment. Install fse from the measured checkout into each environment.

`benchmarks/quality/prepare_models.py` converts the six pinned Hub models. `prepare_classics.py` reproduces the current folded SIF-10 classics and plain-mean paranmt model. The older `prepare_word_models.py` raw-table conversion remains only for historical measurements and same-shape speed proxies. Preparers retain source manifests and refuse existing output files.

`benchmarks/bench_compare.py run` rotates fse A, identical fse B and a counterpart. Supply `--lock`, `--results`, `--threads 1` or the usable CPU count, and at least three rounds. `--against` selects model2vec, st, rust or fastembed. `--other-python` selects the isolated counterpart environment. Build the Rust worker with `benchmarks/compare_rust_build.py --coordination <scratchpad>` before selecting rust. The helper uses the heavy lock and records locked inputs, compiler and binary digest. The worker refuses a changed source or binary.

Every operation and round carries a unique series ID. Local historical measurements require completed rounds, matching ordered-input digests and strict endpoint loads below 5. Shared-pod measurements record host load without rejecting it, and require the global fse A/A band within 10 percent instead. No new local speed measurement is authorized under shared rule 10. Completed, incomplete and rejected rounds remain in the raw journal. Pass percentiles describe repeated calls inside one process. Independent round ratios and the A/A control assess repeatability.

The corpus reader selects one immutable dataset revision. It records explicit splits and a length-delimited XXH3-128 digest of every ordered input text. The journal records the model revision, source commit, complete Python package environment, CPU, OS and Python. Model setup begins after common harness/provenance imports (including fse hashing support); corpus IO, provenance and result serialization are outside encode timing. RSS scopes distinguish Python and native Rust processes. New records capture each executing backend's own package version, install path and installed-file content fingerprint. Rust records the verified worker binary digest. Reports reject changes between rounds or between the two fse controls. Historical records without runtime fingerprints remain explicitly qualified observations.

Quality runs on the CPU with bounded torch, tokenizer and BLAS pools. Use `--coordination` for shared heavy-job locking and `nice -n 10`. The runner saves each task immediately. It yields at a task boundary when a timing window appears, releases after a bounded task batch, and waits at least 120 seconds before reacquiring the heavy lock.

`benchmarks/compare_numerics.py` compares all rows of two ordered matrices. It computes principal angles between centered column spaces using SciPy's implementation, and exact cosine neighborhoods for every eligible query. It excludes self matches, resolves ties by document ID and reports zero-vector disagreements. Neighborhood recall measures numerical stability, not labeled retrieval relevance. No ANN or row sampling enters that calculation.

## Published reference targets

MinishLab's [results at commit 79c6559](https://github.com/MinishLab/model2vec/blob/79c655983241d68f8754dda48246b30bc34b897b/results/README.md) report 17,925 samples/s for model2vec plus logistic regression. This is the historical target named in the brief. Its [October 2026 revision](https://github.com/MinishLab/model2vec/blob/528a84fe92387009a5b0e1be679c7380dc005026/results/README.md) instead reports 32,869 samples/s. These are published reference figures, not measurements of this checkout or this machine. A reproduction must identify the dataset protocol and machine before comparing rates.

The native Rust worker pins [model2vec-rs source](https://github.com/MinishLab/model2vec-rs/tree/19f58bef6ce145f2bc08de7016498f28d521feea). The subspace method follows [SciPy's primary documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html), which cites Knyazev and Argentati (2002). The exact FAISS reference is [IndexFlatIP](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatIP.html), applied to normalized vectors when comparing cosine search.

## Still required

Historical speed cells retain their rotating paired measurements and per-cell A/A spreads. Older counterpart rows with fse code attribution remain provisional. Pod 2 records backend-specific identities and has a separate completed audit. Eight float32 configurations completed all 53 non-deferred tasks: the six Hub models, MiniLM-L6-v2 and plain-mean ParaNMT. Different method cohorts establish no controlled model ranking. MIND remains deferred locally. Missing or noisy speed cells, the remaining quality configurations, matched precision deltas, search/lifecycle measurements and final review remain open. See [PERFORMANCE.md](../PERFORMANCE.md) for raw-source links and qualifications.

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
Both frozen POTION-8M and POTION-32M are tested through fse linear heads and Python Model2Vec embeddings plus sklearn LogReg linear heads.
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
first and warm queries. Common normalization has its own `preprocess_s` timer. `startup_through_first_search_s` measures model load through first search, including preprocessing and intervening overhead. It excludes input/provenance preparation and later recall diagnostics. Do not add the combined build/save field to its component times. All search arms use the same encoder startup mode.

Classifier manifests bind each fitted head to the actual training encoder fingerprint. Quantized tables must match their verified float32 parent. Frozen Model2Vec plus sklearn linear heads bind the weights and tokenizer/config files. Preparation and prediction both validate these identities. Completed quality records write the same normalized identity fields that resume checks, including explicit null fields for non-fse models.

`quality/report_document_tasks.py` joins a compatible quality summary and raw journal to the installed MTEB2.26.25 descriptive statistics. It reports every full task with an upstream evaluation text stream averaging at least1,000 characters. The output preserves task metrics, model limits and source-statistics digests. It never samples evaluation rows or averages different task metrics into a new benchmark score. These upstream lengths are not fresh measurements of the run or truncation frequencies.

The window5 counterpart runtime path/hash was incorrectly copied from fse. Existing fse rates and A/A remain valid. Counterpart rates and final ratios are provisional until re-measurement with backend-specific package identity. Recorder fix02d387c and reporting follow-upd635a12 retain every original observation. The own-package hash covers declared installed files and the module tree, excluding mutable Python caches. Rust records its verified binary digest. Common harness imports precede model-setup timing, so that stage is not a complete cold process startup.


### Native math limits and the FAISS CPU path

New child launches set OpenMP, OpenBLAS, MKL, BLIS, Accelerate and NumExpr maxima
before Python imports NumPy. Each row records these requested limits. A paired
round rejects different limits between its contenders. Supported thread pools
remain recorded separately. These maxima do not measure the exact active
thread count. Model2Vec's tokenizer off/on labels and unmeasured exact tokenizer
thread count remain unchanged. Apple's installed `Accelerate(7)` manual notes
that `VECLIB_MAXIMUM_THREADS` can be cached and must precede Accelerate calls.
The local NumPy build uses Accelerate, which this threadpoolctl inspection does
not enumerate. These explicit launch limits therefore cover search BLAS and
sklearn prediction before their first measurements.

The installed FAISS wheel reports `OPTIMIZE DD ARM_NEON MAC_METAL`. The harness
constructs CPU `IndexFlatIP`. Its pinned implementation calls
`knn_inner_product`, while Metal uses a separate index and cloning API.
[FAISS 1.15.1 IndexFlat implementation](https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/IndexFlat.cpp),
[CPU inner-product search](https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/utils/distances.cpp)
and [Add Metal GPU backend for Apple Silicon (IndexFlat)](https://github.com/facebookresearch/faiss/pull/5144).
No GPU index is constructed by this harness.

### Exported MLP and additional folding comparison

Model2Vec 0.10.0 fine-tunes its embeddings and an MLP with one 512-unit ReLU hidden layer using PyTorch. The exported classifier runs in plain NumPy, with softmax or sigmoid for probabilities. Its label path selects from logits. fse's classifier head is linear. Accuracy and speed therefore compare a fine-tuned MLP against a frozen-embedding linear head. [Pinned MLP implementation](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/inference/mlp.py), [pinned training documentation](https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/train/README.md).

Controlled Pod 2 classifier throughput is now published in [the Pod 2 audit](fse-pod2-results.md). Encoder-only measurements do not establish classifier rates. Raw-mean folding comparisons remain pending Pod 3. Pod 3 adds raw-mean fse folded prediction versus raw `Encoder.encode` plus the exact same linear matmul. It also includes the exported fine-tuned Model2Vec MLP reference. Existing unfolded normalized-linear, LogReg, TF-IDF and fastText rows remain unchanged. Standalone storage preserves fse's automatic raw-head folding algorithm.

Pod 3 also compares Python Model2Vec with the **same frozen sklearn linear head**, including the identical float32 coefficients and bias. This isolates the encoding engines. Its fine-tuned Model2Vec MLP remains a separate trained-model comparison. AG News linear cells are published first, before MLP training, with raw and summary artifacts.

Worker scaling and the expanded large-write protocol are specified in [the pod evidence guide](fse-comparison-pods.md#expanded-pod-3-scaling-write-paths-and-100m-showcase). The [corpus-length evidence](fse-corpus-lengths.md) records exact means and medians of words, Unicode characters and consumed POTION-8M tokens. Counting completed under the shared heavy lock for all 21 published/planned corpus entries, including classifier evaluation sets. No local throughput measurement runs.

Fresh Pod 3 graphics use a complete 1/16/32 grid for table dtypes and same-model library pairs, plus separate six-point float32 curves for POTION-8M and static-retrieval-mrl-en-v1. Selection always retains one core. It adds 32 only when the unrounded primary median32/median16 is at least 1.20, otherwise 16. No intermediate curve setting is substituted. Native ST static outputs remain unnormalized means. POTION outputs retain the model's native normalization. Full-corpus semantic checks precede each timed series. See `docs/fse-comparison-pods.md` for checks and fixed whole-pod limits.

The current local quality plan is `benchmarks/baseline/dr151_quality_priority_plan_20261101.json`. It supersedes the initial 35-configuration plan. The final scope contains eleven full float32 model configurations and twenty reduced quantized configurations. Eight full float32 configurations have completed the local 53-task scope. Reduced float32 references are additional calibrations for matched quantization deltas. They do not add model/dtype combinations. Local full runs exclude MIND, which remains a separate unmodified-loader pod task.

MiniLM and plain-mean ParaNMT now each have 40 completed English tasks plus all 13 NanoBEIR tasks. MIND remains deferred locally. Their English 40-task means are 59.8863 and 50.6940 points, and NanoBEIR means are 56.2214 and 39.3910 points. The official full 41-task English means remain null. Raw task records and their exact method/runtime identities remain separate from these reported means.

Quality-row `seconds` measures time inside model.encode calls. It is not the complete task or configuration duration. Earlier partial-task-average duration scenarios used the wrong label and are withdrawn. `report_elapsed.py` reports encoder totals, individual invocations, the union of overlapping invocation intervals, and the full calendar span separately. ParaNMT's uninterrupted complete invocation took 1,354.38 seconds. MiniLM's resumed invocations occupy 13,449.89 seconds after counting its overlapping retry once. Both include evaluation and waits inside invocations and are diagnostic completion times, never CPU throughput measurements.

Document-task breakouts now also cover both completed MRL float32 cohorts and the completed MiniLM reference. The selection includes every measured full task whose packaged MTEB evaluation text stream averages at least 1,000 characters. [Static retrieval MRL](../benchmarks/baseline/dr151_document_tasks_staticretr_20261101.json) and [static similarity MRL](../benchmarks/baseline/dr151_document_tasks_staticsim_20261101.json) each contain 19 tasks. [MiniLM](../benchmarks/baseline/dr151_document_tasks_minilm_20261101.json) now contains 19. [ParaNMT](../benchmarks/baseline/dr151_document_tasks_paranmt_20261101.json) also contains 19. These files preserve exact task revisions, source-statistics hashes, native task metrics, encoder limits and distinct configuration identities. The upstream length statistics do not measure this run's token truncation. Different configuration cohorts remain separate, with no new aggregate ranking.


Prepared classic artifacts and complete SIF-10 fit receipts are in `benchmarks/baseline/dr151_classic_fitted_artifacts_20261101.json`. FastText and GloVe use all 859,346 pinned WikiText-103 training paragraphs, seed 0, a=0.001 and ten uncentered components. Receipts preserve the tokenized corpus digest, fitted state hash, final encoder fingerprint, original Hub revision and exact gensim import path. ParaNMT uses plain mean with no fitted SIF state. FastText benchmark inclusion makes no release decision. Cross-platform fitted hashes await the MIND pod's own independently generated artifacts.


## Cross-check against model authors

The [machine-readable authors cross-check](../benchmarks/baseline/dr151_author_claims_crosscheck_20261101.json) records sources observed on 2026-11-01 and the local result identities. This adds no inference or timing run.

| Model and evaluation | Author value, points | Measured value, points | Interpretation |
|---|---:|---:|---|
| Static Retrieval MRL, 1024 dimensions, NanoBEIR | 50.31 | 50.3072 | Aggregate agrees at the displayed two decimals; individual task scores differ slightly. |
| POTION-8M, author MTEB versus local English40 | 51.08 | 53.9527 | Different task composition; no direct reproduction or fse quality gain claim. |
| POTION-32M, author MTEB versus local English40 | 52.13 | 55.2573 | Different task composition. |
| MiniLM, MinishLab's MTEB comparison versus local English40 | 55.93 | 59.8863 | Different task composition; author-column value is from MinishLab, not MiniLM's original authors. |
| ParaNMT WORD, STS Benchmark | 79.2 Pearson | 79.0700 Spearman | Different correlation metrics; exact checkpoint/preprocessing equivalence is unverified. |

The [Static Retrieval model card](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1#matryoshka-evaluations) supplies all thirteen NanoBEIR task values. Our aggregate differs by about 0.00285 points. The maximum individual task difference is 0.094 points. Rounded aggregate agreement does not establish bitwise or exact per-task agreement. That model card was observed at main; the local model weights retain their pinned revision.

The [pinned POTION-8M model card](https://huggingface.co/minishlab/potion-base-8M/blob/bf8b056651a2c21b8d2565580b8569da283cab23/README.md) contains the POTION and MiniLM author-table values. Its evaluation includes summarization, while our English40 mean uses the approved v2 task set without MIND. The [ParaNMT paper, Table9](https://www.cs.cmu.edu/~jwieting/wieting2017Millions.pdf) reports Pearson for its WORD model. Our MTEB main score uses Spearman, so numerical proximity is not replication.

The [Static Retrieval speed evaluation](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1#evaluation) reports a 397x CPU ratio against MPNet using GooAQ on an i7-13700K. Our corpus and hardware differ, and we did not run that MPNet comparison. We therefore do not claim to reproduce that ratio. Executing identical pretrained weights faster measures the library, not better model quality.

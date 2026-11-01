The review found seven defects. The main risks concern mixed-run evidence, reproducibility, and output safety. Findings below are verified from source inspection, without execution.

1. **Numerical recipe and scientific comparison, including retrieval and selection**

   **Medium: FastText loses the historical unknown-word weight.**  
   [prepare_hub_classics.py:75](/private/tmp/dr157-review-702f3c0/benchmarks/prepare_hub_classics.py:75) and [prepare_hub_classics.py:202](/private/tmp/dr157-review-702f3c0/benchmarks/prepare_hub_classics.py:202) omit `oov_weight`. Both therefore use `1.0`. The historical implementation uses `max(word_weights)` for unknown FastText words. Consequently, fitting, historical diagnostics, and exported SIF models differ from that recipe whenever the maximum weight is below one.  
   **Fix:** Pass the computed maximum explicitly during fitting and preserve it in the folded encoder. Test sentences containing both known and unknown words.

   **Medium: Selection accepts stale, incompatible, or non-finite scores.**  
   [evaluate_hub_fit.py:77](/private/tmp/dr157-review-702f3c0/benchmarks/evaluate_hub_fit.py:77) averages files by filename. It ignores their recorded encoder fingerprints, dataset revisions, metric names, and MTEB versions. STS records contain no encoder fingerprint. Results from different candidate builds can therefore select the corpus for all five sources. `NaN` also reaches `max()` without rejection.  
   **Fix:** Bind every score to the candidate fingerprint, fit receipt, evaluation inputs, and evaluator version. Validate all 13 expected tasks, metric names, and finite score ranges before selection.

   Otherwise, the inspected code follows the requested arithmetic and comparison structure: English wordfreq best-list counts, `a=0.001`, ten uncentered components, and equal paragraph counts. The selection mean gives equal weight to Pearson, Spearman, and NanoBEIR’s macro mean. The installed MTEB 2.26.25 source confirms cosine retrieval and 13 pinned tasks using `ndcg_at_10`. Sentence recall excludes self and handles ties deterministically. Evaluation-text fitting remains confined to the labelled historical diagnostic.

2. **Persistence, provenance, reproducibility and failure handling**

   **Medium: Existing corpus files acquire unverified provenance.**  
   [hub_classic_corpus.py:28](/private/tmp/dr157-review-702f3c0/benchmarks/hub_classic_corpus.py:28) and [hub_classic_fineweb.py:48](/private/tmp/dr157-review-702f3c0/benchmarks/hub_classic_fineweb.py:48) return existing files without validating their origin or preprocessing. Later fitting hashes those files and labels them with the configured revision and seed. An altered file with the expected row count passes.  
   **Fix:** Store and validate a corpus manifest containing source revision, preprocessing identity, sampling recipe, row count, and checksum. Refuse caches lacking that evidence.

   **Medium: Evaluation inputs are not fully pinned.**  
   [verify_hub_classics.py:40](/private/tmp/dr157-review-702f3c0/benchmarks/verify_hub_classics.py:40) reads STS-B through mutable `origin/1.x`. [hub_classic_corpus.py:40](/private/tmp/dr157-review-702f3c0/benchmarks/hub_classic_corpus.py:40) downloads unpinned `punkt_tab` data. Pinning the NLTK package does not pin its separately downloaded tokenizer data. Scores can change without a benchmark-code change.  
   **Fix:** Pin the STS fixture commit and checksum, and pin or verify tokenizer resources. Record both identities in evaluation receipts.

   **Medium: Rerunning comparison can replace receipts before refusing the existing model.**  
   [compare_hub_fit.py:51](/private/tmp/dr157-review-702f3c0/benchmarks/compare_hub_fit.py:51) overwrites the fitted state and JSON receipt. The subsequent model save at line 63 rejects an existing artifact. A changed rerun can leave new fit evidence beside the old model and scores.  
   **Fix:** Check every destination before fitting, or publish a complete candidate run from a fresh staging directory. Reject incomplete or conflicting runs without changing existing evidence.

   **Medium: Card generation does not verify that measurements describe the model.**  
   [hub_classic_cards.py:150](/private/tmp/dr157-review-702f3c0/benchmarks/hub_classic_cards.py:150) matches verification rows only by repository name. Verification records lack model fingerprints. At line 181, the generator replaces file checksums without first checking the previous manifest. A replaced model can receive stale measurements and newly accepted checksums.  
   **Fix:** Record fingerprints during verification. Before writing cards, validate the model, build manifest, verification records, and fit receipt together. Reject mismatches before updating checksums.

   Source-vector revisions and both training-dataset revisions are explicitly pinned. Model folders receive reload and fingerprint checks. I found no upload calls.

3. **Model-card license accuracy**

   **No findings against the specified terms.** The generated text provides:

   - GloVe `pddl`, separated from its implementation’s Apache-2.0 license. This matches the [upstream README](https://raw.githubusercontent.com/stanfordnlp/GloVe/master/README.md).
   - Google News `other`, the research-only wording, and an explicit distinction between code and vectors.
   - FastText `cc-by-sa-3.0`, Facebook attribution, modification disclosure, and ShareAlike for all derivatives. These align with the [official vector page](https://fasttext.cc/docs/en/english-vectors.html).
   - ParaNMT and Paragram `unknown`, preservation statements, and removal-request links.
   - ParaNMT’s noncommercial-research source-data restriction, consistent with [CzEng 1.6](https://ufal.mff.cuni.cz/czeng/czeng16).

   **Unresolved source question:** ParaNMT’s repository contains a [BSD-3-Clause software license](https://raw.githubusercontent.com/jwieting/para-nmt-50m/master/LICENSE). Its applicability to separately distributed vectors remains unestablished. The card appropriately limits its claim to those vectors. I could not independently inspect the Google archive’s JavaScript-rendered wording.

4. **Test coverage, dependency isolation and repository rules**

   **Medium: The shared test fixture requires undeclared optional SciPy availability.**  
   [test_hub_classics.py:21](/private/tmp/dr157-review-702f3c0/tests/test_hub_classics.py:21) imports `verify_hub_classics` for every test. That module imports `scipy.stats` immediately. The repository’s musllinux test requirements omit SciPy, gensim, and scikit-learn. Where benchmark files are present, these tests fail during fixture setup instead of respecting optional dependency isolation.  
   **Fix:** Split fixtures by dependency. Keep frequency and persistence tests independent of SciPy. Explicitly skip SciPy-specific tests when unavailable.

   Coverage also leaves the selection workflow, corpus-cache validation, interrupted publication, and FastText weighting untested. The existing numerical test checks folding linearity, not the complete fitted recipe. Add focused regression cases alongside the fixes above.

   No findings in the changelog entry, source-file limits, or runtime dependency boundaries. The preparation requirements pin their listed packages. MTEB’s pin exists separately in `benchmarks/quality/requirements.txt`; linking that setup explicitly would make reproduction clearer.

Done: Reviewed the diff, surrounding implementation, historical recipe, and upstream license sources.  
Open: Seven defects and the stated source-verification limits. Tests, timings, uploads, and file changes were not performed.  
Needs from you: nothing.
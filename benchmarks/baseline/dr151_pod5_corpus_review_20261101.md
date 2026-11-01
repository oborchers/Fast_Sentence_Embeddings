The hash alias correctly proves identical consumed tokens and paragraph order. I found one low-priority test coverage gap.

1. **Incorrect acceptance, missing validation, historical provenance: No findings.**

   [pod5_corpus.py:19](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-quality/.review/pod5-corpus/benchmarks/quality/pod5_corpus.py:19) uses the same UTF-8 line iteration and `line.split()` as `classic_corpus.batches`. Its serialization preserves token order, paragraph boundaries, and empty paragraphs. Therefore, it also preserves `frequencies` counts.

   Acceptance requires the historical file’s raw hash, its paragraph count, and every artifact’s recorded raw hash to match the proof. The historical record changes only in a deep copy. `checked_row` still checks the other fit fields and source-specific FastText provenance.

   Independent read-only inspection reproduced every saved proof field:

   | Evidence | Verified value |
   |---|---|
   | Historical raw hash | `xxh3-128:c89c1a2929270add9040835399b8bea4` |
   | Canonical hash | `xxh3-128:ca7a595ff0547a4a847a849d749d3604` |
   | Paragraphs | 859,346 |
   | Changed lines | 6,063 |
   | Bytes, historical → canonical | 497,531,589 → 497,523,659 |

   All 159 saved scores and aggregate means match the collected results. All 338 transfer members match their checksums. Every task’s fit record equals its artifact’s record.

2. **Tests, reproducibility, resource use: One finding.**

   **P3, low: The tests do not independently connect the proof to its consumers.**  
   Locations: [test_pod5_corpus.py:16](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-quality/.review/pod5-corpus/tests/test_pod5_corpus.py:16), [test_pod5_audit.py:181](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-quality/.review/pod5-corpus/tests/test_pod5_audit.py:181).

   **Evidence:** Corpus fixtures derive expected hashes through `corpus_hashes` itself. No test compares actual `batches` or `frequencies` outputs. The full audit test uses identical hashes and supplies no historical corpus, so it bypasses the new alias path.

   **Fix:** Add independently specified corpus pairs and compare their actual consumer outputs. Exercise alias acceptance through `audit(..., historical_corpus=path)`, including rejection when another fit field differs.

   **Reproducibility and resource use: No findings.** The saved proof reproduced exactly. Processing streams two passes, with an 8 MiB raw buffer and per-line normalization.

Done: reviewed `227170a` against `dc69ea5`; no modifications, tests, MTEB, or model loads.  
Open: low-priority test coverage gap.  
Needs from you: nothing.

## Owner response

The P3 finding is valid and fixed. Independently specified corpus pairs now compare the actual `classic_corpus.batches` and `frequencies` consumers, including an empty paragraph and two-row batching. The full collection audit test now covers the alias path and rejects an altered SIF `a` parameter. Production acceptance logic is unchanged. All 57 focused tests and static gates pass after the correction.

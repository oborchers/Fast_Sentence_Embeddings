The numerical recipe matches the inspected contracts. I found six issues in evidence validation, reproducibility, and reporting. These should be fixed before treating generated cards and reports as verified deliverables.

I inspected implementation, focused tests, historical arithmetic, MTEB 2.26.25 source, recorded JSON, and local candidate headers. I ran no tests or benchmarks and changed no files. Earlier reviews and green-test logs were not evidence.

**1. Numerical recipe: No findings**

- Frequency counts, float32 probability normalization, and `a=0.001` match the inspected fse 1.x arithmetic.
- FastText uses `vectors_vocab` for known words. Unknown words use ngram rows and `max(weights)`, including after folding.
- Ten uncentered components are fitted before folding. Both word and ngram tables are projected, then quantized.
- Historical test-set fitting remains separate and does not export deployable models.
- Folding changes rounding relative to sentence-level removal. The surrounding contract explicitly permits that distinction.

**2. Scientific validity and measurement binding**

**Finding 1, High: The report can combine unrelated models, verification results, and corpus selections.**  
Location: [report_hub_classics.py:98](/private/tmp/dr157-review-f320afa/benchmarks/report_hub_classics.py:98), [report_hub_classics.py:81](/private/tmp/dr157-review-f320afa/benchmarks/report_hub_classics.py:81).

`inventory()` checks folder files against their manifests. The report then independently loads `selection.json` and verification rows without checking their relationship to those models. It never checks each source’s `selection_xxh3`. It also does not require six unique production verification rows per source or enriched cards and `preparation.json`.

Consequently, intact folders can receive stale scores or a different corpus winner in the final report. Thirty folders alone also allow claims about card enrichment that has not occurred.

**Fix:** Validate the complete delivery graph before rendering. Match every verification row to its model and fit state. Require the expected rows, enriched cards, and preparation receipts. Match every source’s selection checksum to the supplied selection.

**Finding 2, Medium: Card validation accepts invalid scores and incompatible benchmark evidence.**  
Location: [hub_classic_evidence.py:87](/private/tmp/dr157-review-f320afa/benchmarks/hub_classic_evidence.py:87).

`validate_build()` checks model and state identity, but ignores benchmark identity, pair counts, metric bounds, and finiteness. It also validates only the current model’s row. `measured_text()` takes the raw and SIF float32 comparison scores from other unchecked rows.

Concrete evidence appears in [test_hub_classic_cards.py:72](/private/tmp/dr157-review-f320afa/tests/test_hub_classic_cards.py:72): accepted fixtures omit benchmark identity and pair counts entirely. A matching model row containing `NaN` would reach formatting unchanged.

**Fix:** Apply a shared verification schema before card generation. Check all consumed rows, benchmark/preprocessing identities, counts, finite scores, and bounds. Recompute reported differences from validated parent scores.

The corpus selector itself correctly requires all 13 NanoBEIR tasks per candidate, finite bounded scores, matching dataset revisions, and the specified three-way mean. The recorded STS scores are finite. Local candidate header fingerprints match the committed receipts. That header inspection does not independently verify tensor contents or recorded measurements. The committed evidence does not yet establish a corpus winner.

**3. Reproducibility, safety, provenance, dependencies, and tests**

**Finding 3, Medium: Preprocessing identity omits the custom preprocessing implementation.**  
Location: [hub_classic_evidence.py:36](/private/tmp/dr157-review-f320afa/benchmarks/hub_classic_evidence.py:36), [hub_classic_corpus.py:12](/private/tmp/dr157-review-f320afa/benchmarks/hub_classic_corpus.py:12).

The identity contains NLTK’s version and Punkt resource checksums. It excludes the custom lowercase, punctuation, contraction, and hyphen rules. Changing those rules leaves the identity unchanged. During resumed NanoBEIR evaluation, old and new task results could therefore pass selection together.

**Fix:** Add a preprocessing recipe version or implementation digest to corpus manifests, fit receipts, and every measurement. Reject mixed versions. Include the evaluation wrapper’s identity where its behavior affects scores.

**Finding 4, Medium: Interrupted card generation prevents a clean rerun.**  
Location: [hub_classic_cards.py:185](/private/tmp/dr157-review-f320afa/benchmarks/hub_classic_cards.py:185).

The script replaces `README.md`, writes the preview and preparation receipt, then updates the file manifest. An interruption or preview-write failure after replacing the README leaves its old checksum in the manifest. The next run fails inside `validate_build()` before it can repair the card.

**Fix:** Publish card enrichment and its manifest through a recoverable transaction or staged generation. Add interruption coverage between each publication step.

**Finding 5, Low: Fresh comparison artifacts omit source provenance present in the recorded receipts.**  
Location: [compare_hub_fit.py:25](/private/tmp/dr157-review-f320afa/benchmarks/compare_hub_fit.py:25), [compare_hub_fit.py:63](/private/tmp/dr157-review-f320afa/benchmarks/compare_hub_fit.py:63).

The receipt generator never writes `source_repo` or `source_revision`. Candidate construction also omits model provenance. Both inspected candidate headers consequently lack `fse.source`, while the committed fit JSON contains source fields. Those fields cannot be reproduced by this generator.

**Fix:** Generate the pinned source identity directly in the receipt and candidate metadata. Validate it when consuming candidate evidence.

No findings in optional runtime dependency isolation or upload behavior. Heavy preparation dependencies remain outside core dependencies, and these scripts contain no upload operation. Existing candidate/model overwrite guards fail safely. Focused tests cover useful primitives, but do not exercise complete selection, report validation, or interrupted card publication.

**4. Card/report claims and licenses**

**Finding 6, Medium: The report supplies an inapplicable explanation for raw-model differences.**  
Location: [report_hub_classics.py:90](/private/tmp/dr157-review-f320afa/benchmarks/report_hub_classics.py:90).

Every difference above 0.5 points receives the same explanation about training-corpus components and component-folding rounding. This branch includes raw models, which have neither operation. For SIF, it also presents possible causes without measurements establishing their contribution.

**Fix:** Generate explanations by method. State unexplained differences as unresolved. Attribute causes only where a controlled comparison supports them.

No findings in the requested source-license labels and notices:

| Source | Assessment |
|---|---|
| GloVe | `pddl`, correctly separated from the implementation license. Supported by the [authors’ release page](https://github.com/stanfordnlp/GloVe). |
| Google News | `other`, with the project research-only disclaimer. The card does not assign Apache-2.0 to vectors. |
| FastText | `cc-by-sa-3.0`, attribution, ShareAlike, changes, and non-endorsement are stated. The [official vector page](https://fasttext.cc/docs/en/english-vectors.html) confirms the license. |
| ParaNMT/Paragram | `unknown`, preservation, and removal notices avoid asserting permission. ParaNMT correctly distinguishes vector rights from [CzEng’s research restriction](https://ufal.mff.cuni.cz/czeng/czeng16). |

Wordfreq attribution names Robyn Speer, distinguishes code from frequency-data terms, describes the transformation, and links its [NOTICE](https://github.com/rspeer/wordfreq/blob/master/NOTICE.md). Whether distributing folded weights requires additional licensing or notices remains a release question. The code does not establish that clearance. Likewise, preservation wording does not establish redistribution permission for unknown-license vectors.

Done: Reviewed all four categories directly.  
Open: Six findings and unresolved release questions. Missing production folders/cards were not counted as defects.  
Needs from you: nothing.
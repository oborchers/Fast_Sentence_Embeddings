# Classic word-vector Hub preparation evidence

DR-157 (Ausgewählte Modelle auf den Hugging Face Hub laden), draft PR #201 (2.0: Prepare the classic word-vector models for the Hub).

This folder secures measured results on the task branch. It excludes models, caches and corpus inputs. Large local files remain under `/Users/oliverborchers/fse-dr157`.

The GloVe corpus comparison is in progress. `corpus-comparison/` records both completed component fits and STS-B scores. Both tables passed a complete bitwise comparison against their source vectors, stored wordfreq weights and stored components. `candidate-audit.txt` records that audit and the pinned STS-B/tokenizer identities. The corpus winner is pending the complete 13-task NanoBEIR evaluation for both candidates.

`make-check-6e31282.txt` records 9,134 passing tests and 514 skipped tests. Later source changes require a final gate before delivery.

`reviews/initial-code-review.md` is the independent read-only review of commit 702f3c0. It found seven defects, addressed by subsequent commits. A final review remains required. No model was uploaded, no Hub write occurred, and no pod lifecycle action was taken by this worker.

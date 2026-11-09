# Civil Comments classifier preparation evidence

Work: DR-311 (Toxizitäts-Classifier aus google/civil_comments außerhalb von fse
trainieren und mit Modellkarte vorbereiten). These are local correctness and
quality results on Apple M1 Pro, not speed measurements. No Linux comparison
was run. Source, package and compiler versions are in `verified.json`.

Current state: model selected and artifacts verified. Official test is held by
the coordinator while validation-only threshold proposals are reviewed. No test
result or completed model card is claimed at this stage.

| File | Meaning |
|---|---|
| `protocol.md` | Selection and variant rules declared before outcome inspection. |
| `selection.json` | All six encoder/C candidates, full-validation per-label metrics, sample digest and final small Bekko choice. |
| `probe.json` | Both encoder adapters pass the initial synthetic-label functional checks. These are not quality estimates. |
| `verified.json` | Numerical bounds, persistence bits, score stores, IDs and vectors for the selected actual head. |
| `threshold-proposal.json` | Optional per-label maximum-F1 thresholds on full validation only, before any test access. Saved heads remain unchanged. |

Training is entirely in scikit-learn outside fse. The pinned sources and recipe
are under `tools/hub/heads/`. Data, vectors, sklearn estimators and the complete
and standalone models are external. No model weights or texts are included here.

The validation threshold proposal uses the saved standalone fse model's float32
probabilities. These can tie after sigmoid rounding, especially at one. Its
ranking metrics therefore differ slightly from the sklearn candidate metrics,
which use float64 probabilities. No calibration claim is made.

The first two short functional probes found invalid sink calls: `predict`
requires `resume=True`, and document IDs require signed int64. Both were fixed.
The third probe passed for both encoders. No candidate training or official test
evaluation was repeated. An independent exhaustive threshold oracle passed 200
small tied-score cases, including scores exactly zero, one-half and one. A binary
metric probe confirms actual class decisions are retained at rounded ties and
rejects missing decisions.

Two independent Codex CLI review rounds ran on commits ad210c63 and ebe5b568.
The first found five issues: mutated sklearn probe input, missing memory
admission/watchdog, incomplete 18:00 admission, non-transactional completion and
hardcoded excluded-label counts. All were fixed and confirmed by the second
round. Two new warnings in shared helpers were also fixed: use actual binary
class predictions, and restore non-mutating scaler inference after in-place
training. The later validation-threshold routine is assigned to the coordinating
agent's independent final diff review, with no third Codex round.

Full `make check`, final held-out results, card and manifest remain pending.

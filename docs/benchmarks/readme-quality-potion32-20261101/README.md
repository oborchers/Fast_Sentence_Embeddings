# POTION-32M float32 quality sample, 1 November 2026

This is a separate quality sample, not a controlled model comparison.
DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen) published
it at 7bb78dd5e4af7fdccc6dc8511ae752bd582fb4b2. Raw task records, the original
summary and invocation-wall evidence here are byte-identical snapshots.

## Coverage and means

| Task group | Completed | Mean task score, points |
|---|---:|---:|
| MTEB(eng, v2), completed tasks only | 40 of 41, MIND deferred | 55.2573 |
| NanoBEIR search tasks | All 13 of 13 | 46.1754 |

The English mean is **not the complete 41-task benchmark score**.
MindSmallReranking was deferred exactly as authorized by the coordinator.
Its score and the full-41 English mean remain null. The raw journal has
54 unique task names: 53 completed finite scores and one deferred task.

Each completed task has equal weight. Multiply the native mean by 100 to
get points: English 0.552573112515375 becomes 55.2573112515375, and NanoBEIR
0.4617538461538462 becomes 46.175384615384615. The table rounds to four decimals.
All 53 task scores and metric names match the published summary.
All 54 tasks have pinned dataset revisions. English uses its task-specific
metrics; all 13 NanoBEIR tasks use ndcg_at_10. The two task groups are separate results.

## Configuration and comparison limit

Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4, MTEB 2.26.25,
tokenizers 0.23.3. Measured 1 November 2026 on CPU, with two fse workers.
Model minishlab/potion-base-32M, revision 1e5a03f8eeb2c98b928fbbd846f22f816360919f.
Float32 table, 63,091 rows, 512 output values, saved file 130,145,513 bytes.
Encoder fingerprint xxh3-128:99737b0f0d2dc7cfb4ee240faa2e278f matches the bound
published artifact and the committed saved-file observations.
The recorded encoder uses wordpiece@1, drops unknown ids, takes the first
512 tokens of the first 3,584 characters and normalizes outputs.
No cache model tensors were read or copied for this documentation.

Recorded source: 0de9c3ddc66722581722a5ed732a78cb19b6fe19.
Method fingerprint: xxh3-128:b7cde152974d96e1f2b5baa0fe736d5c.
Runtime fingerprint: xxh3-128:e0f772e2be1bfde21414ce3f403b0e21.
Suite fingerprint: xxh3-128:3382243b7faea5e5c71ba0a52f2bf4a4.

Task names and metadata match the earlier POTION-8M sample. However, its
recorded method and runtime fingerprints differ. A POTION-8M confirmation
with the matching method is queued. These means do not establish a controlled
ranking against that earlier sample. No cross-model difference, quantized
quality delta or library-quality equality is claimed.

The raw records retain differing versions.fse and per-row fse version strings.
They are preserved, not used to infer identical code across runs. Both fingerprint
sets and source commits remain explicit. verified-means.json contains the independent calculation.

## Invocation wall time is not model compute time

invocation-wall.json records 3,064.1069151250413 seconds, rounded to 3,064.107.
It includes the manual priority SIGSTOP hold, timing-marker and foreign-job
waits, imports, setup, evaluation and cooldowns. Approximate state buckets are
not measurements of pure model compute. No throughput, compute duration or
cloud runtime is inferred from this elapsed invocation.

## Evidence and reproduction

- [Original raw task records](dr151-quality-potion32-float32.jsonl)
- [Original published summary](published-summary.json)
- [Independently calculated means](verified-means.json)
- [Original invocation-wall evidence](invocation-wall.json)
- [Producing evaluator](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/0de9c3ddc66722581722a5ed732a78cb19b6fe19/benchmarks/quality/run_mteb.py)
- [Identity protocol](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/0de9c3ddc66722581722a5ed732a78cb19b6fe19/benchmarks/quality/protocol.py)
- [Published coverage reporter](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/7bb78dd5e4af7fdccc6dc8511ae752bd582fb4b2/benchmarks/quality/report_precision.py)
- [Bound artifact manifest](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/7bb78dd5e4af7fdccc6dc8511ae752bd582fb4b2/benchmarks/baseline/dr151_model_artifacts_20261101.json)

From the producing checkout, using its quality environment and prepared artifact:

```sh
nice -n 10 .venv-quality/bin/python benchmarks/quality/run_mteb.py run \
  --models fse/potion-base-32M-float32 --workers 2 --batch-tasks 16 --defer-mind \
  --fse-models /path/to/prepared/models \
  --coordination /path/to/shared/scratchpad --results quality.jsonl
```

Use the evaluator's recorded task settings; no separate seed override is
recorded. Follow the current shared load rules and slot order. No quality run
was repeated by this documentation worker. Per-task seconds and invocation
wall time are retained as provenance, not accepted speed measurements.

# POTION-retrieval-32M float32 quality sample, 1 November 2026

This sample uses the fixed runtime recording. It is a separate evaluation
cohort, not a controlled ranking against the earlier P8/P32 checkpoints.
DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen) published
it at 6e7fc5e6454f41614bb0fe99f1141801ddab2251. Raw task records, original
summary and invocation-wall evidence here are byte-identical snapshots.

## Coverage and means

| Task group | Completed | Mean task score, points |
|---|---:|---:|
| MTEB(eng, v2), completed tasks only | 40 of 41 tasks, MIND not computed | 54.5098 |
| NanoBEIR search tasks | All 13 of 13 | 50.3872 |

The English mean is **not the complete 41-task benchmark score**.
Final MIND status: **not computed: MTEB 2.26.25 MindSmallReranking needs more memory than the 64 GB pod (two attempts, status 137 and earlier ENOSPC)**. No further MIND runs.
The reported English mean covers **40 of 41 tasks**.
The historical snapshot marked MindSmallReranking deferred. Its score and the full-41
English mean remain null. The journal contains 54 unique task names, with
53 completed finite scores and one deferred record. All task dataset revisions are pinned.

Each completed task has equal weight. The native English mean is
0.5450983952160903; the native NanoBEIR mean is 0.5038723076923077.
Multiplying each by 100 and rounding to four decimals gives 54.5098 and 50.3872
points. verified-means.json records independent calculations. All 53 task
scores and metric names match the published summary. English combines its
task-specific metrics; all 13 NanoBEIR tasks use ndcg_at_10.

## Configuration and recorded runtime

Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4, MTEB 2.26.25,
tokenizers 0.23.3. Measured 1 November 2026, CPU, two fse workers.
Model minishlab/potion-retrieval-32M, revision
6fc8051fab2a1e0ee76689cf08c853792ac285e7, float32, 63,091 rows and 512 output values.
The file is 130,145,513 bytes. Encoder fingerprint
xxh3-128:6f78af26bab53a3d464e97d9189f5b20 matches the bound published artifact
and committed saved-file observations. The encoder uses wordpiece@1,
drops unknown ids, takes the first 512 tokens of the first 3,584 characters
and normalizes output. No cache model tensors were read or copied here.

All 54 rows record comparison_identity_version=2, the loaded package location,
and the same runtime/configuration identity. Recorded source:
300bde80d2bb7542f9051da8352707d86dccf9a1.
Method: xxh3-128:42d579e533e130ec31285bc410413e3c.
Runtime: xxh3-128:afed836a8d25546883efce941d590317.
Suite: xxh3-128:3382243b7faea5e5c71ba0a52f2bf4a4.

The raw records preserve differing legacy per-row fse and versions.fse labels;
the deferred row has no legacy fse alias. These labels do not establish code
identity. The captured runtime fingerprint and package location remain explicit.

The earlier POTION-8M and POTION-base-32M samples have different method/runtime
identifiers. Both await matching-code confirmations after the recording fix.
No cross-model ranking, improvement, quantized delta or library equality is
inferred. This is currently one configuration in its recorded cohort.

## Elapsed invocation is not model compute

The matching record in invocation-wall.json has wall_seconds
1102.8503631660715, rounded to 1,102.850 seconds. It includes setup, evaluation,
shared waits, cooldowns and the store timing hold. Approximate own-heavy time
is not pure model compute. Neither this elapsed invocation nor per-task
seconds is promoted as accepted throughput or a cloud runtime estimate.
The source owner's remaining-work planning scenario is not a measured runtime
for any cloud machine and is not used in the README.

## Evidence and reproduction

- [Original raw task records](dr151-quality-retrieval32-float32.jsonl)
- [Original published summary](published-summary.json)
- [Independently calculated means](verified-means.json)
- [Original invocation-wall records](invocation-wall.json)
- [Producing evaluator](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/300bde80d2bb7542f9051da8352707d86dccf9a1/benchmarks/quality/run_mteb.py)
- [Identity protocol](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/300bde80d2bb7542f9051da8352707d86dccf9a1/benchmarks/quality/protocol.py)
- [Published coverage reporter](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6e7fc5e6454f41614bb0fe99f1141801ddab2251/benchmarks/quality/report_precision.py)
- [Bound artifact manifest](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6e7fc5e6454f41614bb0fe99f1141801ddab2251/benchmarks/baseline/dr151_model_artifacts_20261101.json)

With the producing checkout, its quality environment and prepared model:

```sh
nice -n 10 .venv-quality/bin/python benchmarks/quality/run_mteb.py run \
  --models fse/potion-retrieval-32M-float32 --workers 2 --batch-tasks 16 --defer-mind \
  --fse-models /path/to/prepared/models \
  --coordination /path/to/shared/scratchpad --results quality.jsonl
```

Use the recorded evaluator/task settings; no separate seed override is recorded.
Honor current shared locks, timing windows, pause rules and slot priority.
No quality or timing run was repeated by this documentation worker.

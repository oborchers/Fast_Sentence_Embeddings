# First POTION-8M float32 quality sample, 1 November 2026

This is one model configuration, not a comparison or model ranking.
DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen) published
the raw records and summary at 0de9c3ddc66722581722a5ed732a78cb19b6fe19.
Both files here are byte-identical snapshots of that commit.

## Coverage and means

| Task group | Completed | Mean task score, points |
|---|---:|---:|
| MTEB(eng, v2), completed tasks only | 40 of 41, MIND deferred | 53.9527 |
| NanoBEIR search tasks | All 13 of 13 | 43.8485 |

The English mean is **not the complete 41-task benchmark score**.
MindSmallReranking was explicitly deferred by the coordinator. Its raw record
has status deferred and a null score. The full-41 English mean stays null.
The other 53 raw records have status completed and finite task scores.
Task names are unique: 40 English tasks, 13 NanoBEIR tasks and one deferred MIND task.

Each completed task has equal weight. Points are 100 times the mean of the
native main task scores. English tasks use accuracy, v_measure,
cosine_spearman, map_at_1000, max_ap or ndcg_at_10. All 13 NanoBEIR tasks use
ndcg_at_10. The groups contain different tests and metrics, so their means
are separate results. These are not timing measurements or A/A noise bands.

Raw native means: 0.5395270762114082 for the completed English 40 and
0.4384853846153846 for NanoBEIR 13. Multiplying by 100 and rounding to four
decimals gives the table. verified-means.json records this independent calculation.
All 53 per-task scores and metric names were checked against the published summary.
Every completed task's raw metadata records a pinned dataset revision.
No missing task was replaced by zero, a subset estimate or a neighborhood metric.

## Configuration and provenance

Apple M1 Pro, macOS 27.0.1, Python 3.13.16, NumPy 2.5.4, MTEB 2.26.25,
tokenizers 0.23.3. Measurement date: 1 November 2026. CPU, two fse workers.
The model is minishlab/potion-base-8M, revision
bf8b056651a2c21b8d2565580b8569da283cab23, with its float32 table and 256 output values.
Its recorded encoder description uses wordpiece@1, drops unknown ids, takes
the first 512 tokens of the first 3,072 characters, and normalizes outputs.

Encoder fingerprint: xxh3-128:7aed2ca9bf7f9073ffa6778672a7575f.
The raw loaded fingerprint matches the published bound artifact and the
committed saved-file observations. That file is 30,664,410 bytes and has
29,528 rows. No model tensor data was read or copied for this documentation.

All records share source commit 265e70de7d1e343a19d26ef43e911061478f50a1,
method fingerprint xxh3-128:7fe0b281fc9a9bc7b53313a3300fdf41,
runtime fingerprint xxh3-128:3f93e3744d1eda78ee3c5636604ab1cc and
suite fingerprint xxh3-128:3382243b7faea5e5c71ba0a52f2bf4a4.
The raw records retain both versions.fse and per-row fse version strings,
which differ. We preserve both rather than infer code identity from a version label.
The recorded source commit and runtime fingerprint identify this sample.

No other model or precision has quality results in this snapshot. Quantized
float32 deltas remain null in the published summary. No model/dtype ranking,
quality equality with another library or speed conclusion follows.

## Evidence and reproduction

- [Original raw task records](dr151-quality-primary.jsonl)
- [Original published summary](published-summary.json)
- [Independently calculated means](verified-means.json)
- [Producing evaluator](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/265e70de7d1e343a19d26ef43e911061478f50a1/benchmarks/quality/run_mteb.py)
- [Evaluation identity protocol](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/265e70de7d1e343a19d26ef43e911061478f50a1/benchmarks/quality/protocol.py)
- [Published coverage/delta reporter](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/0de9c3ddc66722581722a5ed732a78cb19b6fe19/benchmarks/quality/report_precision.py)
- [Bound artifact manifest](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/0de9c3ddc66722581722a5ed732a78cb19b6fe19/benchmarks/baseline/dr151_model_artifacts_20261101.json)

From the producing checkout with its pinned quality environment and prepared
artifact, run the complete selected suites with the authorized MIND deferral:

```sh
nice -n 10 .venv-quality/bin/python benchmarks/quality/run_mteb.py run \
  --models fse/potion-base-8M-float32 --workers 2 --defer-mind \
  --fse-models /path/to/prepared/models \
  --coordination /path/to/shared/scratchpad --results quality.jsonl
```

The runner uses the pinned evaluator's task settings; no independent seed
override is recorded. Honor the shared heavy lock, pause and timing-window
rules. No quality run was repeated by the documentation worker. The per-task
seconds and load fields are retained as provenance, not accepted throughput.

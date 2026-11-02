# Required rows for the next README speed panels

The current README table uses only `dtypes-02` through `dtypes-07` from the compact `speed-2026-11.json` study.
See [the current evidence guide](readme-evidence.md#current-speed-cohort). These six cells measure 1/16 workers on one EPYC 9965 run.
The maintainer holds the headline and library panel for a benchmark decision. Visible draft placeholders are permitted.
The older 32/16 selection layout below is a future plan, not a result from this current cohort.
`media/readme/speed_panels.py` prepares the table and light/dark library bars.
The empty `speed_panel_data.json` produces no publishable panel. Unmeasured previews live under `media/readme/layout/` only.

## Machine settings

Worker counts follow the target Pod (32 vCPU), not the old Mac setting.
One core always remains selected. Choose the second setting from accepted float32 fse cells for each model.

- Show **all 32 cores** only when the unrounded primary median at 32 divided by that at 16 is **at least 1.20**.
- The boundary is inclusive. Otherwise show **16 cores**.
- Read the 16 and 32 cells directly. No 2, 4 or 8 worker points or scaling curve are required or displayed.
- When 32 fails the threshold, add “More than 16 cores adds little here.” Name the affected model in the library figure.
- Record the exact medians, ratio and immutable sources in `core-selection-evidence.md`.

Each cell retains three accepted rotating fse-a/fse-b rounds. Missing or rejected 16/32 cells cannot justify either choice.
There is no scaling-curve graphic. All README speed values come from RunPod; earlier Mac observations stay in the evidence guide.

## Compact table: one core plus 16 or 32

Use POTION-8M, one pinned model revision and float32 parent artifact, complete WikiText encoding input,
one Pod machine and date, and float32 vector output. Preserve the cohort's exact digest, newline rules and tokenizer limits.

| Public row | Required DR-151 selectors (fse-a and fse-b), W=1,16,32 |
|---|---|
| float32 | encode/minishlab/potion-base-8M/float32/wikitext/W/table |
| float16 | encode/minishlab/potion-base-8M/float16/wikitext/W/table |
| int8 | encode/minishlab/potion-base-8M/int8/wikitext/W/table |

Collect all nine cells during the same Pod run, without a follow-up Pod round.
After core selection the renderer uses only workers=1 and workers=K, keeping the other observations in evidence.
No one-core-only or 32-worker int8 cell can fill a missing K-worker cell.
All selected dtype cells must share one input digest/count, machine/runtime and context with the selection rows for their model.
K is always 16 or 32. The 1/16/32 grid therefore contains all possible selected cells; no second Pod run or per-model follow-up request is needed.

Planned public layout:

| fse table | One core, texts/s | Selected setting, texts/s |
|---|---:|---:|
| float32 | measured value | measured value |
| float16 | measured value | measured value |
| int8 | measured value | measured value |

One caption: Pod machine (32 vCPU), POTION-8M, WikiText, date. Average text: measured words.
The table sits in Speed first, beside the library comparison and the single “How we measured” link.
No placeholder table appears in the main README.

## Library bars: measured core settings per library

Keep **One core**. Add only the setting selected by that model's own 16/32 decision, labelled **All 32 cores** or **K cores**.
Each bar needs its own matched fse baseline. The MRL model needs its own 16/32 cells; POTION-8M cannot justify its 32-core group.
Use float32 tables and float32 vector output, with the same dimension and normalization within each pair.

| Library | Required paired selectors, with W=1,16,32 | Operations | Same model in both libraries |
|---|---|---|---|
| Model2Vec | encode/minishlab/potion-base-8M/float32/wikitext/W/model2vec | fse-a, fse-b, model2vec | minishlab/potion-base-8M |
| model2vec-rs | encode/minishlab/potion-base-8M/float32/wikitext/W/rust | fse-a, fse-b, rust | minishlab/potion-base-8M |
| FastEmbed | encode/minishlab/potion-base-8M/float32/wikitext/W/fastembed | fse-a, fse-b, fastembed | minishlab/potion-base-8M |
| Sentence Transformers | encode/sentence-transformers/static-retrieval-mrl-en-v1/float32/wikitext/W/st | fse-a, fse-b, st | sentence-transformers/static-retrieval-mrl-en-v1 |

Collect all twelve paired series at workers 1, 16 and 32 in the same Pod run.
Collect float32 fse selection cells for both models at 1, 16 and 32 in this run. No separate Pod round is planned.
The renderer filters the measured grid to one core and the selected K for each model.
K is 16 or 32. Settings 2, 4 and 8 are outside this run and these panels.
The Sentence Transformers numerator must run that same MRL model.
Never compare POTION-8M fse with the MRL reference and label it a library speedup.
Keep model names visible under every bar; each bar is a same-model comparison despite different static models across rows.
The generator permits a library only when all of its selected core settings are accepted.
The figure shares one machine, date, source corpus and output dtype. Each bar has its own paired workload, calibrated separately.
Within a pair, fse and the library must share the exact input digest/count, model, output settings and worker budget.
Different bars may use different counts and digests. Never describe them as one shared input.
The common input for the dtype table and 16/32 selection remains separate and unchanged.
A requested thread limit is not proof of observed tokenizer threads. Preserve exact library settings in the evidence.
If output semantics differ, correct or omit that pair before publication.

A bar uses the median of three paired fse/reference rate ratios in round order, equivalent to reference-time/fse-time for equal input counts.
Do not divide two independent medians. All selected groups share the same scale and zero origin.
Public labels say “N times as fast as Library”. The caption names machine, source corpus, output and date.
It states that each bar is its own paired comparison and lists the exact input size for each library/core bar.
The renderer derives these counts from the paired rows and also prints them beside each bar. Group headings give core settings.
MiniLM is absent from this same-model renderer. If shown later, it needs a clearly separate “Different model” group and both model names.

## Remaining Pod 4 scope

Keep the corrected folded-classification comparison, separate from encoding library bars.
Retain its exact head, normalization, dataset, workers and output semantics in evidence.
Do not publish a folded-classification speedup without accepted committed raw comparisons.

Drop the 10M write-path stage. The committed 100M showcase already covers writing.
Keep large encoding/classification inputs and the per-call two-second rule.
Every first and warm fse call must last at least two seconds, including both fse-a and fse-b.
A long series or a median above two seconds cannot hide an individual short call.
This follows the Pod source's fse-call duration gate; it does not impose a new duration rule on reference libraries.

## Acceptance and extraction contract

Fill `speed_panel_data.json` only after auditing committed raw rows and strict summaries against DR-151's current rules.
The renderer does not replace the independent source audit. Keep measurement-process details here, not in the public README.
All fse baselines within a panel must share one executed-runtime fingerprint.
Each cell needs three complete accepted rotating rounds, paired fse-a/fse-b, backend-specific executed identity,
verified artifact lineage, and symmetric A/A upper bound at most 1.10. Apply the source's other validity checks,
including old Mac load rules or Pod host/probe rules. Acceptance never transfers between machines.
Keep publication commit, journal path, exact series and ordered primary rates from the same three rounds.
Reference rows must identify their own package or Rust binary, not fse's fingerprint. Check common output semantics first.

The format is `readme-speed-panels-v1`, with committed `selection_rows` rows, `table` rows and `comparisons` pairs.
Rows contain `complete`, `runtime_attributed`, `aa_upper`, `backend`, `runtime_fingerprint`,
`source` (`commit`, `path`, `series_id`), `context`, and three ordered `texts_per_second` values.
For fse, `call_seconds` is a list of three rounds. Each round has `fse-a` and `fse-b`,
each containing `first` (seconds) and `warm` (all recorded warm-call seconds).
Include every call from the accepted source operations. Missing rounds, backends or first/warm calls fail validation.
Extract actual durations, not the configured `minimum_call_seconds` threshold. The renderer rejects any call below two seconds.
`CONTEXT` in the renderer lists the required model/input/output fields. `parent_artifact` identifies original weights, not quantized bytes.
A pair contains `fse`, `reference` and `output_checked: true`, added only after verifying output semantics.
Both rows name the same combined paired journal and series. Preserve original records in the evidence snapshot.
`table_caption` and `comparison_caption` contain one-line public captions copied from verified context.
The renderer derives core settings from `selection_rows` using the shared input for each model.
Dtype cells must match that input. Library bars may have their own inputs, but both arms of each bar must match exactly.
All displayed cells still require the selection model, runtime, machine, corpus and output context. It then
filters out other settings before rendering. It rejects undeclared missing chosen rows, mixed contexts, noisy/unattributed rows
and stale unselected output files before writing.
It never silently retains withdrawn claims or deletes files.

Run `python media/readme/speed_panels.py --preview` for the explicitly unmeasured layout.
After accepted data arrives, run without `--preview`, export SVGs with librsvg and inspect both themes.
The main visual generator calls the same publish function. Empty data adds nothing to the public README or final figures.

## Earlier preparation superseded

The earlier incomplete dtype preparation is not published in the README.
Use only the six current cells from the [compact speed source](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6cb13ed72321d1207fc106f6c0e7154935164017/benchmarks/baseline/speed-2026-11.json).
No earlier 32-worker cell may be combined with those 1/16-worker cells to claim a current 32/16 ratio.
The current source has all six dtype cells accepted, including int8 at 16 workers.
The previous source-label blocker and blank int8 preparation do not describe the current source.
The old archive remains historical provenance, without its superseded rates in this plan.

## Prepared quality panel: three historical classics

[Compact quality results](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/6cb13ed72321d1207fc106f6c0e7154935164017/benchmarks/baseline/quality-2026-11.json)
provide the following arithmetic task-score means, multiplied by 100:

| Model | MTEB English, 40 of 41 tasks |
|---|---:|
| FastText crawl subwords 300, SIF-10 | 51.42 |
| paragram 300 sl999, SIF-10 | 50.98 |
| word2vec Google News 300, SIF-10 | 49.26 |

The unrounded source means are 51.42144, 50.98332 and 49.26358 respectively.
All three use float32 and the historical WikiText-103 frequency/component protocol.
Their raw evidence is [archived at 660309c4c88930254288495a0bb025fe428870f7](https://github.com/oborchers/Fast_Sentence_Embeddings/tree/660309c4c88930254288495a0bb025fe428870f7/benchmarks/baseline/results/dr151-pod5-20261101).
These are not scores for the Hub's wordfreq/FineWeb SIF-10 models.
Do not combine historical POTION checkpoints with this panel as a controlled ranking without a cohort audit.
MIND is not computed: MTEB 2.26.25 MindSmallReranking needs more memory than the 64 GB pod (two attempts, status 137 and earlier ENOSPC).
The older EPYC 7713 headline is withdrawn. It cannot be mixed with the current speed cohort.

## Historical PERFORMANCE audit

The final maintainer decision withdraws every M1 speed measurement, regardless of input size.
The provenance status is independent of that decision. Missing raw files are not reconstructed from rounded prose.
No new timing was run. Durations below are seconds unless marked otherwise.

| PERFORMANCE section/table | Source | Texts per call or timed unit | Duration per call or timed unit | Provenance status | Decision |
|---|---|---|---|---|---|
| RunPod Linux x86 runtime columns | `benchmarks/baseline/dr190_runpod_20261031.json`, run 2, `runtime`; `synthetic_tokens.sentences` | 200,000 synthetic texts | 0.016802282072603703 to 0.15447962691541761, calculated as recorded count / exact recorded rate | Committed count and rates; duration derived, not an independent raw timer | Remains, large-input RunPod observation on EPYC 9655P, 32 assigned logical CPUs |
| RunPod Linux x86 bandwidth columns | Same source, run 2 `roofline`; `benchmarks/roofline_probe.py` | No text input; 1 GiB table probe | Original summary records GB/s, not individual call seconds | Rate and probe configuration retained | Remains as a memory probe, not a text throughput claim |
| RunPod sync costs | Same source, `runs[].sync_C0` | File/directory sync primitive, no text input | Recorded primitive p50 values, not embedding calls | Committed summary; two separately labelled hosts | Remains as filesystem latency, not a text throughput claim |
| Hardware profiles, M1 measured bandwidth | `benchmarks/roofline_hardware.toml`; original probe recordings absent | No text input; 1 GiB probe | Not independently recoverable from surviving raw files | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1; vendor specifications remain separate |
| Measured memory roofs | Original `roofline.py probe` recordings absent | No text input; 1 GiB probe reported in legacy prose | Not independently recoverable | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1 |
| Measured on M1, WikiText table | Original `roofline.py measure` JSON absent | Legacy prose says 5,352; not raw-verifiable | Not independently recoverable | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1 |
| Measured on M1, MASSIVE table | Same missing source | Legacy prose says 151,674; not raw-verifiable | Not independently recoverable | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1 |
| One-worker stage table and timing-derived limits | Same missing source | Both legacy corpora above | Only ns/token remain in prose; no verified per-call timer | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1 |
| Where workers idle: full and short tables, ablations | Original `runtime_timeline.py` recordings absent; commit `07be44b` contains scripts/prose only | Legacy prose reports 5,352 / 151,674 / short prefixes | Printed wall values remain historical prose, not raw verification | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1 |
| Tail split: corpus and synthetic gate tables | Original recordings absent; commit `d33cc54` contains scripts/prose only | Legacy prose reports corpus counts above and 200,000 synthetic texts | Not independently recoverable | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1 |
| Persistent-pool matrix, both model tables, full MASSIVE rows | `benchmarks/baseline/thread_pool_20261031.json`, `cells[].kept` | 151,674 | 0.023650499992072582 to 0.11612262506969273 across accepted base/control/reuse observations | Raw counts and seconds retained | zurückgezogen: M1 |
| Persistent-pool matrix, WikiText/full rows | Same source | 5,352 | 0.005041375057771802 to 0.022281333920545876 | Raw counts and seconds retained | zurückgezogen: M1; also small input |
| Persistent-pool matrix, short/server/inline rows | Same source | 32 / 500 / 5,000, or server ranges 76 to 260 and 452 to 2,417 | Individual/aggregate per-call values retained in `seconds_per_call`; not a single shared call size | Raw counts and seconds retained | zurückgezogen: M1; also small input |
| Persistent-pool deciding timeline | `benchmarks/baseline/thread_pool_profiles_20261031.json`, `timeline.variants` | 5,000 | Median traced wall 0.0019655 / 0.001819917 | Raw summarized timeline retained | zurückgezogen: M1; also small input |
| Store sync C1 | `benchmarks/baseline/store_sync_20261031.json`, `cells[].kept`; `benchmarks/bench_store_sync.py:encode` | One timed stream of 200,000 or 524,288; batches 1,000 / 10,000 / 65,536 | Complete timed stream: 0.10771750006824732 to 2.3293397499946877 across C1 observations | Raw whole-stream timer retained; batch size is not total timed input | zurückgezogen: M1 |
| Store sync C2/C3 | Same source | 250,000 synthetic texts / real file input; legacy C3 row count checked separately in none study | Whole-operation seconds retained per side | Raw observations retained; no invented per-batch encode duration | zurückgezogen: M1 |
| Store C0 primitive table | Same source, `C0` | No texts, dirty file/directory sync | Individual primitive summaries retained | Raw summary retained | zurückgezogen: M1 |
| Implemented durability none: stream and runtime gate | `benchmarks/baseline/store_none_20261031.json`, `measurements.runs[].command/result` | C1: timed 200,000-text stream, batches of 10,000; runtime gate uses synthetic benchmark corpus | Whole-stream and runtime rate records retained, distinct from process `wall_seconds` | Command and result retained | zurückgezogen: M1 |
| Implemented durability none: verified opens | Same source, `informative.reader/sink` | No text encoding; opens stored vectors or an empty sink | Retained reader/sink timers | Raw summary retained | zurückgezogen: M1 |
| int8 1024 comparison and earlier task-fusion claims | Original `roofline.py compare/measure` JSON absent | Legacy WikiText / MASSIVE counts, not independently raw-verifiable | Not independently recoverable | Herkunft nicht verifizierbar (Rohdaten verloren) | zurückgezogen: M1 |
| CPU shares, allocation/RSS tables, formulas, correctness checks | Corresponding study sources above | Not text-throughput measurements | Not applicable to speed publication | Retain their stated profiling/provenance limits | Remain as memory, correctness or explanatory material; no speed inference |

The original M1 raw files were never committed and were lost during the scratchpad cleanup, as confirmed by fse-main.
No source absence is presented as a measured failure. All M1 speed decisions above follow the maintainer's separate machine-wide withdrawal.

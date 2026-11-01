# Required rows for the next README speed panels

This is a delivery plan, not a benchmark result. Unfinished parts stay outside the main README.
`media/readme/speed_panels.py` prepares the table and light/dark library bars.
The empty `speed_panel_data.json` produces no publishable panel. Unmeasured previews live under `media/readme/layout/` only.

## Machine settings

Worker counts follow the target Pod (32 vCPU), not the old Mac setting.
One core always remains selected. The second setting is chosen automatically from committed accepted scaling cells **for each model**.

- Show **all 32 cores** only when median fse-a throughput at 32 divided by median fse-a throughput at 16 is **at least 1.20**.
- Compare the unrounded medians. The boundary is inclusive; 1.199999 does not qualify.
- Otherwise show **16 cores**, even when 2, 4 or 8 are faster on the curve.
- The public panels always show exactly one core plus either 16 or 32. Smaller settings appear only in the scaling curve.
- The generated `core-selection-evidence.md` records any smaller setting that beats 16, with exact medians and committed sources. Include that generated note in the evidence guide when publishing results.
- When 32 fails the threshold, add one plain line: “More than 16 cores adds little here.” The library figure names the affected model.

Each scaling cell has its own three rotating fse-a/fse-b rounds. The decision compares their primary medians;
it does not pretend the 16/32 cells are an additional interleaved cross-setting experiment.
Every curve must contain all six valid, coherent points. Missing/noisy points cannot justify a 32-core label or the fallback claim.
The renderer refuses a panel without its required curve rather than inventing a choice.
The existing 918,299 texts/s Mac headline at ten workers remains unchanged with its Apple M1 Pro caption.
Do not mix Mac and Pod cells in one table or speedup.

## Compact table: one core plus 16 or 32

Use POTION-8M, one pinned model revision and float32 parent artifact, complete WikiText encoding input,
one Pod machine and date, and float32 vector output. Preserve the cohort's exact digest, newline rules and tokenizer limits.

| Public row | Required DR-151 selectors (fse-a and fse-b), W=1,16,32 |
|---|---|
| float32 | encode/minishlab/potion-base-8M/float32/wikitext/W/table |
| float16 | encode/minishlab/potion-base-8M/float16/wikitext/W/table |
| int8 | encode/minishlab/potion-base-8M/int8/wikitext/W/table |

Collect all nine cells during the same Pod run, not by a follow-up request after each curve.
After scaling selection the renderer uses only workers=1 and workers=K, keeping the other observations in evidence.
No one-core-only or 32-worker int8 cell can fill a missing K-worker cell.
All selected cells must share one machine/runtime and context with the decision curve.
K is always 16 or 32. The 1/16/32 grid therefore contains all possible selected cells; no second Pod run or per-curve follow-up request is needed.

Planned public layout:

| fse table | One core, texts/s | Selected setting, texts/s |
|---|---:|---:|
| float32 | measured value | measured value |
| float16 | measured value | measured value |
| int8 | measured value | measured value |

One caption: Pod machine (32 vCPU), POTION-8M, WikiText, date. Average text: measured words.
The table sits below the existing throughput graphic and above the single “How we measured” link.
No placeholder table appears in the main README.

## Library bars: measured core settings per library

Keep **One core**. Add only the setting selected by that model's own scaling curve, labelled **All 32 cores** or **K cores**.
Each bar needs its own matched fse baseline. The MRL model needs its own curve; POTION-8M scaling cannot justify its 32-core group.
Use float32 tables and float32 vector output, with the same dimension and normalization within each pair.

| Library | Required paired selectors, with W=1,16,32 | Operations | Same model in both libraries |
|---|---|---|---|
| Model2Vec | encode/minishlab/potion-base-8M/float32/wikitext/W/model2vec | fse-a, fse-b, model2vec | minishlab/potion-base-8M |
| model2vec-rs | encode/minishlab/potion-base-8M/float32/wikitext/W/rust | fse-a, fse-b, rust | minishlab/potion-base-8M |
| FastEmbed | encode/minishlab/potion-base-8M/float32/wikitext/W/fastembed | fse-a, fse-b, fastembed | minishlab/potion-base-8M |
| Sentence Transformers | encode/sentence-transformers/static-retrieval-mrl-en-v1/float32/wikitext/W/st | fse-a, fse-b, st | sentence-transformers/static-retrieval-mrl-en-v1 |

Collect all twelve paired series at workers 1, 16 and 32 in the same Pod run.
Also collect both complete six-point model curves before the Pod ends. No separate Pod round or per-curve follow-up request is planned.
The renderer filters the measured grid to one core and the selected K for each model.
K is 16 or 32. Settings 2, 4 and 8 stay in the scaling curve only; they are never substituted into these library panels.
The Sentence Transformers numerator must run that same MRL model.
Never compare POTION-8M fse with the MRL reference and label it a library speedup.
Keep model names visible under every bar; each bar is a same-model comparison despite different static models across rows.
The generator permits a library only when all of its selected core settings are accepted.
The figure shares one machine, date, complete input digest and output dtype, with each pair sharing its own worker budget.
A requested thread limit is not proof of observed tokenizer threads. Preserve exact library settings in the evidence.
If output semantics differ, correct or omit that pair before publication.

A bar uses the median of three paired fse/reference rate ratios in round order, equivalent to reference-time/fse-time for equal input counts.
Do not divide two independent medians. All selected groups share the same scale and zero origin.
Public labels say “N times as fast as Library”. One caption names machine, input, output and date; group headings give core settings.
MiniLM is absent from this same-model renderer. If shown later, it needs a clearly separate “Different model” group and both model names.

## Worker-scaling curve

Required Pod settings: **1, 2, 4, 8, 16 and 32** workers, one fixed float32 model and input on the same Pod.
Record this curve for POTION-8M and, before adding its library bars, static-retrieval-mrl-en-v1.
The POTION curve drives the dtype table; each library bar uses the curve for its own same-model fse baseline.
Use accepted measured points only, with exact machine/date, digest, output settings, runtime identity and A/A in the evidence.
There is no fixed ten-worker point on the Pod. No interpolation or empty rows in the README.

## Acceptance and extraction contract

Fill `speed_panel_data.json` only after auditing committed raw rows and strict summaries against DR-151's current rules.
The renderer does not replace the independent source audit. Keep measurement-process details here, not in the public README.
All fse baselines within a panel must share one executed-runtime fingerprint.
Each cell needs three complete accepted rotating rounds, paired fse-a/fse-b, backend-specific executed identity,
verified artifact lineage, and symmetric A/A upper bound at most 1.10. Apply the source's other validity checks,
including old Mac load rules or Pod host/probe rules. Acceptance never transfers between machines.
Keep publication commit, journal path, exact series and ordered primary rates from the same three rounds.
Reference rows must identify their own package or Rust binary, not fse's fingerprint. Check common output semantics first.

The format is `readme-speed-panels-v1`, with committed `scaling` rows, `table` rows and `comparisons` pairs.
Rows contain `complete`, `runtime_attributed`, `aa_upper`, `backend`, `runtime_fingerprint`,
`source` (`commit`, `path`, `series_id`), `context`, and three ordered `texts_per_second` values.
`CONTEXT` in the renderer lists the required model/input/output fields. `parent_artifact` identifies original weights, not quantized bytes.
A pair contains `fse`, `reference` and `output_checked: true`, added only after verifying output semantics.
Both rows name the same combined paired journal and series. Preserve original records in the evidence snapshot.
`table_caption` and `comparison_caption` contain one-line public captions copied from verified context.
The renderer derives selected core settings from `scaling`, verifies matching machine/runtime/model/input/output context,
and filters out other settings before rendering. It rejects missing chosen rows, mixed contexts, noisy/unattributed rows
and stale unselected output files before writing.
It never silently retains withdrawn claims or deletes files.

Run `python media/readme/speed_panels.py --preview` for the explicitly unmeasured layout.
After accepted data arrives, run without `--preview`, export SVGs with librsvg and inspect both themes.
The main visual generator calls the same publish function. Empty data adds nothing to the public README or final figures.

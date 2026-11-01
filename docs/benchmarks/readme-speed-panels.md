# Required rows for the next README speed panels

This is a delivery plan, not a benchmark result. The main README stays unchanged until accepted rows arrive.
`media/readme/speed_panels.py` prepares the table and the light/dark library comparison layout.
The empty `speed_panel_data.json` produces no publishable panel. Layout previews live under `media/readme/layout/` only.

## Compact table: three ten-worker cells

Use one cohort for the whole table: POTION-8M, the same pinned model revision and float32 parent artifact,
the same complete WikiText encoding input, one machine and date, and float32 vector output.
Use the exact source digest and text rules from that cohort, including newline handling and tokenizer limits.

| Table cell | Required DR-151 row | Primary operations |
|---|---|---|
| fse float32 | encode/minishlab/potion-base-8M/float32/wikitext/10/table | fse-a, fse-b |
| fse float16 | encode/minishlab/potion-base-8M/float16/wikitext/10/table | fse-a, fse-b |
| fse int8 | encode/minishlab/potion-base-8M/int8/wikitext/10/table | fse-a, fse-b |

These are selectors following the Pod 2 `encode_cell` naming convention, not claims that those rows exist.
The reviewed Pod 2 loops currently use 1 and 32 workers. **Add 10 for all three precisions.**
Pod 3's float32-only scaling row at 10 workers does not provide the missing float16 or int8 rows.

Mac observations already accepted: float32 ten workers 918,299 texts/s and float16 ten workers 946,796 texts/s.
The Mac int8 ten-worker observation is noisy (symmetric A/A upper 1.1169), so it cannot fill the third cell.
A fresh Linux int8 cell cannot share the Mac caption. Prefer a complete three-precision Linux cohort and replace the whole table.
The generator deliberately requires all three rows before it writes the compact table. It never substitutes int8 at one worker.

Planned public layout:

| fse table | Texts/s |
|---|---:|
| float32 | measured value |
| float16 | measured value |
| int8 | measured value |

One caption: machine, POTION-8M, 10 workers, WikiText, date. Average text: measured words.
The table sits below the existing throughput graphic and above the single “How we measured” link.
No placeholder table is included in the main README.

## Library speedup bars: one worker, separate matched fse baselines

Prefer one-worker pairs for all four bars. Do not reuse a Mac headline or an unrelated dtype baseline as the numerator.
Each pair runs on the same pod, input, model revision, tokenizer rules and output settings.
Use float32 tables and float32 vectors, with identical dimension and normalization within each pair.

| Bar | Required paired DR-151 row | Operations | Model in both libraries |
|---|---|---|---|
| Model2Vec | encode/minishlab/potion-base-8M/float32/wikitext/1/model2vec | fse-a, fse-b, model2vec | minishlab/potion-base-8M |
| model2vec-rs | encode/minishlab/potion-base-8M/float32/wikitext/1/rust | fse-a, fse-b, rust | minishlab/potion-base-8M |
| FastEmbed | encode/minishlab/potion-base-8M/float32/wikitext/1/fastembed | fse-a, fse-b, fastembed | minishlab/potion-base-8M |
| Sentence Transformers | encode/sentence-transformers/static-retrieval-mrl-en-v1/float32/wikitext/1/st | fse-a, fse-b, st | sentence-transformers/static-retrieval-mrl-en-v1 |

The first three selectors are already scheduled by Pod 2's counterpart loop at one worker.
The Sentence Transformers pair must use its compatible static model, with a fresh fse numerator for that same model.
Keep model names visible under every bar. This makes the different static-model baselines clear.
Do not compare POTION-8M fse directly with the MRL reference and call it a library speedup.
All pairs must share the exact input digest, machine, worker budget, output dtype and date for this graphic.
If one library changes output semantics (for example missing normalization), fix or omit that pair before publication.

A bar uses the median of the three paired fse/reference rate ratios, preserving round order.
With equal input counts this is the median reference-time/fse-time ratio, not the ratio of two independent medians.
The bars share a zero origin. The public label says “N times as fast as Library”.
One caption names machine, input, output, worker budget and date; the model is named on each bar.

MiniLM is deliberately absent from this generator. It is a different model, not a same-model library comparison.
If later requested, give that workload its own graphic/group explicitly labelled “Different model”, with both model names.
Do not add it to this same-model data file or bypass the model-equality check.

## Acceptance and extraction contract

Only fill `speed_panel_data.json` after auditing the committed raw rows and strict summary against DR-151's current validity rules.
The renderer does not replace that independent source audit. Keep full evidence here, not in the public README.

Every cell needs three complete accepted rotating rounds, paired fse-a/fse-b, backend-specific executed-runtime identity,
verified artifact lineage, and a symmetric A/A upper bound at most 1.10. Apply all other source protocol checks,
including load rules for old Mac cells and the recorded host/probe rules for pod cells. Never transfer acceptance between machines.
Record the publication commit, journal path, exact series identity and ordered primary rates from the same rounds.
The reference runtime must identify its own library or Rust binary, not fse's package fingerprint.
Require a checked common output contract before computing any library ratio.

The input format has `format: readme-speed-panels-v1`, `table` rows and `comparisons` pairs.
Each row contains `complete`, `runtime_attributed`, `aa_upper`, `backend`, `runtime_fingerprint`,
`source` (`commit`, `path`, `series_id`), `context` and three ordered `texts_per_second` values.
The context keys are defined by `CONTEXT` in the renderer. `parent_artifact` identifies original weights, not quantized bytes.
A comparison contains `fse`, `reference` and `output_checked: true`, added only after checking output semantics.
Both rows name the same combined paired journal and series. Keep every source record in the evidence snapshot.
`table_caption` and `comparison_caption` hold the one-line public captions, copied from the verified context.
Missing, mixed-model, mixed-input, wrong-worker, noisy or unattributed data fails before any panel is written.

Run `python media/readme/speed_panels.py --preview` for the explicitly unmeasured layout.
Run `python media/readme/speed_panels.py` after audited data lands, then export SVGs with librsvg and inspect both themes.
The main visual generator calls the same publish function. With empty data, it adds nothing to the README or final figures.

# WikiText sample for the fse README

These are unchanged raw snapshots from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen).
The comparison worker produced them on 1 November 2026. The snapshots preserve their original completion and rejection records.
Canonical source: commit 3265b35, benchmarks/baseline/dr151_summary_20261101.json and its raw files.
They are kept here so the README's first comparison is reviewable without an unpublished worker checkout.

Use only operations belonging to a `round_complete` record with both `complete` and `valid` true,
in a `series_complete` with three valid rounds. Each series rotates fse-a, fse-b and model2vec across three rounds.
Each child warms once, measures three calls, and records their median as `p50_s`.
For a child, sentences/s is `texts / p50_s`. The README reports the median and full range across children.
The published comparison uses fse-a as its primary run and fse-b as a separate A/A control.
The displayed fse and model2vec medians each use three child rates, matching the source owner's summary.
No confidence interval is claimed.

Model: minishlab/potion-base-8M, revision bf8b056651a2c21b8d2565580b8569da283cab23.
Input: 5,352 nonblank WikiText validation/test texts, dataset revision b08601e04326c79dfdd32d625aee71d232d685c3.
Each raw operation records the input digest, source commit, model revision, package versions, dtype, load, warm-call times and RSS.
The model's default token/character limits and normalization apply on both sides.
Tables and outputs are float32. Worker settings are one and ten (all logical CPUs on this M1 Pro).
The additional MASSIVE file has three valid one-worker rounds over 151,674 test texts.
Its input digest and dataset revision are recorded in every operation.
The record's `commit` is the source checkout. Its installed fse distribution version may name an earlier editable-build commit.
No claim of bit equality with model2vec, quality superiority or a universal speedup follows from these timings.

`summary.json` stores unrounded calculated medians and ranges. The README rounds sentences/s to whole numbers.
The A/A time-ratio medians and ranges are 1.0141 (0.8998 to 1.0704) for one worker,
and 1.0941 (0.9862 to 1.1562) for ten workers. These controls show variation, not an implementation gain.

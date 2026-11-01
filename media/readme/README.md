# README visuals

The palette comes from the local `brand-identity` repository, `colors/tokens.json`,
revision `648624ce9bed1d08cacd00e32d5a4bf8ca62f271` (25 October 2026).
`palette.json` keeps the exact light/dark tokens needed to reproduce these assets.
The visuals use the brand's editorial explainer mode: warm surfaces, one green accent and clear reading order.

`throughput-*.svg` and its PNG exports show fse configurations, not a library speed ranking.
Values come from `docs/benchmarks/readme-runtime-window5-20261101/summary.json`
and `docs/benchmarks/readme-int8-20261101/summary.json`. The latter adds only its eligible one-worker int8 cell.
Ten-worker int8 exceeds the 10% A/A band and is not plotted. All plotted fse rows have the same runtime content identity.
Bars use exact fse-a medians with a common zero baseline. Whiskers show the full observed range.
The model, corpus, machine, date and measurement scope appear inside the image.
No Model2Vec rate or inferred words-per-second count is plotted.

`vectors-and-labels-*.svg` explains the application flow. It contains no timing claim.
The classifier requires a previously fitted linear model for this encoder's vectors.

Regenerate from any working directory with Python's standard library:

```sh
python media/readme/generate.py
```

Export each SVG to PNG with librsvg, for example:

```sh
rsvg-convert media/readme/throughput-light.svg -o media/readme/throughput-light.png
```

Repeat for the dark throughput chart, both vectors-and-labels variants and both classification-folding variants.
The SVGs request Geist with an Arial/sans-serif fallback. PNG exports use installed fallback fonts when Geist is unavailable.
Light and dark exports were inspected visually at full size.

`classification-folding-*.svg` and PNG exports explain fse's linear classification path.
They show a preparation step followed by repeated text prediction. The bias is added after averaging, not folded into each word.
This applies to heads trained on raw, unnormalized averages when only class outputs are requested.
The shorter-score-list explanation is conditional on fewer scores than input vector values.
The standalone file stores that same folded table. It is not plotted as a separate faster algorithm.
No controlled classification rates are published yet, so this figure has no rate, multiplier or timing bars.

Architecture sources: `src/fse/_runtime/_head.py` (`make_head`, `fold`, `HeadPipeline`),
`src/fse/_head/_model.py` (`save`, `_standalone_artifact`), and `docs/architecture/overview.md` (stage4b and finished tables).
The next measured panel will use committed classifier rows from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen),
with exact head, normalization, dataset, machine, workers, date and a symmetric A/A upper band at most1.10.

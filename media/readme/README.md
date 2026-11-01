# README visuals

The palette comes from the local `brand-identity` repository, `colors/tokens.json`,
revision `648624ce9bed1d08cacd00e32d5a4bf8ca62f271` (25 October 2026).
`palette.json` keeps the exact light/dark tokens needed to reproduce these assets.
The visuals use the brand's editorial explainer mode: warm surfaces, one green accent and clear reading order.

`throughput-*.svg` and its PNG exports show fse configurations, not a library speed ranking.
Values come from `docs/benchmarks/readme-runtime-window5-20261101/summary.json`.
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

Repeat for the dark throughput chart and both vectors-and-labels variants.
The SVGs request Geist with an Arial/sans-serif fallback. PNG exports use installed fallback fonts when Geist is unavailable.
Light and dark exports were inspected visually at full size.

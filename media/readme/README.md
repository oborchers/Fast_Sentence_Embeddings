# README visuals

The palette comes from the local `brand-identity` repository, `colors/tokens.json`,
revision `648624ce9bed1d08cacd00e32d5a4bf8ca62f271` (25 October 2026).
`palette.json` keeps the exact light/dark tokens needed to reproduce these assets.
The visuals use the brand's editorial explainer mode: warm surfaces, one green accent and clear reading order.

Historical Mac throughput assets belong to the evidence archive, not the current README generator.
Their originals remain intact until the archive commit and hashes are verified.
`docs/benchmarks/readme-evidence-manifest.json` records the transfer source and status.
The generator only creates explanatory illustrations and accepted current speed panels.
It does not read historical Mac journals or regenerate their measured plots.

`vectors-and-labels-*.svg` explains the application flow. It contains no timing claim.
The classifier requires a previously fitted linear model for this encoder's vectors.

Regenerate from any working directory with Python's standard library:

```sh
python media/readme/generate.py
```

Export each SVG to PNG with librsvg, for example:

```sh
rsvg-convert media/readme/vectors-and-labels-light.svg -o media/readme/vectors-and-labels-light.png
```

Repeat for the dark vectors-and-labels variant and both classification-folding variants.
The SVGs request Geist with an Arial/sans-serif fallback. PNG exports use installed fallback fonts when Geist is unavailable.
Light and dark exports were inspected visually at full size.

`classification-folding-*.svg` and PNG exports explain fse's linear classification path.
They show a preparation step followed by repeated text prediction. The bias is added after averaging, not folded into each word.
This applies to heads trained on raw, unnormalized averages when only class outputs are requested.
The shorter-score-list explanation is conditional on fewer scores than input vector values.
The standalone file stores that same folded table. It is not plotted as a separate faster algorithm.
This figure explains folding. Its controlled speed comparison remains pending, so it has no rate, multiplier or timing bars.

Architecture sources: `src/fse/_runtime/_head.py` (`make_head`, `fold`, `HeadPipeline`),
`src/fse/_head/_model.py` (`save`, `_standalone_artifact`), and `docs/architecture/overview.md` (stage4b and finished tables).
The next measured panel will use committed classifier rows from DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen),
with exact head, normalization, dataset, machine, workers, date and a symmetric A/A upper band at most1.10.

## Prepared speed table and library comparison

`speed_panels.py` renders the next compact table and light/dark same-model speedup bars.
The accepted-data file starts empty. No unfinished table, fake multiplier or new figure appears in the main README.
See `docs/benchmarks/readme-speed-panels.md` for exact required DR-151 row selectors and the audit/extraction contract.
All three table rows keep one core. Choose 16 or 32 directly from that model's accepted float32 cells.
32 is shown only at an unrounded median gain of at least 1.20 over 16; otherwise use 16.
`core-selection-evidence.md` records the decision and sources. No scaling curve is collected or rendered.
The Pod grid remains 1/16/32. Every first and warm fse call must last at least two seconds.
The separate 10M write stage is dropped; the 100M showcase covers writing.
Each selected library needs the exact same-model pair at the selected setting.
Each library bar has its own calibrated paired input and fse baseline for that same model and output.
Both arms must match input digest/count. Different bars may have different inputs.
The figure caption lists each bar's input size, also shown beside the bar.
Dtype cells and the 16/32 decision keep one shared input per model.
MiniLM is excluded from this same-model renderer.

Generate explicitly unmeasured layout previews:

```sh
python media/readme/speed_panels.py --preview
```

The previews go under `media/readme/layout/` and are never linked by README.
After accepted data arrives, the normal generator writes `speed-table.md` and `library-speedups-{light,dark}.svg`.
The table and graphic each receive one plain caption, with process details kept in the evidence guide.

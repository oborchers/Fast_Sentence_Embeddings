# README illustration sources

`vectors-and-labels-light.png` and `vectors-and-labels-dark.png` are used by README.
Their SVG sources contain no benchmark data. The generator also creates the light and dark `library-speed` SVG sources.
README uses their PNG exports.
Those charts read the eight paired comparisons in `docs/benchmarks/readme-speed.json`.
The one-worker and sixteen-worker panels have separate labelled linear scales.
Source manifests and paired medians are retained in that compact summary.

Colors are copied from `brand-identity/colors/tokens.json`, revision
`648624ce9bed1d08cacd00e32d5a4bf8ca62f271`. `palette.json` contains only the six tokens used in each theme.

Regenerate the SVGs with Python's standard library:

```sh
python docs/assets/readme/generate.py
```

Export either SVG with librsvg:

```sh
rsvg-convert docs/assets/readme/vectors-and-labels-light.svg -o docs/assets/readme/vectors-and-labels-light.png
```

Repeat for the dark theme. Fonts use Geist, then Arial or sans-serif as fallback.

Export the two library charts with the same `rsvg-convert` command, using their `library-speed` file names.

`dtype-speed` shows the six longer-text rates on one shared linear scale.
Its source is the `dtypes` section of the same compact summary.
Export its light and dark SVGs to PNG with `rsvg-convert` too.

`shared-drive` shows three machines that write one store each to a network drive and a fourth that reads a view.
It contains no data. Export its light and dark SVGs to PNG with `rsvg-convert` too.
README links `shared-drive-light.png` and `shared-drive-dark.png` by relative path.

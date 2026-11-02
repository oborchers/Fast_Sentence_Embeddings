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
python media/readme/generate.py
```

Export either SVG with librsvg:

```sh
rsvg-convert media/readme/vectors-and-labels-light.svg -o media/readme/vectors-and-labels-light.png
```

Repeat for the dark theme. Fonts use Geist, then Arial or sans-serif as fallback.

Export the two library charts with the same `rsvg-convert` command, using their `library-speed` file names.

"""Render audited DR-151 cells. Empty input produces no publishable panel."""

import argparse
import html
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DTYPES = ("float32", "float16", "int8")
BACKENDS = {
    "model2vec": "Model2Vec",
    "rust": "model2vec-rs",
    "fastembed": "FastEmbed",
    "st": "Sentence Transformers",
}
CONTEXT = (
    "model",
    "revision",
    "parent_artifact",
    "input_digest",
    "texts",
    "corpus",
    "machine",
    "date",
    "workers",
    "table_dtype",
    "output_dtype",
    "output_dim",
    "normalize",
    "tokenizer_rule",
)
COHORT = ("input_digest", "texts", "corpus", "machine", "date", "workers", "output_dtype")


def positive(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Expected a finite positive measured value")
    return value


def measured(row):
    """The input is an audited extraction, not a replacement for DR-151 validation."""
    if row["complete"] is not True or row["runtime_attributed"] is not True:
        raise ValueError("Only complete, runtime-attributed cells may be rendered")
    if not 1 <= positive(row["aa_upper"]) <= 1.10:
        raise ValueError("Cell exceeds the DR-151 variation limit")
    source = row["source"]
    if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
        raise ValueError("Source must name an immutable publication commit")
    if not source["path"].startswith("benchmarks/baseline/") or not source["series_id"]:
        raise ValueError("Source path and exact series identity are required")
    context = row["context"]
    if any(key not in context for key in CONTEXT):
        raise ValueError("Incomplete model, input, output or machine context")
    if not row["runtime_fingerprint"] or not row["backend"]:
        raise ValueError("Executing backend identity is required")
    rates = row["texts_per_second"]
    if len(rates) != 3:
        raise ValueError("Three accepted primary rounds are required")
    return statistics.median(positive(value) for value in rates)


def same(rows, keys):
    first = rows[0]["context"]
    if any(any(row["context"][key] != first[key] for key in keys) for row in rows[1:]):
        raise ValueError("Rows do not describe the same comparison context")


def same_runtime(rows):
    if len({row["runtime_fingerprint"] for row in rows}) != 1:
        raise ValueError("The fse rows must share one executed runtime")


def table(rows):
    if not rows:
        return ""
    if {r["context"]["table_dtype"] for r in rows} != set(DTYPES) or len(rows) != 3:
        raise ValueError("The compact table needs all three ten-worker precisions")
    same(rows, [key for key in CONTEXT if key != "table_dtype"])
    same_runtime(rows)
    values = {}
    for row in rows:
        rate = measured(row)
        if row["backend"] != "fse" or row["context"]["workers"] != 10:
            raise ValueError("The compact table is fse at ten workers only")
        values[row["context"]["table_dtype"]] = rate
    return "\n".join(
        [
            "| fse table | Texts/s |",
            "|---|---:|",
            *[f"| {dtype} | {values[dtype]:,.0f} |" for dtype in DTYPES],
        ]
    )


def comparison(pair):
    left, right = pair["fse"], pair["reference"]
    measured(left)
    measured(right)
    if left["backend"] != "fse" or right["backend"] not in BACKENDS:
        raise ValueError("Expected fse and a named reference library")
    same([left, right], CONTEXT)
    if left["source"] != right["source"]:
        raise ValueError("Each bar needs its own matched paired series")
    if pair["output_checked"] is not True:
        raise ValueError("Output shape, dtype and numerical semantics must be checked")
    # Preserve pairing. A ratio of two independent medians is not the paired median.
    ratios = [
        a / b for a, b in zip(left["texts_per_second"], right["texts_per_second"], strict=True)
    ]
    return statistics.median(ratios)


def label(x, y, value, colors, size=18, weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{colors["text"]}">{html.escape(value)}</text>'


def svg(rows, caption, colors, *, preview=False):
    values = [] if preview else [comparison(pair) for pair in rows]
    names = (
        list(BACKENDS.values()) if preview else [BACKENDS[p["reference"]["backend"]] for p in rows]
    )
    height = 190 + 100 * len(names)
    title = "fse vs other libraries"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img" aria-labelledby="title desc"><title id="title">{title}</title><desc id="desc">{html.escape(caption)}</desc><rect width="1100" height="{height}" rx="12" fill="{colors["bg"]}"/><g font-family="Geist, Arial, sans-serif">',
        label(40, 52, title, colors, 30, 700),
    ]
    subtitle = (
        "Layout preview. No measurements."
        if preview
        else "Each bar compares the same model in both libraries."
    )
    parts.append(label(40, 84, subtitle, colors))
    scale = max(1, *values) if values else 1
    for index, name in enumerate(names):
        y = 130 + 100 * index
        parts.append(label(40, y, name, colors, 21, 700))
        if preview:
            parts.extend(
                [
                    f'<rect x="370" y="{y - 22}" width="670" height="35" rx="4" fill="none" stroke="{colors["accent"]}" stroke-dasharray="6 5"/>',
                    label(385, y, "measured speedup", colors),
                ]
            )
        else:
            width = 670 * values[index] / scale
            model = rows[index]["fse"]["context"]["model"].split("/")[-1]
            parts.extend(
                [
                    label(40, y + 28, model, colors, 15),
                    f'<rect x="370" y="{y - 22}" width="{width:.4f}" height="35" rx="4" fill="{colors["accent"]}"/>',
                    label(370, y + 42, f"{values[index]:.2f} times as fast as {name}", colors),
                ]
            )
    parts.extend([label(40, height - 25, caption, colors, 15), "</g></svg>"])
    return "\n".join(parts) + "\n"


def publish(data, output, palette):
    if data["format"] != "readme-speed-panels-v1":
        raise ValueError("Unsupported speed-panel format")
    files = {}
    markdown = table(data["table"])
    if markdown:
        files["speed-table.md"] = markdown + "\n\n" + data["table_caption"] + "\n"
    pairs = data["comparisons"]
    if pairs:
        same([p["fse"] for p in pairs], COHORT)
        same_runtime([p["fse"] for p in pairs])
        for theme in ("light", "dark"):
            colors = {key: value["value"] for key, value in palette[theme].items()}
            files[f"library-speedups-{theme}.svg"] = svg(pairs, data["comparison_caption"], colors)
    owned = {"speed-table.md", "library-speedups-light.svg", "library-speedups-dark.svg"}
    stale = sorted(name for name in owned - files.keys() if (output / name).exists())
    if stale:
        raise ValueError(
            f"Unselected panel files already exist: {stale}. Review these outputs before publishing."
        )
    # Validate every requested output before writing any file.
    for name, content in files.items():
        (output / name).write_text(content)


def preview(output, palette):
    output.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        colors = {key: value["value"] for key, value in palette[theme].items()}
        (output / f"library-speedups-{theme}.svg").write_text(
            svg(
                [],
                "Model, input, output, machine and date come from the accepted rows.",
                colors,
                preview=True,
            )
        )
    (output / "speed-table.md").write_text(
        "Layout only, no measurements. Do not include this draft in README.\n\n| fse table (10 workers) | Texts/s |\n|---|---:|\n| float32 | measured value |\n| float16 | measured value |\n| int8 | measured value |\n\nMachine, model, date. Average text: measured words.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "speed_panel_data.json")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    palette = json.loads((HERE / "palette.json").read_text())
    if args.preview:
        preview(args.output / "layout", palette)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        publish(json.loads(args.data.read_text()), args.output, palette)


if __name__ == "__main__":
    main()

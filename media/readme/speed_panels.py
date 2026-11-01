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
PANEL_WORKERS = (1, 16, 32)
MIN_32_GAIN = 1.20
LOW_GAIN_NOTE = "More than 16 cores adds little here."
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
COHORT = ("input_digest", "texts", "corpus", "machine", "date", "output_dtype")


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
    if row["backend"] == "fse":
        calls = row.get("call_seconds", [])
        if len(calls) < 3 or any(positive(value) < 2 for value in calls):
            raise ValueError("Every recorded fse call must last at least 2 seconds")
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


def core_label(worker):
    if worker == 0:
        return "Selected core count"
    if worker == 1:
        return "One core"
    return "All 32 cores" if worker == 32 else f"{worker} cores"


def select_cores(rows):
    """Choose directly from committed fse-a cells at 1, 16 and 32 cores."""
    points = {r["context"]["workers"]: r for r in rows}
    if set(points) != set(PANEL_WORKERS) or len(rows) != len(points):
        raise ValueError("Selection needs accepted cells at 1, 16 and 32")
    same(rows, [key for key in CONTEXT if key != "workers"])
    same_runtime(rows)
    if any(r["backend"] != "fse" or r["context"]["table_dtype"] != "float32" for r in rows):
        raise ValueError("Selection uses fse float32 rows")
    medians = {worker: measured(row) for worker, row in points.items()}
    return (1, 32) if medians[32] / medians[16] >= MIN_32_GAIN else (1, 16)


def selections(rows):
    models = {r["context"]["model"] for r in rows}
    return {
        model: select_cores([r for r in rows if r["context"]["model"] == model]) for model in models
    }


def selected_rows(rows, selection_rows, selected):
    if not rows:
        return []
    models = {r["context"]["model"] for r in rows}
    if not models <= selected.keys():
        raise ValueError("Committed selection rows are required for every displayed model")
    relevant = [r for r in rows if r["context"]["workers"] in selected[r["context"]["model"]]]
    for row in relevant:
        baseline = next(
            r for r in selection_rows if r["context"]["model"] == row["context"]["model"]
        )
        same([row, baseline], [key for key in CONTEXT if key not in {"workers", "table_dtype"}])
        same_runtime([row, baseline])
    return relevant


def selection_notes(selected):
    return [
        f"{model.split('/')[-1]}: {LOW_GAIN_NOTE}"
        for model, workers in sorted(selected.items())
        if 32 not in workers
    ]


def selection_audit(rows, selected):
    lines = [
        "# Core selection evidence",
        "",
        "Public settings are one core plus 32 only at a 32/16 ratio of at least 1.20; otherwise 16.",
        "",
    ]
    for model, workers in sorted(selected.items()):
        decision_rows = [r for r in rows if r["context"]["model"] == model]
        medians = {r["context"]["workers"]: measured(r) for r in decision_rows}
        lines.append(f"## {model}")
        lines.append(
            f"Selected settings: {workers}. Unrounded 32/16 ratio: {medians[32] / medians[16]:.17g}."
        )
        lines.extend(audit_sources(decision_rows))
        lines.append("")
    return "\n".join(lines) + "\n"


def audit_sources(rows):
    lines = [
        "",
        "| Workers | Primary median texts/s | Committed source | Series |",
        "|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["context"]["workers"]):
        source = row["source"]
        url = f"https://github.com/oborchers/Fast_Sentence_Embeddings/blob/{source['commit']}/{source['path']}"
        lines.append(
            f"| {row['context']['workers']} | {measured(row):.17g} | [journal]({url}) | {source['series_id']} |"
        )
    return lines


def table(rows, workers):
    if not rows:
        return ""
    expected = {(dtype, worker) for dtype in DTYPES for worker in workers}
    actual = {(r["context"]["table_dtype"], r["context"]["workers"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("The compact table needs all precisions at the selected core settings")
    same(rows, [key for key in CONTEXT if key not in {"table_dtype", "workers"}])
    same_runtime(rows)
    values = {}
    for row in rows:
        rate = measured(row)
        if row["backend"] != "fse":
            raise ValueError("The compact table is fse only")
        values[row["context"]["table_dtype"], row["context"]["workers"]] = rate
    heading = "| fse table | " + " | ".join(f"{core_label(w)}, texts/s" for w in workers) + " |"
    separator = "|---|" + "---:|" * len(workers)
    body = [
        "| " + dtype + " | " + " | ".join(f"{values[dtype, w]:,.0f}" for w in workers) + " |"
        for dtype in DTYPES
    ]
    return "\n".join([heading, separator, *body])


def comparison(pair):
    left, right = pair["fse"], pair["reference"]
    measured(left)
    measured(right)
    if left["backend"] != "fse" or right["backend"] not in BACKENDS:
        raise ValueError("Expected fse and a named reference library")
    same([left, right], CONTEXT)
    if left["context"]["workers"] not in PANEL_WORKERS:
        raise ValueError("Pod comparisons must use a measured panel setting")
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


def entries(rows, preview):
    if preview:
        return [
            (w, name, "same model in both libraries", None)
            for w in (1, 0)
            for name in BACKENDS.values()
        ]
    return [
        (
            p["fse"]["context"]["workers"],
            BACKENDS[p["reference"]["backend"]],
            p["fse"]["context"]["model"].split("/")[-1],
            comparison(p),
        )
        for p in rows
    ]


def bar(entry, y, scale, colors):
    _, name, model, value = entry
    parts = [label(40, y, name, colors, 21, 700), label(40, y + 28, model, colors, 15)]
    if value is None:
        parts.extend(
            [
                f'<rect x="370" y="{y - 22}" width="670" height="35" rx="4" fill="none" stroke="{colors["accent"]}" stroke-dasharray="6 5"/>',
                label(385, y, "measured speedup", colors),
            ]
        )
    else:
        width = 670 * value / scale
        parts.extend(
            [
                f'<rect x="370" y="{y - 22}" width="{width:.4f}" height="35" rx="4" fill="{colors["accent"]}"/>',
                label(370, y + 42, f"{value:.2f} times as fast as {name}", colors),
            ]
        )
    return parts


def svg(rows, caption, colors, *, preview=False, notes=()):
    content = entries(rows, preview)
    scale = max([1, *(entry[3] for entry in content if entry[3] is not None)])
    groups = sorted({entry[0] for entry in content}, key=lambda w: (w != 1, w))
    height = 150 + 40 * len(groups) + 90 * len(content) + 28 * len(notes)
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
    y = 125
    for worker in groups:
        parts.append(label(40, y, core_label(worker), colors, 21, 700))
        y += 40
        for entry in (item for item in content if item[0] == worker):
            parts.extend(bar(entry, y, scale, colors))
            y += 90
    for index, note in enumerate(notes):
        parts.append(label(40, height - 55 - 28 * (len(notes) - index - 1), note, colors, 15))
    parts.extend([label(40, height - 25, caption, colors, 15), "</g></svg>"])
    return "\n".join(parts) + "\n"


def paired_core_settings(pairs, selected, expected=None):
    requested = pairs if expected is None else expected
    keys = {(p["reference"]["backend"], p["fse"]["context"]["model"]) for p in requested}
    for backend, model in keys:
        workers = [
            p["fse"]["context"]["workers"]
            for p in pairs
            if p["reference"]["backend"] == backend and p["fse"]["context"]["model"] == model
        ]
        if sorted(workers) != list(selected[model]):
            raise ValueError("Each library needs one accepted pair at every selected core setting")


def check_stale(output, files):
    owned = {
        "speed-table.md",
        "library-speedups-light.svg",
        "library-speedups-dark.svg",
        "core-selection-evidence.md",
    }
    stale = sorted(name for name in owned - files.keys() if (output / name).exists())
    if stale:
        raise ValueError(
            f"Unselected panel files already exist: {stale}. Review these outputs before publishing."
        )


def panel_files(data, palette):
    selected = selections(data.get("selection_rows", []))
    table_rows = selected_rows(data["table"], data.get("selection_rows", []), selected)
    if data["table"] and not table_rows:
        raise ValueError("Missing selected precision rows")
    files = (
        {"core-selection-evidence.md": selection_audit(data["selection_rows"], selected)}
        if selected
        else {}
    )
    if table_rows:
        model = table_rows[0]["context"]["model"]
        note = "\n".join(selection_notes({model: selected[model]}))
        files["speed-table.md"] = (
            table(table_rows, selected[model]) + "\n\n" + data["table_caption"] + "\n" + note + "\n"
        )
    pairs = select_pairs(data["comparisons"], data.get("selection_rows", []), selected)
    if pairs:
        for theme in ("light", "dark"):
            colors = {key: value["value"] for key, value in palette[theme].items()}
            shown = {
                p["fse"]["context"]["model"]: selected[p["fse"]["context"]["model"]] for p in pairs
            }
            files[f"library-speedups-{theme}.svg"] = svg(
                pairs, data["comparison_caption"], colors, notes=selection_notes(shown)
            )
    return files


def select_pairs(pairs, selection_rows, selected):
    selected_rows([p["fse"] for p in pairs], selection_rows, selected)
    chosen = [
        p for p in pairs if p["fse"]["context"]["workers"] in selected[p["fse"]["context"]["model"]]
    ]
    paired_core_settings(chosen, selected, pairs)
    if chosen:
        same([p["fse"] for p in chosen], COHORT)
        same_runtime([p["fse"] for p in chosen])
    return chosen


def publish(data, output, palette):
    if data["format"] != "readme-speed-panels-v1":
        raise ValueError("Unsupported speed-panel format")
    files = panel_files(data, palette)

    check_stale(output, files)
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
        "Layout only, no measurements. Do not include this draft in README.\n\n| fse table | One core, texts/s | Selected setting, texts/s |\n|---|---:|---:|\n| float32 | measured value | measured value |\n| float16 | measured value | measured value |\n| int8 | measured value | measured value |\n\nMachine (32 vCPU), model, date. Average text: measured words.\n"
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

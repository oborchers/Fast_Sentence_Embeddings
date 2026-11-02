import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
brand = json.loads((ROOT / "media/readme/palette.json").read_text())
out = ROOT / "media/readme"
out.mkdir(exist_ok=True)


def text(x, y, value, size=18, fill=None, weight=400, anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill or colors["text"]}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>'


for theme in ("light", "dark"):
    colors = {k: v["value"] for k, v in brand[theme].items()}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="370" viewBox="0 0 1100 370" role="img" aria-labelledby="title desc"><title id="title">Run text vectors or a fitted classifier locally</title><desc id="desc">Texts enter fse. An encoder gives vectors for search and clustering. A classifier combines an encoder with a previously trained linear model and gives labels for routing or filtering. Training happens outside fse.</desc><rect width="1100" height="370" rx="12" fill="{colors["bg"]}"/><g font-family="Geist, Arial, sans-serif">',
        text(40, 53, "One local call. Vectors or labels.", 30, weight=700),
    ]
    for x, y, w, h in (
        (40, 134, 230, 100),
        (405, 134, 230, 100),
        (810, 90, 250, 80),
        (810, 218, 250, 80),
    ):
        parts += [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{colors["bg-secondary"]}" stroke="{colors["border"]}"/>'
        ]
    parts += [
        text(155, 175, "Your texts", 23, weight=700, anchor="middle"),
        text(155, 204, "files, batches, streams", 17, colors["text-secondary"], anchor="middle"),
        text(520, 175, "fse", 30, colors["accent"], 700, anchor="middle"),
        text(520, 204, "runs on your CPU", 17, colors["text-secondary"], anchor="middle"),
        text(935, 124, "Vectors", 23, weight=700, anchor="middle"),
        text(935, 151, "search and clustering", 17, colors["text-secondary"], anchor="middle"),
        text(935, 251, "Labels", 23, weight=700, anchor="middle"),
        text(935, 278, "routing and filtering", 17, colors["text-secondary"], anchor="middle"),
    ]
    for path in ("M285 184 H390", "M650 172 H715 V130 H795", "M650 198 H715 V258 H795"):
        parts += [f'<path d="{path}" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>']
    for x, y in ((390, 184), (795, 130), (795, 258)):
        parts += [
            f'<path d="M{x - 9} {y - 6} L{x} {y} L{x - 9} {y + 6}" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>'
        ]
    parts += [
        text(
            40,
            340,
            "For labels, import a linear model trained on your encoder's vectors. fse runs it. Training happens elsewhere.",
            18,
            colors["text-secondary"],
        ),
        "</g></svg>",
    ]
    (out / f"vectors-and-labels-{theme}.svg").write_text("\n".join(parts) + "\n")


def speed_panel(theme, data):
    global colors
    colors = {k: v["value"] for k, v in brand[theme].items()}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="680" '
        'viewBox="0 0 1100 680" role="img" aria-labelledby="title desc">',
        '<title id="title">fse vs other libraries</title>',
        '<desc id="desc">Paired speedups with the same model and input. '
        "The one-worker and sixteen-worker panels have separate linear scales. "
        "Exact values and inputs are in the table below.</desc>",
        f'<rect width="1100" height="680" rx="12" fill="{colors["bg"]}"/>',
        '<g font-family="Geist, Arial, sans-serif">',
        text(40, 48, "Same model. More texts per second.", 30, weight=700),
    ]
    for workers, left, maximum in ((1, 40, 25), (16, 595, 130)):
        width = 465
        parts += [
            text(left, 98, f"{workers} worker" + ("s" if workers != 1 else ""), 24, weight=700)
        ]
        for index, row in enumerate(r for r in data["comparisons"] if r["workers"] == workers):
            y = 143 + index * 118
            ratio = row["paired_speedup_median"]
            model = "POTION-8M" if row["dimensions"] == 256 else "static-retrieval-mrl-en-v1"
            parts += [
                text(left, y, row["library"], 21, weight=700),
                text(left + width, y, f"{ratio:.2f}x", 21, colors["accent"], 700, "end"),
                text(
                    left,
                    y + 25,
                    f"{model}, {row['dimensions']} dimensions",
                    16,
                    colors["text-secondary"],
                ),
                f'<rect x="{left}" y="{y + 38}" width="{width}" height="20" rx="3" fill="{colors["bg-secondary"]}"/>',
                f'<rect x="{left}" y="{y + 38}" width="{width * ratio / maximum:.3f}" height="20" rx="3" fill="{colors["accent"]}"/>',
            ]
        for fraction in (0, 0.5, 1):
            x = left + fraction * width
            parts += [
                text(
                    x,
                    626,
                    f"{fraction * maximum:g}x",
                    15,
                    colors["text-secondary"],
                    anchor="middle",
                ),
            ]
    parts += [
        text(
            40,
            660,
            "Speedup against each library. Each panel uses its own linear scale.",
            17,
            colors["text-secondary"],
        ),
        "</g></svg>",
    ]
    (out / f"library-speed-{theme}.svg").write_text("\n".join(parts) + "\n")


speed_source = ROOT / "docs/benchmarks/readme-speed.json"
if speed_source.exists():
    speed_data = json.loads(speed_source.read_text())
    for speed_theme in ("light", "dark"):
        speed_panel(speed_theme, speed_data)

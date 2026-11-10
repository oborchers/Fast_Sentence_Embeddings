import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
brand = json.loads((ROOT / "docs/assets/readme/palette.json").read_text())
out = ROOT / "docs/assets/readme"
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
        rows = sorted(
            (r for r in data["comparisons"] if r["workers"] == workers),
            key=lambda r: r["paired_speedup_median"],
            reverse=True,
        )
        for index, row in enumerate(rows):
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


def dtype_panel(theme, data):
    global colors
    colors = {k: v["value"] for k, v in brand[theme].items()}
    left, width, maximum = 210, 700, 1_500_000
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="540" '
        'viewBox="0 0 1100 540" role="img" aria-labelledby="title desc">',
        '<title id="title">fse performance on longer texts</title>',
        '<desc id="desc">Absolute texts per second for float32, float16 and int8 model '
        "tables, with one and sixteen workers. All six bars use the same linear scale.</desc>",
        f'<rect width="1100" height="540" rx="12" fill="{colors["bg"]}"/>',
        '<g font-family="Geist, Arial, sans-serif">',
        text(40, 48, "Performance on longer texts", 30, weight=700),
        text(
            40,
            80,
            "fse only. POTION-8M, 256 dimensions. Float32 output.",
            18,
            colors["text-secondary"],
        ),
    ]
    for workers, x, color in ((1, 210, colors["text-secondary"]), (16, 420, colors["accent"])):
        parts += [
            f'<rect x="{x}" y="102" width="18" height="18" rx="3" fill="{color}"/>',
            text(x + 28, 118, f"{workers} worker" + ("s" if workers != 1 else ""), 18),
        ]
    for index, dtype in enumerate(("float32", "float16", "int8")):
        y = 150 + index * 110
        parts += [text(40, y + 43, dtype, 23, weight=700)]
        for row in (r for r in data["dtypes"]["cells"] if r["dtype"] == dtype):
            bar_y = y + (36 if row["workers"] == 16 else 0)
            value = row["texts_per_second"]
            color = colors["accent"] if row["workers"] == 16 else colors["text-secondary"]
            bar_width = width * value / maximum
            parts += [
                f'<rect x="{left}" y="{bar_y}" width="{bar_width:.3f}" height="24" rx="3" fill="{color}"/>',
                text(left + bar_width + 12, bar_y + 19, f"{value:,.0f}", 20, weight=700),
            ]
    parts += [f'<path d="M{left} 465 H{left + width}" stroke="{colors["border"]}"/>']
    for value in (0, 500_000, 1_000_000, 1_500_000):
        parts += [
            text(
                left + width * value / maximum,
                488,
                f"{value:,}",
                16,
                colors["text-secondary"],
                anchor="middle",
            )
        ]
    parts += [
        text(
            40,
            526,
            "Texts per second. All bars use the same linear scale, starting at zero.",
            17,
            colors["text-secondary"],
        ),
        "</g></svg>",
    ]
    (out / f"dtype-speed-{theme}.svg").write_text("\n".join(parts) + "\n")


speed_source = ROOT / "docs/benchmarks/readme-speed.json"
if speed_source.exists():
    speed_data = json.loads(speed_source.read_text())
    for speed_theme in ("light", "dark"):
        speed_panel(speed_theme, speed_data)
        dtype_panel(speed_theme, speed_data)


def shared_drive_panel(theme):
    global colors
    colors = {k: v["value"] for k, v in brand[theme].items()}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="440" '
        'viewBox="0 0 1100 440" role="img" aria-labelledby="title desc">',
        '<title id="title">Three machines write to a network drive and a fourth reads</title>',
        '<desc id="desc">Three machines each write their own store to a network drive. '
        "A fourth machine opens one view over the three stores. "
        "It opens the view again to see new shards.</desc>",
        f'<rect width="1100" height="440" rx="12" fill="{colors["bg"]}"/>',
        '<g font-family="Geist, Arial, sans-serif">',
        text(40, 53, "Three machines write. A fourth reads.", 30, weight=700),
    ]
    boxes = ((40, 100, 250, 80), (40, 200, 250, 80), (40, 300, 250, 80))
    for x, y, w, h in boxes + ((420, 100, 260, 280), (810, 190, 250, 100)):
        parts += [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{colors["bg-secondary"]}" stroke="{colors["border"]}"/>'
        ]
    for index, (x, y, w, h) in enumerate(boxes):
        parts += [
            text(x + 125, y + 35, f"Machine {index + 1}", 23, weight=700, anchor="middle"),
            text(x + 125, y + 62, f"writes store-{index}", 17, colors["text-secondary"], anchor="middle"),
        ]
        parts += [
            f'<rect x="450" y="{y + 12}" width="200" height="56" rx="8" fill="{colors["bg"]}" stroke="{colors["border"]}"/>',
            text(550, y + 47, f"store-{index}", 20, colors["accent"], 700, anchor="middle"),
        ]
        parts += [f'<path d="M290 {y + 40} H{434}" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>']
        parts += [
            f'<path d="M{425} {y + 34} L{434} {y + 40} L{425} {y + 46}" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>'
        ]
    parts += [
        text(550, 85, "Network drive", 23, weight=700, anchor="middle"),
        text(935, 232, "Machine 4", 23, weight=700, anchor="middle"),
        text(935, 262, "reads one view", 17, colors["text-secondary"], anchor="middle"),
        f'<path d="M680 240 H795" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>',
        f'<path d="M786 234 L795 240 L786 246" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>',
        text(
            40,
            412,
            "One store for each writing machine. The reader opens the view again to see new shards.",
            18,
            colors["text-secondary"],
        ),
        "</g></svg>",
    ]
    (out / f"shared-drive-{theme}.svg").write_text("\n".join(parts) + "\n")


for shared_theme in ("light", "dark"):
    shared_drive_panel(shared_theme)

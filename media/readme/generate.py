import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
brand = json.loads((ROOT / "media/readme/palette.json").read_text())
summary = json.loads(
    (ROOT / "docs/benchmarks/readme-runtime-window5-20261101/summary.json").read_text()
)
data = {
    (c["backends"]["fse-a"]["table_dtype"], c["workers"]): c["backends"]["fse-a"][
        "texts_per_second"
    ]
    for c in summary["cells"]
}
out = ROOT / "media/readme"
out.mkdir(exist_ok=True)


def text(x, y, value, size=18, fill=None, weight=400, anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill or colors["text"]}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>'


for theme in ("light", "dark"):
    colors = {k: v["value"] for k, v in brand[theme].items()}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="530" viewBox="0 0 1100 530" role="img" aria-labelledby="title desc"><title id="title">fse throughput on WikiText</title><desc id="desc">POTION-8M: float32 one worker 222701 texts per second; float32 ten workers 918299; float16 ten workers 946796. Bars show medians, ticks show full ranges. Apple M1 Pro, 1 November 2026. This compares fse configurations, not libraries.</desc><rect width="1100" height="530" rx="12" fill="{colors["bg"]}"/><g font-family="Geist, Arial, sans-serif">'
    ]
    parts += [
        text(40, 52, "More text. Same CPU.", 30, weight=700),
        text(
            40,
            83,
            "POTION-8M on WikiText  |  fse configurations, not a library speed comparison",
            18,
            colors["text-secondary"],
        ),
    ]
    x0 = 315
    width = 725
    for v, label in (
        (0, "0"),
        (250000, "250,000"),
        (500000, "500,000"),
        (750000, "750,000"),
        (1000000, "1,000,000"),
    ):
        x = x0 + width * v / 1000000
        parts += [
            f'<path d="M{x} 116 V399" stroke="{colors["border"]}"/>',
            text(x, 426, label, 16, colors["text-secondary"], anchor="middle"),
        ]
    for i, (dtype, w) in enumerate((("float32", 1), ("float32", 10), ("float16", 10))):
        d = data[dtype, w]
        y = 140 + 96 * i
        end = x0 + width * d["median"] / 1000000
        lo = x0 + width * d["min"] / 1000000
        hi = x0 + width * d["max"] / 1000000
        parts += [
            text(40, y + 14, dtype, 22, weight=700),
            text(
                40,
                y + 39,
                f"{w} " + ("worker" if w == 1 else "workers"),
                18,
                colors["text-secondary"],
            ),
            text(1040, y - 6, f"{d['median']:,.0f} texts/s", 22, weight=700, anchor="end"),
            f'<rect x="{x0}" y="{y}" width="{end - x0:.4f}" height="43" rx="4" fill="{colors["accent"]}"/>',
            f'<path d="M{lo} {y + 22} H{hi} M{lo} {y + 13} V{y + 31} M{hi} {y + 13} V{y + 31}" fill="none" stroke="{colors["text"]}" stroke-width="2"/>',
        ]
    parts += [
        text(
            40,
            470,
            "Apple M1 Pro  ·  1 November 2026  ·  5,352 nonblank validation/test texts",
            17,
            colors["text-secondary"],
        ),
        text(
            40,
            502,
            "3 paired rounds  ·  bars: medians  ·  ticks: full ranges  ·  models already loaded",
            17,
            colors["text-secondary"],
        ),
        "</g></svg>",
    ]
    (out / f"throughput-{theme}.svg").write_text("\n".join(parts) + "\n")
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
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="575" viewBox="0 0 1100 575" role="img" aria-labelledby="title desc"><title id="title">Classification: fold the linear head</title><desc id="desc">A trained linear head turns each word vector into class scores once. For each new text, fse looks up those word scores, averages them, adds the bias and returns labels. This applies to heads trained on raw unnormalized averages when only class outputs are needed. Standalone saving preserves that same folded table. Speed measurements are pending.</desc><rect width="1100" height="575" rx="12" fill="{colors["bg"]}"/><g font-family="Geist, Arial, sans-serif">',
        text(40, 52, "Classification: fold the linear head", 30, weight=700),
        text(
            40,
            84,
            "Compute word scores once. Reuse them for each text.",
            19,
            colors["text-secondary"],
        ),
        text(40, 128, "PREPARE ONCE", 15, colors["accent"], 700),
        text(40, 286, "PREDICT REPEATEDLY", 15, colors["accent"], 700),
    ]
    for x, y, w, label, detail in (
        (40, 146, 250, "Word vectors", "the finished encoder table"),
        (360, 146, 285, "Trained linear head", "apply its weights per word"),
        (755, 146, 305, "Word-score table", "ready for new texts"),
        (40, 307, 180, "New texts", "your input"),
        (285, 307, 245, "Look up scores", "for the words in each text"),
        (595, 307, 220, "Average + bias", "combine the word scores"),
        (880, 307, 180, "Labels", "your predictions"),
    ):
        parts += [
            f'<rect x="{x}" y="{y}" width="{w}" height="85" rx="12" fill="{colors["bg-secondary"]}" stroke="{colors["border"]}"/>',
            text(x + w / 2, y + 35, label, 22, weight=700, anchor="middle"),
            text(x + w / 2, y + 62, detail, 16, colors["text-secondary"], anchor="middle"),
        ]
    for path in (
        "M305 188 H345",
        "M660 188 H740",
        "M235 350 H270",
        "M545 350 H580",
        "M830 350 H865",
        "M908 246 V271 H408 V292",
    ):
        parts += [f'<path d="{path}" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>']
    for x, y, down in (
        (345, 188, False),
        (740, 188, False),
        (270, 350, False),
        (580, 350, False),
        (865, 350, False),
        (408, 292, True),
    ):
        points = (
            f"M{x - 6} {y - 9} L{x} {y} L{x + 6} {y - 9}"
            if down
            else f"M{x - 9} {y - 6} L{x} {y} L{x - 9} {y + 6}"
        )
        parts += [f'<path d="{points}" fill="none" stroke="{colors["accent"]}" stroke-width="3"/>']
    parts += [
        text(
            40,
            438,
            "For linear heads trained on raw, unnormalized averages and predicting only class outputs.",
            17,
            colors["text-secondary"],
        ),
        text(
            40,
            468,
            "With fewer scores than vector values, each word contributes a shorter list of numbers.",
            17,
            colors["text-secondary"],
        ),
        text(
            40,
            498,
            "Standalone saving keeps this table. It is not another speed optimization.",
            17,
            colors["text-secondary"],
        ),
        text(
            40,
            546,
            "Classification speed: pending controlled measurements. No rate or speedup is claimed.",
            18,
            colors["accent"],
            700,
        ),
        "</g></svg>",
    ]
    (out / f"classification-folding-{theme}.svg").write_text("\n".join(parts) + "\n")

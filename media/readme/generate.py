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

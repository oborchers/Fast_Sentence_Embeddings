"""Small recipe figures using the maintainer's brand palette."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Source: brand-identity/colors/tokens.json at 648624ce9bed1d08cacd00e32d5a4bf8ca62f271.
PALETTES = {
    "light": {
        "bg": "#FAFAF8",
        "card": "#F0EBE3",
        "accent": "#2D6A4F",
        "text": "#1E1B18",
        "muted": "#4F463E",
        "highlight": "#D4E8DC",
    },
    "dark": {
        "bg": "#0F0D0A",
        "card": "#1C1916",
        "accent": "#5B9E7D",
        "text": "#E8E4DF",
        "muted": "#B0A9A0",
        "highlight": "#1A2E24",
    },
}
MEDIA = Path(__file__).resolve().parent / "media"
STORIES = {
    "00-fse-101": (
        "TEXTS BECOME SOMETHING YOU CAN COMPARE",
        ["Your sentences", "POTION + fse", "Rows of numbers", "Similar texts"],
        ["Words you can read", "A local encoder", "One vector per text", "Compare directions"],
    ),
    "01-sif-usif": (
        "PREPARE ONCE, ENCODE MANY TIMES",
        ["Training texts", "Weights + patterns", "Finished word table", "New sentences"],
        ["Keep test scores out", "Learn outside fse", "Fold changes into rows", "Encode with fse"],
    ),
    "02-linear-heads": (
        "SAME DATA, FOUR TRAINING METHODS",
        ["Labeled texts", "Train four methods", "Unseen test texts", "Saved classifier"],
        [
            "Reviews and news",
            "Compare alternatives",
            "Accuracy and macro-F1",
            "Use inside your app",
        ],
    ),
    "03-files-with-resume": (
        "A RESTART CONTINUES AFTER THE SAVED ROWS",
        ["Text file", "Small batches", "fse + disk sink", "Saved vectors"],
        ["One line, one row", "Bound the memory", "Save complete shards", "Resume at first gap"],
    ),
    "04-kafka-and-qdrant": (
        "SAVE FIRST, ACKNOWLEDGE SECOND, COPY WHEN READY",
        ["News queue", "POTION + fse", "Local disk store", "Qdrant search"],
        ["Ordered messages", "Make real vectors", "Survives a restart", "Replay stable IDs"],
    ),
    "05-folder-listener": (
        "ONLY FINISHED FILES ENTER THE STORE",
        ["Write .part file", "Rename to .txt", "Poll + encode", "Local vector store"],
        [
            "Still being written",
            "Ready and immutable",
            "Wait at missing files",
            "One file, one row",
        ],
    ),
    "06-integrations": (
        "ONE ENCODER, YOUR CHOICE OF TOOLS",
        ["Real articles", "Local fse encoder", "Your library", "Useful results"],
        ["Same texts everywhere", "POTION vectors", "Search, topics, tables", "Retrieve or group"],
    ),
    "07-local-compute-and-http": (
        "COMPUTATION AND TRANSFER ARE DIFFERENT COSTS",
        ["Text in your app", "Encode locally", "Or request HTTP", "Vector bytes back"],
        ["Already in memory", "Direct function call", "Request and response", "JSON transfer cost"],
    ),
    "08-folded-classifier": (
        "ONE TRAINED HEAD, TWO WAYS TO PREDICT",
        ["New text", "Average word scores", "Add the same bias", "Topic scores"],
        ["Same tokenizer", "Four values per word", "After averaging", "Same linear head"],
    ),
}


def folding_diagram(theme):
    colors = PALETTES[theme]
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=colors["bg"])
    ax.set(xlim=(0, 12), ylim=(0, 4))
    ax.axis("off")
    ax.text(
        0.4,
        3.65,
        "ONE TRAINED HEAD, TWO WAYS TO PREDICT",
        color=colors["accent"],
        fontsize=11,
        weight="bold",
    )
    ax.text(
        0.4,
        3.25,
        "Preparation: multiply each word row by the trained head weights once.",
        color=colors["muted"],
        fontsize=10,
    )
    rows = [
        (
            2.1,
            ["New text", "256-value mean", "Linear head + bias", "4 topic scores"],
            ["Separate path", "Average word vectors", "Multiply after averaging", "Choose a label"],
        ),
        (
            0.65,
            ["New text", "4-value mean", "Add the same bias", "4 topic scores"],
            ["Folded path", "Average prepared scores", "After averaging", "Choose a label"],
        ),
    ]
    for y, labels, details in rows:
        for i, (label, detail) in enumerate(zip(labels, details, strict=True)):
            x = 0.35 + i * 3
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    2.3,
                    0.65,
                    boxstyle="round,pad=0.12",
                    facecolor=colors["highlight"] if y < 1 else colors["card"],
                    edgecolor=colors["accent"],
                )
            )
            ax.text(
                x + 1.15,
                y + 0.33,
                label,
                ha="center",
                va="center",
                color=colors["text"],
                fontsize=11,
                weight="bold",
            )
            ax.text(x + 1.15, y - 0.35, detail, ha="center", color=colors["muted"], fontsize=10)
            if i < 3:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + 2.5, y + 0.33),
                        (x + 2.78, y + 0.33),
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color=colors["accent"],
                        linewidth=1.5,
                    )
                )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    path = MEDIA / f"08-folded-classifier-{theme}.svg"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if theme == "light":
        adaptive_svg(path, MEDIA / "08-folded-classifier.svg")


def transport_diagram(theme):
    colors = PALETTES[theme]
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=colors["bg"])
    ax.set(xlim=(0, 12), ylim=(0, 3.5))
    ax.axis("off")
    ax.text(
        0.4,
        3.1,
        "TWO SEPARATE COSTS: ENCODING AND TRANSPORT",
        color=colors["accent"],
        fontsize=11,
        weight="bold",
    )
    nodes = [
        (0.4, 1.3, "Text in your app", "Already in memory"),
        (4.4, 2.1, "Encode locally", "Direct function call"),
        (4.4, 0.5, "Request HTTP", "Separate measurement"),
        (8.4, 2.1, "Vector in memory", "No transfer needed"),
        (8.4, 0.5, "Fixed JSON vector", "No encoding on server"),
    ]
    for x, y, label, detail in nodes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.8,
                0.6,
                boxstyle="round,pad=0.12",
                facecolor=colors["card"],
                edgecolor=colors["accent"],
            )
        )
        ax.text(
            x + 1.4,
            y + 0.3,
            label,
            ha="center",
            va="center",
            color=colors["text"],
            fontsize=11,
            weight="bold",
        )
        ax.text(x + 1.4, y - 0.35, detail, ha="center", color=colors["muted"], fontsize=10)
    for start, end in [
        ((3.4, 1.7), (4.2, 2.4)),
        ((3.4, 1.5), (4.2, 0.8)),
        ((7.4, 2.4), (8.2, 2.4)),
        ((7.4, 0.8), (8.2, 0.8)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                color=colors["accent"],
                linewidth=1.5,
            )
        )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    path = MEDIA / f"07-local-compute-and-http-{theme}.svg"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if theme == "light":
        adaptive_svg(path, MEDIA / "07-local-compute-and-http.svg")


def recipe_icon(name):
    colors = PALETTES["light"]
    number = name[:2]
    source = MEDIA / f"{name}-icon.svg"
    source.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
<defs></defs><rect x="2" y="2" width="60" height="60" rx="14" fill="{colors["highlight"].lower()}"/>
<text x="32" y="42" text-anchor="middle" font-family="sans-serif" font-size="28" font-weight="bold" fill="{colors["accent"].lower()}">{number}</text></svg>''')
    adaptive_svg(source, source)


def diagram(name, theme):
    """Draw a self-contained flow readable on either GitHub background."""
    if name == "08-folded-classifier":
        return folding_diagram(theme)
    if name == "07-local-compute-and-http":
        return transport_diagram(theme)
    colors = PALETTES[theme]
    caption, labels, details = STORIES[name]
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(12, 2.65), facecolor=colors["bg"])
        ax.set(xlim=(0, 12), ylim=(0, 2.65))
        ax.axis("off")
        ax.text(0.4, 2.2, caption, color=colors["accent"], fontsize=11, weight="bold")
        for i, (label, detail) in enumerate(zip(labels, details, strict=True)):
            x = 0.35 + i * 3
            box = FancyBboxPatch(
                (x, 0.75),
                2.3,
                0.95,
                boxstyle="round,pad=0.12,rounding_size=0.12",
                facecolor=colors["highlight"] if i == 1 else colors["card"],
                edgecolor=colors["accent"],
                linewidth=1.2,
            )
            ax.add_patch(box)
            ax.text(
                x + 1.15,
                1.25,
                label,
                ha="center",
                va="center",
                color=colors["text"],
                fontsize=11.5,
                weight="bold",
            )
            ax.text(x + 1.15, 0.35, detail, ha="center", color=colors["muted"], fontsize=10)
            if i < 3:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + 2.5, 1.25),
                        (x + 2.78, 1.25),
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color=colors["accent"],
                        linewidth=1.5,
                    )
                )
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        MEDIA.mkdir(exist_ok=True)
        fig.savefig(MEDIA / f"{name}-{theme}.svg", facecolor=fig.get_facecolor())
        plt.close(fig)
        if theme == "light":
            adaptive_svg(MEDIA / f"{name}-light.svg", MEDIA / f"{name}.svg")


def adaptive_svg(source, target):
    """GitHub strips picture/source in ipynb, so let the SVG choose its own theme."""
    svg = source.read_text()
    for token, color in PALETTES["light"].items():
        svg = svg.replace(color.lower(), f"var(--recipe-{token})")
    rules = []
    for theme, colors in PALETTES.items():
        values = ";".join(f"--recipe-{key}:{value}" for key, value in colors.items())
        rule = f":root{{{values}}}"
        rules.append(rule if theme == "light" else f"@media(prefers-color-scheme:dark){{{rule}}}")
    style = '<style type="text/css">' + "".join(rules) + "</style>"
    svg = svg.replace("<defs>", "<defs>" + style, 1)
    target.write_text(svg)


def quality_chart(results):
    """Render comparable metrics without changing training or test data."""
    colors = PALETTES["light"]
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=colors["bg"])
        labels = ["fse + linear", "TF-IDF + linear", "Model2Vec", "fastText"]
        for ax, dataset, metric in zip(
            axes, ("SST-5", "AG News"), ("macro_f1", "accuracy"), strict=True
        ):
            rows = [r for r in results if r["dataset"] == dataset]
            values = [r[metric] for r in rows]
            ax.set_facecolor(colors["bg"])
            bars = ax.barh(labels, values, color=colors["accent"])
            ax.bar_label(bars, fmt="%.3f", padding=4, color=colors["text"])
            ax.set(xlim=(0, 1), title=f"{dataset}: {metric.replace('_', '-')} (higher is better)")
            ax.invert_yaxis()
            ax.tick_params(colors=colors["text"])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.tight_layout()
        plt.show()


def speed_chart(results):
    colors = PALETTES["light"]
    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=colors["bg"])
    ax.set_facecolor(colors["bg"])
    bars = ax.barh(
        [r["setting"] for r in results],
        [r["sentences_per_second"] for r in results],
        color=colors["accent"],
    )
    ax.bar_label(bars, fmt="%.0f", padding=5, color=colors["text"])
    ax.set_xlabel("Sentences per second, one everyday-use run")
    ax.set_xlim(0, max(r["sentences_per_second"] for r in results) * 1.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    for story in STORIES:
        recipe_icon(story)
        for variant in PALETTES:
            diagram(story, variant)

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
}


def diagram(name, theme):
    """Draw a self-contained flow readable on either GitHub background."""
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
        for variant in PALETTES:
            diagram(story, variant)

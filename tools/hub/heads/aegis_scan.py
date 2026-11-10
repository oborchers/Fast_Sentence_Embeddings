"""Scan deliverables, with an auditable exception for pinned source URL literals."""

from __future__ import annotations

import json
from pathlib import Path

# Only these five recipe-owned dataset/license URLs may supply exceptions.
# Freeze them before reading any prompt. Generic phrases, labels, examples and
# URLs obtained from dataset rows never extend this list.
SOURCE_URL_NAMES = ("train.json", "validation.json", "test.json", "aart.csv", "aart-LICENSE")
CONFIG = Path(__file__).with_name("aegis-prompts.json")


def compact_json(path):
    """Change presentation only; canonical roundtrips preserve types and float bits."""
    original = json.loads(path.read_text())
    encoded = json.dumps(original, separators=(",", ":"), allow_nan=False) + "\n"
    restored = json.loads(encoded)
    canonical = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    if json.dumps(original, **canonical) != json.dumps(restored, **canonical):
        raise ValueError("JSON compaction changed decoded values")
    path.write_text(encoded)


def json_strings(value):
    """Include keys and values, so JSON escaping cannot conceal copied text."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, member in value.items():
            yield key
            yield from json_strings(member)
    elif isinstance(value, list):
        for member in value:
            yield from json_strings(member)


def document_texts(documents):
    for name, text in documents.items():
        yield name, text
        if Path(name).suffix == ".json":
            for index, value in enumerate(json_strings(json.loads(text))):
                # Numeric positions identify our decoded strings without copying
                # potentially rejected prompt text into diagnostic filenames.
                yield f"{name}#decoded-string-{index}", value


def document_windows(documents):
    windows = {}
    for name, text in document_texts(documents):
        for offset in range(max(0, len(text) - 29)):
            window = text[offset : offset + 30]
            windows.setdefault(window, []).append({"file": name, "offset": offset})
    return windows


def matches(prompt, windows):
    """Return only text originating in our documents, never source prompt slices."""
    return {
        window
        for offset in range(max(0, len(prompt) - 29))
        if (window := prompt[offset : offset + 30]) in windows
    }


def scan_rows(rows, split, windows, literals, report):
    for index, row in enumerate(rows):
        for window in sorted(matches(row["prompt"], windows)):
            literal_names = [name for name, literal in literals.items() if window in literal]
            entry = {
                "split": split,
                "row_index_zero_based": index,
                "window_from_deliverable": window,
                "occurrences": windows[window],
            }
            if literal_names:
                entry["own_source_url_literals"] = literal_names
                report["ignored"].append(entry)
            else:
                report["rejected"].append(entry)


def text_scan(work, documents, *, config_path=CONFIG, diagnosis=None):
    """Reject every shared 30-character window except the named own URL literals."""
    config = json.loads(config_path.read_text())
    literals = {name: config["sources"][name]["url"] for name in SOURCE_URL_NAMES}
    windows = document_windows(documents)
    report = {
        "window_characters": 30,
        "exception": "windows wholly contained in pinned recipe-owned dataset/license URLs",
        "own_source_url_literals": literals,
        "ignored": [],
        "rejected": [],
    }
    for split in ("train", "validation", "test"):
        rows = json.loads((work / "raw" / (split + ".json")).read_text())
        scan_rows(rows, split, windows, literals, report)
    if diagnosis is not None:
        diagnosis.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    if report["rejected"]:
        raise ValueError("source prompt substring found in a deliverable")
    return report

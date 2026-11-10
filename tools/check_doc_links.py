"""Check tracked Markdown and notebook references without network access."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

from doc_paths import path_error
from doc_references import markdown_references, notebook_references


def tracked_paths(root):
    data = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    return sorted({part.decode("utf-8") for part in data.split(b"\0") if part})


def inline_examples(root):
    config = root / "pyproject.toml"
    if not config.exists():
        return {}
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    entries = data.get("tool", {}).get("fse", {}).get("doc-links", {}).get("inline-examples", [])
    examples = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"document", "target", "reason"}:
            raise ValueError("inline examples require document, target and reason")
        if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
            raise ValueError("inline examples require nonempty strings")
        key = (entry["document"], entry["target"])
        if key in examples:
            raise ValueError(f"duplicate inline example: {key}")
        examples[key] = entry["reason"]
    return examples


def reference_errors(root, name, tracked, roots, examples, used):
    document = root / name
    text = document.read_text(encoding="utf-8")
    parse = notebook_references if document.suffix == ".ipynb" else markdown_references
    errors = []
    for reference in parse(text):
        key = (name, reference.target)
        if reference.inline and key in examples:
            used.add(key)
            continue
        error = path_error(root, document, reference, tracked, roots)
        if error:
            cell = f":cell {reference.cell}" if reference.cell else ""
            errors.append(f"{name}{cell}:{reference.line}: {error}")
    return errors


def check(root, tracked):
    root = root.resolve()
    roots = {path.split("/", 1)[0] for path in tracked if "/" in path}
    errors = []
    examples = inline_examples(root)
    used = set()
    for name in tracked:
        if Path(name).suffix not in {".md", ".ipynb"}:
            continue
        try:
            errors.extend(reference_errors(root, name, tracked, roots, examples, used))
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"{name}: {error}")
    for document, target in sorted(examples.keys() - used):
        errors.append(f"{document}: unused inline example: {target}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    try:
        errors = check(args.root, tracked_paths(args.root))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"documentation inventory failed: {error}", file=sys.stderr)
        return 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Documentation references: all local targets exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

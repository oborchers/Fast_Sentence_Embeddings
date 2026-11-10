"""Require a new fragment for library and selected package changes in a committed PR."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def changed_paths(root, base, *, added=False):
    flags = ["--diff-filter=A"] if added else []
    raw = git(root, "diff", "--name-only", "--no-renames", "-z", *flags, base, "HEAD")
    return [path.decode("utf-8") for path in raw.split(b"\0") if path]


def package_tables(document):
    tool = document.get("tool", {})
    return [
        document.get("build-system", {}),
        document.get("project", {}),
        tool.get("setuptools", {}),
        tool.get("setuptools_scm", {}),
    ]


def package_changed(root, base):
    try:
        before = git(root, "show", f"{base}:pyproject.toml").decode("utf-8")
    except subprocess.CalledProcessError:
        return True
    path = root / "pyproject.toml"
    if not path.exists():
        return True
    return package_tables(tomllib.loads(before)) != package_tables(tomllib.loads(path.read_text()))


def needs_fragment(root, base, paths):
    if any(path.startswith("src/") or path in {"setup.py", "MANIFEST.in"} for path in paths):
        return True
    return "pyproject.toml" in paths and package_changed(root, base)


def require_new_fragment(root, base):
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["towncrier"]
    types = {entry["directory"] for entry in config["type"]}
    for name in changed_paths(root, base, added=True):
        path = Path(name)
        match = re.fullmatch(r"DR-[0-9]+\.([a-z]+)\.md", path.name)
        if path.parent == Path(config["directory"]) and match and match[1] in types:
            fragment = root / path
            if not fragment.is_symlink() and fragment.is_file() and fragment.read_text().strip():
                return
    raise ValueError(
        "library/package changes require a new nonempty changelog.d/DR-<nnn>.<type>.md"
    )


def check(root, base):
    # Fail on an unknown base even for otherwise exempt changes.
    base = git(root, "merge-base", base, "HEAD").decode().strip()
    paths = changed_paths(root, base)
    if not needs_fragment(root, base, paths):
        print("No library or selected package changes; automatic fragment requirement skipped.")
        return
    # Towncrier alone skips its requirement when CHANGELOG.md changes. Keep the
    # explicit new-fragment check so editing the changelog cannot bypass this gate.
    require_new_fragment(root, base)
    subprocess.run(
        [sys.executable, "-m", "towncrier", "check", "--compare-with", base],
        cwd=root,
        check=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="PR base ref, resolved to its merge base")
    args = parser.parse_args(argv)
    try:
        check(Path.cwd(), args.base)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

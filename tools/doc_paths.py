"""Resolve local documentation targets inside one checkout."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

# These are syntactic examples, not repository references. Commands are not paths.
EXAMPLE = re.compile(r"[<>*{}\[\]?]|\$|^~|^/|^[A-Za-z]:")
SUFFIXES = {
    ".py",
    ".pyx",
    ".pxd",
    ".pxi",
    ".h",
    ".md",
    ".ipynb",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".sh",
}
ROOT_FILES = {"Makefile", "LICENSE", "NOTICE", "MANIFEST.in"}
# Keep recognizing a repository directory after its last tracked file is deleted.
REPOSITORY_ROOTS = {
    "src",
    "tests",
    "tools",
    "benchmarks",
    "docs",
    "recipes",
    "integrations",
    ".github",
    ".agents",
    ".claude",
    "media",
}


def inline_path(target, roots):
    if not target or target in {"./", "../"} or EXAMPLE.search(target) or "://" in target:
        return False
    if "/.../" in target:
        return False
    path = Path(target.split("::", 1)[0])
    if target.startswith(("./", "../")):
        return True
    if "/" in target:
        return target.split("/", 1)[0] in roots
    return not any(character.isspace() for character in target) and (
        path.suffix in SUFFIXES or target in ROOT_FILES
    )


def local_target(reference):
    target = reference.target.split("::", 1)[0] if reference.inline else reference.target
    if target.startswith("attachment:"):
        name = unquote(target.removeprefix("attachment:"))
        return None, None if name in reference.attachments else f"missing cell attachment: {target}"
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None, None
    return unquote(parsed.path), None


def candidates(root, document, target, inline, tracked):
    if target.startswith("/"):
        return [root / target.lstrip("/")]
    paths = [document.parent / target]
    if not inline or target.startswith(("./", "../")):
        return paths
    if "/" in target:
        return [root / target]
    paths.append(root / target)
    if "/" not in target:
        matches = [root / path for path in tracked if Path(path).name == target]
        if len(matches) == 1:
            paths.extend(matches)
    return paths


def path_error(root, document, reference, tracked, roots):
    if reference.inline and not inline_path(reference.target, roots):
        return None
    target, error = local_target(reference)
    if error or target is None:
        return error
    paths = candidates(root, document, target, reference.inline, tracked)
    resolved = [path.resolve() for path in paths]
    if any(not path.is_relative_to(root) for path in resolved):
        return f"target escapes checkout: {reference.target}"
    if any(path.exists() for path in resolved):
        return None
    return f"missing local target: {reference.target}"

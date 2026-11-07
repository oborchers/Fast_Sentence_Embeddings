"""Check a release tag against CHANGELOG.md and the built files, and write the release notes."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

TAG = re.compile(
    r"v(?P<version>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?)"
)
PRERELEASE = re.compile(r".*(?:a|b|rc)[0-9]+")
HEADING = re.compile(r"## \[(?P<name>[^\]]+)\]")
FENCE = re.compile(r" {0,3}(?P<fence>`{3,}|~{3,})")


def version_of(tag):
    """Return the version of a release tag `vX.Y.Z` (or `vX.Y.ZaN`, `bN`, `rcN`)."""
    match = TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"tag {tag!r} is not of the form vX.Y.Z or vX.Y.Z{{a,b,rc}}N")
    return match["version"]


def is_prerelease(version):
    return PRERELEASE.fullmatch(version) is not None


def headings_outside_code(changelog):
    """Return the section headings that are not inside a fenced code block."""
    found = []
    fence = None
    position = 0
    for line in changelog.splitlines(keepends=True):
        marker = FENCE.match(line)
        if marker is not None:
            if fence is None:
                fence = marker["fence"]
            elif marker["fence"][0] == fence[0] and len(marker["fence"]) >= len(fence):
                fence = None
        elif fence is None:
            heading = HEADING.match(line)
            if heading is not None:
                found.append((heading["name"], position, position + len(heading[0])))
        position += len(line)
    return found


def section(changelog, version):
    """Return the body of the section `## [version]`, without its heading and outer blank lines."""
    headings = headings_outside_code(changelog)
    matches = [i for i, found in enumerate(headings) if found[0] == version]
    if not matches:
        raise ValueError(f"CHANGELOG.md has no section '## [{version}]'")
    if len(matches) > 1:
        raise ValueError(f"CHANGELOG.md has more than one section '## [{version}]'")
    index = matches[0]
    end = headings[index + 1][1] if index + 1 < len(headings) else len(changelog)
    body = changelog[headings[index][2] : end]
    body = body.partition("\n")[2].strip()
    if not body:
        raise ValueError(f"the CHANGELOG.md section '## [{version}]' is empty")
    return body


def check_files(directory, version):
    """Require built files, and that every one of them carries the version of the tag."""
    files = sorted(path for path in Path(directory).iterdir() if path.is_file())
    if not files:
        raise ValueError(f"no built files in {directory}")
    pattern = re.compile(rf"^fse-{re.escape(version)}(?:-|\.tar\.gz$)")
    wrong = [path.name for path in files if pattern.match(path.name) is None]
    if wrong:
        raise ValueError(f"files do not belong to version {version}: {', '.join(wrong)}")
    return [path.name for path in files]


def prepare(tag, changelog_path, output_path, dist=None):
    version = version_of(tag)
    body = section(Path(changelog_path).read_text(encoding="utf-8"), version)
    if dist is not None:
        check_files(dist, version)
    Path(output_path).write_text(body + "\n", encoding="utf-8")
    return version, is_prerelease(version)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--changelog", default="CHANGELOG.md")
    parser.add_argument("--output", required=True, help="file for the release notes")
    parser.add_argument("--dist", help="directory of the built files, checked against the tag")
    args = parser.parse_args(argv)
    try:
        version, prerelease = prepare(args.tag, args.changelog, args.output, args.dist)
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"Release notes for {version} written to {args.output} (pre-release: {prerelease}).")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a") as stream:
            stream.write(f"version={version}\nprerelease={str(prerelease).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

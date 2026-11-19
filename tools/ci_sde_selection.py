"""Select expensive ISA emulation only for changes to its native/build inputs."""

from __future__ import annotations

import subprocess
import sys
import tomllib

PREFIXES = (
    "src/fse/_native",
    ".github/workflows/",
    "tests/test_native",
    "tests/test_conformance",
)
FILES = {
    "setup.py",
    "tools/check_sde.py",
    "tools/sde_shard.py",
    "tests/test_sde_shard.py",
    "tools/check_isa_isolation.py",
    "tools/wheel_smoke.py",
    "tools/ci_sde_selection.py",
    "tests/test_sde_evidence.py",
    "tests/test_isa_isolation.py",
    "tests/support/conformance.py",
    "tests/support/tiles.py",
    "benchmarks/config/avx512-layout-b-diagnostic.patch",
}


def build_settings(config):
    tool = config.get("tool", {})
    return [config.get("build-system", {})] + [
        tool.get(name, {}) for name in ("setuptools", "setuptools_scm", "cibuildwheel")
    ]


def selected(paths, before=None, after=None):
    if any(path.startswith(PREFIXES) or path in FILES for path in paths):
        return True
    if "pyproject.toml" not in paths:
        return False
    return before is None or after is None or build_settings(before) != build_settings(after)


def from_git(base):
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, "HEAD"], text=True
    ).split("\0")
    if "pyproject.toml" not in paths:
        return selected(paths)
    try:
        before, after = (
            tomllib.loads(
                subprocess.check_output(["git", "show", f"{ref}:pyproject.toml"], text=True)
            )
            for ref in (base, "HEAD")
        )
    except (subprocess.CalledProcessError, tomllib.TOMLDecodeError):
        return True
    return selected(paths, before, after)


def main():
    print(str(from_git(sys.argv[1]) if sys.argv[1:] else True).lower())


if __name__ == "__main__":
    main()

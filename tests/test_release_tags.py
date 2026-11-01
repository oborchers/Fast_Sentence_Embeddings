"""Package release tags cannot become the version of the core distribution."""

import subprocess
import tomllib
from pathlib import Path

import pytest


@pytest.mark.parametrize("package", ["langchain-fse", "llama-index-embeddings-fse", "fse-haystack"])
def test_core_version_ignores_integration_tags(tmp_path, package):
    config_path = Path(__file__).parents[1] / "pyproject.toml"
    with config_path.open("rb") as file:
        command = tomllib.load(file)["tool"]["setuptools_scm"]["scm"]["git"]["describe_command"]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    # Tags exist only in this throwaway repository, never in the project checkout.
    git("init", "--initial-branch=master")
    git("config", "user.name", "Release test")
    git("config", "user.email", "test@example.invalid")
    git("commit", "--allow-empty", "-m", "Core release")
    git("tag", "v2.0.0")
    git("commit", "--allow-empty", "-m", "Adapter release")
    git("tag", f"{package}-v0.1.0")
    # The unfiltered command proves the latest adapter tag could otherwise win.
    assert git("describe", "--tags", "--long").startswith(f"{package}-v0.1.0-0-")
    assert git(*command[1:]).startswith("v2.0.0-1-")
